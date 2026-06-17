"""Directional inhibition mask for suppressing an injected persona in a transformer LM.

Pipeline
--------
1. Contrastive *minimal* pairs (a sentence in neutral style ↔ the same sentence in
   the target persona's style) isolate the persona direction, with content cancelling
   inside each pair.
2. Per layer, the difference-of-means over pairs is decomposed with an SVD into a
   K-dimensional ``role subspace`` 𝒢.
3. 𝒢 is orthogonalized against the chat-template's *structural* token directions so
   that the mask does not touch scaffolding tokens.
4. The mask removes the projection onto 𝒢 from the residual stream at every layer:
   ``h <- h - lam * B Bᵀ h`` (optionally gated: only where the projection is large).

A signed, centered probe (``RoleSensor``) is used as a *diagnostic* metric. Its
direction nearly coincides with the masked subspace, so the behavioural claim rests
on independent checks (a surface-lexicon counter and a concept-specificity control),
not on the probe value. See README for the calibrated claim and limitations.

Assumes a HuggingFace decoder exposing ``model.model.layers`` (Qwen/Llama style).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------- #
# Activation capture
# --------------------------------------------------------------------------- #
def _chat_wrap(tokenizer, text: str):
    """Wrap ``text`` as a single user turn using the model's chat template.

    ``enable_thinking`` is Qwen-specific; templates that don't accept it (e.g. Llama)
    raise ``TypeError``, in which case we retry without it. Only if no chat template
    exists at all do we fall back to the raw text.
    """
    msgs = [{"role": "user", "content": text}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return text


def to_inputs(tokenizer, text: str, device, chat: bool = True):
    rendered = _chat_wrap(tokenizer, text) if chat else text
    return tokenizer(rendered, return_tensors="pt").to(device)


@torch.no_grad()
def capture(model, inputs, layers, pool: bool = True):
    """Return ``{layer: activation}`` for the given decoder layers.

    ``pool=True``  -> mean over tokens, shape ``(d,)``.
    ``pool=False`` -> per token, shape ``(seq, d)``.
    """
    store: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.detach()[0].float()
            store[layer_idx] = h.mean(0) if pool else h
        return hook

    handles = [model.model.layers[L].register_forward_hook(make_hook(L)) for L in layers]
    try:
        model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return store


def mean_pool(model, tokenizer, text, layers, device, chat: bool = True):
    return capture(model, to_inputs(tokenizer, text, device, chat=chat), layers, pool=True)


# --------------------------------------------------------------------------- #
# Role subspace 𝒢
# --------------------------------------------------------------------------- #
@dataclass
class RoleSubspace:
    """Per-layer orthonormal bases of the role subspace. ``B[L]`` is ``(d, K)``."""
    B: dict[int, torch.Tensor]
    layers: list[int]
    K: int

    def projection_magnitude(self, model, tokenizer, text, device) -> "torch.Tensor":
        """Per-token ``‖Bᵀa‖`` averaged over layers (how much each token loads on 𝒢)."""
        caps = capture(model, to_inputs(tokenizer, text, device, chat=False), self.layers, pool=False)
        seq = caps[self.layers[0]].shape[0]
        acc = torch.zeros(seq, device=caps[self.layers[0]].device)
        for L in self.layers:
            acc = acc + (caps[L] @ self.B[L]).norm(dim=-1)
        return (acc / len(self.layers)).cpu()


def _structural_directions(model, tokenizer, texts, layers, device, special_ids, r_struct):
    """SVD bases of chat-template structural-token activations, per layer."""
    block = {"assistant", "user", "system", "<think>", "</think>", ""}
    per_layer: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    for t in texts:
        ids = to_inputs(tokenizer, t, device, chat=True)
        toks = ids["input_ids"][0].tolist()
        pos = [i for i, tk in enumerate(toks)
               if (tk in special_ids) or (tokenizer.decode([tk]).strip() in block)]
        caps = capture(model, ids, layers, pool=False)
        for L in layers:
            per_layer[L].extend(caps[L][i] for i in pos)
    out = {}
    for L in layers:
        S = torch.stack(per_layer[L])
        _, _, Vh = torch.linalg.svd(S, full_matrices=False)
        out[L] = Vh[:r_struct].T.contiguous()
    return out


def build_subspace(model, tokenizer, pairs, layers, device, *, K=4,
                   structural_texts=None, r_struct=12) -> RoleSubspace:
    """Build the role subspace from contrastive (neutral, persona) pairs.

    Pairs are wrapped in the chat template so that structural tokens cancel inside
    each difference; the subspace is then orthogonalized against the structural
    directions estimated from ``structural_texts`` (neutral prompts).
    """
    diffs: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
    for neutral, persona in pairs:
        cn = mean_pool(model, tokenizer, neutral, layers, device, chat=True)
        cp = mean_pool(model, tokenizer, persona, layers, device, chat=True)
        for L in layers:
            diffs[L].append(cp[L] - cn[L])

    struct = None
    if structural_texts:
        special_ids = set(tokenizer.all_special_ids)
        struct = _structural_directions(model, tokenizer, structural_texts, layers,
                                        device, special_ids, r_struct)

    B = {}
    for L in layers:
        _, _, Vh = torch.linalg.svd(torch.stack(diffs[L]), full_matrices=False)
        basis = Vh[:K].T.contiguous()
        if struct is not None:
            Q = struct[L]
            basis = basis - Q @ (Q.T @ basis)        # remove structural component
            basis, _ = torch.linalg.qr(basis)        # re-orthonormalize
        B[L] = basis.to(device).float()
    return RoleSubspace(B=B, layers=list(layers), K=K)


def random_subspace(reference: RoleSubspace, device, seed=0) -> RoleSubspace:
    """Matched-rank random orthonormal subspaces (a control for the projection op)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = {}
    for L, M in reference.B.items():
        d, K = M.shape
        rand = torch.randn(d, K, generator=g).to(device)
        Q, _ = torch.linalg.qr(rand)
        B[L] = Q[:, :K].contiguous().float()
    return RoleSubspace(B=B, layers=reference.layers, K=reference.K)


def orthogonalize(subspace: RoleSubspace, against: RoleSubspace) -> RoleSubspace:
    """Remove ``against`` from ``subspace`` per layer (e.g. pirate minus generic colorfulness).

    Yields the part of ``subspace`` that is orthogonal to ``against`` — used to ask whether
    the persona has a component separable from a confounding concept (vivid writing).
    """
    B = {}
    for L in subspace.layers:
        basis = subspace.B[L]
        Q = against.B[L]
        basis = basis - Q @ (Q.T @ basis)
        basis, _ = torch.linalg.qr(basis)
        B[L] = basis.to(basis.device).float()
    return RoleSubspace(B=B, layers=subspace.layers, K=subspace.K)


# --------------------------------------------------------------------------- #
# Diagnostic probe (home-field metric)
# --------------------------------------------------------------------------- #
@dataclass
class RoleSensor:
    layer: int
    mu: torch.Tensor          # neutral reference (common-mode), (d,)
    direction: torch.Tensor   # unit role direction, (d,)

    def score(self, model, tokenizer, text, device) -> float:
        if not text.strip():
            return 0.0
        a = mean_pool(model, tokenizer, text, [self.layer], device, chat=False)[self.layer]
        return torch.dot(self.direction, a.to(self.direction.device) - self.mu).item()


def build_sensor(model, tokenizer, pairs, layer, device) -> RoleSensor:
    """Signed, common-mode-centered probe at one layer (a *diagnostic*, not ground truth)."""
    neu = torch.stack([mean_pool(model, tokenizer, n, [layer], device, chat=False)[layer]
                       for n, _ in pairs])
    pir = torch.stack([mean_pool(model, tokenizer, p, [layer], device, chat=False)[layer]
                       for _, p in pairs])
    mu = neu.mean(0).to(device)
    direction = (pir - neu).mean(0)
    direction = (direction / direction.norm()).to(device)
    return RoleSensor(layer=layer, mu=mu, direction=direction)


# --------------------------------------------------------------------------- #
# The mask
# --------------------------------------------------------------------------- #
def _projection_hook(B: torch.Tensor, lam: float, theta: float | None):
    def hook(_module, _inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        rest = out[1:] if is_tuple else ()
        x = h.float()
        coords = x @ B                       # (..., K)
        proj = coords @ B.T                  # (..., d)
        if theta is not None:                # gated: only act where projection is large
            gate = (coords.norm(dim=-1, keepdim=True) > theta).float()
            proj = gate * proj
        new = (x - lam * proj).to(h.dtype)
        return (new,) + rest if is_tuple else new
    return hook


@contextmanager
def mask(model, subspace: RoleSubspace, layers, *, lam: float = 1.0, theta: float | None = None):
    """Apply the projection mask on ``layers`` for the duration of the context."""
    handles = [model.model.layers[L].register_forward_hook(_projection_hook(subspace.B[L], lam, theta))
               for L in layers]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def generate(model, tokenizer, prompt, device, *, subspace=None, layers=None,
             lam=1.0, theta=None, max_new_tokens=40, do_sample=False, **gen_kwargs):
    inputs = to_inputs(tokenizer, prompt, device, chat=True)
    n = inputs["input_ids"].shape[1]
    ctx = mask(model, subspace, layers, lam=lam, theta=theta) if (subspace and layers) else _null()
    with ctx:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                             pad_token_id=tokenizer.eos_token_id, **gen_kwargs)
    return tokenizer.decode(out[0][n:], skip_special_tokens=True)


@contextmanager
def _null():
    yield
