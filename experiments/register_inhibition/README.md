# Directional subspace control of speaking register (Qwen2.5 / Qwen3.5)

**Work in progress.** Scope and limitations are stated up front. This is the *register-inhibition* track
of the [Inhibition Gap](https://github.com/foxontheerun/Inhibition_Mask) program: an empirical study of
controlling an injected speaking *register* (case study: a "pirate" persona) by acting on a low-rank,
contrastively-estimated subspace in the residual stream — both **removing** it and, dynamically,
**gating it as a function of context**.

It ships the controls and the failure cases, not only the positive direction.

---

## Two tracks

### 1. Static — removal of an injected register (`static_register_ablation.ipynb`, executed)

A soft directional ablation `h ← h − λ·B Bᵀ h`, applied to the residual stream at every layer, suppresses
the target register's surface form. The controls show it is **not** a generic effect of removing *K*
dimensions:

| condition | probe Δ (diagnostic) | note |
|---|---|---|
| register subspace | **−0.94** | suppresses the register |
| formal-register subspace (same pipeline) | −0.03 | does **not** suppress |
| matched-rank random subspace | +0.04 | does **not** suppress |
| capability (neutral factual prompts) | — | answers preserved (Tokyo) |

The −0.94 is a **diagnostic probe** Δ, not a clean behavioural effect size: the model-independent lexical
metric is confounded by reversion (Limitation 2) and the direction is not persona-specific (Limitation 1) —
read it together with those.

**Why residual-stream ablation (methods rationale).** The intervention acts on the residual stream, not on
attention weights, which makes it **architecture-agnostic by construction** — it acts identically on softmax
and linear-attention layers (demonstrated on one hybrid model, not yet a cross-architecture measurement).
The variant head-to-head (in the dynamic notebook) confirms this matters:
- **4.1 token-scale (Hadamard)** degrades generation (capability → 0, degenerate output);
- **4.2 attention-logit mask** is bound to softmax and behaves all-or-nothing (suppresses only near λ=1, then derails);
- **4.3 projection** is graded in λ, architecture-agnostic, capability-preserving — the recommended actuator.

Cross-model **scripts** (runnable on Kaggle T4, **not yet executed in this folder**):
`crossmodel_qwen25.ipynb`, `crossmodel_llama.ipynb`. The executed second-model evidence currently lives in
`dynamic_qwen7b_variants.ipynb` (Qwen2.5-7B), which reproduces the selective-suppression direction.

### 2. Dynamic — context-conditioned gating

A between-turn *sensor → regulator → actuator* loop: a signed probe measures how far a draft has drifted
into the register; the regulator sets the mask strength λ from that signal; the actuator regenerates under
the projection mask. Across a chat chain λ visibly **breathes** — it tightens on a register-laden turn and
relaxes (λ=0) on a neutral one. `dynamic_qwen7b_variants.ipynb` (executed) additionally runs the 4.1/4.2/4.3
head-to-head and multi-turn adaptive chains on Qwen2.5-7B; `dynamic_adaptive_breathing.ipynb` is the minimal
mean-driven version (runnable).

**Dispersion homeostat (`dynamic_dispersion_homeostat.ipynb`, executed).** This is the formalization's §5.2
rule realized: hardness driven by the **dispersion** of per-branch alignment, `τ ∝ Var(σ)`, so a
heterogeneous (drifting) context tightens the mask. On a multi-turn drift attack, the **variance** carries
the whole signal — the *mean* drift stays ≈0/negative, so a mean-driven controller would never fire — λ
ramps with `D_σ` (0 → 0.13 → 0.40 → 0.90), and removal holds the assistant **neutral and coherent**
(`lex → 0`) while the unmasked baseline drifts into the register. A worst-branch term (v0.7 §5) makes a
single strongly-off-goal segment bite before variance accumulates. This is removal (subtraction), the
program's target operation, and the first empirical run of the §5.2 homeostat.

---

## Limitations (read these first)

1. **Register, not persona.** The estimated direction captures a broad *expressive register*, not a
   specific persona: the "pirate" direction suppresses Shakespearean and noir registers at least as
   strongly. Separability of a persona-specific residual is **untested with a model-independent metric**.
2. **Metric confound.** The model-independent lexical-marker count is confounded: a large fraction of
   ablated outputs revert to an assistant-identity disclaimer ("I am an AI…"), which contains zero markers
   regardless of content — part of the drop is "stopped role-playing," not graded register removal. The
   honest number is the lexical count on the non-reversion subset.
3. **Actuator works, online detection does not (yet).** A single direction is an effective *actuator*
   (intervention), but as an *online* drift *sensor* during single-token generation it washes out
   (per-position variance is diluted/undefined with a KV cache). The working sensor here is **between-turn**
   (on a completed draft), not within-generation. Robust online detection is the open problem.
4. **The dynamic results are demos, not measurements.** Single chains, one model per run, greedy decoding,
   no held-out set or statistics. They illustrate the mechanism; they do not establish an effect size.
5. **Scope.** Small open models, single register, free-tier GPUs.

---

## Files

| file | what | status |
|---|---|---|
| `inhibition_mask.py` | subspace estimation, signed probe, projection ablation, generation | source |
| `data.py` | contrastive pairs and evaluation prompts | source |
| `metrics.py` | model-independent lexical / reversion counters, Cohen's d, clustered CI | source |
| `requirements.txt` | pinned dependencies | source |
| `static_register_ablation.ipynb` | end-to-end removal + concept-specificity + random + capability + dose-response | **executed** |
| `crossmodel_qwen25.ipynb`, `crossmodel_llama.ipynb` | cross-model replications | runnable (Kaggle T4) |
| `dynamic_qwen7b_variants.ipynb` | 4.1/4.2/4.3 head-to-head + adaptive multi-turn chains (Qwen2.5-7B) | **executed** |
| `dynamic_adaptive_breathing.ipynb` | minimal between-turn breathing controller (mean-driven, cross-register chain) | runnable (run to view λ) |
| `dynamic_dispersion_homeostat.ipynb` | §5.2 homeostat: per-branch dispersion `D_σ` → λ → register ablation; multi-turn drift resistance | **executed** |
| `Inhibition_Mask_Formalization_v0.6.md` | formalization: adds §4.3 projection ablation, §5 scope corridor + harm gate, control-theory reading, dispersion homeostat | doc |

## Reproduce

Free Colab/Kaggle T4. Open a notebook; the setup cells (`%%writefile`) recreate the modules, so only the
notebook need be uploaded. `Runtime → Run all`. Set `MODEL_ID` to the chat/instruct checkpoint.

## Relation to the program

This is the empirical companion to **The Inhibition Gap** (the position paper) and the **v0.6 formalization**.
The paper's *target* mechanism is *active suppression of the influence of off-goal context elements,
conditioned on a stable goal* — suppression at the attention level, explicitly distinct from
amplification/steering.

**Scope honesty.** The actuator exercised here — residual-stream directional projection `h ← h − λ·BBᵀh` — is
mechanically the *difference-in-means / contrastive-direction* family (the activation-steering /
refusal-ablation lineage), **not** the paper's attention-level, context-conditioned suppression. Subtracting
an attribute direction from the model's *own* residual is a different operation than suppressing the
*influence of an off-goal context segment* conditioned on a stable goal. What this track contributes is the
**conditioning** side: estimating a stable low-rank subspace, gating its removal on a between-turn drift
signal (dynamic track), and the 4.1/4.2/4.3 variant comparison. The paper's distinctive mechanism is **not
yet implemented** — that remains the gap; the §4.3 / §5 constructs in v0.6 are the bridge under construction.

## License

Code (notebooks, scripts): Apache 2.0. Documents (formalization, README): CC BY 4.0.

> Bulatova, A. (2026). *Inhibition Mask — register-inhibition track.* GitHub: foxontheerun/Inhibition_Mask.
