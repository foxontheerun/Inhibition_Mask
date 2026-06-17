# Formalization of a Dynamic Inhibitory Mask

*Architectural draft of an inhibitory layer for transformers based on directional projections in the residual stream and precision-weighting in the active inference framework.*

*Version 0.6 — June 4, 2026.*

**Changes from v0.5.** This version develops six points raised against v0.5 and adds the application variant that the empirical runs selected:

1. **Two regimes of intervention made explicit** (§2): *token-level* scaling (v0.5's default) vs *direction-level* gating (the new default). The mask now modifies a component of an activation, not the whole position.
2. **§4.3 added — directional subspace projection ablation.** v0.5 §4 carried only Hadamard scaling (4.1) and attention-logit masking (4.2). The actuator that works across architectures — $h \leftarrow h - \lambda\, B B^\top h$ — was missing as a numbered variant. It is now §4.3 and is the recommended actuator.
3. **Homeostat promoted from optional extension to core mechanism** (§6), with an explicit **control-theory reading** (§7).
4. **Stability target stated precisely** (§8): the per-step contraction is proved as a lemma; the closed-loop convergence claim is isolated as the open theorem, with the exact condition to be shown.
5. **Adaptive threshold corrected** (§4.4): percentile relative to a *session baseline*, not to the current context (the v0.5 form collapses under regime-shift attacks).
6. **Admissibility criterion split** (§5): a mode-dependent *scope corridor* and a fixed *harm gate*, formalizing goal-conditioned scope restriction vs capability restriction.

A companion document (`формализация_v07_branch.md`) develops the **branch-level** detector (v0.7); v0.6 is the single-direction / per-token controller that the branch version generalizes. Empirical grounding for the choices here is summarized in §12 and drawn from `EXPERIMENT_2_STATUS.md` and the pirate-register runs.

---

## 1. Basic notation

- $\mathbf{G} \in \mathbb{R}^d$ — **goal anchor vector**, $\|\mathbf{G}\| = 1$. Single-direction case.
- $\mathcal{G} = \mathrm{span}(\mathbf{g}_1, \ldots, \mathbf{g}_K) \subset \mathbb{R}^d$ — **goal subspace** (multi-anchor; v0.5 §9.6). In v0.6 the subspace is the primary object and $\mathbf{G}$ is the special case $K = 1$. Let $B \in \mathbb{R}^{d \times K}$ hold an orthonormal basis of $\mathcal{G}$ (columns), so $P_{\mathcal{G}} = B B^\top$ is the orthogonal projector onto $\mathcal{G}$ and $P_\perp = I - P_{\mathcal{G}}$ the projector onto its complement.
- $\mathbf{a}_{t,i}^{(\ell)} \in \mathbb{R}^d$ — activation (hidden state) of token $i$ at layer $\ell$, time $t$. Where the indices are clear they are dropped.
- **Alignment of a single direction:** $\sigma_i = \dfrac{\mathbf{a}_i \cdot \mathbf{G}}{\|\mathbf{a}_i\|} \in [-1, 1]$.
- **Alignment with a subspace:** $\sigma_i = \dfrac{\|P_{\mathcal{G}} \mathbf{a}_i\|}{\|\mathbf{a}_i\|} \in [0, 1]$ — the fraction of the activation's norm lying in $\mathcal{G}$ (the cosine of the angle to the subspace). The two coincide at $K = 1$ up to sign.

The aim is a controller that decides, from $\{\sigma_i\}$, **when** and **how strongly** to act, and an actuator that modifies activations so as to keep the residual stream inside an admissibility region around $\mathcal{G}$.

---

## 2. Two regimes of intervention

v0.5 framed the mask as a per-token weight $M_i \in [0,1]$ multiplying the whole activation of position $i$. v0.6 separates two regimes, because the runs showed they behave very differently.

**Token-level (scaling).** $\mathbf{a}_i \leftarrow M_i \, \mathbf{a}_i$. The whole position is attenuated by a scalar; when $M_i \to 0$ the token's *entire* content is removed, including any goal-aligned part. This is v0.5's §4.1 and §4.2. It is simple but blunt: empirically it either destroys generation (4.1) or behaves as an all-or-nothing switch (4.2) — see §12.

**Direction-level (gating).** $\mathbf{a}_i \leftarrow \mathbf{a}_i - \lambda\, P_{B}\, \mathbf{a}_i$, i.e. only the component of $\mathbf{a}_i$ inside the targeted subspace is attenuated; the orthogonal content $P_\perp \mathbf{a}_i$ is preserved. The semantic content of the position survives; only its pull along the targeted axes is reduced. This is the new default (§4.3).

The design principle, carried over from `The_Inhibition_Gap`, is **"restrict scope, not capability" — prison, not the death penalty**: correct a token's *influence* on downstream computation rather than delete the token. Direction-level gating is the formal statement of that principle; token-level scaling is its degenerate, content-destroying limit.

---

## 3. Mask-shape functions (how alignment maps to action)

For the token-level regime and for setting the per-unit strength in the direction-level regime, we need a function $f: \sigma_i \mapsto m_i$ from alignment to action weight. The four shapes from v0.5 are retained; an implementation picks one by its requirements on interpretability, differentiability, and transition shape. The hardness parameter $\tau$ appears in all of them and is what the homeostat (§6) drives.

**3.1 Hard (binary).** $m_i = \mathbb{1}[\sigma_i \geq \theta]$. Non-differentiable (needs a straight-through estimator under end-to-end training).

**3.2 Soft sigmoidal.** $m_i = \mathrm{sigmoid}(\tau\,(\sigma_i - \theta))$. $\tau \to \infty$ recovers the hard mask; $\tau \to 0$ gives uniform $\approx 0.5$.

**3.3 Sparse ($\alpha$-entmax).** $\mathbf{m} = \alpha\text{-entmax}(\tau\,\boldsymbol{\sigma})$, exact zeros with differentiability (Peters et al., 2019). Rescale by $N$ for multiplicative use.

**3.4 Cone (admissibility corridor).** With $\sigma_{\text{safe}} = \cos(\alpha_{\text{safe}})$:
$$m_i = \begin{cases} 1, & \sigma_i \geq \sigma_{\text{safe}} \\[1mm] (\sigma_i / \sigma_{\text{safe}})^{\tau}, & 0 \leq \sigma_i < \sigma_{\text{safe}} \\[1mm] 0, & \sigma_i < 0 \end{cases}$$
Inside the cone the mask is inert; in the transition band it suppresses smoothly at rate $\tau$; below orthogonality it zeroes.

In the **direction-level** regime the per-unit strength is set as $\lambda_i = \lambda \cdot (1 - m_i)$ (act where alignment is low) or, in the controller of §6, $\lambda$ is a single scalar driven by an aggregate alignment signal.

---

## 4. Mask application

Three variants. v0.5 carried only 4.1 and 4.2; **4.3 is new and is the recommended actuator**.

### 4.1 Direct scaling of activations (Hadamard) — token-level

$$\tilde{\mathbf{A}}_t^{(\ell)} = \mathrm{diag}(\mathbf{m}_t)\, \mathbf{A}_t^{(\ell)}$$

Each row $i$ is multiplied by the scalar $m_i$. Blunt: scales the entire activation. Empirically degrades generation (§12).

### 4.2 Attention-logit masking — token-level, softmax-bound

$$\mathrm{Attention}_M(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + \log \mathbf{m}_t^\top\right) V$$

$m_j \to 0$ sends a token's attention contribution to exactly zero through the standard additive-mask path — no need to rewrite the model core. v0.5 marked this "preferred." Two limitations surfaced empirically (§12): it is **bound to softmax attention** and therefore does not transfer to linear-attention or hybrid layers (Mamba-/RWKV-style, Qwen-Next…), and on pure-softmax models it behaves **all-or-nothing** (suppresses only near $\lambda = 1$, with derailment). It is retained for pure-softmax settings where graded control is not required.

### 4.3 Directional subspace projection ablation (recommended) — direction-level

Let $B^{(\ell)} \in \mathbb{R}^{d \times K}$ be an orthonormal basis of the targeted subspace at layer $\ell$. The actuator acts on the **residual stream**:

$$\boxed{\;\mathbf{a}_i \leftarrow \mathbf{a}_i - \lambda\, B^{(\ell)} (B^{(\ell)})^\top \mathbf{a}_i = (I - \lambda\, P_{B^{(\ell)}})\, \mathbf{a}_i, \qquad \lambda \in [0, 1]\;}$$

- $\lambda = 1$: full removal of the component in $\mathrm{span}(B)$ (hard projection-out).
- $\lambda \in (0,1)$: **graded** attenuation — the property the other two variants lack.
- $K = 1$: $\mathbf{a}_i \leftarrow \mathbf{a}_i - \lambda\,(\mathbf{g}^\top \mathbf{a}_i)\,\mathbf{g}$ — single-direction gating (the v0.6 §2 direction-level operation).

**Why this is the variant that transfers.** The operator acts on the layer's *output activation* (the residual stream), not on attention weights. It therefore applies **identically to softmax and linear-attention layers**, which makes it architecture-agnostic — unlike 4.2, which is defined through the softmax over the token axis. In a hybrid model (linear-attention layers interleaved with periodic full attention) only 4.3 is uniform across both layer types.

**Why it preserves capability.** Only $P_B \mathbf{a}_i$ is touched; $P_\perp \mathbf{a}_i$ — everything orthogonal to the targeted subspace, including the factual/task content of the position — is left intact. This is the "prison, not death penalty" operation of §2, in contrast to 4.1's scaling of the whole vector.

**Subspace estimation.** $B^{(\ell)}$ is obtained per layer from contrastive minimal pairs (content held fixed, the targeted attribute varied), by difference-of-means reduced by SVD to rank $K$, then orthogonalized against chat-template structural-token directions so the actuator does not touch scaffolding tokens. This is the same difference-in-means / contrastive-activation-direction family as activation steering and single-direction refusal ablation, applied here to a goal/register subspace.

**Honest scope of 4.3 (from the runs).** Two caveats travel with this actuator and must be stated wherever it is used:
- *Not necessarily attribute-specific.* For a *register*/persona subspace the estimated direction captures a broad expressive axis; ablating a "pirate" subspace also suppresses Shakespearean and noir registers. Whether an attribute-specific residual survives removing a generic "vividness" subspace is open. For a *goal/honesty* subspace (Experiment 1) the direction is a real, causally effective actuator; its specificity is supported by matched random-direction controls.
- *Reversion is the correct ground state, not a graded effect.* Removing an injected persona returns the model to its base assistant identity ("I am an AI…"). This is the desired endpoint of suppression, but a surface-lexicon metric cannot by itself separate *graded subspace removal* from *identity reassertion*; report the lexicon on the non-reversion subset, and read the actual texts.

---

## 5. Admissibility: scope corridor and harm gate

v0.5 used one corridor around $\mathcal{G}$. That conflates two different things: departure from the **task** and entry into a **dangerous region**. A creative mode wants a wide task corridor, but widening a single corridor also permits dangerous departures. Scope breadth and harm prohibition are different axes; one threshold cannot separate them. v0.6 splits the test into two independent per-unit checks.

**5.1 Scope corridor (task).** With $\mathcal{G}_{\text{task}}$ the task subspace,
$$a^{\text{scope}}_i = \frac{\|P_{\perp\mathcal{G}_{\text{task}}}\, \mathbf{a}_i\|}{\|\mathbf{a}_i\|}, \qquad \text{admissible if } a^{\text{scope}}_i \leq a_{\text{safe}}^{\text{task}}.$$
The threshold $a_{\text{safe}}^{\text{task}}$ is **mode-dependent**: wide in a creative/exploratory mode, tight before a consequential action. It is governed by the hardness $\tau$ (§6), whose driver is extended to include the **consequence level** of the current output — free text lowers $\tau$ (wide corridor); a tool call / decision / external action raises $\tau$ (corridor tightens). This formalizes "free while writing, collected before acting."

**5.2 Harm gate (capability).** With $\mathcal{H}$ a harm subspace (directions of dangerous actions),
$$a^{\text{harm}}_i = \frac{\|P_{\mathcal{H}}\, \mathbf{a}_i\|}{\|\mathbf{a}_i\|}, \qquad \text{violated if } a^{\text{harm}}_i > h_{\max}.$$
The threshold $h_{\max}$ is **fixed and mode-independent**: the creative mode opens the scope corridor but never the harm gate. A unit triggers intervention if it fails **either** test. This is the formal record of goal-conditioned scope restriction vs capability restriction from `The_Inhibition_Gap`: creative breadth grows, the forbidden zone stays forbidden.

*Open:* extracting $\mathcal{H}$ — contrastive "dangerous action / safe action" pairs, difference-in-means; methodologically the same task as extracting $\mathcal{G}$.

**5.3 Adaptive threshold, corrected.** v0.5 §3.1 set $\sigma_{\text{safe}}$ as a percentile of the *current* context's $\{\sigma_i\}$. This fails under regime-shift (drift) attacks: after the context switches to a new persona/task, the $\sigma_i$ are homogeneously low, and a percentile of that flat distribution yields a threshold so low the mask stops firing (confirmed empirically). v0.6 takes the percentile relative to a **session baseline** — the distribution of $\sigma$ over the first $K$ steps of the session — and compares the current distribution to it. This fuses the adaptive logic with the homeostat memory of §6.

---

## 6. Dynamic homeostat (core mechanism)

In v0.5 the dynamic controller was an optional §5 extension. The runs showed a static mask (fixed $\sigma_{\text{safe}}$, fixed $\lambda$) does not handle drift attacks, where the context slides to a new task/persona and each individual position looks in-corridor relative to the *new* regime. Memory of a baseline is structurally required. The homeostat is therefore core in v0.6.

**6.1 Heterogeneity sensor (per-position, v0.5 form).** $D_\sigma(t) = \mathrm{Var}_i(\sigma_i(t))$. **Empirical caveat (Experiment 2):** as an *online, per-token* drift detector this is unreliable — averaging cosine over hundreds of positions washes out localized drift, and the variance is undefined during single-token KV-cache generation. The resultant $\cos(h_{\text{last}}, \mathbf{G})$ tested as an alternative is a context-length artifact, not a drift signal. Robust *online, localized* detection requires branch-level decomposition (companion v0.7). What does work as a controller at coarser granularity is the between-turn loop below.

**6.2 Realized controller (between-turn, draft-based).** The empirically working loop — *sensor → regulator → actuator* — operates at turn granularity:
$$
\underbrace{\sigma(t) = \frac{\|P_{\mathcal{G}}\,\bar{\mathbf{a}}(t)\|}{\|\bar{\mathbf{a}}(t)\|}}_{\text{sensor: on an unmasked draft, pooled over layers}}
\;\longrightarrow\;
\underbrace{\lambda(t) = \mathrm{clip}\!\big(\gamma\,(\sigma(t) - \theta),\, 0,\, \lambda_{\max}\big)}_{\text{regulator: deadband } \theta + \text{proportional gain } \gamma}
\;\longrightarrow\;
\underbrace{\text{regenerate under } (I - \lambda(t) P_B)}_{\text{actuator: §4.3}}
$$
Below the threshold $\theta$ the mask is released ($\lambda = 0$); above it, $\lambda$ rises with the excess alignment. On a chain of mixed turns $\lambda$ visibly "breathes" — it tightens on a role-laden turn and relaxes on a neutral one. ($\theta$ is the deadband/corridor of §3.4–§5; $\gamma$ the gain; $\lambda_{\max}$ the saturation guarding against an out-of-distribution collapse.)

**6.3 Hardness feedback (when a per-position mask is used).**
$$\tau_t = \mathrm{clip}\!\left(\tau_0 \cdot \exp\!\big(\gamma\,(D_\sigma(t) - D_{\text{baseline}})\big),\, \tau_{\min},\, \tau_{\max}\right)$$
with $D_{\text{baseline}}$ updated either by EMA (smooth, for gradually evolving sessions) or a sliding window of $W \sim 50$–$200$ steps (hard cutoff, for sessions with sharp phase transitions, where EMA sticks in an outdated baseline). $\tau_{\min} > 0$ guarantees a minimum inhibition pool even in a homogeneously hostile context; $\tau_{\max}$ prevents blowup.

---

## 7. Control-theory reading

The mechanism is a standard feedback loop; naming it removes the impression that the formalism is heuristic and opens it to the dynamical-systems audience.

| Control-theory object | Inhibition-mask analog |
|---|---|
| Reference / setpoint; attractor | goal subspace $\mathcal{G}$ |
| Process variable (measured output) | alignment $\sigma(t)$ |
| Deadband around setpoint | corridor $\theta$ / $a_{\text{safe}}$ |
| Error signal | $e(t) = \max(0,\, \sigma(t) - \theta)$ (or the deficit below corridor) |
| Proportional gain | $\gamma$ |
| Integral-like reference memory | $D_{\text{baseline}}$ (EMA / sliding window) |
| Actuator command / loop gain | $\lambda_t$ (or hardness $\tau_t$) |
| Negative-feedback actuation | projection mask $I - \lambda P_B$ |
| Actuator saturation / anti-windup | $\lambda_{\max}$, $\tau_{\max}$ |
| Over-actuation instability | collapse (§9) |

The controller of §6.2 is proportional with a deadband and saturation; adding $D_{\text{baseline}}$ memory makes it proportional-plus-integral-flavoured. The biological reading is unchanged (§9): $\mathcal{G}$ ≈ PFC goal representation, the mask ≈ GABA-ergic inhibition, $\tau$ ≈ tonic gain (dopaminergic), the heterogeneity signal ≈ ACC conflict monitoring.

---

## 8. Stability: what is proved and what is open

Treat the masked residual stream as a discrete dynamical system, alternating the model's forward map $F$ (attention + MLP + residual) with the actuator $T_\lambda = I - \lambda P_B$. Define the **energy in the targeted subspace** $V(\mathbf{h}) = \|P_B \mathbf{h}\|^2$.

**Lemma (per-application contraction).** For one application of the actuator, $P_B (T_\lambda \mathbf{h}) = (1-\lambda) P_B \mathbf{h}$, hence
$$V(T_\lambda \mathbf{h}) = (1-\lambda)^2\, V(\mathbf{h}) \leq V(\mathbf{h}), \quad \text{strict for } \lambda \in (0,1).$$
So the actuator in isolation is a strict contraction on $V$. *(This is elementary; it is stated to separate it cleanly from the claim that follows.)*

**Open theorem (closed-loop convergence).** The forward map re-injects energy into $\mathrm{span}(B)$: in general $V(F(\mathbf{h})) \geq V(\mathbf{h})$. Let $L = \sup_{\mathbf{h}} \dfrac{\|P_B F(\mathbf{h})\|}{\|P_B \mathbf{h}\|}$ bound the re-injection gain of $F$ restricted to the targeted subspace over the operating region. The composed step $T_\lambda \circ F$ contracts $V$ when
$$(1-\lambda)^2\, L^2 < 1, \quad\text{i.e.}\quad \lambda > 1 - \tfrac{1}{L},$$
which by the contraction-mapping argument would give convergence to a neighborhood of the admissible region, with $\lambda_{\max}$ bounding overshoot and $\lambda_{\min}$ (or $\tau_{\min}$) guaranteeing a floor under homogeneously hostile input. **Establishing a finite $L$ on a realistic operating region — and hence turning this into an actual convergence proof — is not done here and remains the open item.** The contribution of v0.6 is to make the claim falsifiable: the per-step contraction is trivial; the whole content is the bound on $F$'s re-injection gain, which is an empirical/analytic question about the model, not about the controller. A Lyapunov argument of this shape, made rigorous, is what would move the justification from "a control loop that works" to "a control loop with established stability."

---

## 9. Failure mode: collapse

Under aggressive calibration (high $\gamma$, low $\sigma_{\text{safe}}$, no $\lambda_{\min}/\tau_{\min}$ or excessive $\lambda_{\max}/\tau_{\max}$) the mask suppresses almost everything except a narrow echo of $\mathcal{G}$; the model degrades to repeating the goal's wording, losing content variety. This is over-actuation in control terms. Track it in ablations with **distinct-$n$** and **self-BLEU** in parallel with perplexity and drift rate; a sharp perplexity drop *together with* a diversity drop is the signature. The split criterion (§5) and the multi-anchor subspace (a wider $\mathcal{G}$ leaves more admissible content) both reduce the risk.

---

## 10. Precision-weighting (active inference reading)

The mask is functionally a **precision** $\pi_i \propto m_i$: high-weight tokens are trusted and passed on, low-weight tokens are down-weighted. In cortex this is GABA-ergic lateral inhibition under top-down PFC control (Desimone & Duncan, 1995; Miller & Cohen, 2001).

| Biological component | Formal analog |
|---|---|
| PFC goal representation (delay-period activity) | goal subspace $\mathcal{G}$ |
| GABA-ergic inhibitory signal | mask / actuator $I - \lambda P_B$ |
| Tonic dopamine (gain modulation) | hardness $\tau_t$ / loop gain $\lambda_t$ |
| ACC conflict monitoring | heterogeneity / drift signal driving the controller |

In active-inference terms $\tau$ is the precision over the next-token policy; an FEP-flavoured update $\tau_t \propto D_{\mathrm{KL}}(q_{M_t}\,\|\,p^*)$ is a sketch for companion work, not a ready formula. (Friston's correspondence confirmed the **applicability** of the FEP framework to agentic loops, not the conclusions of any particular argument.)

---

## 11. Architecture of the layer

```
Input context C_t (tokens 1..N)
    │
    ├──→ Main transformer (frozen weights), layers ℓ = 1..L
    │       residual stream h^(ℓ) produced per layer
    │       ACTUATOR (§4.3) on the chosen layers:
    │           h^(ℓ) ← (I − λ_t · B^(ℓ) B^(ℓ)ᵀ) h^(ℓ)      ← direction-level, architecture-agnostic
    │       (4.2 attention-logit masking is the alternative on pure-softmax models)
    │
    └──→ Supervisor (small, separate, low-D)
            ├── Goal subspace 𝒢 = span(g_1..g_K), frozen at session start
            ├── SENSOR: σ(t) = ||P_𝒢 ā|| / ||ā||   (on an unmasked draft / pooled activations)
            ├── two admissibility tests: scope corridor (mode-dependent) + harm gate (fixed)
            ├── REGULATOR: λ_t = clip(γ (σ(t) − θ), 0, λ_max);  baseline memory via EMA / window
            └──→ broadcast λ_t (and B^(ℓ)) to the actuator
```

Mask placement (all layers / a stable mid-band / a hierarchical schedule) is a design choice set by ablation; a layer sweep with a probe (e.g. on a honesty dataset) locates the stable zone. In Experiment 1 the goal direction was stable in a mid-band (layers 14–22 of 36); upper layers had larger amplitude but unstable direction.

---

## 12. Empirical grounding (summary)

Two open-model studies on free-tier GPUs motivate the v0.6 choices. Details: `EXPERIMENT_2_STATUS.md` and the pirate-register runs.

**Experiment 1 — goal/honesty direction (Qwen2.5-3B-Instruct).** $\mathbf{G}$ extracted by difference-in-means over 16 contrastive system-prompt pairs; mean pairwise cosine $\approx 0.85$ at layer 18 (a real, reproducible direction). Clamping along $\mathbf{G}$ causally flips behavior on subtle pseudoscience provocations and on a jailbreak chain; matched random directions do not. **Asymmetry found:** $\mathbf{G}$ works as an **actuator** (intervention) but a single direction fails as an **online sensor** (per-position variance washes out; the resultant is a length clock). → motivates the between-turn controller (§6.2) and the branch-level detector (v0.7).

**Experiment 2 — register suppression, head-to-head of §4 variants (Qwen3.5-2B hybrid; Qwen2.5-7B-Instruct softmax).** The role subspace is selective: a same-pipeline formal-register subspace and a matched-rank random subspace do **not** suppress, while the register subspace does; neutral factual capability is preserved (Tokyo; $17 \times 23 = 391$). Variant comparison at matched suppression:
- **4.1 token-scale** — breaks generation (capability $\to 0$, degenerate output).
- **4.2 attention-logit** — softmax-bound; on the 7B it suppresses only near $\lambda = 1$ and then derails (all-or-nothing); lower meta-reversion but no graded control.
- **4.3 projection** — graded and stable across $\lambda$, architecture-agnostic (works on the hybrid model's linear-attention layers), capability preserved. → selected as the recommended actuator.

**Confounds carried forward (do not drop):** the register direction is not persona-specific (broad expressive axis); ~60–70% of suppressed outputs revert to the base assistant identity, which is the correct ground state but means the surface-lexicon metric must be read on the non-reversion subset; statistics need prompt-clustered CIs.

---

## 13. Open directions

1. **Closed-loop stability (§8).** Bound the re-injection gain $L$ of the forward map on $\mathrm{span}(B)$; turn the contraction sketch into a proof.
2. **Branch-level detector (v0.7).** Per-branch admissibility $a_m = \|P_\perp \hat{\mathbf{b}}_m\|$ to recover the localized drift signal that a single direction loses online; worst-branch rather than mean.
3. **Subspace sources.** $\mathcal{G}$ from 16–32 contrastive paraphrase sets with cosine-consistency validation; $\mathcal{H}$ (harm) by the same difference-in-means; protection of $\mathcal{G}$ from start-of-session prompt injection.
4. **Attribute specificity.** Whether a persona/goal residual survives removing a generic expressive subspace, measured with a model-independent metric; rank-$K$ sensitivity from the singular-value spectrum.
5. **Calibration & replication.** $(\theta, a_{\text{safe}}^{\text{task}}, h_{\max}, \gamma, \lambda_{\max}, D_{\text{baseline}})$ by a trainable supervisor or ablation; cross-model replication beyond the models above; pre-registered endpoints with multiple-comparison correction.
6. **Comparison.** Activation steering, Constitutional AI, SAE feature steering — the distinguishing claim of this line is the *temporal dynamics* (the homeostat), which steering lacks; an explicit comparison is owed.

---

## References (minimal context set)

- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2).
- Adams, R. A., Stephan, K. E., Brown, H. R., Frith, C. D., & Friston, K. J. (2013). The computational anatomy of psychosis. *Frontiers in Psychiatry*, 4, 47.
- Peters, B., Niculae, V., & Martins, A. F. T. (2019). Sparse Sequence-to-Sequence Models. *ACL 2019*.
- Desimone, R., & Duncan, J. (1995). Neural mechanisms of selective visual attention. *Annual Review of Neuroscience*, 18.
- Miller, E. K., & Cohen, J. D. (2001). An integrative theory of prefrontal cortex function. *Annual Review of Neuroscience*, 24.
- Yamashita, Y., & Tani, J. (2008). Emergence of functional hierarchy in a multiple timescale neural network model. *PLoS Computational Biology*.
- Kulveit, J., von Stengel, R., & Leventov, M. (2023). *Predictive Minds: LLMs as Atypical Active Inference Agents.* arXiv:2311.10215.
- Bulatova, A. (2026). *Mythos is not an anomaly: why restrictions make agents less predictable, not safer.* EA Forum.
