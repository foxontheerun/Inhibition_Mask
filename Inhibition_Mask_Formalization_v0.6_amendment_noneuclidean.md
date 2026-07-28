# Formalization v0.6 — empirical amendment: the non-Euclidean metric

2026-07. Amends §4.3 (projection ablation) and §1 (the alignment sensor). Does not replace v0.6;
adds one correction confirmed empirically on Qwen2.5-3B. The detailed derivation, citations, and
pre-registered run logs are kept in a separate research repository and summarized here.

## The gap

§4.3 writes the actuator as `h ← h − λ·B Bᵀ(h − μ)` and §1 the sensor as `σ = ‖P_B(h−μ)‖/‖h−μ‖`.
Both use `B Bᵀ`, which is the orthogonal projector **only under the Euclidean inner product**
`⟨x,y⟩ = xᵀy`. Hidden-state space is not isotropic: a few coordinates carry orders-of-magnitude more
variance than the rest (extreme on larger models — "massive activations"). Under anisotropy the
Euclidean projector removes/reads the wrong direction.

## The correction

Estimate the metric `G = Σ̂⁻¹` from the **within-class** covariance (the background/content
variation), per layer, from per-token residuals, shrunk toward the diagonal
`Σ̂ = (1−α)Σ_full + α·diag(Σ)`, α fixed once by sensor stability on a holdout. Use one `Σ̂` across
the whole pipeline (build B, sensor, actuator). Then:

- **subspace:** whitened diff-in-means (`Σ̂⁻¹δ`; at rank 1 this is the Fisher direction), not the
  raw SVD component;
- **sensor:** σ computed in the whitened coordinates `x̃ = W(x−μ)`, `W = Σ̂^{−1/2}`;
- **actuator (graded, metric-correct):** `h ← h − λ·B(BᵀG B)⁻¹BᵀG(h − μ)` — the oblique projector
  onto span(B) along the G-orthogonal complement. λ stays the continuous knob (the §6 homeostat
  needs a graded actuator; LEACE-style closed-form erasure is a useful *ceiling*, not the actuator,
  because it has no λ and its guarantee is single-layer).

The gain of whitening is in **which direction is removed**, not in measuring collateral under a
different norm.

## Evidence (Qwen2.5-3B, layer 18)

- The raw leading component sits at cos² ≈ 0.58 with a punctuation/orthography axis — i.e. the
  specificity criterion that failed four times across earlier experiments was rejecting a
  *metric artifact*.
- Whitened diff-in-means clears it: K=7 punct overlap **0.158 → 0.040** (near the whitened chance
  floor ≈ 0.024) while holdout separability is **maintained** (AUC ≈ 0.92–0.95). Pre-registered gate
  passes. Pure diagonal whitening leaves the leading overlap at 0.46 vs 0.035 for full whitening, so
  **off-diagonal structure matters** — diagonal (z-score) is a floor, not the fix.

## Status and caveats

- **Confirmed:** the punctuation confound is anisotropy, not register structure; whitening the
  *sensor / subspace* dissolves it. This is the amendment's firm claim.
- **Plausible, not yet confirmed:** that the *actuator* also benefits (metric-correct ablation
  suppresses at lower collateral). A first 3B run points that way (+0.13 at matched disruption) but
  rests on one seed with over-suppression overshoot, after an audit already caught a basis bug in the
  same experiment — so it is held as plausible pending robustness draws and a 7B replication.
- **Open:** normalization geometry (RMSNorm makes the consumed metric post-norm; a clean pre-norm
  ablation still perturbs `rms(h)` and rescales every coordinate) — theory only, no verified
  literature yet; operationalized as a renormalization test in the follow-up experimental plan.
