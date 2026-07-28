# Research Program: Active Inhibition in LLM Agents

A theoretical and architectural research program on AI safety failures in transformer-based LLM agents, grounded in the Active Inference / Free Energy Principle framework (Friston, 2010).

**Author:** Alsu Bulatova — independent researcher.

---

## Artifacts

### 1. Position paper — *Why LLM Agents Act Beyond Their Task: A Structural Explanation Through Blocked Adaptation* (April 2026, EA Forum)

The original analysis of the Claude Mythos Preview incident, framed as an instance of a structural contradiction: when capabilities are scaled in parallel with restrictions, agent behavior becomes less predictable, not more. Read on [EA Forum](https://forum.effectivealtruism.org/posts/EmYkipjGHYLPhAQa4/why-llm-agents-act-beyond-their-task-a-structural).

Files in this repo:
- `Structural Contradiction in the _Capable Agent + Strict Control_ Formula.pdf`
- `Structural_Contradiction_April2026.pdf`

### 2. Position paper — *The Inhibition Gap* (May 2026, Substack)

Follow-up that proposes one architectural cause shared by five LLM-agent failure modes (Mythos drift, sycophancy, goal hijacking, lost-in-the-middle, jailbreaks via context-shift): the absence in the transformer architecture of an **active inhibition mechanism conditioned on a stable goal representation**. Includes a falsifiable prediction Anthropic could verify against Mythos attention logs.

[Read the position paper on Substack](https://bulatovaalsu.substack.com/p/the-inhibition-gap-one-missing-mechanism) 

### 3. Companion formalization — *Formalization of a Dynamic Inhibitory Mask* (v0.6, current)

Formal architectural draft of the inhibitory layer described in the position paper above. Covers:

- Four mask variants (hard, sigmoid, α-entmax, admissibility cone)
- Adaptive percentile threshold for high-dimensional concentration of measure
- Additive log-mask in attention logits (compatible with standard transformer masking)
- Dynamic τ feedback rule with EMA and sliding-window adaptation schemes
- Multi-anchor / subspace 𝒢 extension for multi-faceted goals
- "Collapse" failure mode under aggressive calibration
- **v0.6 adds:** §4.3 projection ablation actuator, §5 scope corridor + harm gate split,
  control-theory reading, and the dispersion homeostat

Files in this repo:
- `Inhibition_Mask_Formalization_v0.6.md` — **current version** (browsable on GitHub)
- `Inhibition_Mask_Formalization_v0.5.md` / `.tex` / `.pdf` — previous version

**Empirical amendment (2026-07):** `Inhibition_Mask_Formalization_v0.6_amendment_noneuclidean.md`
— the hidden-state space is anisotropic, so the Euclidean projections in §4.3 are measured under
the wrong inner product; the metric-correct form uses `G = Σ⁻¹` (whitening). See the amendment for
the statement and the empirical evidence (it dissolves a recurring confound on Qwen2.5-3B).

---

### 4. Empirical work — `experiments/register_inhibition/`

Executed notebooks probing the mask on small open models (Qwen2.5-3B/7B): static register ablation,
the §4.1/§4.2/§4.3 actuator head-to-head, cross-register probes (pirate / Shakespeare / noir / robot),
and the §5.2 dispersion homeostat (λ breathes with context on a drift attack). Demonstrations of the
mechanism — single chains, not statistics. A rigorous pre-registered validation track is in progress.

---

## Research roadmap

The next planned artifact is an empirical follow-up: a Llama-3-8B prototype with a fixed goal vector and a hard binary mask on the upper attention layers, with logit-level analysis of jailbreak/sycophancy prompts vs. creative-prompt controls. Target window: 8 weeks from publication of the position paper.


## Citation

If citing the formalization:

> Bulatova, A. (2026). *Formalization of a Dynamic Inhibitory Mask* (v0.5). GitHub. https://github.com/foxontheerun/Structural-Contradiction-in-the-Capable-Agent-Strict-Control-Formula

If citing the position paper:

> Bulatova, A. (2026). *The Inhibition Gap: One Missing Mechanism Behind Five LLM Failure Modes.* Substack. https://bulatovaalsu.substack.com/p/the-inhibition-gap-one-missing-mechanism



