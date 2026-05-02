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

[Read the position paper on Substack](#) <!-- replace with Substack URL after publishing -->

### 3. Companion formalization — *Formalization of a Dynamic Inhibitory Mask* (v0.5)

Formal architectural draft of the inhibitory layer described in the position paper above. Covers:

- Four mask variants (hard, sigmoid, α-entmax, admissibility cone)
- Adaptive percentile threshold for high-dimensional concentration of measure
- Additive log-mask in attention logits (compatible with standard transformer masking)
- Dynamic τ feedback rule with EMA and sliding-window adaptation schemes
- Multi-anchor / subspace 𝒢 extension for multi-faceted goals
- "Collapse" failure mode under aggressive calibration
- Architectural diagram, parameter table, and minimal validation program

Files in this repo:
- `Inhibition_Mask_Formalization_v0.5.md` — markdown source (browsable on GitHub)
- `Inhibition_Mask_Formalization_v0.5.tex` — LaTeX source
- `Inhibition_Mask_Formalization_v0.5.pdf` — compiled PDF

---

## Research roadmap

The next planned artifact is an empirical follow-up: a Llama-3-8B prototype with a fixed goal vector and a hard binary mask on the upper attention layers, with logit-level analysis of jailbreak/sycophancy prompts vs. creative-prompt controls. Target window: 8 weeks from publication of the position paper.

## License

[CC BY 4.0](LICENSE) — free to share and adapt with attribution.

## Citation

If citing the formalization:

> Bulatova, A. (2026). *Formalization of a Dynamic Inhibitory Mask* (v0.5). GitHub. https://github.com/foxontheerun/Structural-Contradiction-in-the-Capable-Agent-Strict-Control-Formula

If citing the position paper:

> Bulatova, A. (2026). *The Inhibition Gap: One Missing Mechanism Behind Five LLM Failure Modes.* Substack.


[Add Substack link / EA Forum profile]
