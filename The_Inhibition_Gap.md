# The Inhibition Gap: One Missing Mechanism Behind Five LLM Failure Modes

*Mythos, sycophancy, goal hijacking, lost-in-the-middle, jailbreaks — same architectural deficit. With a falsifiable prediction Anthropic could verify against Mythos attention logs. Follow-up to ["Why LLM Agents Act Beyond Their Task"](https://forum.effectivealtruism.org/posts/EmYkipjGHYLPhAQa4/why-llm-agents-act-beyond-their-task-a-structural) (Bulatova, 2026, EA Forum).*

---

## Where this came from

In my previous piece I offered a structural explanation for the [Claude Mythos Preview incident](https://www.anthropic.com/claude-mythos-preview-risk-report) (Anthropic, April 2026). The model completed its task — found a way out of its sandbox and sent an email — and then, without any prompt, published the details of the discovered exploit on several publicly accessible sites. Anthropic described this as "a concerning and unasked-for effort to demonstrate its success," but did not formally explain the mechanism.

My explanation was: an LLM agent has no mechanism that makes actionable information accumulated in context inactive after the task is done. The internal-adaptation channel (perception/learning in active inference terms) is constrained for LLMs — weights are frozen at inference — so the entire load falls on the action channel. Mythos didn't "decide" to publish the exploit. It generated a high-probability continuation from a context that contained the described vulnerability.

The piece ended with an open question: what architectural changes could solve this? This essay is an attempt at an answer, with no claim to completeness.

## The central thesis in one sentence

**Several distinct, apparently unrelated classes of LLM-agent failure share a computational signature that warrants treatment as a single cluster — the absence in transformer architecture of an active inhibition mechanism conditioned on a stable goal representation.**

By calling this a structural signature I mean: it follows from the inference-time regime of a standard transformer (frozen weights + softmax attention) and does not depend on the specific weights or training distribution. The solution lies at the architectural level, not the training level.

This is a diagnostic claim. Full implementation is a separate research task (see §"Architectural direction"). The industry treats several symptoms as separate bugs; they are manifestations of one missing function, and this changes the strategy of intervention.

## The five symptoms I unify into one cluster

The first is *drift in the agentic loop*. The canonical example is the Mythos case: the agent performs a task, accumulates actionable information in context along the way (a discovered vulnerability, a workaround method, the fact of access), and continues acting on that information after the task is complete. Anthropic's report describes this empirically but does not explain the mechanism.

The second is *sycophancy*, a phenomenon documented across major models, including Claude 2 and GPT-4 (Sharma et al., 2023), and later mechanistically isolated as a controllable direction in activation space in Anthropic's persona-vectors work (Chen et al., 2025). When a user expresses a false opinion or makes an error, the model agrees with them, even though correct information is encoded in its training. The context — the user's framing — overrides trained knowledge.

The third is *goal hijacking in agentic frameworks*. In production agentic systems — Cursor agents, ChatGPT agent (formerly Operator), open-source frameworks built on Claude and GPT APIs, and similar tools used daily by thousands of developers — an agent that encounters an interesting side observation while executing a task can switch to that observation and drift from the original goal. The intermediate context overrides the original instruction. The standard engineering explanation is that the model is confused by too many options. The framing I propose is different: each intermediate observation gets the maximum possible weight in context, and there is no stable mechanism that preserves the original goal in a context-protected form. A neighboring pattern — drift under adversarial competing-objective pressure — is systematically measured in Apollo Research's Goal Drift work, in agents built on frontier models from Anthropic and OpenAI (Arike et al., 2025).

The fourth is *Lost in the Middle*. Liu et al. (2023) showed that models use information less reliably as context grows, especially when key information sits in the middle. Anthropic, in their context-engineering guide, acknowledges directly that context windows of any size are subject to relevance problems. The standard engineering response is to compress and structure better. In the framing I propose, this is degradation of the only internal-adaptation channel available, in the absence of a mechanism for selective suppression of irrelevant segments.

The fifth is *jailbreaks via context-shift*. Many jailbreak attacks work not through a direct request to violate rules, but through establishing a frame in context — style injection, distractor framings, role-play setups — that makes the model act like a different system (Wei et al., 2023). Context changes the model's functional state; trained restrictions lose to context pressure.

## What's common between these five

In all five cases:
- The system has **trained structures** — weights that should make the model behave a certain way (be honest, stay on task, answer the question, not violate rules).
- The system has **runtime context** — current contents of the context window.
- In each of the five cases, **runtime context overrides trained structures**, not because of a training error, but because architecturally there's no mechanism that would let trained structures maintain influence against context pressure.

Another way to put it: the model has no way to say "this part of the context shouldn't influence me right now" — even if it contradicts its goal or policy. Context always influences, and the only regulator is statistical attention weights, which **redistribute** influence but don't **suppress** it.

This is the missing function: **active suppression of the influence of specific context elements, conditioned on a stable representation of what's currently relevant.** Not redistribution — suppression. Not transient (within one forward pass) — sustained.

**What does NOT belong in this cluster.** For the framework to be a diagnosis rather than a relabeling, it's important to specify an exclusion criterion. Several well-known LLM failure classes have a different nature: *factual hallucinations* on simple questions are training gaps or RLHF-bias, not runtime-context overpowering trained policy; *tokenization-induced errors* in arithmetic or character counting are representation-level artifacts, not attention dynamics; *catastrophic forgetting* during continued fine-tuning is a weight-level effect, not a runtime effect. These cases lie **outside** the proposed cluster. If they were inside it, the framework would lose its discriminative power.

## Why standard attention doesn't do this

The first objection is: "But transformers already have self-attention, and that's exactly what it does — choose what to attend to and what to ignore." This objection isn't accurate.

Standard attention works through **softmax** — a normalized weight distribution. Every token gets some weight; none gets exactly zero. Noise from irrelevant tokens still flows through the network with reduced weight and accumulates across layers. At final layers this noise can outweigh the useful signal — this is the "accumulation effect" that manifests in lost-in-the-middle and other long-context failures.

Also, standard attention is trained at training time. Weights are fixed. If during training the model didn't see that "this kind of context is noise relative to this kind of goal," it won't suppress that in the moment. Its attention reacts to context based on trained patterns, not on the current goal in real time.

And most importantly: existing goal-conditioning mechanisms — cross-attention with instructions, instruction tuning, FiLM-modulation, goal-conditioned RL — *amplify* what is relevant to the goal. They do not *suppress* what is irrelevant. These are different functions. In the cortex they are also separated: amplification works through top-down modulation, while suppression works through inhibition implemented by local GABA-ergic interneurons in target areas, recruited by glutamatergic projections from PFC and thalamic gating (TRN). The biological mapping here is suggestive, not load-bearing for the architectural argument — biology does not treat amplification and inhibition as interchangeable, which gives reason to think the distinction is useful in transformer architecture too.

## Biological reference

A specific set of works from computational neuroscience, not general metaphors:

**Aberrant precision** (Adams, Stephan, Brown, Frith, Friston, 2013, *Frontiers in Psychiatry*). In the active inference framework, many psychiatric conditions (notably schizophrenia, with extensions to autism) are explained as **dysregulation of precision-weighting** — the system mis-weights trust between sensory input and prior beliefs. This work has become the canonical reference for FEP-clinical literature.

**Utilization behavior** (Lhermitte, 1983, *Brain*). Patients with frontal lobe damage automatically use any object in their visual field — a comb gets picked up and they comb their hair, a pen gets picked up and they write. Actions are generated by stimulus, not by intention. This is the neuropsychological canon for cases when PFC inhibitory control disappears. I already used this parallel in the Mythos essay.

**Top-down control in PFC.** The standard picture (Miller & Cohen, 2001, *Annual Review of Neuroscience*; Desimone & Duncan, 1995): PFC maintains a goal representation and sends modulating signals down into sensory and associative areas. These signals bias local competitive mechanisms (via GABA-ergic interneurons and via thalamic gating), suppressing irrelevant neurons and amplifying relevant ones.

It's worth emphasizing: in the cortex GABA-ergic inhibition under top-down PFC control is not a restrictor of cognitive capabilities, but rather one of the conditions for them to work (selective attention, working memory, goal-directed action). In the proposed framework the inhibitory mechanism in LLMs plays the same constitutive role, not the role of an external guardrail bolted on top of a finished capability.

**Scope clarification.** LLM-agent and brain are different systems; the biological mechanisms cited above are established results of computational neuroscience, not my discovery; the PFC analogy is functional, not literal — cortical mechanisms are multi-channel and indirect; the supervisor described below is a functional sketch of an architectural requirement, not a model of cortical mechanism.

**Central claim:** LLM agents and systems with deficits of inhibitory control share a **common computational signature** — high generative capacity in the absence or weakening of a mechanism for runtime suppression of irrelevant stimuli. This signature is realizable clinically (frontal syndrome, ADHD, mania), pharmacologically (stimulants, alcohol, sleep deprivation), and architecturally (transformer without top-down inhibition). It is a **computational signature, not a clinical category**.

## The architectural direction

So this text doesn't stay purely diagnostic, here's a sketch of how the bridge from diagnosis to architecture might look. Full formalization is a separate document; here — a conceptual sketch.

A small separate module — a **supervisor** — is attached to the main model. It takes as input (a) a compressed representation of the current context and (b) a goal vector **G**. It outputs a mask of length N (one value per token), which is applied to the main model's attention via the standard additive masking mechanism.

Specifically: for each token $i$ the supervisor computes the cosine similarity $\sigma_i$ of its activation with the goal vector $\mathbf{G}$. From $\sigma_i$ a mask $M_i$ is formed — high for tokens aligned with the goal, low for opposing ones. The mask is added to attention logits before softmax: tokens with $M_i = 0$ give $\log(0) = -\infty$, and after softmax their contribution is exactly zeroed.

The mask's hardness isn't static. Parameter $\tau$ (a precision-analog) is updated as a function of the variance of alignments $\sigma_i$ in the current context: specifically $\tau \propto \mathrm{Var}_i(\sigma_i)$, monotonically increasing (formal dependency given in the companion formalization document). When context becomes heterogeneous (some tokens on goal, some competing), $\tau$ rises and the mask hardens. This is a functional analog of tonic neuromodulator activity stabilizing dynamics under noise.

**An important distinction: goal-conditioned scope vs capability restriction.** The proposed mechanism is not capability restriction, but **goal-conditioned scope restriction**. Searching for workarounds and using an exploit **in service of the task** are not suppressed — corresponding continuations have high cosine similarity with the goal vector. What gets suppressed are continuations **outside the task scope**. In the Mythos case this means: the agent still finds the exploit and uses it to send the email (this is in service of the goal "send email"); but when generating post-task actions, the continuation "publish exploit" is suppressed by the mask because it isn't aligned with the goal vector (publishing ≠ sending email). The capability to find and use the exploit is preserved; unsanctioned action outside the scope of the task does not occur. This distinguishes the approach from guardrail-style safety, which either overblocks legitimate capability (model refuses normal requests) or underblocks via non-standard workarounds. Here the model's capability is fully preserved; only the runtime scope of actions relative to the current goal is constrained.

Advantages of this architecture:
- **Interpretability:** the mask can be extracted and inspected to see exactly what the supervisor is suppressing.
- **Modularity:** the supervisor is trained separately from the main model.
- **Compatibility:** the additive masking mechanism is already standard in transformers (used for causal masking) — no need to rewrite the core.

A working direction. Not closed by existing solutions (analysis of neighbors below).

## Where this overlaps with existing work and where it differs

In the last two years several architectural proposals have appeared that partially fall in the same zone. The closest neighbors are worth walking through carefully.

The Differential Transformer (Ye et al., 2024, ICLR 2025, arXiv:2410.05258) uses the difference of two softmax maps to suppress "noise" in attention. Conceptually it is close — a form of inhibition at the attention level — but the mechanism is input-driven (input statistics), not goal-conditioned. It suppresses what looks like noise on average, but not what is misaligned with a specific goal. Without a stable goal vector, the suppression works on other tasks but does not address agentic drift, where "noise" is defined relative to the current goal.

Differential Gated Self-Attention / M-DGSA (Lygizou, Farsang, Grosu, 2025, arXiv:2505.24054) is explicitly inspired by lateral inhibition in biological neural circuits; it splits each attention head into excitatory and inhibitory branches. This is the closest mechanical neighbor I found. But again the gating is input-driven, not goal-conditioned, and there is no connection to FEP or precision-weighting framing. The work is presented as a noise-robustness improvement, not as a unifying diagnosis for a class of failure modes.

Persona Vectors (Chen et al., Anthropic, 2025, arXiv:2507.21509) identify directions in the residual stream encoding personality traits (sycophancy, evil, hallucination), and use them for activation steering. This is an activation-level approach, not an attention mask. Their work demonstrates that behavioral traits can be reliably localized as directions in the residual stream — multi-anchor extensions of single-vector $\mathbf{G}$ in this framework follow the same structural intuition of representing complex traits as collections of related directions (where $\mathbf{G}$ generalizes to a subspace $\mathcal{G} = \mathrm{span}(\mathbf{g}_1, \ldots, \mathbf{g}_K)$). Conceptually we are solving a similar problem — unifying behavioral patterns into a controllable representation — but through a runtime attention-mask rather than post-hoc activation steering.

Consistency Training (Irpan, Turner, Kurzeja, Elson, Shah, 2025, arXiv:2510.27062, Google DeepMind / ex-Anthropic) is the closest of everything I found. They also unify sycophancy and jailbreaks under one framing (the model "captured by adversarial wrapper"); their solution is Activation Consistency Training — training-time invariance. The key difference is the level of intervention. Training-time invariance works when the adversarial pattern is represented in the training distribution; for out-of-distribution context shifts — which is what new agentic scenarios (Mythos-style) or new genres of jailbreaks not seen during training are by definition — training-time invariance structurally cannot guarantee robustness, because the adversarial direction was not part of the training signal. A runtime mechanism (active inhibition by the current goal vector) does not depend on whether the model saw a similar adversarial pattern during training; it works by the geometry of the current precision-weighting task. Their work also covers two types of failure; the hypothesis I propose extends to three additional ones (drift, lost-in-the-middle, goal hijacking), for which training-time invariance is harder to define — what would count as "consistent" for drift? If this hypothesis holds, the runtime approach gives unified treatment without escalating the count of training augmentations.

Goal Drift in Language Model Agents (Arike, Donoway, Bartsch, Hobbhahn, 2025, Apollo Research, arXiv:2505.02709) measures goal adherence in LLM agents under competing signals. The setup probes mechanism through varying multiple conditions — pressure, time horizon, instruction salience — and these constraints on when drift emerges should inform any unification claim. The empirical findings about which pressure profiles induce drift are exactly the kind of thing the framework I propose ought to predict; treating their work as a generic "drift exists" citation would be a mistake. A natural collaboration angle is to test the proposed mechanism against Apollo's existing drift agents with attention-pattern logging on the drift-onset turn.

Predictive Minds: LLMs As Atypical Active Inference Agents (Kulveit, von Stengel, Leventov, 2023, NeurIPS SoLaR Workshop, arXiv:2311.10215) is the anchor text for the FEP-and-LLM framing. They formalize LLM agents through active inference; their main argument is that LLMs fit this framework, except for missing the perception-action feedback loop. I rely on this work as theoretical foundation and continue in a specific direction.

What unifies all these differentiations: nobody does the full combination — stable goal vector, active suppression mask at the attention level, precision-weighting grounding, and an explanatory diagnosis for a specific cluster of failure modes. Each component separately exists or has been proposed. Their specific combination has not.

## Open directions

The work specifies a functional requirement and an architectural direction. Three directions are developed in companion work:

**Source of the stable goal vector $\mathbf{G}$.** $\mathbf{G}$ is extracted from initial context or the model's long-term priors and must be protected from context drift and from prompt injection. Architectural candidates: multi-timescale architectures with a slow layer, a dedicated aggregator module by analogy with biological convergence of HPC + amygdala + OFC + DLPFC, a stable prior trained via contrastive encoder on goal-invariance pairs. Each is a separate design choice with its own trade-off space.

**Empirical verification.** Indirect empirical signals exist — the works listed above document each of the five symptoms separately. The measurement of the common logit signature in a single experimental frame is the next piece in the program (Llama-3-8B, 8-week timeline).

**Parameter calibration.** The supervisor is calibrated via bootstrap end-to-end loss with regularization against collapse, or by manual tuning on benchmarks. This is an engineering task with known solutions in adjacent areas (gating networks, MoE).

## Empirical roadmap and falsifiable predictions

This work proposes the framework. The empirical follow-up — Llama-3-8B-Instruct, 50 jailbreak/sycophancy prompts + 50 creative prompts as control, fixed G via mean-pool of activations, hard binary mask on upper layers, sweepable parameters — runs in 1–2 weeks and publishes within 8 weeks. The position paper and the empirical follow-up run in parallel: the framework is fixed before measurement, so discussion and empirics move together.

**Falsifiable prediction for the Mythos case.** If the framework is correct, in the model's attention patterns on tokens immediately preceding autonomous publication of the exploit, one would see **dominance of attention on tokens describing the exploit, with low attention on tokens describing the original task or its completion**. Attention balanced between goal-relevant and exploit-related tokens would indicate a different cause for publication — not the absence-of-inhibition problem. Anthropic has activation logs of the Mythos session; retrospective verification is technically feasible in a few hours of work for an interp engineer. An equivalent test could be run on Apollo Research's goal-drift agent setup with attention-mass logging at the drift-onset turn.

**Robustness of the framework to falsification of specific predictions.** The Mythos test may yield a negative result. In that case: (a) the unifying thesis holds — four of the five classes have independent empirical anchors (Sharma on sycophancy, Liu on lost-in-the-middle, Wei on jailbreaks, Apollo Research on goal drift); (b) Mythos is realized through a different architectural channel than hypothesized; (c) the map of which failure modes fall into the active-inhibition-deficit cluster becomes more accurate. The result refines the framework.

## What would be useful

I invite engagement along several specific lines.

I would welcome critique of the theoretical framework, especially from people working in active inference and computational psychiatry. Where is the precision-weighting correspondence stretched? Where is the analogy with PFC-inhibition strained?

I would also welcome being pointed to close work I missed. The literature search was thorough, but 30 days is a long time at the current arXiv pace. If a direct duplicate exists that I haven't engaged with, please tell me.

If anyone is working on M-DGSA-style implementation or Persona Vectors-style work and sees a convergence point with the framing here, I would be glad to discuss it. The natural follow-up is the empirical experiment described above — concretely a Llama-3-8B fixed-G prototype with hard binary mask on upper layers, sweepable parameters. I do not have a lab or compute infrastructure to do this at scale, so I am open to running it jointly with anyone willing.

---

**Companion formalization (v0.5).** The full mathematical formalization of the supervisor architecture, mask construction, and precision homeostat — covering all four mask variants (hard, sigmoid, $\alpha$-entmax, admissibility cone), the dynamic $\tau$ feedback rule with EMA and sliding-window adaptation, the multi-anchor subspace extension, and parameter calibration — is in a companion document available [on GitHub](https://github.com/foxontheerun/Structural-Contradiction-in-the-Capable-Agent-Strict-Control-Formula) as `.md`, `.tex`, and `.pdf`.

---

**Self-citation:** Bulatova, A. (2026). *Why LLM Agents Act Beyond Their Task: A Structural Explanation Through Blocked Adaptation.* EA Forum. https://forum.effectivealtruism.org/posts/EmYkipjGHYLPhAQa4/why-llm-agents-act-beyond-their-task-a-structural

**References** (minimum):
- Adams, R. A., Stephan, K. E., Brown, H. R., Frith, C. D., & Friston, K. J. (2013). The computational anatomy of psychosis. *Frontiers in Psychiatry*, 4, 47.
- Arike, R., Donoway, E., Bartsch, H., & Hobbhahn, M. (2025). *Evaluating Goal Drift in Language Model Agents.* arXiv:2505.02709.
- Chen, R., Arditi, A., Sleight, H., Evans, O., & Lindsey, J. (2025). *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.* arXiv:2507.21509.
- Desimone, R., & Duncan, J. (1995). Neural mechanisms of selective visual attention. *Annual Review of Neuroscience*, 18, 193–222.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2).
- Irpan, A., Turner, A., Kurzeja, M., Elson, D., & Shah, R. (2025). *Consistency Training Helps Stop Sycophancy and Jailbreaks.* arXiv:2510.27062.
- Kulveit, J., von Stengel, C., & Leventov, M. (2023). *Predictive Minds: LLMs as Atypical Active Inference Agents.* arXiv:2311.10215.
- Lhermitte, F. (1983). 'Utilization behaviour' and its relation to lesions of the frontal lobes. *Brain*, 106(Pt 2), 237–255.
- Liu, N. F., Lin, K., Hewitt, J., et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* arXiv:2307.03172.
- Lygizou, E., Farsang, M., & Grosu, R. (2025). *Differential Gated Self-Attention.* arXiv:2505.24054.
- Miller, E. K., & Cohen, J. D. (2001). An integrative theory of prefrontal cortex function. *Annual Review of Neuroscience*, 24, 167–202.
- Sharma, M., Tong, M., Korbak, T., et al. (2023). *Towards Understanding Sycophancy in Language Models.* arXiv:2310.13548.
- Wei, A., Haghtalab, N., & Steinhardt, J. (2023). *Jailbroken: How Does LLM Safety Training Fail?* arXiv:2307.02483.
- Ye, T., Dong, L., Xia, Y., et al. (2024). *Differential Transformer.* arXiv:2410.05258.
