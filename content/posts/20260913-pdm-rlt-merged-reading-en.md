---
id: 20260913-pdm-rlt-merged-reading-en
title: "The checkpoint is not the policy: a merged reading of PDM and RLT"
slug: pdm-rlt-merged-reading-en
date: 2026-09-13
lastmod: 2026-09-13
draft: false
format: "long"
domain: deep-learning
subdomain: rl-training-infrastructure
summary: >-
  A merged reading of two 2026 project reports from the same author line,
  Prefill–Decode Kernel Mismatch (an execution-level diagnosis of why
  same-weight sampler and trainer paths define different policies, plus
  contracts, audits, and four remedies) and Recurrent Looped Transformer (an
  architecture that designs the sampler-trainer mismatch away with one
  history-dependent transition): what each actually solves, what is actually
  measured, and the end-to-end causal loop neither has closed yet.
confidence: working
prerequisites:
  - Autoregressive decoding and KV caches
  - On-policy RL basics: importance ratios, off-policy correction, support conditions
  - A rough picture of state-space and linear-attention models
related: []
tags:
  - execution-fidelity
  - rl-scaling
  - importance-sampling
  - recurrent-architecture
  - numerical-precision
  - audit-discipline
bibliography: ""
code_repo: ""
---


Why this exists: In September 2026 I ingested two project reports from the same author line into my vault, a diagnostics report on Prefill–Decode Kernel Mismatch ("PDM") and an architecture report on a Recurrent Looped Transformer ("RLT"), and cross-linked them. Reading the second right after the first, the two stopped looking like separate papers. One says the mismatch between rollout and training computation is measurable and must be contractualized; the other says it can be designed out. I drafted a merged reading in Chinese for the ingest session; this is the English archive version, reorganized around three questions: what problem each report claims, what it does about it and what that delivers, and what survives a critical read.

The thesis: PDM turns RL execution fidelity into something you can measure, record, and audit, and RLT turns it into something you can design, but neither closes the causal loop from mismatch to end-to-end training outcomes.

Scope: This covers the two reports as of September 2026 (PDM August 6, revised August 24; RLT September), their problem statements, mechanisms, evidence, and weaknesses. It does not re-derive their math, does not independently reproduce any number, and treats both as non-peer-reviewed project reports. The cross-architecture hotspot hypothesis in the critique is mine, not theirs.

Prerequisites: This assumes familiarity with autoregressive decoding and KV caches, on-policy RL basics (importance ratios, off-policy correction, support conditions), and a rough picture of state-space and linear-attention models.

## The problem PDM names: the checkpoint is not the policy

The assumption I carried for years is that a checkpoint defines a policy: load the weights, and every runtime that reads them computes the same distribution. PDM opens by breaking that assumption into pieces. RL pipelines generate rollouts with a sampler (parallel prefill builds a KV cache; decode steps extend it one token at a time, under quantization, custom kernels, and a sampling transform) and score the same tokens with a trainer (usually one teacher-forced parallel pass, another operator, another reduction order, sometimes another precision). Two engines with identical parameters end up defining two token distributions.

Concretely, the report writes the sampled-token log ratio as a sum of two terms:

$$\log \rho_t = \underbrace{\log \pi^{\mathrm{trainer}}_{\theta_n}(a_t \mid h_t) - \log \pi^{\mathrm{beh}}_{\theta_n}(a_t \mid h_t)}_{\Delta_{\mathrm{kernel}}} + \underbrace{\log \pi^{\mathrm{beh}}_{\theta_n}(a_t \mid h_t) - \log \pi^{\mathrm{beh}}_{\theta_v}(a_t \mid h_t)}_{\Delta_{\mathrm{stale}}}.$$

The second term is the familiar staleness of asynchronous RL: it vanishes when the sampler and trainer are in sync. The first term does not. A fully synchronous pipeline can still be off-policy for the trainer's own target. That sentence changed how I read every "we use a fast inference engine for rollouts" line.

The emblematic bug deserves its own paragraph, because it is three lines away from any real repo. If you sample with the sampler operator and later reconstruct the "old" probability with the trainer operator, then at zero lag your recomputed ratio is exactly 1 by construction, while the correct ratio can still differ from 1. The recomputation does not fix a drift. It hides the gap, which is the report's phrase, and it does so structurally rather than by accident.

Then there are the cases where the gap is not a rounding detail. For softmax attention, the algebra of a KV-cached recurrence and a parallel causal pass is equivalent in exact arithmetic, but that identity is not a certificate: tiling, fusion, accumulation order, cache layout, batch shape, parallel communication, precision, quantization, and the post-logit transform all live between the two computations. For linear attention and state-space models the situation is worse: tokenwise, scan, and chunkwise implementations are different schedules, chunk size C runs from 1 (tokenwise) to L (one chunk), and changing the schedule changes floating-point parenthesization. One boundary error propagates into every later state. DeltaNet makes it personal: its state update is a data-dependent corrective write, so an error in the previous state participates in the next write rather than merely being observed downstream. Recurrent sampling against a parallelized delta-rule kernel is two policies until proven otherwise.

What I find sharpest is the contract split. Distributional parity (same forward distribution on every history the sampler reaches) and score parity (the trainer differentiates the deployed policy's score function, or an equivalent forward-and-backward implementation) are logically distinct. Equal forward distributions at one parameter value do not imply equal Jacobians. And support is part of the contract: if hard top-k, top-p, or a precision underflow assigns zero behavior probability to an action the target retains, no downstream likelihood ratio is defined, and sampled log-probabilities cannot repair that after the fact. I had filed support under "implementation detail". It is not; it is a precondition for the estimator to exist.

## The problem RLT names: one model, four execution paths

RLT's problem statement reads like the same diagnosis one level up. A model in a production lifecycle passes through pretraining, supervised fine-tuning, rollout sampling, and replay, and each stage has historically used its own computation path, its own masking, its own caching. The report quotes the PDM line explicitly: matching forward probabilities at one parameter value is not sufficient to establish matching policy gradients, and a shared architectural transition alone does not prove numerical kernel parity. So the replay contract in RLT is not an ops checklist bolted onto an architecture; the architecture is chosen so that the checklist becomes cheap to satisfy.

The second half of RLT's problem is depth. In a standard transformer, the computation depth available to a token is the layer count, full stop. RLT wants reasoning depth that scales with the sequence, not with the stack: it applies a single history-dependent transition to every observed or sampled token, and the recurrent state carries continuous intermediate computation from token to token. After t tokens, the state path has traversed t times the decoder depth in logical block evaluations.

The two halves connect. If one transition serves every token and every lifecycle stage, then the structural part of the sampler-trainer mismatch disappears by construction. There is no prompt path and response path to disagree; the serving split is just a boundary in the token history. I noticed here that RLT does not claim this makes things easier. It claims this makes things consistent, and consistency is the property its own replay contract needs.

## What PDM does about it

The report separates measurement from repair, and its machinery falls into five layers. I keep the table because I will want it the next time I set up an RL training stack.

| Layer | What it gives you |
|---|---|
| Concepts | The $\Delta_{\mathrm{kernel}} / \Delta_{\mathrm{stale}}$ decomposition, distributional vs score parity, the support condition |
| Records | Immutable sampler-emitted log-probabilities plus a per-token execution signature; a trainer may append diagnostics but never replace the stored probability |
| Audits | A four-part fixed-weight parity check: state, forward-policy, score, and sampling parity, run independently of any optimization |
| Remedies | Four alignment routes, described below |
| Operations | Two declared modes: recurrent-target and declared parallel-target, each with guardrails |

The four remedies are where opinions will differ. The first is exposure: log the raw gap and its decomposition (on an audit subset, since separating the terms requires replaying the sampler operator at current weights). The second is to make one recurrence or chunk rule part of the model semantics and execute it everywhere, so there is nothing to reconcile; the report is careful that chunking does not create stochastic tokens in advance, so exact parallel generation still needs proposal and verification. The third is precision, with a ladder I now keep as a default checklist: FP32 baseline for recurrent states, decay products, and state-update reductions; FP64 as an audit reference rather than a training default; and no hope of recovering information by casting an already-rounded BF16 state up. The report captures the route's status in a sentence I quote a lot: precision is a mitigation, not an on-policy certificate. The underlying error recursion is worth writing once, since it explains why precision alone cannot close the contract:

$$\|e_t\| \le L_t\,\|e_{t-1}\| + \delta_{\mathrm{round}} + \delta_{\mathrm{schedule}}.$$

Here $L_t$ is a local state-sensitivity factor, and the two additive terms cover arithmetic rounding and a changed schedule (scan tree, chunk boundary, materialization rule). Higher precision shrinks $\delta_{\mathrm{round}}$. It does not shrink $\delta_{\mathrm{schedule}}$, and it does not stop amplification when products of the $L_t$ are large. That is the whole argument against treating FP32 as a proof.

The fourth remedy is the one I did not expect. Take speculative decoding's machinery and invert its roles: the recurrent implementation is the correctness authority, a cheap operator drafts, and modified rejection sampling (accept with probability $\min(1, p_i(y_i)/q_i(y_i))$, resample from the normalized positive part of $p_i - q_i$ on rejection) recovers the recurrent target up to the declared hardware numerics. The report says plainly that this helps only when drafting is cheap, acceptance is high, and verification batches well, and that it can provide no speedup when recurrent verification remains the serial bottleneck. It also draws a line I would have blurred: the proposal acceptance ratio is not the RL importance ratio. After exact verification, the committed tokens are distributed according to the recurrent target, so the record stores the target's log-probability as the behavior log-probability.

## What PDM's measurements actually show

Four evidence blocks and one registration. I care more about the "does not establish" column than about the headline numbers.

| Block | Headline | Establishes | Does not establish |
|---|---|---|---|
| Estimator check (vocab 4, horizon 6, exhaustive) | Full-trajectory IS matches the exact target gradient to $2.33 \times 10^{-16}$; raw token-local stays $0.1551$ away in $L^2$ | The estimator algebra is airtight in a toy world; local IS is a valid estimator of a declared local surrogate only | Anything about full-size models |
| Production-path audit (two H100 nodes, 8,192-token prompt) | p95 of the absolute log-ratio: 0.09293 (dense), 0.04999 (FP16 cache), 0.04433 (FP32 cache); no support violations | The gap is measurable and, in these configurations, small | Causality; generality; the two models are not matched runs |
| Mechanism test (128-route gated SSM) | FP32 passes forward and score checks at about $6 \times 10^{-7}$; BF16 fails both, with up to 4.69% top-1 and 0.670% gradient-sign disagreement | Precision is a real switch between passing and failing the checks; throughput rises from 13.0K to 136.0K tok/s across chunk sizes | That these error levels matter at scale |
| Full-distribution stress (16 histories) | Top-1 agrees in 100% of cases; total variation is still roughly 350x the dense value for the hybrid model; FP32 lowers TV by only 4.11% | Agreement at the argmax is not agreement in distribution | Production importance of the tail |
| Registered experiment | DAPO-17k-Eng; 512 trajectories per step, 200 steps, four arms, three seeds, 8xH100 per arm | Nothing yet; explicitly no results | Everything causal |

The registered protocol is the part I will watch. It is a full pre-registration (data gate, models, arms, seeds, evaluation) with the honest footnote that no arm is accepted and the earlier pilot is excluded from claims. I expected a scaling paper to bury that; instead it sits near the top.

## What RLT does about it, and what it costs

The mental model I keep from RLT is one transition, no reset. Every token, prompt or response, advances a single state $H_t = (s_t, C_t^D)$: a recurrent output plus layerwise sliding-window caches. The end of the prompt is a serving boundary in the bookkeeping, not a change in the conditional model. Gradients follow the same path: the reference training computation differentiates through the full history (full backpropagation through time with checkpointing), and the report is precise about where approximations leak in (detaching only the recurrent output still leaves gradient paths through the KV cache and encoder memory; SWA eviction is not a stop-gradient).

Three properties matter for anyone trying to falsify or adopt this. First, serving-split invariance: under identical kernels and precision, where you split prompt from response does not change the conditional distribution. The report itself flags this as mathematical equivalence only, so numerical discrepancy is exactly the door through which PDM's problem can walk back in. Second, the state depends only on the prefix, and the appendix warns that gradient products through $\partial s_t / \partial s_{t-1}$ alone miss paths through the cache; normalization does not bound products of the transition Jacobians. Third, cache exactness: a cached prefix can replace replay only when it matches the current parameters, positions, window semantics, and execution settings; a cache built under older weights satisfies neither the current-policy value nor the gradient requirement.

The cost side is where the report is unusually forthright. The decoder path is sequential through all T transitions, so encoder parallelism does not make prefill fully parallel, and the report states it makes no claim of a reduced-prefill speedup. Cache storage grows with encoder depth plus a bounded decoder window. Training wants full BPTT. And the depth story carries its own asterisk: unbounded structural depth is not unbounded useful depth, because gates, contraction, and learned projections can suppress what long paths contribute. I find that asterisk more interesting than the headline. The design makes depth available; learning is a separate question.

So what does RLT deliver, as of September 2026? A mechanism specification. No measured reasoning quality, no measured hardware efficiency, no scaling curves. Its claim is that if the discipline is followed, the fixed-weight parallel-recurrent mismatch has no place to live, because there is only one recurrence. That is a real design-level effect and a zero-evidence empirical effect.

## The seam between the two reports

Put PDM's contract checklist next to RLT's design, one row at a time. This is the table I check when someone tells me an architecture "solves" the mismatch.

| PDM module | RLT's response | Status |
|---|---|---|
| Canonical recurrent rule | One transition for all tokens, shared by pretraining, SFT, rollout, replay | Adopted at design level |
| Score parity | Explicitly inherited, citing PDM | Adopted |
| Support and behavior records | Full record semantics; metadata cannot restore missing support | Consistent |
| Precision ladder | Only a note that mathematical equivalence is not numerical equality | Gap |
| Kernel-level audit | Cache exactness only covers "is this state current-policy" | Gap |
| Proposal plus recurrent verification | Not addressed | Gap |

The convergence is in the diagnosis, and it is close to verbatim: both reports treat forward-probability parity as insufficient and score parity as a separate contract. The divergence is in coverage. RLT borrows PDM's logic but not PDM's three remaining modules, which means that by PDM's own standard, an RLT deployment is not automatically exact; it is exactly as auditable as any other pipeline, with a better starting structure. Structural mismatch (prompt path against response path) is designed away. Execution mismatch (the same RLT transition running on different kernels, precisions, and schedules) stays in PDM's jurisdiction.

I read that as complementarity rather than competition. RLT makes PDM's recurrent-target mode implementable at scale, because "canonical recurrence" stops being an audit nightmare and becomes the model. PDM remains the reason you cannot call the result exact without measuring it. Each report is the other's checklist.

## Critical assessment

### PDM: a strong evidence chain with an open causal loop

The measurement design is careful and the self-limiting is real: the report refuses to claim that gap magnitude implies capability loss, marks the cross-model numbers as descriptive, and labels the support stress as appendix-level evidence with a predeclared disposition. What I cannot yet see is the value proposition closed end-to-end. If production-path p95 log-ratios sit at 0.04 to 0.09 and even the full-distribution stress shows top-1 agreement across the board, the honest reading of the current data is that the gap is real, measurable, and modest in tested configurations. That makes this agenda insurance and auditability first, performance repair second, until the registered experiment says otherwise. I would also want a cost ledger: recurrent replay gives up sequence parallelism, the precision ladder costs compute, and proposal-verification can provide no speedup at all. Nobody has priced the discipline against the gap it removes.

Two smaller critiques. First, the empirical surface is small: one 8,192-token prompt path, sixteen stress histories, and a 128-route toy SSM. These demonstrate measurability, not prevalence. Second, much of the toolbox is assembled from known parts (importance sampling, modified rejection, FP32 baselines); the new contribution is the execution-level framing and the systematic contractualization. That is a framework contribution, and expecting an algorithmic surprise from it would be a category error.

### RLT: design discipline with zero measurements

The report is unusually honest for a design proposal, and that honesty is also its evaluation. It states no measured results, concedes that structural depth is not a reasoning guarantee, and concedes that neither weight tying nor temporal recurrence alone establishes novelty. What remains is a system-consistency argument: a lifecycle with one transition, replay semantics that follow from the architecture, and explicit interfaces for caches, multi-turn serving, and gradient approximations. The costs point the other way from its selling point: prefill loses parallelism, memory grows, and training wants full BPTT. Whether a quality-per-cost sweet spot exists is exactly the experiment that does not exist yet. And by PDM's standards, RLT's exact-replay claim is itself pending audit: no precision ladder, no kernel-level audit, and no treatment of stochastic computation beyond a deterministic-forward assumption.

### What both leave open together

The full narrative chain reads: diagnose the execution mismatch (PDM), design it away (RLT), show the training outcome change. The first link is measured, the second is specified, the third is empty. And there is no economics anywhere in either report. Both add cost in the name of correctness, and neither answers at what training scale the discipline pays for itself. My own working hypothesis, flagged as inference rather than result, is that the payoff is highly architecture-dependent: I expect dense softmax configurations to stay in the modest regime, while chunkwise state-space and delta-rule models carry structural gaps where alignment discipline could be the difference between a stable run and a mysterious one. A systematic gap taxonomy (architecture family against kernel pair against precision) does not exist. That matrix is the most valuable missing artifact I can see in this area.

## What I take from reading them together

Two hooks to keep. PDM: the checkpoint name is not the policy; the policy is the executable distribution, and an exact deployment gradient also includes its score function. RLT: one transition for the whole lifecycle, or replay fidelity becomes someone else's audit problem.

Three operational items, in order of how soon they apply to any RL stack I touch. One, audit for the recomputation bug: if old probabilities are rebuilt with the trainer engine, the kernel gap is being hidden, and the fix is to store sampler-emitted probabilities at sampling time. Two, adopt a minimal record contract before scaling: sampler log-probability, version, execution signature, sampling metadata, all immutable. Three, declare the target mode (recurrent-target or declared parallel-target) and do not blur it; the blur is where silent off-policyness lives.

The watch point is the registered DAPO experiment. When it runs, it becomes the first causal data on whether any of this changes training outcomes, and which arm (naive, raw local IS, stabilized IS, FP32 mitigation) moves first. My prior is mild: I expect FP32 state handling to matter more than ratio corrections for hybrid models, and neither to be dramatic on dense ones. I would like to be wrong in a specific direction. If the stabilized arm separates from naive at 200 steps, the priority of this whole line changes from insurance to intervention.

## Boundary conditions

- Every number in this reading is as of September 2026, and PDM is already on its second revision (August 24). Treat figures as versioned, not fixed.
- The "modest gap" reading is bounded by the tested configurations: 4B-class models, one 8,192-token prompt path, sixteen stress histories. It does not extend to frontier scale or other architectures.
- The hotspot hypothesis (structural architectures carry larger gaps) is mine. It could be wrong in either direction, and the registered experiment will only test two Qwen models.
- RLT's cost profile is argued, not measured. A working implementation could come out better or worse than the design reasoning suggests.
- Both reports share one unexamined premise: that execution fidelity, once measurable, deserves to be optimized for. If audit magnitude turns out uncorrelated with outcomes, the agenda shifts from engineering priority to bookkeeping.
- I read both as non-peer-reviewed project reports; the absence of external review is a boundary on everything above.

## Open questions

- When the DAPO registration runs, which arm separates from naive first: stabilized token-local IS, the FP32 recurrent-state mitigation, or none of them within 200 steps?
- What does the actual delta-kernel matrix look like across architecture families at fixed scale, and is there a cheap pre-training predictor for "this kernel pair will bite"?
- Can the four-part parity audit run as a CI gate inside a training stack, and what does one audit step cost relative to one optimizer step?
- Does RLT's serving-split invariance survive a production decode stack, or does hardware entropy re-open PDM's gap the moment the sampler is not the reference kernel?
- Is there a regime where full-BPTT recurrent replay is cheaper than the mismatch it removes, for example short-response verifiable-reward RL where outcome variance dominates?

[[Q]] Six months from now: has the DAPO-17k-Eng registration produced accepted results, and did any kernel-mismatch arm beat the naive baseline in evaluation? If not, check whether the blocker was infrastructure, a support violation, or a genuinely null effect.

## References

1. Yifan Zhang et al., "Reliable RL Scaling Requires Accounting for Prefill–Decode Kernel Mismatch", Pretraining-RL-Science project report, August 6, 2026 (revised August 24, 2026). https://github.com/yifanzhang-pro/Pretraining-RL-Science
2. Yifan Zhang et al., "Recurrent Looped Transformer", project report, September 2026. https://github.com/yifanzhang-pro/recurrent-looped-tranformer
3. Vault reading notes used for this draft: wiki/concepts/prefill-decode-kernel-mismatch.md and wiki/concepts/recurrent-looped-transformer.md, with source extracts at sources/prefill-decode-kernel-mismatch.md and sources/recurrent-looped-transformer.md (claim ranges c-34903 through c-34926 and c-34883 through c-34902).

