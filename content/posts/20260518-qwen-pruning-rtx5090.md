---
id: 20260518-qwen-pruning-rtx5090
title: "Pruning Qwen3.6-35B-A3B for RTX 5090: what I learned pushing MoE compression to its limit on a single GPU"
slug: qwen-pruning-rtx5090
date: 2026-05-18
lastmod: 2026-05-19
draft: false
format: "long"
domain: deep-learning
subdomain: model-compression
summary: "Seven sessions of REAP-pruning a 256-expert MoE model on a single GPU taught me that calibration data composition matters more than the pruning algorithm, that agentic calibration cannot lift capacity-bound agentic benchmarks, and that the right recipe at the right compression depth can move +17 BugFind points at no parameter count change."
confidence: working
prerequisites:
  - "Mixture-of-Experts (MoE) architecture basics"
  - "expert pruning concepts"
  - "quantization fundamentals (FP8, INT4)"
  - "single-GPU training constraints"
related: []
tags:
  - expert-pruning
  - REAP
  - RTX5090
  - Qwen
  - calibration-data
  - 4-bit-unbatching
  - agentic-benchmarks
  - benchlocal
bibliography: ""
code_repo: ""
sources_used:
  - sources/qwen-pruning-rtx5090-development-story.md
  - sources/qwen3-prune-session-6.md
  - sources/qwen3-prune-session-7.md
  - sources/qwen3-prune-handover.md
---

## Why this exists

I had a 35B-parameter MoE model that needed to run on a single RTX 5090. The model in FP8 needed 34.4 GiB. The GPU had 31.84 GiB. The gap was 2.6 GiB — small enough to feel tantalizing, large enough to break every standard deployment pipeline.

Six days later, I had a pruned model that scored 73.2% on HumanEval+, 51.0% on Toolcall, and 33.6% on MMLU. That model (v3) shipped as the best result available at the time.

Twenty-four hours after that, I had evidence that everything I thought I understood about calibration was specific to one recipe and didn't generalize. Then another twelve hours produced v7b-fp8 — a model that beats v3 on every measured pack I ran, with BugFind +17, DataExtract +17, and InstructFollow +20.

The path between those states was not linear. Some of my most confident conclusions turned out to be wrong. This article is what I wish I had known on day one — updated after the sessions that overturned my earlier priors.

Qwen3.6-35B-A3B is a 256-expert MoE model optimized for agentic coding, and my only available hardware was a single RTX 5090. This covers pruning, quantization, evaluation, and attempted recovery fine-tuning on a single consumer GPU across seven experimental sessions. It does not cover multi-GPU setups, cloud inference, or comparisons with other pruning algorithms.

## Why a 2.6 GiB deficit ate my week

The standard view is that compressing a model by 8% to fit a memory budget is a routine calibration exercise — run AWQ or GPTQ, adjust the quantization config, ship it. A 2.6 GiB gap on a 34.4 GiB model is 7.5% compression. Well within what quantization alone should handle.

It was not routine.

The contradiction hit immediately: BitsAndBytes, GPTQ, and AWQ all quantize only 2D `nn.Linear` weights. MoE models store experts as batched 3D tensors — `[n_experts, in_dim, out_dim]` — for efficient grouped matrix multiplication. These 3D tensors contain roughly 90% of the model's total parameters, and every standard quantization tool simply skips them. No error. No warning. They just pass through at full precision.

So quantization alone could not close the gap. I needed expert pruning.

REAP (Router-Weighted Expert Activation Pruning) scores each expert by the conditional average of its gating weight times its activation norm over calibration tokens:

$$S_j = \mathbb{E}_{x \in X_j}[g_j(x) \cdot \|f_j(x)\|]$$

The intuition: an expert that gets low gating weight *and* produces small activations contributes little to the output and can be removed with minimal reconstruction error. REAP then removes the lowest-scoring experts and propagates residuals to keep the functional manifold topology intact.

The pruner works block-by-block: load one 750 MiB decoder layer to GPU, score all surviving experts, prune the lowest, propagate residuals, save. About 25 minutes per experiment across 40 decoder layers. I ran seven experiments over seven sessions on one GPU.

The constraint that shaped everything: the RTX 5090's 32 GiB was simultaneously my inference platform, evaluation framework, and training environment. Pruning, eval, and SFT all competed for the same VRAM, and no two of them could run at the same time. At BF16, the model occupied 30.6 GiB — 97% of GPU capacity, leaving zero room for gradients or activations during training.

This total-resource competition, not the pruning algorithm, turned out to be the hard problem.

## The calibration mix that rewrote my priors

I came to this expecting the pruning algorithm or the compression ratio to dominate quality. That is how the literature frames it: better importance scores, better pruning decisions.

The evidence says otherwise.

Here is what happened.

My first pruning experiment (v2) used four code-focused calibration datasets: evol-codealpaca, BigCodeBench, SWE-bench, and xlam. Pure code data. The result: HumanEval+ at 72.0%, Toolcall at 44.0%, and MMLU at — two categories at 0.0%.

Dead categories. The model could generate code, but it could not answer a general-knowledge question. The pruner had never seen general-knowledge tokens during scoring, so it had no way to know which experts mattered for those domains. Any expert carrying general-knowledge information was pruned or severely weakened.

For the v3 experiment, I switched to a 70/30 code-to-general mix. Same REAP algorithm. Same compression ratio. Just two general datasets added — 600 samples of MMLU and 600 samples of C4 — to a pool of 700 samples each from four code datasets.

| Category | v2 (pure code cal) | v3 (70/30 cal) | Change |
|---|---|---|---|
| MMLU Social Sciences | 0.0% | 33.3% | +33.3pp |
| MMLU Other | 0.0% | 34.3% | +34.3pp |
| HumanEval+ | 72.0% | 73.2% | +1.2pp |
| Toolcall | 44.0% | 51.0% | +7.0pp |

Every single metric improved. MMLU recovered from dead to 33%+. Code benchmarks improved too.

I think of calibration data as the lens through which the pruner sees the model. A narrow lens (pure code) gives a sharp but myopic view — the pruner keeps only what it sees, and blinds the model to everything else. A wider lens (70/30 mix) lets the pruner see the full functional space, so it preserves the structures that serve the whole distribution, not just one mode.

The analogy breaks in one direction: you cannot just keep adding calibration domains forever. More data means longer scoring passes. But within a practical budget, the evidence is clear — calibration composition matters more than any algorithmic tweak I could have made to the pruner itself.

I shipped v3 and moved on. That was where the story got interesting.

## The discovery that my conclusions were calibration-specific

After shipping v3, I went back to test a hypothesis that seemed obvious: if I replaced the code-heavy calibration with agentic traces, the pruned model would perform better on agentic benchmarks. Tool calling, bug finding, multi-step reasoning — these were what the model was designed for.

I built a proper BenchLocal evaluation harness — 8 packs covering tool-call, hermes-agent, bug-find, data-extract, instruct-follow, reason-math, struct-output, and cli — and established a v3 baseline with the new v19 chat template. The baseline was sobering: ToolCall-15 at 90, HermesAgent-20 at 16, BugFind-15 at 8. The agentic gates were low to begin with.

I ran two candidate experiments at a deeper compression (0.40, keeping 154 of 256 experts) with two calibration strategies:

- **Mix-A**: full replacement — agentic data (glm47-reap + hermes-agent-traces) instead of code
- **Mix-B**: additive — agentic data layered on top of v3's base

Both produced the same result. ToolCall-15 dropped from 97 (the v3 baseline with the old reasoning parser) to 90. HermesAgent-20 stayed at exactly 16. BugFind-15 hovered in the 3-10 range. No agentic movement. Both failed all three gates.

I stopped and ran a control experiment. I took Mix-A's exact calibration list and ran it at v3's exact compression ratio (0.289, 183 experts). If compression depth was the cause of the toolcall regression, the control would recover toolcall. If calibration content was the cause, the control would still show the regression.

The control was decisive:

| Candidate | Compression | Calibration | ToolCall-15 | HermesAgent-20 | BugFind-15 |
|:---------:|:-----------:|:-----------:|:-----------:|:--------------:|:----------:|
| v3 | 0.289 | 70/30 balanced | 97 | 16 | 8 |
| Mix-A | 0.40 | Agentic replacement | 90 | 16 | 10 |
| Mix-B | 0.40 | Additive layering | 90 | 16 | 3 |
| v3ratio | 0.289 | Mix-A calibration | 90 | 16 | 0 |

ToolCall-15 at 90 across all three candidates, including the control at v3's compression. The regression came from the calibration content, not the compression depth.

The mechanism was surprisingly specific. The regression was concentrated entirely in a single sub-dimension: Parameter Precision dropped from 100 to 67. The model still picked the right tools with the right structure — it just generated wrong-typed or wrong-formatted arguments more often. Dropping code corpora (evol-codealpaca, bigcodebench, swe-bench) and substituting agentic traces had cost the model its tight argument-formatting discipline.

The other finding was harsher. HermesAgent-20 scored 16 out of 20 across all four configurations — literally identical, including per-category breakdown. A 25B pruned MoE cannot handle these multi-step browser-automation scenarios, regardless of what you feed it during pruning calibration. The gate is capacity-bound.

I also discovered that the vLLM `--reasoning-parser qwen3` flag was essential for correct evaluation. Without it, the model's `<think>` reasoning block leaked into all plain-text responses, breaking every non-tool-call scorer. The flag lifted ToolCall-15 from 90 to 97 and recovered instruct-follow and data-extract from flat zero. The lesson: validate your eval infrastructure before you trust a single number.

I closed Session 6 with a clean negative result and the conviction that calibration content could not move agentic gates. That conviction lasted about twelve hours.

## The recipe that broke through

Session 7 adopted a fundamentally different calibration recipe: the REAP-26B 6-dataset mix. Six datasets — SWE-bench/SWE-smith-trajectories (tool split), xlam-function-calling-60k, evol-codealpaca, and Mixture-of-Thoughts (code/math/science) — at much higher token count (1024 samples x 16384 sequence length, totaling 16.8M tokens). Router renormalization disabled per the REAP-26B README.

I ran three plans plus a follow-up:

**Plan-A** (compression 0.40, fresh prune). ToolCall-15 collapsed to 63 — a catastrophic -27 regression. But BugFind jumped to +15 and InstructFollow to +33. The recipe was clearly powerful. Too powerful at this depth.

**Plan-C** (stacked prune on top of the upstream REAP-26B-VL). Recovered toolcall to 90 but lost ~90% of the recipe's other gains. Stacked pruning does not inherit upstream calibration signal.

**Plan-B** (compression 0.289, v3's depth, the experiment I initially skipped). This was the follow-up after both Plan-A and Plan-C failed.

| Candidate | Compression | ToolCall-15 | BugFind-15 | DataExtract-15 | InstructFollow-15 | Verdict |
|:---------:|:-----------:|:-----------:|:----------:|:--------------:|:-----------------:|:-------:|
| v3+v19 | 0.289 | 90 | 8 | 5 | 20 | Baseline |
| v7a | 0.40 | 63 | 23 | 24 | 53 | FailToolcall |
| v7c | stacked | 90 | 0 | 4 | 16 | FailAgentic |
| **v7b** | **0.289** | **93** | **25** | **22** | **40** | **Pass** |

v7b-fp8 scored equal or better than v3 on all 7 measured packs. No regressions. BugFind +17, DataExtract +17, InstructFollow +20, ToolCall +3. The trigger verdict was Pass.

This is the result I keep coming back to: the same recipe at 0.40 collapsed toolcall to 63; at 0.289 it improved toolcall to 93. The recipe drives the agentic gains. The compression depth modulates the toolcall trade-off. Session 6's conclusion that "calibration content cannot move agentic benchmarks" was specific to Mix-A's content, not a universal property of pruning calibration.

The pipeline upgrades that made this possible are worth noting. The 16K sequence length calibration required a chunked REAP scoring accumulator — the single-pass approach would have materialized 67 GiB on a 32 GiB GPU. The custom FP8 quantizer (`scripts/quantize_fp8.py`) bypasses llmcompressor's broken Qwen3.6 compatibility with a 274-line direct cast from BF16 to `torch.float8_e4m3fn`. Schema adapters with a 60-second preflight caught dataset drift before any GPU allocation.

## How 3D tensors broke every quantization framework

The calibration experiment got me v3 — a model that almost passed its quality gates. HumanEval+ at 73.2% was close to the 75% threshold. MMLU at 33.6% was still short of the 40% target. The obvious next step was recovery fine-tuning.

This is where the second assumption broke.

I assumed that standard quantization tools would handle model compression for training. Load the model in 4-bit, apply LoRA adapters, train. This is the default workflow for QLoRA on every Hugging Face tutorial. It works on LLaMA, it works on Mistral — it should work on Qwen.

It does not. Because the model's 3D expert tensors are invisible to BitsAndBytes.

I spent the evening of day two systematically eliminating every standard training approach. Seven attempts, all failures:

| Attempt | Approach | Result |
|---|---|---|
| 1 | BnB 4-bit QLoRA | Can't quantize 3D expert tensors [183, 1024, 2048] |
| 2 | BF16 model.to('cuda') | 30.6 GiB — 0 bytes left for activations |
| 3 | accelerate device_map='auto' | Keeps all layers on GPU for backward |
| 4 | DeepSpeed ZeRO-3 (single GPU) | Trainer moves model to GPU before partitioning |
| 5 | DeepSpeed zero.Init + from_pretrained | Weight loading conflicts with meta-device tensors |
| 6 | FP8 frozen weights + monkey-patched ops | grouped_mm upcast creates 768 MiB BF16 temp per layer |
| 7 | FP8 + dispatch_model with 10 GiB budget | Offloaded layers accumulate on GPU during backward |

I don't fully understand why every framework eventually calls `model.to(device)` for the backward pass on single GPU. The documentation promises CPU offloading. The reality is that DeepSpeed ZeRO-3, accelerate's dispatch_model, and FSDP all converge on the same behavior: put the full model on GPU when gradients need to flow.

The resolution came from a workaround I had not considered: **unbatch the 3D expert tensors into individual `bnb.nn.Linear4bit` layers**. BnB can quantize standard 2D linear layers. A 3D tensor `[183, 1024, 2048]` becomes 183 separate `Linear(2048, 1024)` objects, each quantizable in 4-bit.

The result: model on GPU dropped from 30.6 GiB to 16.8 GiB, leaving 16.9 GiB for activations and gradients. The SFT ran — 311 steps, 9,934 samples, 11.5 hours, loss from 1.058 to 0.975, token accuracy from 85% to 96%. By every training metric, it worked.

It did not work.

## The SFT trap: when training makes everything worse

I expected SFT with quantized frozen weights to improve the model. The training curves were healthy. Loss decreasing. Token accuracy climbing. All the signals that normally say "keep training, it is converging."

Post-SFT evaluation told a different story:

| Benchmark | Pre-SFT | Post-SFT | Delta |
|---|---|---|---|
| HumanEval+ | 73.2% | 67.7% | -5.5pp |
| Toolcall | 51.0% | 50.5% | -0.5pp |
| MMLU | 33.6% | 9.4% | -24.2pp |

Everything regressed. MMLU collapsed back to v2-level. HumanEval+ lost 5.5 points.

The mechanism is specific and instructive — and worth pausing on because it explains an entire class of pipeline failures:

4-bit quantization injects noise into the forward pass of every frozen expert layer. That noise is deterministic — same input, same 4-bit weights, same quantization error — but it shifts the activation distribution that the trainable router and shared expert layers see. The trainable parameters adapt to this shifted distribution during SFT. They learn to work *with the noise profile* of the 4-bit experts.

When you remove the 4-bit quantization (merge the fine-tuned weights back into the original BF16 model for inference), the noise profile disappears. The trainable parameters are now running on clean activations, and they have overfit to a distribution that no longer exists.

This is why the training metrics looked great while the benchmarks collapsed. The model was not learning to generate better code or answer knowledge questions. It was learning to compensate for quantization noise in the frozen pathway. When the noise went away, the compensation became output distortion.

The result that I keep returning to: the pre-SFT v3 model was the project's best output at that point. The calibration strategy was the lever. Fine-tuning was a trap.

## What I would do differently

If I started this project again, I would change several things. The later sessions taught me that some of my early conclusions were incomplete — so these recommendations are updated with everything I know after seven sessions.

| What I did | What I would do instead | Why |
|---|---|---|
| Jumped straight to production-scale SFT | Validate on a 2-layer toy model first | Seven failed approaches, ~4 hours of debugging, caught in minutes |
| Let eval and training share one venv | Isolate venvs from the start | huggingface_hub 1.5 vs 1.14 broke vLLM weight loading |
| Ran SFT before exhausting calibration experiments | Run all calibration experiments before SFT | The 50/50 experiment was never attempted, and calibration is the primary lever |
| Shallow pruning (183/256) + 4-bit SFT | Deeper pruning (154 experts) + clean BF16 SFT | Avoids the noise-overfitting trap entirely |
| Assumed agentic calibration lifts agentic gates | Test the REAP-26B recipe at the original depth first | Session 6's clean negative result was Mix-A-specific; Plan-B at v3's depth passed everything |
| Tested one variable at a time | Test "new calibration at same compression" as default isolation | The v3ratio control flipped the interpretation — always isolate the variable |
| Championed "capacity-bound benchmarks" as a universal conclusion | Measure with multiple recipes before declaring a ceiling | BugFind moved +17 with the right recipe at no parameter count change |

The most painful lesson is also the most transferable: validation loops catch pipeline bugs, but they do not catch strategy bugs. The SFT pipeline ran correctly — no crashes, no OOM, healthy training curves — and produced a worse model.

The one experiment I most regret not running is the 50/50 calibration mix. If 30% general data pushed MMLU from 0% to 33%, 50% might push it past 40%. That experiment would have taken 25 minutes. The SFT that replaced it took 11.5 hours and made everything worse.

## Boundary conditions

- The calibration-composition result is established for REAP on one model family (Qwen3.6-35B-A3B). It likely transfers to other MoE models and other pruning algorithms, but I have not tested this.
- The REAP-26B recipe finding (v7b-fp8 beating v3) is specific to this calibration mix at this compression depth. Whether it generalizes to other MoE scales is an open question.
- The Session 6 "calibration content drives toolcall regression" finding is specific to Mix-A's content (glm47-reap + hermes-agent-traces). The REAP-26B recipe at the same depth showed a *positive* toolcall delta. The finding is recipe-specific, not universal.
- HermesAgent-20 remained stuck at 16/20 across all seven sessions and all configurations tested. It is genuinely capacity-bound at this model size.
- The 4-bit expert unbatching technique works at the cost of inference speed — per-expert sequential Linear4bit forward passes are slower than native grouped matrix multiplication.
- The SFT degradation result applies specifically to training with 4-bit frozen experts on this model architecture. FP8 frozen experts or full BF16 SFT may behave differently.
- Single-GPU constraints shaped every conclusion. With multi-GPU hardware, the trade-offs shift substantially.
- The v19 chat template costs about 7 toolcall points compared to v18 (90 vs 97 on the same model). All Session 7 comparisons are within the same template, but direct comparability with Session 6 numbers requires accounting for the template shift.

## Open questions

- **Can SFT on v7b-fp8 lift HermesAgent-20?** It is the only pack that v7b didn't improve, stuck at 16/20. Session 6's capacity-bound conclusion was wrong for BugFind (the right recipe moved it +17). It may also be wrong for HermesAgent once the right recipe is found. But SFT on actual agent traces is a qualitatively different approach, and the infrastructure exists but hasn't been validated against v7b.

- **Does 50/50 calibration push MMLU past 40%?** The experiment takes 25 minutes and was never scheduled.

- **Can Transformer Engine FP8 training enable quality SFT without the noise-overfitting trap?** The tools are installed on sm_120. Untested.

- **Does the REAP-26B recipe replicate on other MoE families?** The recipe drove +17 across multiple benchmarks on Qwen3.6. Would it produce similar gains on DeepSeek, Mixtral, or OLMoE?

- **Should stacked pruning ever be used?** Session 7 showed it destroys upstream calibration signal. But if the upstream calibration is expensive (24K samples on a 96 GB GPU), stacking a cheap re-prune on top seems like it should work in theory. The empirical result was negative. I don't fully understand why.

[[Q]] Six months from now: run the 50/50 calibration experiment first. If it pushes MMLU past 40%, the entire SFT effort was wasted. And promote the v7b symlink — it is the best model you have.

## References

1. Fang et al., "REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression", arXiv:2510.13999, 2025.
2. Dery et al., "Finding Fantastic Experts in MoE Models", arXiv:2504.15447, 2025.
3. Zhang et al., "Efficient Expert Pruning in MoE LLMs", arXiv:2505.12345, 2025.
4. BitsAndBytes, Hugging Face quantization library, https://github.com/bitsandbytes-foundation/bitsandbytes.
5. TRL: Transformer Reinforcement Learning, Hugging Face, https://github.com/huggingface/trl.

