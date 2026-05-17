---
id: 20260518-qwen-pruning-rtx5090
title: "Pruning Qwen3.6-35B-A3B for RTX 5090: what I learned pushing MoE compression to its limit on a single GPU"
slug: qwen-pruning-rtx5090
date: 2026-05-18
lastmod: 2026-05-18
draft: false
format: "long"
domain: deep-learning
subdomain: model-compression
summary: "Six days of REAP-pruning a 256-expert MoE model to fit 32 GiB taught me that calibration data composition — not the pruning algorithm — is the primary quality lever, and that SFT with 4-bit frozen experts actively degrades generalization."
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
  - post-SFT-degradation
bibliography: ""
code_repo: ""
sources_used:
  - sources/qwen-pruning-rtx5090-development-story.md
---

## Why this exists

I had a 35B-parameter MoE model that needed to run on a single RTX 5090. The model in FP8 needed 34.4 GiB. The GPU had 31.84 GiB. The gap was 2.6 GiB — small enough to feel tantalizing, large enough to break every standard deployment pipeline.

Six days later, I had a pruned model at 26.3 GiB FP8 that scored 73.2% on HumanEval+, 51.0% on Toolcall, and 33.6% on MMLU. The path between those two states was not what I expected. The calibration data mattered more than the pruning algorithm. The SFT I assumed would improve quality made everything worse. And 90% of the model's parameters were invisible to every standard quantization framework on the market.

This article is what I wish I had known on day one. While the next round of pruning is on the going, I write down this article.

Qwen3.6-35B-A3B is a 256-expert MoE model optimized for agentic coding, and my only available hardware was a single RTX 5090. This covers the full pipeline — pruning, quantization, evaluation, and attempted recovery fine-tuning — on a single consumer GPU. It does not cover multi-GPU training setups, cloud-based inference, or comparisons with other pruning algorithms.

This assumes familiarity with MoE architecture, the basics of expert pruning, and quantization formats (FP8, INT4).

## Why a 2.6 GiB deficit ate my week

The standard view is that compressing a model by 8% to fit a memory budget is a routine calibration exercise — run AWQ or GPTQ, adjust the quantization config, ship it. A 2.6 GiB gap on a 34.4 GiB model is 7.5% compression. Well within what quantization alone should handle.

It was not routine.

The contradiction hit immediately: BitsAndBytes, GPTQ, and AWQ all quantize only 2D `nn.Linear` weights. MoE models store experts as batched 3D tensors — `[n_experts, in_dim, out_dim]` — for efficient grouped matrix multiplication. These 3D tensors contain roughly 90% of the model's total parameters, and every standard quantization tool simply skips them. No error. No warning. They just pass through at full precision.

So quantization alone could not close the gap. I needed expert pruning.

REAP (Router-Weighted Expert Activation Pruning) scores each expert by the conditional average of its gating weight times its activation norm over calibration tokens:

$$S_j = \mathbb{E}_{x \in X_j}[g_j(x) \cdot \|f_j(x)\|]$$

The intuition: an expert that gets low gating weight *and* produces small activations contributes little to the model's output and can be removed with minimal reconstruction error. REAP then removes the lowest-scoring experts and propagates residuals to keep the model's functional manifold topology intact.

The pruner works block-by-block: load one 750 MiB decoder layer to GPU, score all 183 surviving experts, prune the lowest, propagate residuals, save. About 25 minutes per experiment across 40 decoder layers. I ran three major experiments over six days.

The constraint that shaped everything: the RTX 5090's 32 GiB was simultaneously my inference platform, evaluation framework, and training environment. Pruning, eval, and SFT all competed for the same VRAM, and no two of them could run at the same time. At BF16, the model occupied 30.6 GiB — 97% of GPU capacity, leaving zero room for gradients or activations during training.

This total-resource competition, not the pruning algorithm, turned out to be the hard problem. And the variable that ultimately solved it was not what I expected.

## The calibration mix that rewrote my priors

I came to this expecting the pruning algorithm or the compression ratio to dominate quality. That is how the literature frames it: better importance scores, better pruning decisions.

The evidence says otherwise.

The key insight — and the one I keep coming back to — is that calibration dataset composition is the #1 quality lever for expert pruning. Not the algorithm. Not the ratio. The data you feed the pruner during the scoring pass determines more about the final model than any other decision you can make.

Here is what happened.

My first pruning experiment (v2) used four code-focused calibration datasets: evol-codealpaca, BigCodeBench, SWE-bench, and xlam. Pure code data. The result: HumanEval+ at 72.0%, Toolcall at 44.0%, and MMLU at — two categories at 0.0%.

Dead categories. The model could generate code, but it could not answer a general-knowledge question to save its life. The pruner had never seen general-knowledge tokens during scoring, so it had no way to know which experts mattered for those domains. Any expert that carried general-knowledge information was pruned or severely weakened.

For the v3 experiment, I switched to a 70/30 code-to-general mix. Same REAP algorithm. Same compression ratio. Just two general datasets added — 600 samples of MMLU and 600 samples of C4 — to a pool of 700 samples each from four code datasets.

| Category | v2 (pure code cal) | v3 (70/30 cal) | Change |
|---|---|---|---|
| MMLU Social Sciences | 0.0% | 33.3% | +33.3pp |
| MMLU Other | 0.0% | 34.3% | +34.3pp |
| HumanEval+ | 72.0% | 73.2% | +1.2pp |
| Toolcall | 44.0% | 51.0% | +7.0pp |

Every single metric improved. MMLU recovered from dead to 33%+. Code benchmarks improved too — HumanEval+ went up by 1.2 points, Toolcall by 7 points.

This is worth sitting with: adding 30% general-knowledge data to the calibration set did not just fix general knowledge. It also improved code performance. The pruner, given a more representative view of the model's operating distribution, made better pruning decisions *for both domains*.

I think of calibration data as the lens through which the pruner sees the model. A narrow lens (pure code) gives a sharp but myopic view — the pruner keeps only what it sees, and blinds the model to everything else. A wider lens (70/30 mix) lets the pruner see the full functional space, so it preserves the structures that serve the whole distribution, not just one mode.

The analogy breaks in one direction: you cannot just keep adding calibration domains forever. More data means longer scoring passes. The 70/30 mix already took 25 minutes per experiment. But within a practical budget, the evidence is clear — calibration composition matters more than any algorithmic tweak I could have made to the pruner itself.

## How 3D tensors broke every quantization framework

The calibration experiment got me a model that almost passed its quality gates. HumanEval+ at 73.2% was close to the 75% threshold. MMLU at 33.6% was still short of the 40% target. The obvious next step was recovery fine-tuning — SFT on the pruned model to lift the remaining benchmarks.

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

Everything regressed. MMLU collapsed back to v2-level. HumanEval+ lost 5.5 points. The only metric that stayed flat was Toolcall, and "flat" meant it did not participate in the collapse rather than being positive news.

The mechanism is specific and instructive — and worth pausing on because it explains an entire class of pipeline failures:

4-bit quantization injects noise into the forward pass of every frozen expert layer. That noise is deterministic — same input, same 4-bit weights, same quantization error — but it shifts the activation distribution that the trainable router and shared expert layers see. The trainable parameters adapt to this shifted distribution during SFT. They learn to work *with the noise profile* of the 4-bit experts.

When you remove the 4-bit quantization (merge the fine-tuned weights back into the original BF16 model for inference), the noise profile disappears. The trainable parameters are now running on clean activations, and they have overfit to a distribution that no longer exists.

This is why the training metrics looked great while the benchmarks collapsed. The model was not learning to generate better code or answer knowledge questions. It was learning to compensate for quantization noise in the frozen pathway. When the noise went away, the compensation became output distortion.

The result that I keep returning to: the pre-SFT v3 model is the project's best output. The calibration strategy was the lever. Fine-tuning was a trap.

## What I would do differently

If I started this project again, I would change four things. None of them are about the pruning algorithm.

| What I did | What I would do instead | Why |
|---|---|---|
| Jumped straight to production-scale SFT | Validate on a 2-layer toy model first | Seven failed approaches, ~4 hours of debugging, caught in minutes |
| Let eval and training share one venv | Isolate venvs from the start | huggingface_hub 1.5 vs 1.14 broke vLLM weight loading — a dependency interaction that cost hours to diagnose |
| Ran SFT before exhausting calibration experiments | Run all calibration experiments before SFT | 50/50 calibration never run; given calibration is the primary lever, it might have pushed MMLU past 40% |
| Shallow pruning (183/256 experts) + 4-bit SFT | Deeper pruning (154 experts) + BF16 SFT | At 0.40 compression ratio, model would be ~24 GiB BF16 — trainable without quantization tricks, avoiding the noise-overfitting trap entirely |

The most painful lesson is also the most transferable: validation loops catch pipeline bugs, but they do not catch strategy bugs. The SFT pipeline ran correctly — no crashes, no OOM, healthy training curves — and produced a worse model. A toy model would not have caught this either, because the noise-overfitting mechanism is a property of the real model's scale and quantization configuration.

The one experiment I most regret not running is the 50/50 calibration mix. If 30% general data pushed MMLU from 0% to 33%, 50% might push it past 40%. That experiment would have taken 25 minutes. The SFT that replaced it took 11.5 hours and made everything worse.

I remain unconvinced that any amount of recovery fine-tuning on this hardware configuration can beat a well-calibrated pruned model. The most promising path for quality SFT on a single GPU — Transformer Engine FP8 training — is installed and verified on Blackwell architecture but remains untested for full SFT.

## Boundary conditions

- The calibration-composition result is established for REAP on one model family (Qwen3.6-35B-A3B). It likely transfers to other MoE models and other pruning algorithms, but I have not tested this.
- The 4-bit expert unbatching technique works at the cost of inference speed — per-expert sequential Linear4bit forward passes are slower than the native grouped matrix multiplication. Not a problem for training, significant for production serving.
- The SFT degradation result applies specifically to training with 4-bit frozen experts. FP8 frozen experts or full BF16 SFT may behave differently. Transformer Engine FP8 training remains the untested most promising path.
- Single-GPU constraints shaped every conclusion. With multi-GPU hardware, the trade-offs between pruning ratio, quantization, and fine-tuning shift substantially.
- The v3 model at 26.3 GiB FP8 fits the RTX 5090 with headroom, but 128K+ context serving has not been validated. On-the-fly FP8 OOMs at 131K context.

## Open questions

- Does 50/50 calibration push MMLU past 40%? The experiment takes 25 minutes to run and was never scheduled.
- Can Transformer Engine FP8 training enable quality SFT on a single GPU without the noise-overfitting trap? The tools are installed on sm_120. The experiment is waiting.
- Is there a sweet spot in the pruning-ratio vs training-feasibility curve? At 0.40 compression (154 experts), the model hits ~24 GiB BF16 — barely trainable without quantization. Is the quality trade-off of deeper pruning + clean BF16 SFT better than shallow pruning + no SFT?
- Does the calibration-composition finding replicate across model families (DeepSeek, Mixtral, OLMoE) and pruning algorithms (Min-EAN, soft counting)?
- For the 256K context target: does offline FP8 quantization hold up at long context, or does activation sparsity at long inputs change the pruning field?

[[Q]] Six months from now: have you ran the 50/50 calibration experiment? If it pushes MMLU past 40%, was the entire SFT effort wasted?

## References

1. Fang et al., "REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression", arXiv:2510.13999, 2025.
2. Dery et al., "Finding Fantastic Experts in MoE Models", arXiv:2504.15447, 2025.
3. Zhang et al., "Efficient Expert Pruning in MoE LLMs", arXiv:2505.12345, 2025.
4. BitsAndBytes, Hugging Face quantization library, https://github.com/bitsandbytes-foundation/bitsandbytes.
5. TRL: Transformer Reinforcement Learning, Hugging Face, https://github.com/huggingface/trl.

