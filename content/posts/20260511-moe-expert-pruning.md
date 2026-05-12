---
id: 20260511-moe-expert-pruning
title: "MoE Expert Pruning: What Works, What Doesn't, and What We Still Don't Know"
slug: moe-expert-pruning
date: 2026-05-11
lastmod: 2026-05-12
draft: false
format: "long"
domain: deep-learning
subdomain: model-compression
summary: "A survey of expert pruning techniques for sparse Mixture-of-Experts language models, covering why pruning works, the pruning-vs-merging debate, how to score expert importance, the strategies that matter, and where the standard story breaks."
confidence: working
prerequisites:
  - mixture-of-experts architecture
  - transformer language models
  - basic concepts of model compression
related: []
tags:
  - moe
  - expert-pruning
  - sparsification
  - efficient-inference
  - model-deployment
  - mixtral
  - nllb
bibliography: ""
code_repo: ""
sources_used:
  - /data/feng/weave/wiki/expert-pruning.md
---

I spent the last week reading seven papers on expert compression for Mixture-of-Experts models. I went in assuming the landscape was settled: expert pruning was a useful technique for shrinking models, expert merging was a promising alternative, and the choice between them was mostly a matter of taste. I came out with a very different picture. The pruning-vs-merging debate flipped in later 2025, and the reason it flipped tells you something fundamental about how these models actually work.

The core tension is simple. Sparse Mixture-of-Experts models activate only a fraction of their parameters per token, but they still occupy all of them in memory. Mixtral 8×7B activates 2 of 8 experts per layer — yet all 8 experts (45B of 47B total parameters) must sit on the GPU. NLLB-200 has 1,536 experts; you need four 32GB GPUs just to load the thing. Expert compression asks: can we drop or combine the experts that rarely get used, and how much does it cost?

The answer, across all seven papers, is surprising in two ways. First, you can drop far more experts than I expected, and the costs are concentrated in specific capabilities rather than spread evenly. Second, and more unexpectedly, **when it comes to actual generative tasks — code, math, creative writing — expert pruning is decisively better than expert merging.** The merging methods that looked good on multiple-choice benchmarks collapse on tasks that require the model to actually generate tokens. The reason is structural, not empirical: merging removes the router's fine-grained control over experts, and on generative tasks, that control matters.

## Why expert pruning works at all

Here's the one-sentence model: **expert utilization in MoE models is long-tailed — a handful of experts do most of the work, and the rest are along for the ride.**

Think of it like a restaurant kitchen during a dinner rush. You have eight chefs at eight stations. On a given night, two chefs handle 80% of the orders. The other six are standing by, occasionally contributing a garnish, mostly drawing salary. If you fire four of them and redistribute their stations, dinner still gets served — maybe even faster, because the remaining chefs spend less time coordinating. That is expert pruning.

The numbers back this up. Heatmap analysis of Mixtral 8×7B on MMLU shows stark unevenness: Expert #2 in Layers 26 and 30 is heavily activated while Expert #7 in Layers 22 and 23 is barely touched [1]. The same pre-trained MoE model produces substantially different expert contribution patterns when fine-tuned on different tasks [2]. This task-specificity cuts both ways — it means pruning must be calibrated to the deployment domain, but it also means aggressive pruning is possible for narrow use cases.

**The distribution is not just long-tailed — it's task-dependent.** An expert that dominates on MNLI might be silent on CoLA. This is the core insight behind every pruning strategy: you are not removing universal knowledge. You are removing specialists that your specific task does not call.

Where the analogy breaks: experts are not independent chefs. The router learns to distribute tokens across experts during pre-training, and removing experts changes the routing distribution for the survivors. This is why naive pruning based on activation frequency alone performs *worse than random pruning* [3]. The router expects a full kitchen.

### Knowledge redundancy: the surprising overcapacity result

Here is the finding that made me stop and re-read: pruning 4 of 8 experts in Mixtral 8×7B-Instruct *improves* SQuAD accuracy from 53.4% to 75.4%, without updating any remaining expert parameters [4]. This is not a typo. Removing half the experts makes the model *better* at question answering.

The mechanism: pruning simplifies the routing problem. With 8 experts, the router must learn to partition the hidden space across many specialists — a hard optimization problem. With 4, the routing is easier, and the remaining experts each get a cleaner slice of the input distribution. The router stops sending ambiguous tokens to the wrong specialist.

This overcapacity effect is not universal, but it recurs: on data-limited downstream tasks, a single-expert model can outperform the full multi-expert counterpart. After fine-tuning a pruned Mixtral 8×7B on MetaMathQA, the 7-expert model slightly *exceeds* the original 8-expert model on GSM8K (81.50 vs. 81.43) [3]. A single expert in Mixtral 8×7B-Instruct operates without model collapse [4].

## Pruning vs. merging: the debate that flipped in 2025

Until mid-2025, the story seemed clear. Expert merging — clustering and averaging experts rather than discarding them — was winning. M-SMoE and HC-SMoE showed that merging outperformed pruning when measured by perplexity and multiple-choice question answering benchmarks. If you only looked at those numbers, merging was the smarter choice. Retain information from all experts. Avoid the binary brutality of pruning.

Then REAP showed up and asked: what happens when you actually make these models *generate* tokens?

**The answer is a head-on collision between the two approaches.** On code generation, REAP achieves a mean accuracy decrease of only 1.9% at 25% compression and 6.9% at 50% compression. Merging methods? HC-SMoE and M-SMoE degrade more than 5% at 25% and more than 20% at 50% [7]. On creative writing and mathematical reasoning, the same pattern holds. Merging is not slightly worse — it is qualitatively broken on generative tasks at 50% compression.

Lasby et al. didn't just report the numbers. They derived *why* this has to be the case. When a router selects two experts f_i and f_j for a token, it produces a dynamic mixture r(x)·f_i(x) + (1−r(x))·f_j(x), where the mixing ratio r(x) depends on the input. After merging, the router must apply the summed gate to a constant convex combination — a static merged expert. The merged model must approximate a dynamic, input-dependent target with a static one. The resulting irreducible error is proportional to the router's policy variability Var[r(x)] and the functional gap between the merged experts ∥Δ_ij∥ [7].

Pruning doesn't have this problem. When you prune expert j, the router still controls each surviving expert independently. Pruning only incurs error when the pruned expert was in the top-k set, and that error is proportional to its gate-value g_j — it does not penalize policy variability at all [7]. The mathematical difference is clean: pruning is a coordinate subspace operation that preserves the functional manifold's topology. Merging introduces novel functions and collapses the manifold toward its center — by up to 100× reduction in spread in late layers of high-granularity models [7].

**Here's what this means in practice.** I can now look at a compressed model and predict failure modes based on the operation, not just the sparsity level. Merged model outputs have significantly lower N-gram diversity and their logits diverge from the original model more rapidly during auto-regressive generation [7]. The tokens drift. The model stops sounding like itself. MC benchmarks missed this entirely because they never asked the model to string tokens together — they only asked it to rank answer choices in a single forward pass.

One more uncomfortable finding: when merging *does* work well, look closer. HC-SMoE produces a high prevalence of **singleton clusters** — single-expert clusters that are functionally indistinguishable from keeping the expert unmerged [7]. The "merging" that succeeds is pruning plus a few mega-clusters of the truly redundant experts. And those mega-clusters are fragile: restricting the maximum cluster size to 32 experts causes large accuracy drops [7].

A separate problem compounds this. The L2-distance between clustered expert weights, even after weight-matching permutation, greatly exceeds the distance between pretrained and instruction-fine-tuned checkpoints. Singular-vector alignment remains poor [7]. Merging experts is fundamentally harder than the widely successful technique of model merging, and we should stop assuming the two are similar problems.

## How to score expert importance

The choice of importance criterion is the single biggest lever in expert pruning. I organize the criteria by what information they use.

| Criterion | Source | What it measures | Best result |
|---|---|---|---|
| Alpha score (accumulated gating weight) | Chen et al. 2022 [2] | Weighted contribution to output | Single expert preserves 99.3% of full MoE |
| Soft counting (accumulated softmax) | Muzio et al. 2024 [1] | Confidence margin of selection | 25% sparsity: 3.85 pp MMLU drop |
| Min-EAN (activation norm) | Jaiswal et al. 2025 [5] | Minimum activation magnitude | 14.02 PPL at 75% sparsity |
| **REAP** (conditional g_j∥f_j∥) | Lasby et al. 2025 [7] | Gate × activation, conditional | Near-lossless at 50%, up to 1T params |
| Importance product (top1 × exp(conf)) | Koishekenov et al. 2023 [6] | Combined activity and confidence | 80% pruning, chrF++ Δ = 0.29 |
| Activation frequency alone | Lu et al. 2024 [3] | Simple token count | Worse than random |

REAP deserves special attention because it's the first criterion explicitly designed to minimize the reconstruction error bound. Its saliency score computes the conditional average of g_j(x)·∥f_j(x)∥ over only those tokens where expert j is active [7]. This decouples functional impact from usage frequency — a specialist expert that activates rarely but contributes heavily when it does won't be pruned just because it's infrequent. Min-EAN held the previous crown among 16 criteria benchmarked by MC-Suite [5]. REAP now looks like the new baseline for generative tasks, especially at scale.

**The easy heuristic is still wrong.** Simple activation frequency — counting how many tokens each expert processes — does worse than random selection [3]. The router's assignment frequency is not the same as contribution.

Domain-specific calibration delivers the biggest gap I've seen in any compression result. When REAP calibrates on C4 (general pre-training data) instead of domain-specific data (evol-codealpaca for code), code generation accuracy collapses — some compressed models produce 0% accuracy, failing to output coherent code at all [7]. This is not a matter of degree. The calibrating dataset determines whether the pruned model works or is completely useless on the target task. And this was already visible in earlier work: using MATH instead of C4 for calibration shifts expert selections in 28 of 32 layers of Mixtral 8×7B [3].

## Pruning strategies: the choices that matter

Once you have an importance score, you need to decide *how* to use it. Three choices define your strategy.

| Choice | Options | Trade-off |
|---|---|---|
| **Scope** | Global vs. layer-wise | Global: better quality but variable per-layer counts. Layer-wise: fixed memory layout but lower ceiling |
| **Schedule** | One-shot vs. iterative | One-shot: fast but importance rankings are stale post-pruning. Iterative: ~2× better PPL but needs re-estimation |
| **Timing** | Eager vs. staged | Eager: more optimization steps for survivors. Staged: better importance estimates from longer observation |

**Global vs. layer-wise.** Global pruning — sorting all experts across all layers by a single importance ranking — outperforms layer-wise on quality because it avoids the constraint of keeping a fixed number per layer [1]. But it creates deployment headaches: variable per-layer expert counts mean variable memory usage across tasks, requiring model recreation for each configuration [6]. Layer-wise pruning gives predictable memory layouts at the cost of some quality.

**One-shot vs. iterative.** One-shot pruning drops experts in a single pass. The problem: after you remove experts, the importance rankings of the survivors change. Iterative pruning re-estimates importance after each round, achieving ~2× better perplexity. Add task-agnostic finetuning between rounds and you get ~3× better [5]. One-shot and iterative pruning identify *substantially different subsets* of experts at the same sparsity level — they produce effectively different subnetworks [5]. REAP demonstrates that with the right criterion, one-shot pruning can be remarkably effective even at 50% compression on models up to 1T parameters [7], but the iterative advantage likely still holds.

**Eager vs. staged.** Eager (progressive) pruning drops experts early using a dynamic threshold T = β / Z where Z is the number of surviving experts [2]. The earlier you drop, the more training steps you can dedicate to the selected expert. Eager consistently wins [2].

### The NLLB-200 special case: language-specific pruning

The NLLB-200 translation model surfaces a phenomenon that the Mixtral papers miss: **language-specific expert emergence**. In the decoder, Jaccard similarity of selected experts is 68–87% for the same target language versus only 13–39% for different target languages [6]. Per-language pruning (source language for encoder, target language for decoder) performs as well as per-language-pair pruning while requiring only L configurations instead of L² [6]. An unbalanced 3:1 encoder-to-decoder ratio yields the best quality [6].

## Beyond pruning: complementary techniques

Static expert pruning rarely stands alone. Four complementary techniques compound its gains, and there's now a clearer distinction between approaches that help and approaches that hurt.

### Expert merging (the post-pruning variant — and why it's different)

EEP's expert merging is not the same thing as HC-SMoE or M-SMoE. EEP merges pruned expert knowledge into survivors *after* pruning, using learned Router Mapping and Expert Merging matrices, adding 5–7% accuracy improvement [4]. This is a knowledge transfer operation — the pruned experts are already gone and their useful information is folded into the survivors. It's fundamentally different from the HC-SMoE/M-SMoE approach of replacing entire expert groups with merged averages, which removes router independence and causes the collapse described above. The EEP variant is a net positive. The HC-SMoE variant is not, unless you're only evaluating on multiple-choice.

### Dynamic expert skipping

Static pruning removes experts permanently. Dynamic skipping removes them *conditionally* — dropping the second-ranked expert for a token when its routing weight is below a threshold β times the top expert's weight, yielding ~50% skipping probability [3]. The key finding: skipping is *complementary* to pruning. A model pruned to 6 experts with skipping achieves the same speedup as pruning alone to 4 experts, but with higher accuracy [3]. You get the speedup without the full accuracy cost.

### Active expert reduction and finetuning

Switching from top-2 to top-1 expert activation reduces forward-pass FLOPs by ~27% in Mixtral [1], but zero-shot top-1 routing drops SST5 accuracy from 50.8% to 42.6%. Recovery via entropy-based gating regularization plus annealing top-k reduction closes most of this gap (51.8% vs. 53.6% top-2) [1].

Task-agnostic finetuning (~1M tokens; benefits saturate) corrects the skewed load distribution caused by removing router entries. It doesn't change which experts are selected — it mitigates impact through load rebalancing. This finetuning is central enough that iterative prune-estimate-finetune cycles produce what Jaiswal et al. call **MoE Lottery Subnetworks** [5].

### Quantization after pruning

Pruning combines naturally with quantization without additional steps, unlike merging which requires block-scale reconciliation for block quantization formats [7]. Combining REAP with 4-bit quantization on Kimi-K2 achieves 87.5% total size reduction — a compression rate neither technique can reach alone [7].

## What the numbers actually say

Across all seven papers, the efficiency-performance trade-off is more favorable than I expected.

**At moderate sparsity (25–50% experts removed), the accuracy cost on generative tasks is remarkably low — provided you prune, not merge.** REAP achieves a 1.9% mean accuracy decrease at 25% compression and 6.9% at 50% on coding benchmarks [7]. On Qwen3-Coder-480B and Kimi-K2, pruning 50% of experts drops code generation accuracy by only 1.2% [7]. On SWE-Bench (agentic software engineering), REAP-pruned Kimi-K2 at 50% compression actually slightly *exceeds* the baseline (0.576 vs. 0.554) [7].

Compare with merging at the same compression: HC-SMoE and M-SMoE see >5% accuracy decrease at 25% and >20% at 50% on the same coding benchmarks [7]. Merging looks reasonable on MC benchmarks (~4% decrease at 25%) but the MC numbers don't predict generative performance. This gap — between discriminative and generative evaluation — is what the pre-REAP literature missed.

**At high sparsity (75–80% experts removed), the numbers depend heavily on task type and recovery technique.** At 75% sparsity, Min-EAN achieves 14.02 PPL versus 34.47 random [5]. NLLB-200 at 80% pruning achieves chrF++ 36.61 versus 36.81 full — a delta of −0.2 [6]. Expert dropping predominantly degrades instruction-following, not pretraining knowledge or reasoning; these capabilities can be substantially restored through K-shot examples or fine-tuning [5].

**The fastest path to deployment, based on the evidence, is: Base model → expert pruning → finetuning → instruction tuning.** Expert dropping yields greater benefits before instruction tuning than after [5]. With SFT after pruning, high-sparsity models can outperform full counterparts on easier tasks like BoolQ and ARC-easy.

## Where the standard story breaks

**The standard story:** expert utilization is long-tailed. You prune the tail. Light finetuning recovers the loss. Any compression method that reduces the expert count should work about as well.

This story is wrong in ways that matter, and REAP is the paper that forced the correction.

**Pruning and merging are not interchangeable.** They produce qualitatively different models with different failure modes. Merging loses the router's input-dependent control — an irreducible error proportional to the router's policy variability. Pruning preserves it. On discriminative tasks, the difference is hidden because ranking answers in a single forward pass doesn't require the model to maintain coherent generation. On generative tasks, the difference is dramatic [7].

**Discriminative metrics like perplexity and MC accuracy are poor proxies for generative quality.** This sounds obvious in retrospect, but the field relied on these metrics to claim merging > pruning. Jaiswal et al. had already warned that perplexity can be misleading for compressed LLMs [5]. REAP proved it with a clean experiment: merging methods that looked competitive on MC benchmarks collapsed on code generation to the point of producing 0% accuracy outputs [7]. If you evaluate a compressed model only on MC, you haven't evaluated it at all for real-world use.

**Expert-level sparsification still beats weight pruning, and the argument is now stronger.** Across equivalent sparsity levels, dropping whole experts outperforms Wanda by ~3.6% average accuracy and ~16.2% on ARC-c [5]. And expert pruning preserves manifold topology while weight pruning may not — the geometric analysis from REAP [7] provides a structural argument for why whole-expert removal is the more principled approach.

**High-vocabulary-coverage experts hurt when dropped** — the specialist-generalist tension is real. If an expert handles many distinct tokens, removing it does outsized damage [5]. This suggests that pre-training methods that push experts toward specialization may make pruning easier in one sense (more experts are "dispensable") but harder in another (the remaining generalist experts carry structural load that can't be removed).

**Dominant experts have lower stable-rank** — a clean signal for identification but not yet exploited for additional compression [5].

**The second-pass degradation puzzle.** Two-pass eager-drop pruning degrades performance compared to a single pass, with average GLUE dropping by 0.58 points [2]. More iteration is not always better. But the REAP paper shows that a strong criterion in a single pass can go remarkably far — one-shot REAP on a 1T-parameter model at 50% compression is near-lossless on code [7]. The lesson isn't "one-shot > iterative." It's that criterion quality and scale-appropriate calibration dominate the one-shot vs. iterative trade-off.

## Boundary conditions

- The enumeration-based pruning approach in Lu et al. [3] works for 4–8 experts per layer but becomes computationally intractable at 32+ experts. The combinatorial explosion is unresolved.
- Gradient-free methods like EEP's evolutionary strategy [4] have been studied only on the Mixtral family. Whether they generalize to architectures with many more experts is unknown.
- HC-SMoE's mega-clusters containing tens of experts are fragile — restricting maximum cluster size to 32 causes large accuracy drops [7]. Coherently merging many experts remains an open problem.
- Hallucination and over-generation have been observed in pruned translation models, with global threshold methods more sensitive than fixed-per-layer pruning [6].
- All seven papers study static expert counts per layer. None address dynamic architectures where expert count varies by input complexity.
- Qwen2-MoE experts are notably homogeneous — the "expert specialization" narrative is architecture-dependent [4].
- Merging methods require recording activations from every expert for every token during calibration, making them more expensive at scale than pruning methods [7].
- The pruning-vs-merging analysis from REAP [7] applies to one-shot, no-fine-tuning compression. Whether fine-tuning after merging can recover the policy variability loss is not addressed.

## Open questions

1. The REAP criterion's derivation minimizes a reconstruction error bound assuming one-shot pruning. Can the same router-gate × activation-norm logic be extended to iterative pruning, and does it produce even better results?
2. Merging fails on generative tasks because it removes router independence. But could you *train* a model to be merge-friendly — by regularizing expert functional similarity or router policy smoothness — and get the memory savings of merging without the generative collapse?
3. What is the interaction between expert pruning and quantization at scale? REAP showed the combination works [7], but only on one model family (Kimi-K2 at 4-bit). Do pruned experts tolerate lower-bit quantization better or worse than full experts?
4. The "MoE Lottery Subnetworks" framing [5] has only been studied up to Mixtral 8×22B. Does it hold at the scale REAP demonstrated (480B–1T parameters)?
5. The vocabulary coverage finding [5] — high-coverage experts hurt when dropped — implies a tension with specialization. If you make experts more specialized, you might make pruning easier, but you risk creating fragile specialists that cannot be removed. Which direction wins?
6. No paper in this set studies pruning during pre-training rather than post-training. Could you train an MoE from scratch knowing it will be pruned and get a better result?

[[Q]] Six months from now: has the community converged on REAP as the default one-shot pruning criterion, or has the merging community produced a variant that recovers router independence and closes the generative-task gap?

## References

1. Muzio et al., "SEER-MoE: Sparse Expert Efficiency through Regularization for Mixture-of-Experts", arXiv:2404.05089, 2024.
2. Chen et al., "Task-Specific Expert Pruning for Sparse Mixture-of-Experts", arXiv:2206.00277, 2022.
3. Lu et al., "Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models", ACL 2024.
4. Liu et al., "Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs", arXiv:2407.00945, 2024.
5. Jaiswal et al., "Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations", arXiv:2504.05586, 2025.
6. Koishekenov et al., "Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model", ACL 2023.
7. Lasby et al., "REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression", arXiv:2510.13999, 2025.

