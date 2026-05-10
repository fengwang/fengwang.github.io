---
id: 20260510-llm-vlm-compression-foundations-clean
title: LLM/VLM Compression Foundations
slug: llm-vlm-compression-foundations-clean
date: 2026-05-10
lastmod: 2026-05-10
revision_of: 20260510-llm-vlm-compression-foundations
draft: false
format: "long"
domain: deep-learning
subdomain: model-compression
summary: A working notebook on compressing LLMs and VLMs — why overparameterization is the precondition, how pruning, quantization, and distillation interact, why P-KD-Q ordering dominates, and where compression breaks.
confidence: working
prerequisites:
- basic neural network architecture
- transformer attention mechanism
- floating-point representation
- cross-entropy loss
related: []
tags:
- pruning
- quantization
- knowledge-distillation
- token-compression
- vision-language-models
- neural-architecture-search
- hardware-aware-ml
bibliography: ""
code_repo: ""
sources_used:
- /data/feng/weave/wiki/llm-vlm-compression-foundations.md
- /data/feng/weave/sources/
- /data/feng/weave/claims/
---


I started looking at model compression because the numbers didn't add up. My GPU has 24GB of VRAM and the models I want to run need 40GB. The gap is a factor of two, which quantization claims to solve. But then I found papers about pruning, and distillation, and token compression, and hardware-aware NAS, and suddenly the question wasn't "which technique" but "which combination, in what order, for which hardware."

This article is my attempt to organize what I've learned into a coherent map. It is not a survey — there are good surveys for that. It is a working notebook: what I understand, what surprised me, and what I still can't explain.

**Thesis:** Compression works because neural networks are overparameterized for the expressivity they actually use. The hard part is knowing which bits are the ones that don't matter — and that answer depends on what you're compressing (text vs. vision-language), how you remove it (prune, quantize, or distill), what order you apply the steps (P-KD-Q), and what hardware runs the result.

**Scope:** This covers the foundations of LLM and VLM compression — the three pillars (pruning, quantization, distillation), token compression, NAS for compression, the empirical ordering evidence, failure modes, and hardware decision rules. It does not cover training-from-scratch efficiency, inference serving systems (vLLM, TensorRT-LLM) beyond their connection to compression, or retrieval-augmented generation.

**Prerequisites:** This assumes familiarity with transformer architectures, basic neural network training (backpropagation, gradient descent), floating-point representation, and cross-entropy loss.

## 1. Overparameterization is the precondition

If models weren't overparameterized, compression wouldn't work. The Lottery Ticket Hypothesis established this formally in 2018: dense, randomly-initialized networks contain subnetworks that, trained in isolation, match the full network's accuracy. For modern LLMs, the numbers are concrete — up to 30% of parameters can be pruned with negligible loss, and models hold 98-99% of original capabilities at just 15% pruning.

This overparameterization isn't a mistake. Sparse architectures are hard to train from scratch. We train dense and then compress because that's what the optimization surface allows.

The shape of the redundancy matters, and it differs by modality. This is something I initially underestimated — I thought all redundancy was weight-level, but the token-level and modality-dependent patterns are just as important for practical compression decisions.

| Modality | Redundancy pattern | Scale |
|----------|-------------------|-------|
| Images | Spatial — neighboring patches share textures/colors | — |
| Video | Spatiotemporal — consecutive frames share backgrounds; at 10fps, 1000 tokens/frame, a 90-min video yields ~54M tokens | 54M tokens/video |
| Audio | Salient info concentrates in sparse, brief segments and specific frequency bands | — |
| MLLM sequences | >50% of tokens get minimal attention; multimodal tokens are >80% of sequences in reasoning tasks | >80% of sequence |

All compression exploits some version of this: there are bits you can throw away because they don't change the output. The question is which bits.

## 2. The three pillars, and two newer additions

The literature converges on five categories. Three dominate practice:

| Method | Mechanism | What it reduces | Tuning required |
|--------|-----------|-----------------|-----------------|
| **Quantization** | Lower-bit weight/activation representation | Memory, potentially speed | Often tuning-free for LLMs |
| **Pruning** | Remove unimportant weights or structures | Parameters, compute | Recovery training at high ratios |
| **Distillation** | Transfer knowledge from large → small model | Parameters, compute | Training a student |



Token compression and Neural Architecture Search sit alongside these — newer, less universal, but important for specific scenarios.

### 2.1 Quantization: the hardware-sensitive frontier

Quantization converts float32/float16 weights to fewer bits. The fundamental tension: non-uniform quantization achieves higher accuracy because weights aren't uniformly distributed, but uniform quantization gets hardware support. You cannot have both accuracy and hardware efficiency simultaneously with existing methods.

A critical asymmetry drives the research: **weights are easy to quantize, activations are hard** because of outlier distributions. SmoothQuant addresses this by migrating quantization difficulty from activations to weights via per-channel scaling:

$$Y = X \cdot \text{diag}(s)^{-1} \times \text{diag}(s) \cdot W$$

This enables W8A8 quantization with minimal accuracy loss and a 2× throughput gain. The idea is simple — smooth the activation outliers into the weights where they do less damage — but the execution requires careful per-channel scaling factors.

#### The outlier problem, quantified

ICQuant reveals the structure of the problem in a way I find unusually clean: **the top 5% of weight outliers consume about 50% of the total value range** — meaning one full quantization bit gets wasted on just 5% of the weights. About 97% of weight channels have uniformly-distributed outlier positions (verified across Llama2/3/4 and Qwen2.5 families), which enables a per-channel partitioning strategy: separate codebooks for outliers and inliers, combined with index coding that costs ≈0.3 bits/weight vs. ≈1 bit for prior approaches.

#### The production baseline and the frontier

**FP8 (E4M3) on NVIDIA H100/B200** is the modern production baseline — essentially lossless 50% memory reduction from FP16. **4-bit PTQ (AWQ, GPTQ)** achieves virtually lossless quantization for models above 70B parameters. QuIP/QuIP# pushes to 2 bits by multiplying weight and Hessian matrices with randomized Hadamard transforms to make entries approximately i.i.d. Gaussian, enabling E8 lattice codebook quantization.

At the extreme frontier: LittleBit reaches 0.1 bits/weight through latent factorization; iFairy uses complex numbers {±1, ±i} for 2-bit "multiplication-free" inference via sign flips.

#### Edge and VLM-specific quantization

Edge deployment demands specialized methods. Q-VLM minimizes cross-layer dependency errors in LVLMs using activation entropy as a proxy. MBQ accounts for differential sensitivity between vision and language tokens, achieving up to 1.4× decoding speedup with a custom W3 kernel. P4Q introduces learnable prompts and a lightweight low-bit adapter to realign post-quantization feature distributions.

KV-cache quantization deserves separate mention. In PaLM-540B with batch size 512 and context length 2048, the KV cache alone needs 3TB — three times the model parameters. KIVI-style KV-cache quantization is now table-stakes for long-context serving.

### 2.2 Pruning: three strategies, three hardware outcomes

Pruning's real-world impact depends entirely on the pattern, because hardware can only exploit certain sparsity structures:

| Pruning type | What's removed | Hardware speedup | Examples |
|-------------|----------------|------------------|----------|
| **Unstructured** | Individual weights | None without sparse kernels | SparseGPT, Wanda |
| **Semi-structured** | Fixed patterns (2:4, 4:8) | Yes on NVIDIA Ampere+ | SparseGPT 2:4, Wanda N:M |
| **Structured** | Whole layers/heads/channels | Yes on commodity hardware | LLM-Pruner, NIRVANA, UKMP |



The key insight — and the one I keep coming back to — is that unstructured sparsity achieves the best accuracy but delivers zero speedup without special hardware. Structured pruning physically reduces matrix dimensions — immediate gains on any hardware, at higher accuracy cost. Semi-structured 2:4 sparsity is NVIDIA's compromise: hardware-supported on Ampere GPUs, but one-shot methods like SparseGPT and Wanda still suffer at 60-80% sparsity or with tight 2:4 constraints.

#### Beyond uniform sparsity: per-dimension pruning

A critical limitation of prior methods is uniform sparsity within layers — all output dimensions of a weight matrix get the same pruning ratio. TRIM demonstrates this is deeply suboptimal: individual output dimensions differ significantly in sensitivity. By assigning unique per-row sparsity ratios via iterative metric-driven adjustment, TRIM reduces OPT-13B perplexity at 80% sparsity from 6461 (Wanda-based OWL) to 324 — **over 95% reduction in perplexity** at the same sparsity level.

**NIRVANA** redesigns structured pruning by combining magnitude-scaled gradient saliency ($|\partial f / \partial W \cdot W|$) with Adam-based NTK stability guarantees. The dual criterion balances output preservation with training stability — Proposition 4.1 proves $\|\hat{\Theta} - \Theta\| \leq O(\varepsilon)$ under the SignGD kernel.

Key design choices:
- **Adaptive sparsity allocation**: parameter $\gamma$ controls MLP vs. attention pruning rates ($v_{\text{MLP}} = \gamma \cdot v_{\text{Attn}}$)
- **Hardware-aware dimension alignment**: all hidden dimensions forced to multiples of 8 for Tensor Core compatibility
- **Global joint ranking** across all layers/modules with a safeguard retaining ≥1 unit per layer to prevent collapse

At 50% sparsity, NIRVANA achieves WikiText2 PPL of 48.94 vs. 215.94 for LLM-Pruner on Llama3.1-8B. Ablation reveals that magnitude-based scoring alone causes extreme collapse (PPL ≈ 10⁵–10⁶), and removing adaptive allocation $\gamma$ raises PPL from 48.94 to 102.00.

**FastForward Pruning** reformulates sparsity allocation as a single-step RL problem. The RL state is defined solely by the global target sparsity $\sigma_t$ (enabling transfer learning), with a ratio-based reward function ( $\text{PPL}_{\text{dense}} / \text{PPL}_{\text{pruned}}$ ) that is scale-invariant for portability across model sizes. Results: 3.4× faster than EAS (6.13 vs. 23.6 GPU-hr) on LLaMA-V1 7B at 20% sparsity, with better PPL (6.64 vs. 6.89).

#### VLM-specific structured pruning: UKMP

Text-only pruning methods fail on LVLMs because they treat the language backbone in isolation, ignoring the vision-language interface. **UKMP (AAAI 2025)** introduces the first unified structured pruning framework purpose-built for LVLMs.

The **UKMI metric** combines three innovations:
- **Adaptive dual normalization**: block-wise normalization (by parameter volume) prevents large modules from dominating; modality-wise normalization balances vision and language components
- **First-order gradient saliency**: UKMP discards the second-order Fisher term because the convergence assumption of second-order derivatives is invalid when parameters are frozen — they retain first-order gradients
- **Angle distribution entropy**: entropy over 100 cosine bins weights the Taylor importance, penalizing parameters whose removal would cause large angular shifts in feature space

Recovery uses a **weight recalling module**: low-rank $P_2 Q_2 W^p$ transformation parallel to LoRA, trained through three-phase progressive distillation (vision-only MSE → vision+language MSE → task loss + KL). This module is reparameterizable — it folds into base weights after training at no inference cost.

At 50% pruning, UKMP achieves 47.81% VQAv2 accuracy (vs. 36.40% next-best) and 96.92 NoCaps CIDEr (vs. 85.51). Even at 20% pruning, the pruned BLIP-2 beats similarly-sized full BLIP-2 on OK-VQA and GQA.

### 2.3 Distillation: transfer without the baggage

Knowledge distillation trains a smaller student model to mimic a larger teacher. The three challenges: what knowledge to transfer, which algorithm to use, and how to design the student-teacher pair.

White-box distillation using KL-divergence at high temperature ($\tau = 4.0$) reveals the teacher's confidence across the full vocabulary, enabling finer-grained transfer than black-box methods relying on text outputs alone.

#### Curriculum distillation with selective reflection

**SRD (Selective Reflection Distillation)** demonstrates that not all training samples contribute equally — and that curriculum ordering matters. Easy-to-hard curriculum significantly outperforms reverse hard-to-easy ordering. An increasing temperature schedule ($\tau_0 = 1 \to \tau_n = 2$) is a key effectiveness driver; reversing it severely degrades results.

SRD achieves up to 39% training time reduction while using 75% of data, and consistently improves ROUGE-L by 3.92–15.53% across all 7 tested KD methods on 5 benchmarks. It is plug-and-play — no changes to model architectures, loss functions, or KD algorithms. It even enables distilled students to surpass teacher performance (26.07 vs. 25.15 ROUGE-L for OpenLLaMA2).

#### VLM-specific distillation

VLMs present unique challenges because cross-modal alignment must be preserved.

**Switch-KD (CVPR 2026)** unifies vision-language knowledge transfer within a shared text-probability space. The Visual-Switch Distillation pathway switches student visual outputs into the teacher's language pathway ($S\text{-ViT} \to T\text{-Projector} \to T\text{-LLM}$), producing visual-switch logits that represent the teacher's output distribution conditioned on student-encoded visual representations. This is supervised by **DBiLD loss**, which uses the Kneedle algorithm for adaptive top-k boundary detection and bidirectional reverse KL alignment on pairwise logit differences — outperforming forward KL by 0.5 points.

Switch-KD-0.5B achieves +3.6 Avg10 over TinyLLaVA-0.5B across 10 multimodal benchmarks and matches the 3B teacher with half the parameters. However, it requires feature-space and vocabulary consistency between teacher and student.

**Align-KD** rests on a critical architectural finding: cross-modal alignment in VLMs occurs primarily at the first attention layer's text-query-vision component ($A_{1, t \leftarrow v}$). Distilling only this targeted attention map achieves the same performance as distilling all maps while saving up to 50% computation. Distilling the wrong component is harmful: vision-query-vision attention KD collapses performance to 43.7 (vs. 64.4 baseline).

#### Bridging black-box and white-box distillation

The strongest teachers (GPT-4, proprietary models) are black-box — only text outputs available via API. White-box KD requires internal parameters. **GrayKD (AAAI 2026)** bridges this with a single-stage framework using no proxy teacher. Black-box rationales are injected through a lightweight cross-attention module — student hidden states as queries, rationale embeddings as keys/values, with 15% random masking for augmentation.

The efficiency gain is dramatic: GrayKD uses 610M parameters total vs. 2.06B for conventional KD pipelines. GrayKD Triple achieves 27.64 Avg Rouge-L, beating PromptKD + White Teacher (26.44) — despite using the same black-box GPT-4o-mini teacher as lower-scoring methods. Rationale diversity is the dominant factor: switching from multi-rationale to single-rationale reuse drops Rouge-L by 1.14 points.

### 2.4 Token compression: compressing the input, not the model

Token compression operates upstream of the three traditional pillars: instead of compressing model weights, it compresses the input. Approaches are categorized by modality (image/video/audio) and mechanism (transformation-based, similarity-based, attention-based, query-based). The key advantage: token compression is post-optimization, requiring no retraining.

I find this category theoretically elegant but practically limited — it only helps when tokens dominate the compute budget, which is true for video and long-context multimodal tasks but less so for standard image+text inference.

### 2.5 NAS for compression

**CompressNAS** treats Tucker rank selection as a global search problem, using an MSE-based accuracy proxy comparing decomposed vs. reference layer feature vectors. Existing zero-cost proxies (NASWOT, GraSP, SNIP, ZiCo) fail monotonic trends at higher ranks. CompressNAS builds two lookup tables ($\Delta\text{acc}$, $\Delta\text{flash}$) and uses ILP-based NAS to select ranks globally given a hardware budget — 8× compression of ResNet-18 on ImageNet with <4% accuracy drop.

**LLM-NAS** solves a problem I hadn't considered: LLM-driven architecture search exhibits exploration bias, repeatedly proposing designs within a narrow region of the search space. The fix is three innovations:
1. **Complexity-driven partitioning** into 6 disjoint niches defined by architectural complexity (nor_conv_3×3 count)
2. **LLM-powered prompt co-evolution** — prompts and architectures co-evolve across rounds
3. **XGBoost zero-cost predictor** aggregating 13 proxy metrics with Spearman correlation ~0.90 to ground truth

Search takes 3 minutes and 120 API calls vs. 2–17 GPU-days for supernet baselines. Removing partitioning drops hypervolume from 0.978 to 0.516. Removing the LLM entirely drops it to 0.843.

## 3. The P-KD-Q ordering: sequence matters

A systematic study on Qwen2.5-3B shows that compression ordering is not a detail — it determines whether the pipeline works at all.

| Sequence | Compression | G-Eval | PPL | Verdict |
|----------|-------------|--------|-----|---------|
| **P-KD-Q** | 3.68× | 0.733 | 5.048 | Best |
| KD-P-Q | 3.68× | 0.644 | — | Intermediate |
| P-Q-KD | 3.68× | 0.610 | — | Intermediate |
| KD-Q-P | 3.68× | — | 53.4 | Collapse |
| Q-P-KD | 3.68× | 0.060 | 34.5 | Near-zero |
| Q-KD-P | 3.68× | 0.080 | 24.1 | Near-zero |



The mechanism is specific and instructive — and worth pausing on because it explains an entire class of pipeline failures: NF4 quantization produces inference-only models incompatible with gradient-based training. Any sequence with Q before training steps is dead on arrival. The P-KD-Q sequence lets each step compound: pruning reduces the search space, distillation transfers knowledge to the pruned architecture, quantization reduces precision with minimal added loss.

A practical note: quantization alone achieves 3.00× compression (5886→1959 MB). Adding pruning and distillation adds only 0.68× more (to 3.68×) at significant complexity cost. For many use cases, quantization alone is the right answer.

## 4. Where compression fails

### 4.1 The alignment cliff in VLMs

VLM compression has a failure mode absent in text-only LLMs. At low compression ratios, structural pruning damages multimodal alignment (vision ↔ language) more than the language backbone; at high ratios, both degrade. This means for mild compression, fine-tuning only the multimodal projector is sufficient — you are repairing the alignment bridge, not the entire model.

UKMP addresses this directly through modality-wise adaptive normalization and its weight recalling module's progressive three-phase distillation. Text-only importance metrics (magnitude, gradient) cannot detect which parameters mediate the vision-language interface. The convergence assumption of Fisher information is also invalid for VLMs: frozen parameters retain first-order gradients, making second-order importance estimates actively misleading.

### 4.2 Extreme sparsity collapse

One-shot pruning methods degrade severely at 60-80% sparsity with semi-structured patterns. NIRVANA's ablation shows magnitude-based scoring alone causes PPL ≈ 10⁵–10⁶ at 50% sparsity. Attention-only pruning causes catastrophic collapse; joint pruning of attention and MLP yields the smoothest degradation.

### 4.3 Early quantization destroys trainability

Applying NF4 quantization before any other technique destroys trainability. Q-KD-P and Q-P-KD sequences achieve near-zero G-Eval scores (0.080, 0.060). The gradient-free nature of NF4-quantized models means they cannot participate in subsequent distillation or pruning recovery.

### 4.4 Layer sensitivity isn't uniform

In partial 2:4 sparsification, later layers are more sensitive than earlier ones — skipping the last third of the model yields the best accuracy. For LVLMs, widthwise pruning of attention heads and MLP neurons outperforms wholesale layer removal. And within a single layer, individual output dimensions differ dramatically in sensitivity.

## 5. How hardware changes everything

### 5.1 The hardware-taxonomy mismatch

The compression technique that looks best on paper often delivers zero real-world speedup. Models optimized for GPU do not run fast on CPU and mobile, and vice versa.

| If your target is... | Prefer... | Avoid... |
|---------------------|-----------|----------|
| Datacenter GPU (A100/H100) | Semi-structured 2:4 + quantization | Pure unstructured |
| Edge/CPU/Mobile | Structured pruning (widthwise) | Any unstructured or semi-structured |
| Long-context serving | KV-cache quantization | — |
| Extreme compression (≤45%) | Structured + distillation recovery | One-shot pruning alone |



### 5.2 Memory bandwidth is the real bottleneck

During autoregressive decode, each token generation requires loading the entire model from memory — a classic memory-bandwidth-bound operation. This explains why quantization helps more than pruning for decode latency (smaller weights mean less data movement), why KV-cache quantization becomes critical at long contexts, and why joint algorithm-hardware optimization is the only path to order-of-magnitude gains.

The Titanus accelerator takes this to the extreme: chiplet-based digital computing-in-memory stores all static weights on-chip, eliminating repeated weight reloading during decode — a 39.4× reduction in off-chip memory access.

### 5.3 The edge reality

CLIP-B/16 at 149.6M parameters already exceeds Jetson Nano's 4GB RAM (no dedicated GPU), causing frequent memory swaps that kill real-time performance. Edge deployment demands the full toolbox: pre-deployment compression, efficient fine-tuning, runtime optimization, and careful security/privacy handling.

### 5.4 A practical decision flow

For compressing an existing LVLM:

1. **Extremely low resources, no recovery training**: Widthwise pruning only. Accept accuracy loss.
2. **Moderate compression (≤30%)**: Layerwise pruning + multimodal projector fine-tuning (5% of original data suffices).
3. **High compression (≤45%)**: Widthwise pruning + supervised fine-tuning + hidden-state distillation.
4. **For any combination**: 4-bit quantization adds ~+0.1 PPL on top of sparsity.
5. **Always: P-KD-Q ordering.** Never quantize before training.

## 6. Where we are in 2025

The compression field has matured from experimental techniques into an engineering discipline with clear tiers:

| Tier | Category | Examples | Status |
|------|----------|----------|--------|
| **Production** | Foundational pruning | SparseGPT, Wanda, LLM-Pruner | Deployed |
| **Production** | 4-bit quantization | AWQ, GPTQ, NF4 (QLoRA) | Deployed |
| **Production** | Inference engines | vLLM (PagedAttention), TensorRT-LLM | Deployed |
| **Production** | FP8 baseline | H100/B200 hardware-native | Deployed |
| **Experimental** | Extreme pruning | TRIM, NIRVANA, FastForward | Active research |
| **Experimental** | VLM-specific pruning | UKMP (UKMI + weight recalling) | Active research |
| **Experimental** | Ultra-low-bit quant | LittleBit (0.1-bit), ICQuant, iFairy | Active research |
| **Experimental** | Curriculum distillation | SRD | Active research |
| **Experimental** | VLM distillation | Switch-KD, Align-KD | Active research |
| **Experimental** | Black-box KD | GrayKD (610M params, no proxy) | Active research |
| **Experimental** | NAS for compression | CompressNAS, LLM-NAS, HAT | Active research |



I expect several of the experimental rows to move to production within 12-18 months. TRIM-style per-dimension pruning and ICQuant-style index coding are both conceptually simple enough to integrate into existing pipelines. UKMP's modality-aware pruning is clearly the right approach for VLMs — the question is whether it generalizes beyond BLIP-2 to LLaVA-style architectures.

## 7. A unifying mental model

Compression works because networks use fewer bits of information than they allocate parameters. The art is knowing which parameters carry that information. The answer depends on four things:

1. **What redundancy exists**: token-level, weight-level, layer-level — and modality-dependent. Video has spatiotemporal redundancy. MLLM sequences are >80% multimodal tokens that receive minimal attention. Within a single weight matrix, individual rows differ in sensitivity by orders of magnitude.

2. **How you remove it**: quantize, prune, or distill. Quantization targets precision. Pruning targets structure. Distillation targets knowledge transfer. Token compression targets the input directly.

3. **What order you apply techniques**: P-KD-Q is empirically optimal. Any sequence with Q before training steps fails catastrophically. This is not a heuristic — it follows from NF4's gradient-free nature.

4. **What hardware runs it**: This is what determines whether a 50% parameter reduction translates to a 50% latency reduction or no reduction at all. Unstructured sparsity wins on accuracy but loses on every hardware metric. Structured pruning is the opposite. Semi-structured 2:4 is NVIDIA's compromise.

What I find striking about the failure modes is how cleanly they carve the parameter space. In VLMs, the vision-language alignment is more fragile than the language backbone. In deep transformers, later layers carry disproportionate importance, and individual output dimensions within the same layer differ dramatically. Hardware is not an implementation detail — it defines which removal patterns become faster.

The frontier is hybrid, sequential, and precision-extreme: combining pruning, distillation, and quantization in the right order, with per-dimension granularity, pushing quantization to fractions of a bit — while ensuring the pipeline remains trainable throughout.

## Boundary conditions

- This model assumes the pretrained model is available. If you are training from scratch with a compression target, the entire framework shifts — you would design the architecture sparse from the start rather than compressing post-hoc.
- The P-KD-Q ordering evidence comes from a single systematic study on Qwen2.5-3B. I have not seen replication on larger models or different architectures. The mechanism (NF4 gradient-free) is general, but the magnitude of the ordering effect at other scales is unknown.
- UKMP has been validated primarily on BLIP-2. Its generalization to LLaVA, InternVL, or other VLM architectures is an open question.
- Token compression is effective when tokens dominate compute (long video, long context). For single-image QA, the gains are modest.
- The practical decision flow (Section 5.4) assumes access to fine-tuning resources. For truly zero-shot deployment, only quantization and token compression apply.
- I have not covered training-time efficiency (mixed precision, gradient accumulation, ZeRO, FSDP), which interacts with compression in deployment pipelines but is a separate topic.

## Open questions

- Does the P-KD-Q ordering effect replicate on models above 70B parameters? The mechanism is architecture-agnostic, but the magnitude could scale differently.
- Why do some layers tolerate 4-bit quantization while structurally similar layers fall apart at 6-bit? I suspect effective rank or singular value distribution, but I have not tested this.
- Can UKMP's modality-aware importance metric be adapted to video-language models, where the redundancy patterns are spatiotemporally structured rather than spatially structured?
- What is the minimum viable recovery data for structured pruning of LVLMs at 50%+ sparsity? The current evidence says 5% of original data for moderate compression and full SFT for high compression, but the boundary between these regimes is fuzzy.
- At what point does token compression become more practical than model compression for video understanding? The 54M-token number for 90-minute video suggests the crossover exists but I do not know where.

[[Q]] Six months from now: has UKMP been extended to LLaVA-style architectures, and does the modality-aware importance metric generalize beyond BLIP-2?

## References

1. Frankle & Carbin, "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", ICLR 2019.
2. Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", ICML 2023.
3. Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot", ICML 2023.
4. Sun et al., "Wanda: A Simple and Effective Pruning Approach for Large Language Models", ICLR 2024.
5. Ashkboos et al., "SliceGPT: Compress Large Language Models by Deleting Rows and Columns", ICLR 2024.
6. Ma et al., "LLM-Pruner: On the Structural Pruning of Large Language Models", NeurIPS 2023.
7. NIRVANA: "NIRVANA: Neural Implicit Removal via Verifiable Adam-based NTK Alignment for Structured Pruning", 2025.
8. TRIM: "TRIM: Per-Dimension Structured Pruning for Large Language Models", 2025.
9. FastForward: "FastForward Pruning: Efficient LLM Pruning via Single-Step Reinforcement Learning", 2025.
10. UKMP: "UKMP: Unified Knowledge Maintenance Pruning for Vision-Language Models", AAAI 2025.
11. ICQuant: "ICQuant: Index Coding Quantization for Large Language Models", 2025.
12. Switch-KD: "Switch-KD: Knowledge Distillation with Visual Switch for Efficient Vision-Language Models", CVPR 2026.
13. Align-KD: "Align-KD: Shallow-Layer Attention Alignment for Mobile Vision-Language Models", 2025.
14. GrayKD: "GrayKD: Gray-Box Knowledge Distillation for Large Language Models", AAAI 2026.
15. SRD: "Selective Reflection Distillation: Curriculum Knowledge Distillation for LLMs", 2025.
16. CompressNAS: "CompressNAS: Neural Architecture Search for Model Compression", 2025.
17. LLM-NAS: "LLM-NAS: Large Language Models for Neural Architecture Search", 2025.
18. Compression Ordering: "A Systematic Study of Compression Ordering for Large Language Models", 2025.
19. Multimodal Token Compression Survey, arXiv 2507.20198, 2025.
20. Efficient VLM Survey: "Efficient Vision-Language Models: A Survey", 2025.

