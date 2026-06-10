---
id: 20260610-dflash-vs-diffusiongemma
title: "Two ways to diffuse text: DFlash block diffusion vs DiffusionGemma"
slug: dflash-vs-diffusiongemma
date: 2026-06-10
lastmod: 2026-06-10
draft: false
format: "long"
domain: deep-learning
subdomain: efficient-inference
summary: "A side-by-side comparison of two recent approaches that apply diffusion to text generation — DFlash using a tiny block diffusion model as a speculative drafter, and DiffusionGemma building a standalone text diffusion model on the Gemma 4 backbone."
confidence: working
prerequisites:
  - "autoregressive language model basics (tokens, logits, causal attention)"
  - "speculative decoding concept (draft-then-verify)"
  - "basic diffusion model intuition (noise → denoise iteratively)"
related: []
tags:
  - diffusion-language-models
  - speculative-decoding
  - efficient-inference
  - text-generation
  - parallel-decoding
bibliography: ""
code_repo: ""
sources_used: []
---

## Why this exists

I've spent the whole day reading papers at the intersection of diffusion models and text generation, and two names kept surfacing in different contexts: DFlash and DiffusionGemma. Both apply diffusion to discrete text tokens, but they solve completely different problems. DFlash uses a tiny block diffusion model as a fast draft generator inside a speculative decoding loop. DiffusionGemma is a full-scale text diffusion model that replaces the autoregressive decoder entirely. I kept getting them confused — which one uses bidirectional attention? Which one generates 256 tokens at once? Which one guarantees lossless output? This article works through the details side by side so I don't have to reconstruct the comparison from scratch next time.

Autoregressive language models generate one token at a time. That serial dependency makes them memory-bound at small batch sizes — most of the time goes to loading weights, not computing. Diffusion offers a way out: generate many tokens in parallel and refine them iteratively. DFlash and DiffusionGemma are two concrete implementations of this idea, but they occupy opposite ends of the system-design spectrum.

**Scope:** This covers the mechanism of DFlash's block diffusion drafter and DiffusionGemma's text diffusion model, then compares them across architecture, training, inference, and use case. It does not cover non-diffusion speculative decoding methods (EAGLE-3, Medusa) or image/video diffusion models.

**Prerequisites:** This assumes familiarity with autoregressive language model basics (causal attention, tokens, logits), the speculative decoding pattern (a small draft model proposes tokens that a large target model verifies), and the general diffusion concept (reverse process that iteratively removes noise).

## The bottleneck that diffusion addresses

Autoregressive decoding has a fundamental throughput problem. For each token, the model must load its full set of weights from memory into compute units, do a forward pass, and output a single logit vector. At small batch sizes — typical for interactive applications — the memory-bandwidth bottleneck dominates: the compute units sit idle waiting for weights to arrive. Batching many requests together amortizes this cost, but it does nothing for single-request latency.

Diffusion models flip this dynamic. Instead of one token at a time, they generate an entire block of tokens in parallel during each denoising step. The computation shifts from memory-bandwidth-bound to compute-bound. The question is how to make this work for text, where tokens are discrete and the sequential structure of language matters.

Two distinct answers have emerged. One treats diffusion as a fast approximator inside an existing autoregressive system. The other treats diffusion as the primary generation mechanism, replacing the autoregressive decoder entirely. DFlash is the first approach. DiffusionGemma is the second.

## DFlash: block diffusion as a speculative drafter

DFlash was developed at UC San Diego and published at ICML 2026. Its core insight is simple: the hidden states of a large autoregressive language model already encode information about multiple future tokens. Rather than training a separate draft model to predict tokens from scratch, DFlash extracts these hidden features and uses them to condition a tiny block diffusion model.

### Architecture

The draft model is a shallow bidirectional Transformer — typically 5 layers — that operates on blocks of $\gamma = 16$ tokens. The generation process works in five steps:

1. **Feature extraction.** During the target model's prefill pass, DFlash extracts hidden representations from a fixed set of layers uniformly sampled from shallow to deep (e.g., layers 3, 10, 17, 24, 31 of a 32-layer model). These hidden states capture information at different levels of abstraction.

2. **Feature fusion and injection.** The extracted features are concatenated and passed through a lightweight projection layer that fuses cross-layer information into a compact target context feature. This feature is then injected directly into the Key and Value projections of every draft model layer. The result is stored in the draft model's KV cache and persists across drafting iterations.

   This persistent per-layer injection is the key design choice. Prior work like EAGLE-3 fuses target features only at the input layer, letting the signal dilute as the draft model deepens. DFlash's approach keeps the conditioning strong regardless of draft depth.

3. **Parallel diffusion drafting.** Starting from $\gamma$ random token embeddings, the draft model performs 1-3 bidirectional denoising steps. Unlike autoregressive drafters that must run $\gamma$ sequential forward passes, the diffusion drafter generates all $\gamma$ tokens in a single forward pass per step. Draft latency is nearly independent of $\gamma$.

4. **Parallel verification.** The target model processes all $\gamma$ draft tokens in a single forward pass, computing acceptance probabilities for each prefix position. Because the attention computation is parallel within a single sequence, verifying $\gamma$ tokens costs barely more than generating one.

5. **Token acceptance.** The longest prefix where all tokens match the target model's distribution is accepted. One additional "bonus" token is generated for free at the acceptance boundary, and the process repeats from step 1.

### Training

The draft model shares the target model's token embedding and language modeling head — only the draft Transformer layers are trained. Training data comes from response text: random anchor tokens are selected, a contiguous block of $\gamma$ tokens is masked with noise, and the model learns to denoise the entire block in one shot. A position-dependent exponential decay loss weights early tokens more heavily — if the first draft token is wrong, all subsequent tokens in the block are wasted, so the model should prioritize correctness at the beginning of the block.

Multiple blocks can be trained in a single forward pass using sparse attention masks (Flex Attention), with cross-block attention disabled. This makes training efficient despite the block structure.

### Why it works

DFlash achieves the best of both worlds. The target model provides high-quality next-token predictions; the block diffusion drafter generates many candidates at near-zero marginal latency. The combination yields lossless 5-6x speedups on models like Qwen3-8B, compared to the 2-3x ceiling of autoregressive drafters like EAGLE-3. On Chain-of-Thought reasoning tasks, where sequences are long and the acceptance distribution matters more, the speedup drops to 4-5x — still far above what AR drafters achieve.

The unconditional baseline (a block diffusion model trained without target features) reaches only 2-3x speedup. The target context conditioning is what makes the difference.

## DiffusionGemma: standalone text diffusion

DiffusionGemma is an experimental open model from Google DeepMind, announced in early 2026. It represents a much more ambitious bet: replace the autoregressive decoder entirely with a diffusion process, while building on the Gemma 4 26B A4B backbone.

### Noise type: uniform state diffusion

Text diffusion requires defining what "noise" means for discrete tokens. Earlier approaches use masked diffusion, where tokens are replaced with a special `[MASK]` token. Once a masked position is predicted, it stays fixed — there is no self-correction.

DiffusionGemma uses uniform state diffusion instead. Noise means replacing a token with a random token drawn uniformly from the vocabulary. This has two consequences:

- The model must first identify which tokens are noise before it can predict the correct tokens. The denoising task combines detection and correction.
- A token predicted in step 1 can be replaced in step 11 if its probability drops. Self-correction is built in.

Rejected tokens (those where the model's confidence drops) are replaced with fresh random tokens, not the old wrong ones. This keeps the input distribution close to what the model saw during training.

### Shared encoder-denoiser architecture

Diffusion models typically need two components: an encoder that processes the user's query, and a denoiser that cleans the noisy canvas. DiffusionGemma patches a single pre-trained Gemma 4 26B A4B model to serve both roles, rather than training separate networks.

**Encoder mode.** The model uses its native causal attention to process the user's query. The resulting KV cache is computed once and stored.

**Denoiser mode.** Attention is patched from causal to bidirectional, allowing every token in the canvas to attend to every other token. The model uses the logits of all canvas positions to predict replacements. The pre-computed KV cache from the encoder is injected into the denoiser, providing context about the query without requiring cross-attention layers.

This design choice — reusing a single pretrained checkpoint instead of training from scratch — dramatically reduces training cost. The model starts from a capable autoregressive backbone and learns to work in bidirectional mode as a fine-tuning task.

### Canvas-based inference

DiffusionGemma operates on a canvas of 256 tokens, generated as follows:

1. **Self-conditioning.** The denoiser needs memory of its previous predictions. It takes the probability distribution from the previous step, multiplies it by the embedding matrix (producing one weighted embedding per position), passes it through a small feedforward network, and adds the result as a memory vector to the current step's token embeddings.

2. **Entropy-bounded sampling.** The canvas starts as 256 random uniform tokens. At each denoising step, the model computes entropy for every position. Positions are sorted from lowest entropy (most confident) to highest. Tokens are accepted as long as the cumulative sum of their entropies stays below a threshold. Rejected tokens are replaced with new random values.

3. **Scheduler.** A decreasing temperature controls exploration over steps. Early steps use high temperature for broad token search; later steps use low temperature for peaked, confident predictions. Adaptive stopping halts denoising early if the model is stable (top tokens unchanged for N steps) and confident (entropy below 0.005).

4. **Block extension.** For text longer than 256 tokens, the generated block is appended to the prompt, the KV cache is cheaply updated (it is causal on the encoder side), and diffusion begins on the next 256-token canvas.

The result is a model that generates up to 4x faster raw token throughput than an autoregressive baseline on dedicated GPUs, though the quality on complex reasoning tasks trails the full Gemma 4 autoregressive model by a small margin.

## Side-by-side comparison

With both mechanisms on the table, the differences snap into focus.

| Dimension | DFlash block diffusion | DiffusionGemma |
|---|---|---|
| **Role** | Draft model in speculative decoding | Standalone text generation model |
| **Output guarantee** | Lossless — target model verifies | None — quality depends on denoising |
| **Model size** | 5-layer bidirectional Transformer (~tens of millions params) | Full 26B backbone with bidirectional patch |
| **Block size** | 16 tokens | 256 tokens |
| **Denoising steps** | 1-3 | Many (scheduler-controlled, adaptive) |
| **Noise type** | Masked diffusion (predict corrupted tokens) | Uniform state diffusion (random tokens) |
| **Self-correction** | Not needed (target catches errors) | Yes — tokens can be revised |
| **Attention** | Bidirectional within block | Dual: causal (encoder) / bidirectional (denoiser) |
| **Conditioning source** | External: target model hidden features | Internal: self-conditioning from previous step |
| **Conditioning injection** | Into K/V projections of every draft layer | Added as memory vector to token embeddings |
| **KV cache** | Draft model builds its own with injected features | Encoder computes once, denoiser reuses |
| **Extended generation** | Speculative decoding loop (draft → verify → accept) | Block diffusion (append canvas → update KV cache → next canvas) |
| **Speedup over AR** | 5-6x system-level (draft + verify combined) | Up to 4x raw token throughput |
| **Primary bottleneck** | Acceptance length (how many draft tokens are accepted) | Denoising step count |

### Where they converge

Both models exploit the same fundamental idea: bidirectional attention within a block of tokens allows parallel prediction, and the block structure enables integration with autoregressive generation (DFlash through verification, DiffusionGemma through block extension). Both achieve their best speedups when compute, not memory, is the limiting factor — which means dedicated GPUs, not CPU inference or heavily loaded servers.

### Where they diverge

The divergence is best understood through the lens of error handling. DFlash accepts that its draft model will make mistakes and handles them through verification — the target model catches errors, so draft quality just affects throughput, not correctness. DiffusionGemma cannot afford to make mistakes because there is no verifier. It must allocate enough denoising steps for the model to self-correct, which creates a direct quality-speed tradeoff.

This difference cascades through every design choice. DFlash can use a tiny draft model because correctness is not its job. DiffusionGemma needs the full 26B backbone because the model must produce acceptable text on its own. DFlash can use 1-3 denoising steps because rough drafts are fine. DiffusionGemma needs many more because each step should improve the output. DFlash uses masked diffusion (simpler, cheaper) because errors are caught downstream. DiffusionGemma uses uniform state diffusion (more flexible, more expensive) because self-correction is essential.

## When to use which

DFlash is the right choice when you already have a deployed autoregressive model and want to reduce latency without changing the output distribution. The integration cost is moderate — you need to train the small draft model and set up the speculative decoding pipeline — but the output is guaranteed identical to the original model. It is particularly effective for long generations (chat, reasoning chains) where the acceptance distribution has time to average out.

DiffusionGemma is the right choice when you are building a new system from scratch and want the fastest possible raw generation speed, or when you are experimenting with text diffusion as a research direction. The quality gap relative to autoregressive models is small and narrowing, and the 4x speedup at small batch sizes is genuine. But you are committing to a different output distribution, and there is no verifier to catch mistakes.

The two approaches are not direct competitors. They operate at different levels of the system stack and solve different constraints. DFlash accelerates an existing model. DiffusionGemma replaces one.

## Where the comparison breaks

The speedup numbers are not directly comparable. DFlash's 5-6x includes both draft generation and target verification in an end-to-end system. DiffusionGemma's 4x is raw token throughput from a single model. If you were to run DiffusionGemma through a speculative decoding loop with a separate verifier, the effective speedup would be different.

The acceptance length metric ($\tau$) that DFlash reports has no direct analogue for DiffusionGemma. A diffusion model does not produce a sequence that an external verifier accepts or rejects — it produces text directly. Comparing $\tau = 6.8$ to a 256-token canvas is comparing apples to power plants.

Both approaches assume you have a GPU with enough memory. DFlash needs memory for both the target model and the draft model (though the draft model is tiny). DiffusionGemma needs memory for the full 26B backbone plus the bidirectional attention workspace. Neither works well on CPU or edge devices.

## Open questions

- Does the block diffusion draft model benefit from larger block sizes ($\gamma > 16$) in expectation, or does the acceptance distribution saturate? DFlash's block-size transfer experiment (large blocks train → small blocks infer) suggests a non-trivial relationship.
- Can DiffusionGemma's encoder-denoiser sharing be applied to smaller backbones (4-7B) without unacceptable quality loss? The Gemma 4 26B is a large starting point.
- What happens when you stack the two approaches — use a block diffusion model as a draft model for DiffusionGemma as the target? Would the uniform state diffusion's self-correction interact well with a separate verifier?
- How does the quality-speed Pareto frontier of DiffusionGemma shift with fewer denoising steps plus a small external verifier (a hybrid approach)?
- The position-dependent loss decay in DFlash is motivated by speculative decoding's sequential constraint. Does a similar non-uniform loss help DiffusionGemma's training, even though all positions matter for final output quality?

[[Q]] Six months from now: has any work combined the two approaches — using a block diffusion drafter conditioned on a diffusion target model's hidden states, or alternatively, has DiffusionGemma's uniform state diffusion been explored as a training objective for draft models?

## References

1. Chen, Liang, Liu, "DFlash: Block Diffusion for Flash Speculative Decoding", ICML 2026, arXiv:2602.06036.
2. Google DeepMind, "DiffusionGemma: An experimental open text diffusion model", 2026. Blog post and developer guide.
3. Li et al., "EAGLE-3: Scaling Speculative Decoding via Training Objectives", 2025.
4. Samragh et al., "Hidden Features of Large Language Models Encode Future Tokens", 2025.
5. Nie et al., "LLaDA: Large Language Diffusion with Masked Language Modeling", 2025.
6. Arriola et al., "Block Diffusion Models", 2025.


