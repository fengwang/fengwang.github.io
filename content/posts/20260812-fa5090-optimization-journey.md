---
id: 20260812-fa5090-optimization-journey
title: "The FA5090 Optimization Journey: From 3.23% to 94.6% of Roofline on a Consumer Blackwell GPU"
slug: fa5090-optimization-journey
date: 2026-08-12
lastmod: 2026-08-12
draft: true
format: "long"
domain: deep-learning
subdomain: gpu-kernel-optimization
summary: "A seven-version profiling-driven walkthrough of taking a FlashAttention forward kernel on the RTX 5090 (sm_120) from a naive 3.23% to 94.6% of the measured GEMM ceiling, and what each rung of the ladder cost to establish."
confidence: confident
prerequisites:
  - "FlashAttention tiling and online softmax"
  - "CUDA shared memory and warp-level programming"
  - "Tensor core instructions (mma.sync vs wgmma vs tcgen05)"
  - "basic Nsight Compute profiling"
related: []
tags: [gpu-kernel-optimization, flashattention, cuda, roofline, profiling, sm_120, tensor-cores, kernel-ladder]
bibliography: ""
code_repo: ""
sources_used:
  - "/data/feng/weave/wiki/concepts/flashattention-5090.md"
  - "/data/feng/weave/wiki/concepts/fa-family-optimization-comparison.md"
  - "/data/feng/weave/wiki/concepts/aa-null-methodology.md"
  - "/data/feng/weave/wiki/concepts/kernel-scheduling-optimization.md"
  - "/data/feng/weave/wiki/concepts/softmax-exponential-optimization.md"
  - "/data/feng/weave/wiki/concepts/cuda-core-vs-tensor-core.md"
  - "/data/feng/weave/sources/fa5090-r1-final-report.md"
---


Why this exists: I spent two weeks profiling a FlashAttention forward kernel on an RTX 5090, one version at a time, and the most valuable output was not the final kernel but the sequence of *why* each rung of the ladder worked or failed. FA3 and FA4 are written for H100 and B200; neither instruction set exists on the consumer part, and I wanted to know how far profiling-driven iteration could push a from-scratch kernel against the machine's real ceiling.

The thesis: on `sm_120`, the first ~60 points of speed-of-light come from correcting fundamentals (addressing hardware that sits idle), the next ~30 from removing instructions and hiding latency, and the last stretch requires calibrating the measurement instrument itself before any small margin can be trusted.

Scope: this covers the forward pass only, FP16/BF16 with FP32 accumulation, head dimension 128 primary, sequence lengths 512-32768. It does not cover the backward pass, FP8 (measured and rejected), or production hardening.

Out-Of-Scope: CUDA fast math, bf16 accumulation in softmax and `sm_120a`.

Prerequisites: this assumes familiarity with FA1 and FA2, including FlashAttention tiling, online softmax, CUDA shared memory, and the difference between warp-level `mma.sync`, Hopper `wgmma`, and Blackwell `tcgen05`.

## 1. Motivation: a current-generation GPU that current-generation kernels cannot address

[FlashAttention-3](https://arxiv.org/abs/2407.08608) reaches its throughput on Hopper through the warpgroup instruction `wgmma`; [FlashAttention-4](https://arxiv.org/abs/2603.05451) targets datacenter Blackwell through `tcgen05` and its tensor memory (TMEM). Neither exists on the consumer RTX 5090. Its `sm_120` part drives fifth-generation tensor cores with the warp-level, warp-synchronous `mma.sync.aligned` (`m16n8k16` for FP16/BF16/TF32) plus `ldmatrix`, a repertoire closer to Ampere than to the datacenter part sharing its architecture name. The absence was [verified](https://github.com/fengwang/ncu-report-skill-rtx-5090/blob/main/blackwell-cuda-programming.md), not assumed: assembling a minimal `wgmma.fence.sync.aligned` for `sm_120` fails at `ptxas` with an exit-255 error.

Upstream FlashAttention runs on the RTX 5090, but only through a thin SM80 fallback of roughly 61 lines that caps shared memory at 99 KB and reuses the SM80 `mma.sync` path, with no `sm_120`-specific optimization. The project therefore framed its gap as the absence of a teaching-grade, profiling-driven walkthrough taking attention from a naive kernel to a competitive one on this GPU, explaining at each step why the hardware behaves as it does.

Two questions drove everything:

1. If I apply every lesson FA3 and FA4 taught, can a hand-written kernel beat the cuDNN baseline on this card?
2. How far up the roofline can profiling-driven iteration actually climb?

The answer, stated at the top: the kernel reached 94.6% of the *measured* GEMM ceiling (204.164 TFLOP/s, with GPU frequency locked to a stable but lower level) and 1.0447x over a re-baselined cuDNN comparator (184.785 TFLOP/s) at its own configuration, but the interesting part is not the final number. It is that the first ~60 points of speed-of-light were corrections of omissions, the next ~30 were instruction removal and latency hiding, and the last stretch was mostly a measurement problem.

The machine itself is an awkward target in a way worth spelling out: 170 SMs, 680 fifth-generation tensor cores, 32 GB GDDR7 at 1,792 GB/s, 96 MB L2, 128 KB L1/shared per SM, 255 registers per thread, and a 99 KB opt-in dynamic shared memory ceiling per block (vs 227 KB on `sm_100`) with 48 resident warps per SM. One numerical trap: FP32 accumulation on the legacy FP16/BF16 path halves throughput, so an FP32-accumulate attention kernel is bounded by ~209.5 TFLOP/s, not the headline 419. The project's own measured FP16 peak was 208.5 TFLOP/s, and its measured GEMM ceiling 204.164, three denominators that must never be conflated.

## 2. Approach

### 2.1 Prepare: lock the instrument before trusting any number

The series starts by freezing the environment and the correctness gate, not by writing a kernel. I remember resisting this at first: I wanted to see a kernel run, and the setup felt like paperwork. It turned out to be the highest-leverage hour of the whole project. GPU clocks locked at 2400/14001 MHz (achieved ~2385/13801) on driver 610.43.03 with `ncu` counters verified non-zero; a frozen tolerance, pass iff `max_abs(kernel, oracle) <= 2 * max_abs(fp16_ref, oracle) + 1e-4`, roughly 100x below the bug error floor.

The correctness gate deserves its own sentence because it shaped everything downstream: two references, both required. The oracle is PyTorch attention in fp32; a second same-dtype FP16 reference defines the per-configuration floor, the irreducible error any correct fp16 kernel must be permitted. Both constants in the tolerance policy were derived empirically over 3,904 configurations rather than chosen: the worst correct kernel sat at 0.9635 of the bound, the smallest injected defect at 22.3x it. That margin is what made the whole ladder possible: a bug had to be 23x worse than the worst correct kernel before the gate would pass it.

At the frozen shape (D=128, non-causal MHA, B=2, H=16), attention is compute/tensor-core-bound, not HBM-bound, a fact established by profiling, not assumption. The comparator set: cuDNN 186.7 / FA wheel 180.9 / SDPA 175.8 TFLOP/s, where an earlier-cited 203.6 TFLOP/s cuDNN figure was corrected downward by re-measurement. This is the first lesson of the whole project: the baseline itself is a measurement to be taken, not a number to be quoted.

Three instruments coexist across the lineage and figures from one are not comparable with those from another: the frozen core instrument (25 warmup / 100 timed CUDA-event iterations, median, sequential and unpaired), the per-version margin script (10 warmup, 20 alternations, A/B/A bracketing, geometric mean of per-alternation ratios), and the v7 protocol (120 alternations, warmup 25 / measure 100, order-halving, second paired run in a fresh process). One rule governs how ratios may be combined: a margin over a grandparent is a *product of paired links*, never a ratio of two aggregate medians. Pairing is the only thing that removes drift; dividing two medians from different alternation blocks compounds both while still wearing the word "paired."

### 2.2 [v0](https://github.com/fengwang/FA5090/blob/main/v0/flash_v0.cu): the naive kernel is not bandwidth-bound

The baseline kernel materializes the O(N²) score matrix, which makes naive attention impossible at the frozen long-context config (256 MB at N=8192; 103 GB at N=32768). It uses online softmax for overflow robustness, 64-row CTAs in 48 KB shared memory, scalar strided global-to-shared staging, and the precise `expf`.

At the N=2048 profile: 9.52 ms = 3.23% of speed-of-light, 8.3% occupancy (against a theoretical 8.33%), 159 registers. The opening hypothesis that a naive kernel is HBM-bandwidth-bound was falsified by v0's own first profile: 5.3 GB/s DRAM, 95.3% L2 hit, short-scoreboard stalls dominating, tensor pipe at 0.0%. The kernel was memory-*latency*-bound, not bandwidth-bound, at 8.3% occupancy with 95% of its traffic served from L2.

Why this matters: the first and largest lesson of the ladder is that the profiler finds omissions faster than intuition. The naive kernel was not "slow but correct", it was leaving the machine's most expensive hardware completely idle while waiting on its cheapest. The 8.3% occupancy meant the SM had almost nothing in flight to hide latency; the 95.3% L2 hit meant the memory system was barely being exercised; the 0.0% tensor pipe meant the arithmetic engine that defines this GPU never woke up.

### 2.3 v0 → [v1](https://github.com/fengwang/FA5090/blob/main/v1/flash_v1.cu): tensor cores, and why the jump is 6.9x

The first rung moves both matmuls onto the fifth-generation tensor cores via `mma.sync.aligned.m16n8k16` + `ldmatrix`, with the grid, tiles, shared-memory budget and staging *byte-identical* to v0. Result: 6.92x at N=512 non-causal (6.01 → 41.60 TFLOP/s), 51.03 TFLOP/s / 24.4% at the frozen config.

To see why one instruction change is worth 6.9x, it helps to be precise about what the two execution units are. A CUDA Core is a scalar FMA unit: `d = a*b + c`, one instruction handles one number for one thread. A Tensor Core is a matrix multiply-accumulate array: `D = A×B + C`, where one instruction completes 64 MACs (Volta-generation definition) or more. The RTX 5090 packs 128 CUDA Cores per SM but only 4 Tensor Cores per SM; the tensor core wins not by count but by work per instruction: one MMA instruction amortizes instruction overhead by roughly 20x over a scalar loop, because issuing and decoding an instruction costs the same whether it moves one MAC or sixty-four.

The precision spectrum makes the mismatch concrete. On this card, FP32 (CUDA Core) peaks at 104.8 TFLOPS; FP16/BF16 (Tensor Core) peaks at 209.5 TFLOPS, exactly 2x. FP8 reaches 419, INT8 838, FP4 ~1676 (library-only, no PTX path). Nearly all of a modern GPU's AI peak lives on the tensor cores, and v0 was running attention as if the tensor cores did not exist: its inner loop issued scalar FMAs on the CUDA cores, capping the kernel at the 104.8 TFLOP/s FP32 ceiling while the 209.5 TFLOP/s engine sat at 0.0% utilization.

Why such a large jump from one instruction change? Four mechanisms, in increasing order of subtlety:

1. **Attention's QK^T is a "TN" GEMM.** The C-fragment layout of `mma.sync` maps element-for-element onto the A-fragment that P needs, no layout conversion required. The operand layouts the tensor core demands and the layouts attention naturally produces are the same layouts. This is a property of attention's matrix orientation, and it is why the tensor-core transition for attention costs near-zero layout overhead while a random GEMM would pay a transpose tax.

2. **`ldmatrix.x4` moves 512 B per instruction** against a 128 B shared-memory wavefront. The warp-level load path exists precisely because the tensor core wants 16x16 fragments; the naive scalar staging was reading 2 B at a time through the register file. The data movement was not merely slow, it was *instruction-starved*: the same issue bandwidth that could feed one `ldmatrix.x4` had to be spent on 256 scalar loads.

3. **The pipe that had been at 0.0% became the pipe doing the work.** The tensor-pipe counter reads 0.0% at v0 and 27.58% at v1; the kernel finally addressed the hardware that defines the card. This is the clearest single number in the whole ladder: before v1, the most expensive execution unit on the GPU never woke up once.

4. **The scalar path was asking the wrong unit to do the wrong precision.** v0's FP32 FMAs ran at the CUDA Core rate and precision; v1's HMMA runs at the tensor-core rate with FP16 inputs and FP32 accumulation. Moving to mixed precision doubles the peak available while keeping the accumulation exact.

The sm_120 boundary matters for what v1 could *not* do. The fifth-generation tensor cores on this part expose only the warp-cooperative, synchronous `mma.sync` path: no Hopper `wgmma` (async, SMEM-sourced), no datacenter-Blackwell `tcgen05` (single-thread, TMEM-destined). Measured SASS contains only HMMA/IMMA encodings, and `wgmma.fence` fails at ptxas with exit-255. The 6.9x jump is thus the *synchronous* ceiling: the tensor pipe runs, but overlap with the CUDA-core work relies on warp-level parallelism (one warp's HMMA overlapping another warp's softmax), not on hardware-asynchronous MMA. That constraint shows up two rungs later, when the exponential-overlap question is decided by the serial data dependency QK → softmax → PV, and again in the v7 search where the three barrier domains of exp1 matter precisely because `mma.sync` is synchronous.

Two counterintuitive cases from the same hardware sharpen the picture. First, FP32 accumulation halves the FP16/BF16 peak: choosing FP32 accumulators yields 209.5 instead of 419 TFLOPS on the legacy path, so an FP32-accumulate attention kernel is bounded by ~209.5, and v1's 51.03 TFLOP/s must be read against that ceiling, not the headline 419. Second, tensor cores are idle during decode: a GEMV has arithmetic intensity ~1.0 FLOP/byte, ~117x below the roofline breakpoint (RTX 5090 BF16: 116.9 FLOPs/byte), so the tensor pipe staying under 5% in ncu is expected, not a bug. Both cases explain why the v0→v1 jump was attention-specific: attention's tiled GEMMs clear the intensity breakpoint, the naive kernel simply never used the unit that was built for exactly this workload.

The overlap view completes the story. CUDA cores and tensor cores are independent execution pipes in hardware (FMA pipe / tensor pipe / XU-SFU pipe / LD-ST pipe), issued in parallel by multiple warp schedulers; different warps' instructions can execute on different pipes simultaneously, with the constraint that a single warp issues in order, so overlap relies on warp-level parallelism. v1's HMMA instructions freed the CUDA cores to do addressing and control flow *while* the tensor pipe worked, instead of serially. The measured evidence on this card: H100's exponential unit delivers 3.9 TFLOPS vs tensor cores' 989 TFLOPS FP16 (a 256x gap) that FA3's pingpong hides by running one warpgroup's softmax concurrently with another's GEMM; on `sm_120` the XU pipe runs concurrently with the tensor pipe: exp2 is ~4.99% of dynamic instructions and replacing it with FFMA drops the runtime share to only 0.59%, which is precisely the concurrency v1's tensor-core move unlocked.

The lesson generalizes: v1's ~21-point jump came purely from addressing a different execution unit. It was not optimization, nothing was tuned. The grid was identical; the tiles were identical; the code simply stopped asking the CUDA cores to do tensor-core work with scalar FMA instructions. The first and largest rung of the ladder was not a cleverness; it was the removal of an omission.

### 2.4 v1 → [v2](https://github.com/fengwang/FA5090/blob/main/v2/flash_v2.cu): shared memory swizzle, and why bank conflicts are a factor of 8

v2 XOR-swizzles the shared-memory Ks/Vs tiles. The shipped swizzle `offset = r*D + (c ^ ((r&7)<<3))` uses exactly 3 bits of XOR. It drops the `ldmatrix.x4` wavefronts from 32.00 to 4.00 and conflicts per instruction from 28.00 to 0.00; the "8.00x → 1.00x" change is the conflict *factor*. Throughput goes 51.03 → 85.63 TFLOP/s, a 1.68x jump that closes the gap to 40.9% of speed-of-light.

The mechanism matters more than the formula, so start with the bank structure. Shared memory on `sm_120` is 32 banks of 4 B, with `bank(addr) = (addr/4) mod 32` and a 128-byte bank period: addresses 128 B apart land on the same bank. A warp access is broken into wavefronts, each wavefront servicing at most one distinct 4-byte word per bank; when lanes pile onto the same banks the hardware serializes, and the cost lands in the LSU pipe, not the math. The canonical failure mode is any data structure whose row stride is a multiple of 128 B: an fp16 D=64 row is exactly 128 B, an fp16 D=128 row exactly 256 B, so every row's same column sits on the same banks.

That is precisely v1's layout: row-major `[64][D]` fp16 tiles, D=128 row stride 256 B, both exact multiples of the 128 B bank period. `ldmatrix.x4` (warp-cooperative shared-memory load into register fragments) measured 32.00 wavefronts per instruction against a theoretical floor of 4: an 8x bank-conflict factor, with 28 conflicts per instruction. The tensor core is now fed by `ldmatrix`, and `ldmatrix` was serializing on bank conflicts that the scalar staging loop never exposed; it was too slow to care. This is why swizzling is the second fundamental correction rather than a micro-optimization: v1 handed the tensor core a food pump that jams on every fourth rotation.

Why XOR, and why exactly 3 bits? The formula borrows 3 bits of entropy from the row index and XORs them into bit 3 of the column address:

```
offset(r, c) = r*D + (c ^ ((r & 7) << 3))
                  ↑      ↑        ↑
             row base  col XOR  row bits 3..5 shifted to bit 3
```

Each design decision has a reason. Three bits of entropy cover all eight bank phases: the 128 B bank period divided by the 16 B per-ldmatrix-lane run gives 8 windows, and 3 bits index all of them. The bits come from the row index because the row index contributes *nothing* to the bank phase (the row stride is a multiple of 128, i.e. D % 64 == 0); the swizzle borrows entropy from an axis that is otherwise wasted. The XOR lands at bit 3, above the in-fragment half-index bits, so each lane's contiguous 16 B run stays intact. And the mask permutes within each row, never across rows, so writer and reader using the same formula lose no data. The result: for gather rows r0 = 0 mod 8, the phase runs over all eight windows, one wavefront per gather, four per x4, zero conflicts, independent of D and of the tile base, for one XOR and zero extra shared-memory bytes.

The measured before/after tells the whole story:

| Metric | v1 (row-major) | v2 (swizzle) | Change |
|---|---|---|---|
| wavefronts / ldmatrix.x4 | 32.00 | 4.00 | 8x down |
| conflicts / instruction | 28.00 | 0.00 | zeroed |
| short_scoreboard stall | 5.285 (35.3% stall weight) | 0.358 (3.8%) | -93% |
| throughput (N=2048 non-causal) | 51.03 TFLOPS | 85.63 TFLOPS | 1.68x |

The most instructive number is the last one. An 8x conflict reduction buys only 1.68x; this is Amdahl applied to stalls. In v1, shared-memory stalls were 35.3% of stall weight and global-latency stalls 44.7%; removing the first bottleneck caps the gain at roughly 1/0.65 ≈ 1.5x, and the measured 1.68x sits close to that bound, far from the 8x. After swizzling, `long_scoreboard` (global-load latency) becomes 65.0% of stall weight: the bottleneck moves rather than disappears. That single observation is the design driver for v3's cp.async pipelining, and it is the recurring shape of the whole ladder: every rung removes one bottleneck and reveals the next.

The textbook alternative, padding, reaches the same conflict-free floor but pays in a different currency. Adding 8 halves per row inflates the per-CTA shared memory (49,152 → 52,224 B at D=128); under the 99 KB budget that drops occupancy from 2 to 1 CTA/SM; measured throughput falls to 53.23 vs 85.63 TFLOP/s, a 1.61x penalty, half an SM idle. The reason to prefer a swizzle over padding is cost, not capability: both reach the floor, only the swizzle keeps the occupancy. There is a mirror-image caveat at D=64, where the swizzle's +16 registers (128 → 144) cost a CTA/SM through the register account, the same currency padding spends at D=128, just a different account.

The correctness trap deserves emphasis: conflict-free and correct are independent properties requiring independent tests. A one-sided swizzle (stores swizzled, loads row-major) is perfectly conflict-free and completely wrong: data misplaced. A `r&3` mutation that writer and reader both agree on is correct but still 8x conflicted. The actual correctness test was bitwise: 23 configs x 3 layouts with max-abs error delta exactly +0.000e+00. And a swizzle bug that misplaces one element perturbs one fp16 mantissa bit, inside tolerance; reassuring, but a warning that correctness gates must be stronger than bit-level equality when layouts are permuted.

One more fact from this rung that paid off later: the hand-derived XOR is not a hack, it is what the hardware itself does. On `sm_120` with `CU_TENSOR_MAP_SWIZZLE_128B`, TMA writes `slot(r,c) = r*64 + (c ^ ((r&7)*8))` at D=64, exactly the first-principles formula, 4096/4096 cells matching bit-for-bit. At D=128 the hardware diverges (a row spans two 128 B segments, so TMA uses a group-major layout `slice = 2r + g` where the row-keyed XOR matches only 128/8192 cells). The convergence at D=64 is independent confirmation that the XOR phase remapping is the natural answer to the 128 B bank period, and the D=128 divergence explains why FA3/FA4's TMA swizzle path cannot simply be copied onto this kernel.

Result: 85.63 TFLOP/s / 40.9%, with an in-binary RowMajor control arm at 50.478 confirming the attribution.

### 2.5 v2 → [v3](https://github.com/fengwang/FA5090/blob/main/v3/flash_v3.cu): `cp.async` staging, and why double buffering loses

v3 replaces the 2 B scalar staging loop with 16 B/lane [`cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy) (LDGSTS), 512 B/warp, single-buffered as the shipped configuration. Result: 129.46 TFLOP/s / 61.8%; the copy-width lever in isolation measures 1.512x (the composed figure over scalar is 1.317x).

Why widening the copy helps, mechanically. The scalar loop moved 64 B/warp per instruction with address computation and issue overhead per element; `cp.async` moves 512 B/warp per instruction with the address computed once. The instruction count falls by exactly the width ratio: the staging loop's 17,039,360 two-byte global loads became exactly 2,129,920 LDGSTS instructions at D=128 (1,064,960 at D=64). And the deeper reason is that `cp.async` writes no register: the data path has nothing to scoreboard against, so the issuing warp does not block on the copy. The scalar loop's load had to complete before the FMA that consumed it could issue; the LDGSTS's consumer waits on the *commit group*, not the individual copy, so the latency moves off the critical path entirely.

Completion is positional, which is both the enabling property and a trap. The hardware counts outstanding issue groups, so the pipeline discipline is: NS-1 prologue commits, exactly one commit per loop iteration, `wait_group<NS-1>` as the kb-independent invariant, and a final empty commit on the epilogue. That last detail is where bugs hide: skipping the empty commit makes the final wait return immediately and the consumer read a half-filled stage, a nondeterministic wrong output only under load, invisible to a tolerance check on a single run. This is the kind of defect a loose gate certifies rather than catches.

Two more properties of the instruction shaped the design. `cp.async` 16 B uses L1 BYPASS mode, which is the right cache policy here: a staged K/V tile is read once per CTA, so the reuse that matters is across CTAs in L2 (measured 95.16% hit), and caching in L1 would only pollute it. And there is a scope boundary worth stating: cp.async changes *when* bytes arrive, not *where* they land. A bank conflict is an address-map problem, and a deeper or wider copy pipeline cannot fix it: the `cp_async_2_rowmajor` binary (same depth, swizzle removed) measured wavefronts per shared load 4.00 → 32.00 and duration 567,648 → 912,320 ns against the swizzled arm. Pipelining and swizzling are independent levers; v2's swizzle had to be in place before v3's pipeline could pay.

The interesting result is the negative one: **double buffering measured 12.9% slower** (0.871x, 112.79 vs 129.46). The three-step decomposition explains why:

1. **It did what it promised.** `long_scoreboard` collapsed from 1.1571 to 0.1727; the global-load latency v2's swizzle had exposed as the successor bottleneck (65% of stall weight) was genuinely hidden.
2. **The second stage ate occupancy.** The per-CTA footprint went 49,152 → 81,920 B; 2 × 81,920 = 163,840 > 101,376 B (`sm_120`'s 99 KB per-block ceiling), so occupancy dropped 2→1 CTAs/SM and waves per multiprocessor doubled (3.0118 → 6.0235).
3. **The hidden latency was not worth half the occupancy.** At a 95% L2 hit rate, the prefetch hides only a few hundred cycles of L2 latency, not the 1000+ cycles of a DRAM miss. The 2→1 CTA/SM cliff, not the stage count, is what costs time.

The depth-axis measurements at D=64 make the occupancy cliff unmistakable. Depth 2 (single → double, 4→2 CTAs/SM) is nearly free: +0.65% duration, with `long_scoreboard` collapsing 8.1x. Depth 3 (double → triple, 2→1 CTAs/SM) loses 32%: duration 263,488 → 348,096 ns. At D=128, depth 3 (114,688 B) exceeds the 99 KB ceiling and refuses to launch outright. The depth axis is an occupancy axis, not a latency axis. On this part, at this hit rate, "deeper pipeline is always better" from the datacenter playbook is reversed, purely because the shared-memory budget is 99 KB not 227 KB.

The Amdahl pattern from the previous rung repeats. The v2→v3 gain came from collapsing `long_scoreboard` (the global-latency share), and after the fix the successor bottleneck is compute-side: `math_pipe_throttle` rises 3.02x to 49.1% of stalls, at only 7.1% of the 1,792 GB/s bandwidth roof. A deeper prefetch cannot recover the remaining 38%; the pipe that was starved for data is now the pipe doing all the waiting. This is the same shape as 2.4's "8x conflict reduction buys only 1.68x": every rung removes one bottleneck and reveals the next, and the revealed bottleneck determines what the next rung must be. Here it points at the softmax's ALU shell, which is exactly what v4 attacks.

The 99 KB ceiling deserves its own paragraph because it is the single most decisive constraint on the whole ladder. Datacenter Blackwell gives 227 KB; `sm_120` gives 99 KB (queried via `cudaFuncSetAttribute`). A Q+K+V triple tile at Br=Bc=64, D=128, bf16 is 48 KiB, which fits with room for double-buffering but only barely. A 128x128 fp16 tile with double-buffered K/V would need ~160 KB, forcing asymmetric tiling (e.g. 128x64) or reduced pipeline depth; exceeding the limit fails the launch with `CUDA_ERROR_INVALID_VALUE` or fails silently at runtime. Every later decision on this card, v7's exp1 tile choice, the TMA rejection, the split-KV rejection, traces back to this number.

Also measured and rejected on this rung: TMA on `sm_120`. At matched occupancy (both arms depth-2, one CTA/SM) the TMA copy engine is genuinely faster than cp.async: 1.03189x, won 60/60 paired alternations. But TMA's swizzle-atom padding plus static barrier memory inflates shared memory enough to cost the second CTA per SM (occupancy 1 vs 2 under the 99 KB budget), so the best TMA whole-kernel configuration loses to the best cp.async configuration by 1.099x. Copy-engine speed does not survive the occupancy cost. The two measurements measure different things and must never be collapsed into "TMA is slower": on a 227 KB part the arithmetic would very likely go the other way. There is also an alignment hazard that makes TMA quietly dangerous: swizzle mode is keyed off the absolute shared-memory address, and a 128 B misalignment silently rotates the layout; copies complete, values are real tensor elements, but in the wrong places, invisible to output-equality checks in general.

The decision framework this rung established is compact: measure the stall share first (`long_scoreboard` below ~20% means pipelining gains are capped); ask what latency you are hiding (DRAM pays for deep pipelines, L2 at high hit rate does not); budget smem before choosing depth (Q + stages x (K+V) ≤ 101,376 B); prefer wide copies over deep pipelines, since the 8x width is the main driver of the 1.512x; and always check the occupancy cliff before the stall table.

### 2.6 v3 → [v4](https://github.com/fengwang/FA5090/blob/main/v4/flash_v4.cu): `exp2` softmax, and why the compiler shell was the enemy

v4 replaces the precise `expf` with `ex2.approx.ftz.f32` and folds `log2(e)` into the host-side score scale. Result: 141.37 TFLOP/s / 67.5%, paired 1.0906x over the in-binary bitwise-v3 control arm, won 40/40. This rung produced the project's deepest profiling find, and also its best methodological lesson: the hypothesis we pre-registered was wrong, and the three-layer evidence pyramid that proved it wrong is the part worth reusing.

**Layer 1, the static instruction census.** A SASS census across all 34 `expf` call sites counts instructions per family, collapsing to a closed form verified to the integer on all 24 v4 instantiations:

```
MUFU.EX2 = 0 if arm == 5 else 34        # INDEPENDENT of head dim
HMMA     = head_dim                     # D=128: 128 instructions
FMUL     = 32 + head_dim + (34 if precise expf else 0)
```

The `MUFU.EX2` count is head-dim independent because the score tile is BR×BC, not head-dim-shaped. The census has a prediction built in: the pre-registered closed form predicted 204 extra instructions; the census measured 238. The model committed first, `expf` lowers to 8 instructions not 6, and the census located the missing FFMAs instead of being refitted to match. That discipline matters: a census that adjusts its formula after seeing the answer proves nothing.

**Layer 2, the timing estimate.** MUFU.EX2 measures 8.00003 SMSP-cycles per warp-instruction, data-independent (alternating MUFU.LG2 gives the same 1.000000 ratio). At folklore constants (8 cycles/MUFU, 32 cycles/HMMA), exp2 is ~6% of the cycle budget. The hidden assumption: Layers 1 and 2 divide an instruction count by a cycle count, which *assumes* the XU work sits on the critical path. Layer 3 tests that assumption and finds it false by roughly a factor of nine.

**Layer 3, the causal ablation (authoritative).** Replace MUFU.EX2 with a same-shape FFMA, an intentionally wrong kernel, unreachable from the shipped path, failing its correctness gate on 31/32 envelope configs with 110x-1569x headroom ratios over the bound, but pipe-shape identical. The runtime share of exp2 drops to **0.59% at D=128** (0.31% at D=64), against a 4.99% dynamic-instruction share and a pre-registered 15% threshold. Verdict: REFUTED; verdict held, magnitude missed: the pre-registered prediction P1 anticipated the refuted verdict, but its magnitude band (f ≈ 6-8%) was wrong by ~10x. The XU pipe runs concurrently with the tensor pipe that defines the ceiling; a 5% instruction share buys 0.59% when removed. And Layer 3 is an *upper* bound: deleting an instruction also deletes its dependency edges, so the true share sits below 0.59%.

The methodological note underneath is worth its own sentence: `sm_120` has no XU cycle counter, only an instruction counter. An instruction share is not a time share, and you cannot convert one into the other by reading a counter. You have to delete the instruction and time the difference.

**The mechanism reversal: where expf's cost actually lives.** Precise `expf(x)` lowers to eight instructions per call, of which exactly one is on the XU pipe:

```
FFMA.SAT     R47, x, R66, 0.5                    // overflow / denormal guard
FFMA         R52, x, 1.4426950216293334961, -R52 // x log2(e), fused with range reduction
FADD         R52, R47, -12583039                 // exponent bias
IMAD.SHL.U32 R54, R47, 0x800000                  // build 2^k as float bit pattern
MUFU.EX2     R50, R50                            // <-- the ONLY XU instruction
FMUL         R50, R50, R54                       // x 2^k
```

Across 34 call sites: +238 FMA/ALU instructions (FFMA +136, FMUL +34, FADD +34, IMAD +34), with the MUFU.EX2 delta exactly zero. The exponential's cost is 7/8 on the pipes the SFU-bound framing says are not the problem; the fast-exp lever's payoff is an instruction-count reduction on the FMA/ALU pipes, not SFU relief. That is the opposite of the hypothesized mechanism: the same lever paying for a different reason.

The fix itself is two pieces, and both matter. The first is `ex2.approx.ftz.f32` as inline PTX, deliberately *not* `--use_fast_math`: a compiler flag changes all six arms at once, including the control, making the control useless and every delta unattributable. The approximation stays visible in the source diff and selectable per arm. The PTX-spec guarantees are asserted on device rather than trusted: `ex2.approx.ftz.f32(−inf) = +0.0` (a masked key contributes exactly nothing) and `ex2.approx.ftz.f32(0) = 1.0` exactly (what keeps the rescale-skip bitwise-exact when both levers are on). Accuracy was validated over a 96-config envelope with worst headroom ratio 0.8292 (about 17% of margin, the D=64 causal corner).

The second piece is folding `log2(e)` into the host-side score scale. `ex2` needs log2-domain input; the naive implementation multiplies every score by log2(e), 32 extra FMULs per warp per key block, eating a seventh of what the lever just saved. Instead the host folds log2(e) into `scale`, which the kernel already multiplies by, so the base change costs zero instructions. The running max m is carried in log2 domain too, which is self-consistent: every exponential takes a *difference* of same-scaled quantities, and the final 1/l normalization is a ratio, so the base cancels.

**Why it works: the 1.85 cycles/instruction dependency-chain signature.** The fast-exp kernel executes a byte-identical 4,489,216 XU instructions, the XU pipe untouched, yet is 9% faster. The 238 removed instructions sat *between* the two mmas in a strictly sequential QK-mma → softmax → PV-mma body at 2 warps per SMSP, with no second warp to fill the slots they occupied. Removing 46,839 warp-instructions per SMSP returned 86,652 cycles, 1.85 cycles per removed instruction. No pipe charges 1.85 cycles for an FFMA; that ratio is the signature of a dependency chain (latency), not pipe occupancy (throughput). The stalls worth attacking at 2 warps/SMSP are wait (1.6579) and barrier (0.3969), the signature of a serial chain with at most one other warp on the SMSP to cover it.

The overlap dimension rounds out the picture. The rescale can be split across the PV loop (+0.42%, arm 4, 1.004183x at +83 static instructions), but the exponential itself **cannot** overlap the PV mma within a key block: PV's A-operand *is* the exponentiated tile, a data dependency. Overlapping the exponential requires cross-block pipelining: a second score accumulator, at 186 registers. This is the same serial QK → softmax → PV chain that later blocks warp specialization on this part, and it is the structural reason FA3's pingpong does not transfer.

Two negative results on this rung are as instructive as the win. `defer_rescale`, postponing the rescale to the epilogue, bitwise-exact, saving 64 FMULs, measured 0.998325x, won 13/40: the pure schedule cost of the `__any_sync` vote plus branch exceeds the 64 FMULs it skips, a null for this placement of the branch (it was not hoisted out of the PV loop). And the predicted additivity of `fast_exp2` x `defer_rescale` was falsified: predicted ~1.088, measured 1.060055x, 2.77% slower than `fast_exp2` alone. Levers do not add, and only in-binary arm decomposition can attribute a two-lever result. The combined-arm table tells the whole story:

| Pair (vs precise control) | speedup | won |
|---|---|---|
| precise → `fast_exp2` | 1.090631x | 40/40 |
| precise → combined (`fast_exp2` + `defer_rescale`) | 1.060055x | 40/40 |
| precise → defer_rescale | 0.998325x | 13/40 |

The v4 outcome: 141.37 TFLOP/s, tensor pipe 76.33% → 82.56% of peak, registers 186 → 171 with zero spills. `math_pipe_throttle` *rose* 28% (3.7103 → 4.7622), counterintuitive until you know it aggregates FMA, ALU, XU and tensor pipes: on a tensor-bound kernel a rising value is a symptom of health. The conclusion that softmax is hidden behind the tensor pipe is conditional on occupancy and could change sign at higher warps/SM, a caveat that v7's exp1 (12 warps/SM) later tests directly.

### 2.7 v4 → [v5](https://github.com/fengwang/FA5090/blob/main/v5/flash_v5.cu): scheduling, and why warps were the wrong lever

At this point the kernel is tensor-pipe-bound at 8 warps/SM (v4's canonical kernel uses 209 registers/thread → 1 CTA/SM). Nsight showed two visible gaps, and both suggested obvious fixes. Tensor-pipe utilization read 82.66% of peak-sustained-active: 17.3% of the tensor pipe idle while the SM is running. SM activity read 72.41/82.66 = 0.876: no resident warp at all for 12.4% of elapsed time, grid-bound wave quantization (a later NCU pass puts the same idle at 21.7%). The first gap says feed more warps; the second says fill waves with a persistent grid. This rung is the story of sound reasoning, wrong prediction, and the one-command pre-flight that would have predicted the answer. Three experiments, each answering a different "why", and a fourth that also failed.

**Experiment 1: more warps failed.** The reasoning looked airtight. v4's occupancy is min(smem, regs, CTA slots, warp slots); t64x32 halves BC to 32, dropping smem to 32,768 B and landing registers at exactly 168, driver-confirmed via cudaOccupancyMaxActiveBlocksPerMultiprocessor: 12 warps/SM, 50% more supply for a 17.3% idle pipe. Result: 1.0065x at the primary config and 0.9079-0.9757x at the other four: 50% more warps loses to a 26.4% rise in total instructions.

The instruction count exposes why. The tensor instruction count is identical across all four tiles: 16,777,216, which it must be, since it is the same MMA count for the same matrices. The total rises 26.4% from BC=64 to BC=32, and that column *is* the per-block fixed cost: block-constant work doubles when the block halves. The transferable rule: when a tiling change alters the number of iterations, measure `smsp__inst_executed.sum` alongside the pipe you care about. The pipe count stays flat and the total does not; the difference is the fixed cost you just multiplied.

The one-command pre-flight: if the tensor-pipe active counter reads in the 80s at the occupancy you already have, the residual is not warp supply, and no tile change will find more than fractions of a percent. v4 read 82.66%. Every step of the reasoning about the 17.3% gap was correct; the inference "therefore supply more warps" required the pipe to be the *follower* in the relationship, and it is the leader.

The discriminating measurement was the derivative, not the level: supply a little more occupancy and watch `math_pipe_throttle`. It rises (5.041 → 5.610 → 7.224) while barrier waits triple then quadruple (0.334 → 1.037 → 1.357) as more CTAs contend for `__syncthreads`. Rising means the pipe leads and the warps follow, so you are adding contention, not supply; falling would have meant headroom to convert. The three highest-occupancy configurations in the 288-candidate search (t128x32 at 16 warps/SM) are among the slowest at 0.561-0.849x. You cannot decompose "the pipe is 85% utilized" into "85% work, 15% missing warps": the complement is whatever the instruction stream's dependency structure costs, and warps do not pay it.

**Experiment 2: a persistent grid failed.** The second gap was real: at N=2048, B=2, Hq=16, BR=64 the grid is 1,024 CTAs over 340 slots (170 SMs × 2 CTAs) = 3.0118 waves, with a fourth wave of four CTAs while 336 slots idle through it. A persistent grid removes that by construction, launching exactly `ctas_per_sm` × `sm_count` CTAs and pulling work items until the list is empty. Waves/SM went 3.0118 → 1.0000 and SM-active 0.8776 → 0.9172. It measured 0.9875x, the largest fill improvement in the session, and it is slower.

Why: every work item pays a per-item tax that the grid version does not have. The persistent loop decodes each item (`fa5090::decode(it, ...)`) and executes a `__syncthreads()` that is not optional: the loop reuses one shared-memory Q buffer across items, so item n+1's staging load must not overtake item n's last read, a write-after-read dependency the launch-per-tile grid cannot have because each CTA handles exactly one item and exits. At N=512 the grid is 256 CTAs each running 8 key blocks, so the two per-item costs are amortized over the least work in the config set, and every configuration there measures 0.937-0.999x. A perfect wave count at the price of a dependency in the hot loop loses.

**Experiment 3: reordering the same CTAs won.** The real lever was cost skew under causal masking. Under causality, per-CTA cost is not uniform: `live_blocks(q_tile) = q_tile + 1` for BR = BC, so at N=2048/BR=64 a CTA does between 1 and 32 key blocks, a 32x spread. And `blockIdx.x` is the fastest-varying dimension of the launch, so the hardware dispatches each head's cheapest tile first and its most expensive last: the worst order a greedy scheduler can be handed, since the long jobs arrive when there is nothing left to overlap them with and the makespan is dominated by one CTA finishing alone.

Longest-processing-time-first is the textbook answer and a 4/3-approximation for makespan on identical machines, and it required no new work and no new kernel: the same items, indexed differently. HeaviestFirst (primary key DESCENDING `q_tile`, secondary `b*Hq+hq`) measured 1.326-1.351x across three runs at causal N=2048 (frozen 1.349182x, won 20/20), 1.388-1.400x at causal D=64, and 0.920-1.084x centered on 1.0 at non-causal shapes where there is no cost skew; the lever is inert exactly where the mechanism predicts it should be, an effective control. Instruction counts are identical to within 0.019%, output bitwise-equal, 195 registers, 0 spill bytes, 1,872 SASS instructions: a pure scheduling change with zero arithmetic change. SM-idle fraction of elapsed falls from 21.7% to 2.8%, the makespan-dominated-by-one-CTA evidence erased directly. NCU corroborates the persistent+LPT arm to a tenth of a percent (+13.9% vs +14.0% across two independent instruments).

**The fourth lever, split-KV, also failed.** Three of four scheduling levers on this rung are negative: warps (1.0065x), persistent grid (0.9875x), and split-KV work partitioning at every setting (0.9354/0.9788/0.7670x at 2/4/8 splits under-filled, 0.47-0.86x at filled configs), because the fp32 workspace round trip transits HBM at a 9.47% L2 hit rate. One large positive against three negatives is the honest shape of the rung.

**The shipped FLASH_V5: conditional activation.** `select_config` sets `item_order`=HeaviestFirst only when causal, because the lever is disabled where there is no skew. Causal-only keying also reflects that the per-shape winner did not reproduce across five search runs (six of seven configs crowned a different winner on re-run), so the contract's requested per-shape table was deliberately left open rather than overfit. Shipped numbers: 1.367720x at causal N=2048 (156.90 TFLOPS, 74.89% of the 209.5 SoL, won 20/20, a two-link chain 1.349182 × 1.013740), 1.034235x non-causal (3.4-3.9% unattributed code-gen difference), and 0.981604x at N=512 non-causal, a launch- and tail-bound regression (0.7529 waves) deliberately accepted after the owner waived contract clause 7, which fails at two configs. Pricing lesson from the persistent arm: the persistent+LPT variant measures 1.168005x, less than plain-grid LPT's 1.349182x, because the persistent scheduler's strided assignment already mixes cheap and expensive work by construction; price a lever at the setting the shipped path uses, and the plain grid IS the shipped setting.

Why LPT works where warps failed: the kernel was not resource-starved, it was *makespan-imbalanced*. The scheduler's job was not to add supply but to reorder demand, and scheduling order changes the combinatorial structure of the makespan at zero cost. This is the same insight FA4's paper lists as LPT scheduling for causal masking; the project derived it independently from makespan theory, and the convergence is the strongest evidence that the mechanism is real.

**The v7 confirmation refined the story in two ways.** First, with a calibrated instrument the 12-warp tile was worth keeping after all: v7's `exp1` moved from t64x64 (8 warps/SM) to t64x32 (12 warps/SM, `MIN_CTAS`=3 forcing ≤168 registers) and measured 1.003818 paired, a 0.38% keep made believable by the A/A null (`t_noise` 0.0591%, `t_keep` 1.000591, cleared at 6.5x `t_noise`). v5's 1.0065x at N=2048 non-causal is directionally consistent with v7's calibrated +0.38% at a different config. Second, the mechanism is independent barrier domains, not warp count: t32x32 reaches the same 12 warps/SM and measured −17%; t96x32 (2 CTAs × 6 warps, also 12 warps/SM) measured −14%. What t64x32 supplies is three independent `__syncthreads()` domains per SM, so one CTA's barrier does not stall another's issue. The derivative test said "pipe is leader" (still true); v7 adds that when warps DO help, the channel is barrier-domain parallelism, not warp supply. And a dead end resurrected: S5's deferred-rescale measured −0.17% at 8 warps/SM and was shelved; v7's exp3 re-screened the same idea at 12 warps/SM and measured +0.98%; the dead end was conditional, and the condition was the occupancy exp1 had just changed.

One correction the final report forced, independent of the mechanism: v5's published headline (1.367720x) is measured on the *causal* axis while v4's is *non-causal*, the shared non-causal axis margin is 1.034235x. The difference is dominated by the 1.605x causal block skip that has existed since v2 and is contained in every causal figure from v2 onward. Quoting 1.367720x credited v5 with a v2 saving and a change of denominator. When the regime a headline is taken under changes between versions, the ratio prices the regime change silently, always flatteringly, since nobody switches to the regime that reads worse.

### 2.8 v5 → [v7](https://github.com/fengwang/FA5090/blob/main/v7/flash_v7.cu): four keeps, and why 0.38% became believable

v7 is one version beyond the original plan, seeded bit-identically from v5. The v6 FP8 side branch is deliberately excluded, it is accuracy-gated and optional, and v7 was seeded from v5, not v6. The v7 search ran 17 numbered experiments (exp1 through exp18 with exp7 absent from the ledger) plus two re-anchors, in about two hours at a cost of roughly $0.2. Four were kept.

The four keeps, each gated by the full keep protocol (paired+interleaved median margin >= `t_keep`, second paired run in a fresh process, AB/BA order-consistency, correctness tier):

| keep | lever | paired margin | why it works |
|---|---|---|---|
| exp1 | `t64x32` tile (3 CTAs x 4 warps = 12 warps/SM) | 1.003818 | not more warps — three independent `__syncthreads()` domains per SM; same-12-warp controls t32x32 −17%, t96x32 −14% |
| exp3 | `rescale_skip=1` (skip `4*NT_O` FMULs when correction factor is exactly 1.0) | 1.009784 | S5's deferred-rescale resurrected — it was conditional, not wrong: −0.17% at 8 warps/SM, +0.97% at 12 |
| exp10 | `MaskMode::None` template specialization of `Softmax::apply` (function, not kernel) | 1.021838 | deletes the mask predicate from ~1023/1024 blocks; no kernel instantiation added; kernel-level 3-way alternative lost 5.4% |
| exp15 | split the key-block loop at the diagonal with `integral_constant` type tags | 1.012469 | mode becomes a type, not a value: 168→161 regs, spill 16B→0; exists only because exp10 proved the specialization pays |

The product of the four margins is 1.048690; the independent re-anchor against the untouched seed measured 1.048680, a 0.001% gap, one statistic ruling out double-counting, negative interaction and machine drift at once. All four keeps pass the C2 oracle at the identical 0.00774 `max_abs` (output-neutral).

**Why 0.38% became believable: the A/A null.** This is the rung's methodological core, so the setup deserves the full treatment. On a consumer GPU with unlocked clocks, a single-arm benchmark has 0.42% per-iteration dispersion at the v7 config, defined as (p90 − p10)/median at warmup 25 / measure 100: the spread of the measurement, not the uncertainty of the median. Almost all of that dispersion is *shared drift*: clock sagging as the card warms, memory boost states, another process on the device, thermal state. A margin smaller than ~0.4% is indistinguishable from a warm card, which naively caps what optimizations are credible.

The A/A null calibrates the instrument against itself before any result exists: ten paired A/A experiments, each in a fresh process, both arms the *same* unmodified kernel, same source, same build, same binary, pushed through exactly the path a real keep test uses (paired and interleaved, 120 alternations, warmup 25 / measure 100, order-halving, the same parser). The true margin is 1.000000 by construction. The ten signed margins came back 0.99982 … 1.00059, split 5 positive / 5 negative, balanced; `t_noise` = p95 of |margin − 1| = 0.0591% (at n=10 the p95 is effectively the maximum, which the artifact itself notes rather than implying a fitted quantile).

Why the paired floor is 7x tighter than the single-arm dispersion: interleaving both arms inside one process puts each measurement of A microseconds from a measurement of B under the *same* clock state. The reported statistic is the ratio, so the common factor divides out. What survives is only the noise not shared between neighbouring launches: 0.0591% instead of 0.42%.

The keep gate follows, and the ordering is the whole point: the A/A null is not the gate by itself, it *sets* the gate. `t_keep = 1 + t_noise = 1.000591` is read out of the artifact on every decision, so no threshold is ever chosen after seeing a result. A keep requires, per program rule: a paired median margin ≥ `t_keep` with `paired_lo/paired_hi` logged; a second paired run in a fresh process also clearing `t_keep` (under the A/A null, a *single* paired run clears `t_keep` with probability ~0.025, so the replicate exists); order-consistency, the AB and BA halves agreeing in sign; and a passing correctness tier, C1 reduced-sequence oracle on every experiment, C2 full frozen config before every keep.

The instrument also returns nulls, which is evidence it is not echoing whatever it is pointed at. The .cs epilogue-store change measured 0.99997 / 1.00020 across order halves (split sign, reverted); the .wt store 0.99980 (consistently slower, reverted). A margin between 1.000x and `t_keep` is "indistinguishable from the A/A null" and reverts, the sole exception being a change that is strictly simpler, logged as a simplification rather than a gain.

At v5-era precision, exp1's +0.38% was indistinguishable from a warm card; with the calibrated instrument it became a keep, clearing the gate at 6.5x `t_noise`. This is the project's most underrated contribution: the A/A null turned a coin flip into an argument, and it is the reason the smallest result on the ladder rests on the firmest evidence: the only era guarded by a pre-read threshold rather than a 5% dispersion warning.

**Why each keep works, mechanistically:**

exp1 works because the SM's `__syncthreads` domains are the resource, not warp count. Three CTAs give three independent barrier domains; one CTA's barrier no longer blocks another's issue. The same-12-warp controls (t32x32: 4 CTAs x 2 warps; t96x32: 2 CTAs x 6 warps) prove it is not the number 12, it is the number of independent domains.

exp3 is the resurrection of a v4-era dead end. S5's deferred-rescale measured −0.17% at 8 warps/SM and was shelved; at 12 warps/SM it measures +0.97%. The dead end was not wrong, it was conditional on occupancy. The identity test must compare against `m_eff` (the effective max) not `m_new`, because a fully-masked tile's -inf would wrongly short-circuit; the decision must be warp-uniform (one `__any_sync` vote); and the running-sum expression stays bitwise-identical across settings so the lever is verifiable with torch.equal, not tolerance.

exp10 is the largest keep and the purest "why" lesson: 1023 of 1024 blocks in a causal kernel need no mask predicate at all, but a runtime branch forces every block to evaluate it. Specializing `Softmax::apply` as a *function* template (not a kernel template) deletes the predicate from the interior blocks with zero new kernel instantiations, while the kernel-level 3-way alternative lost 5.4% (code-size cost charged to every instantiation).

exp15 exists only because exp10 proved the specialization pays: splitting the key-block loop at the causal diagonal with `std::integral_constant` type tags turns the mode into a type instead of a value, letting the compiler actually eliminate the branch, 168→161 registers, spill 16B→0. The two specializations' live ranges no longer coexist in one body.

The conditional dead end is the methodology's most reusable licensing: re-screen the shelf after anything moves occupancy. S5's deferred-rescale measured −0.17% at 8 warps/SM and was shelved; v7's exp2 knob screen flagged the same idea at +0.97% arm-major once exp1 moved the kernel to 3 CTAs/SM, and exp3 confirmed it paired at 1.009784. The dead end was not wrong; it was conditional on occupancy. Later re-screens priced `rescale_skip=0` at −2.1% (exp12) then −1.4% (exp17), re-confirming the keep from both sides. The lesson is to record the condition beside the rejection, because the shelf is where future wins hide.

The re-anchor is the product validation, and its failure modes all land in the same statistic. Each margin is measured against the then-best shipped configuration, appropriate for attributing a delta but not for stating a total, because multiplying margins assumes no keep was counted twice and no keep changes in the presence of others. A paired run of the current best against the untouched frozen seed should equal the product. RA1 after three keeps: product 1.035775 vs measured 1.035868, gap 0.009% against a tolerance of 0.118% (`= sqrt(4)·t_noise`). RA2 after all four keeps: product 1.048690 vs measured 1.048680, gap 0.001% against 0.132%. Double-counting would make the product *overshoot* the anchor; negative interaction would make the anchor *undershoot* the product; drift across the run would miss in whichever direction the card went. None appears, twice, at two stack depths, with AB and BA agreeing to 0.007%. A weaker second independence argument comes from correctness: all four keeps passed the C2 oracle at the identical 0.00774 `max_abs`, so none trades accuracy for speed and none can explain another's gain numerically.

Rejected at this config (13 labels): 16 warps/SM refuted by arithmetic before running (register file forces a 72B spill); split-KV −7.5/−9.6/−13.3% (the 170-CTA grid is never under-filled at N=32768); cp.async depth-2 double buffer −2.7% (smem 32768→49152 B drops 3 CTAs/SM to 2, the same occupancy cliff as v3, at higher occupancy); t96x32 −14% (same warp count, fewer barrier domains); fixed-domain softmax 0.99896; both epilogue-store cache policies null (the .cs/.wt sign-split pair above). The run stopped under the plateau rule, which is a diagnosis, not an exhaustion: headline reached and plateaued, three consecutive no-gain experiments at the final state, plus a complete bottleneck diagnosis (tensor pipe 95.4%, memory un-starved, L2 hit 99.7%), with every remaining structural lever either arithmetic-blocked or measured-null at the frozen config.

Every v7 number carries provisional=true, clocks were not locked (achieved 2392/13801 MHz vs requested 2400/14001, recorded never asserted), no compute-sanitizer was run (correctness rests on the fp32-oracle gate). The paired+interleaved protocol is the reason unlocked clocks were survivable at all; the run's own report still calls for a --lock-clocks re-run to finalise. The v7 track's statistical discipline is the project's strongest while its environmental control is its weakest; best-characterised noise is not least noise. And the A/A null quantifies the instrument, it does not replace the correctness gate.

The final score at the v7 config: 193.04 TFLOP/s arm-major = 1.0447x the re-baselined cuDNN comparator (184.785), 1.0487x vs its own frozen seed, 94.6% of the measured 204.164 GEMM ceiling, tensor pipe at 95.4% of sustained peak. The 1.03x cuDNN target was beaten; the 195 TFLOP/s stretch was not (arithmetically blocked).

## 3. Comparison with FA3 and FA4

Mapping the FA5090 ladder against the datacenter papers, ten strategies in all: 3 hardware-generational (tensor cores, swizzle, pipelining), tile-shape/barrier structure, 2 same-idea-different-method (softmax exponential, rescale skipping), 1 independent convergence (LPT CTA ordering), 3 FA5090-unique (function-level mask specialization, loop split with type tags, A/A-null methodology). FA3 is Hopper (`sm_90a`, `wgmma`/TMA), FA4 is datacenter Blackwell (`sm_100`, `tcgen05`/TMEM); the FA5090 ladder starts from the FA2 structure and adds one technique per rung, so each strategy can be asked: does FA3 have it? does FA4?

| # | Strategy | FA5090 | FA3 | FA4 | Relationship |
|---|---|---|---|---|---|
| 1 | Tensor-core matmul | v0→v1 mma.sync + ldmatrix | wgmma (async, SMEM-sourced) | tcgen05 (async, TMEM-destined) | hardware-generational: same idea, three instruction generations |
| 2 | SMEM swizzle | v1→v2 hand-derived XOR | TMA 128B swizzle | same | same goal, different mechanism; D=64 matches bitwise |
| 3 | Data pipelining | v2→v3 cp.async single-buffer | TMA circular buffer (core pillar) | TMEM multi-stage, deeper | same concept, different engines; FA5090's double-buffer measured NEGATIVE (−12.9%, 99 KB ceiling) |
| 4 | Softmax exponential | v3→v4 fast ex2.approx + host log2(e) fold (4.99% instr → 0.59% time) | pingpong hides softmax (256x exp:tensor gap) | software-emulated exponential (Cody-Waite + poly on FMA) | same problem, three solutions: remove / hide / redistribute |
| 5 | CTA ordering | v4→v5 HeaviestFirst (LPT, 4/3-approx, 1.349x, SM-idle 21.7%→2.8%) | — | LPT for causal + SPT for backward + variable-length pre-sort | **independent convergence** (page-level inference: the vault does not establish which came first) |
| 6 | Tile shape / barrier | v7 exp1 t64x32, 3 independent `__syncthreads()` domains | 128x64, 4 warpgroups, mbarrier async sync | 128x128 TMEM (frees registers) | all tune tile shape; FA5090's barrier-domains-per-SM split is not recorded in FA3/FA4 papers |
| 7 | Rescale skipping | v7 exp3 identity-only (skip only when correction = exactly 1.0f, bitwise-safe) | — | conditional tau = log2(256) = 8.0 threshold (lazy + final norm) | same idea family, different aggression: FA4 skips more via proven equivalence; FA5090 skips only the provably-identity case |
| 8 | Compile-time mask specialization | v7 exp10 MaskMode::None (function-level, +2.18%, 1023/1024 blocks) | — | — | **FA5090-unique** (paper-level FA3/FA4 do not record it) |
| 9 | Loop split with type tags | v7 exp15 integral_constant (168→161 regs, spill 16B→0) | — | — | **FA5090-unique** (not recorded in FA3/FA4 papers) |
| 10 | Measurement methodology | A/A-null + paired+interleaved + re-anchor (t_noise 0.0591%, 0.001% product gap) | — | — | **FA5090-unique** (no FA3/FA4 paper records it) |

What surprised me most was not the gaps but the convergences. I had expected a consumer kernel to be a pale imitation of the datacenter papers; instead it independently rediscovered two of their strategies.

The convergences are the story. FA5090's HeaviestFirst is FA4's LPT, derived independently from makespan theory, at zero instruction cost, and the strongest evidence that causal-mask load imbalance is a real and addressable structure. FA4's version is richer: LPT for the forward pass, SPT for the backward, and variable-length pre-sorting of the work list. FA5090 found the same scheduling principle without any of that machinery, purely from the observation that live blocks under causality equal q+1 and the hardware dispatches the cheapest first. The rescale-skip family shows the design-space spectrum: FA4's tau = log2(256) = 8.0 threshold skip (aggressive, skips rescale within a tolerance band, licensed by a lazy-rescale + final-normalization equivalence proof) vs FA5090's identity-only skip (conservative, skips only the provably bitwise-identity case, no proof required beyond exp2(0) = 1.0 exactly). Same problem, two points on one continuum, different risk postures.

The missing-set analysis is where the comparison gets honest, because each absence has a distinct reason: hardware-absent, measured-and-rejected, or not-implemented.

| Missing strategy | In FA3 | In FA4 | Why FA5090 lacks it |
|---|---|---|---|
| Warp specialization (producer/consumer + setmaxnreg) | pillar | — | `sm_120` has only synchronous mma.sync (no async `wgmma`/`tcgen05`), so warp-role split pays far less; measured overlap arm 4 at only +0.42% |
| Pingpong cross-warpgroup scheduling | pillar | — | same sync constraint; exponential cannot overlap PV mma (data dependency: PV's A-operand is the exponentiated tile) |
| TMA adoption | pillar | — | measured and rejected: matched-occupancy TMA +3.19% faster but whole-kernel lost 1.099x (99 KB smem ceiling eats occupancy) |
| FP8 + block quantization | peak mode | — | measured and failed: FP8 peak is 416.6 TOPS (2.00x FP16's 208.5) but the shipped FP8 arm is 2.87x SLOWER than the same source without FP8 (0.319-0.434x vs the bitwise-v5 control, 47.6 TOPS = 11.4% of peak) — the mechanism is the data path (no ldmatrix 8-bit form on sm_120), 566/612 rows fail tolerance |
| TMEM everything (128x128 tiles, direct MMA results) | — | core | hardware absent on `sm_120` |
| Software-emulated exponential | — | ✓ | FA5090 used hardware ex2.approx instead (simpler, single instruction) |
| Conditional rescale tau=8.0 | — | ✓ | FA5090 implemented only the identity version; threshold version requires the lazy+final-norm equivalence proof FA4 provides |
| 2-CTA MMA mode + DSMEM | — | ✓ | not implemented: 2-CTA MMA is a tcgen05 co-operative mode absent from `sm_120`'s synchronous mma.sync; DSMEM itself exists on sm_120 (clusters validated up to 8 blocks) but FA5090 never used it |
| CuTe-DSL (2.5s compile) | — | ✓ | FA5090 uses CUDA C++ |
| Head/batch swizzling (L2 optimization) | — | ✓ | only CTA ordering done, no L2 layout |
| Deterministic mode (lock-based) | — | ✓ | not implemented |

The gaps are hardware-generational, not idea-generational. FA5090 lacks FA3's warp specialization and pingpong because `sm_120` has only synchronous mma.sync, and the exponential cannot overlap the PV mma because PV's A-operand *is* the exponentiated tile (data dependency, measured: overlap arm 4 at only +0.42%). It lacks TMA because the 99 KB ceiling eats the occupancy TMA needs (measured and rejected). It lacks FP8 because ldmatrix has no 8-bit form on `sm_120` (the shipped FP8 arm measured 2.87x slower than the fp16 control; 566/612 datasheet rows fail tolerance). It lacks FA4's TMEM everything because the hardware does not exist. The asymmetry of evidence is worth stating: FA5090's absences are backed by measurements on this card, FA3/FA4's presence are backed by their papers on their cards. The direction of the comparison matters: FA5090 is "measured and rejected" where FA3/FA4 are "shipped."

Four key insights close the section. First, FA5090 converges with FA4 on strategy level: 7 of 10 strategies have an FA4 counterpart, including two independently discovered ones on the FA5090 side (LPT CTA ordering, rescale skipping), page-level inference, with the vault not establishing which came first. Second, the gap is hardware-generational, not idea-generational: the missing items (TMEM, warp-spec with async MMA, deep pipelining) are either hardware-absent or measured-negative on `sm_120`. Third, FA5090's unique contributions live at the compiler and methodology layer: function-level mask specialization (+2.18%), loop-split type tags (7 registers back), and A/A-null calibration, none recorded in the FA3/FA4 papers. Fourth, the rescale family shows the design-space spectrum: FA4's aggressive threshold skip versus FA5090's conservative identity skip are two points on one continuum, not rival designs.

The synthesis: a consumer-grade hand-written kernel arrived at the same scheduling and softmax-rescale answers as the data-center paper, with the vault not establishing which came first, and the parts it could not copy from the datacenter playbook are exactly the parts that are hardware-absent or measured-negative on its own card. The convergence validates the mechanism-level thinking; the divergence validates the measurement discipline.

## 4. Outlook: will we customize CUDA kernels for each GPU?

The economics changed during this project. v5 → v7 cost roughly $0.2 of compute and about one hour of wall time with [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731); the search ran 17 experiments plus two re-anchors and stopped under the plateau rule at 94.6% of the measured GEMM ceiling. The question is no longer "can we afford to hand-tune a kernel?" but "can we afford not to, when the tuning loop is this cheap?"

Three observations shape my answer.

First, the structure of the gains is stable across the whole ladder: the first ~60 points of speed-of-light are corrections of omissions, the next ~30 are instruction removal and latency hiding, and the last stretch is dominated by measurement quality. Any GPU that has ever been shipped has at least one generation of kernels written before its quirks were understood, the omission-correction phase is not a one-time event per vendor, it is a per-architecture event. Each new part (`sm_90a`,` sm_100`, `sm_120`, `sm_120a`) reopens it.

Second, the leverage of profiling-driven iteration grows as the tooling improves. The A/A-null methodology turned a 0.38% margin from noise into a keep; the re-anchor product check turned four correlated experiments into one structural statement about independence. The expensive part of kernel work is not the optimization, it is the measurement, and measurement tooling is compounding faster than the hardware.

Third, the customization question has a boundary condition the project measured directly: at ~95% tensor-pipe utilization, the remaining levers are worth 0.4–2.2% each, and 13 of 17 labels bought nothing. There is a point past which per-GPU customization stops paying, the kernel hits the pipe's structural ceiling and the search becomes a plateau. The interesting economic zone is the one *before* that ceiling, where the ladder climbs in double-digit steps per session.

My expectation: we will not customize a kernel for every GPU in the traditional sense, but we will run an automated profiling-driven search for every GPU in every generation: because the search is now cheaper than the manual tuning it replaces, and because the same 31 falsified levers that cost this project months are the most transferable asset: a lever that loses structurally (occupancy paid for a conflict fix, an instruction removed that costs more in schedule than it saves) loses again for the same reason on the next kernel.

The datacenter playbook does not transfer to consumer parts, TMA staging and L2 residency control measured negative here for structural reasons. But the *method* transfers: lock the instrument, correct the omissions, remove the instructions, calibrate the measurement, stop at the plateau. That sequence is the customization.

## Boundary conditions

- The ladder's numbers live on two non-interchangeable tracks: the fp16 non-causal N=2048 frozen config (v0 6.768 → v5 146.40 TFLOP/s, 3.23 → 69.88% of 209.5) and the bf16 causal N=32768 ladder config (v0 7.97 → v5 184.45 TFLOP/s, 90.34% of roofline). No genuine v0-to-v7 progression curve exists, the final report states this as the highest-value missing measurement, and I should not imply one.
- Every v7 number is provisional, clocks were not locked and compute-sanitizer was not run. Re-running under --lock-clocks could move the headline.
- The 204.164 TFLOP/s GEMM ceiling has no derivation on the branch, it is quoted, not derived, in the v7 work.
- The A/A-null methodology was demonstrated once, at n=10 (p95 effectively the maximum). Its transfer to other benchmarks is a page-level inference, not a verified generalization.
- The v4-to-v5 headline correction cuts both ways: 1.367720x was never fake (it is real on the causal axis), but it was not attributable to v5 alone. Precision about regimes is not the same as debunking.

## Open questions

1. Does the A/A-null threshold transfer to memory-bound kernels, or is `t_noise` a tensor-bound-only artifact?
2. Would warp specialization become viable on `sm_120` with a split-KV data layout that breaks the QK-softmax-PV serial dependency, or is the synchronous mma.sync fundamentally the blocker?
3. At what utilization does the plateau rule's "three consecutive no-gain" trigger become too conservative, would the search have found a fifth keep with a longer patience?
4. cuDNN's own profiled kernel on this card is `sm120_flash_fprop_f16_..._cga1x1x1`, the same machinery. If both kernels are on the same architecture, what is the residual 5.4% and where does it live?

[[Q]] Six months from now: re-read this article after re-running an even longer v7 search under locked clocks, does the 1.0447x cuDNN margin survive, and does the plateau rule still hold at 95% tensor-pipe utilization on a fresh part?

## References

1. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," arXiv:2205.14135, 2022. https://arxiv.org/abs/2205.14135.
2. Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning," arXiv:2307.08691, 2023. https://arxiv.org/abs/2307.08691.
3. Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," arXiv:2407.08608, 2024. https://arxiv.org/abs/2407.08608.
4. Zadouri et al., "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling," arXiv:2603.05451, 2026. https://arxiv.org/abs/2603.05451.

