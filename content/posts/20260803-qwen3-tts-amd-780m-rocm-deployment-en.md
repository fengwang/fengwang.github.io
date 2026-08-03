---
id: 20260803-qwen3-tts-amd-780m-rocm-deployment-en
title: "Qwen3-TTS on AMD Radeon 780M (gfx1103): A Docker + ROCm Deployment Autopsy"
slug: qwen3-tts-amd-780m-rocm-deployment-en
date: 2026-08-03
lastmod: 2026-08-03
draft: false
format: "long"
domain: deep-learning
subdomain: inference-deployment
summary: >-
  A full post-mortem of deploying Qwen3-TTS-Openai-Fastapi on an AMD Radeon
  780M (gfx1103) APU via Docker + ROCm. Central finding: per-step .item()
  GPU-to-CPU syncs inside transformers generate() deadlock permanently on this
  APU because of lost HSA completion signals; routing around them with the
  project's fast codebook path made the service stable.
confidence: confident
prerequisites:
  - Docker and container basics
  - PyTorch inference fundamentals
  - ROCm / HIP basics
  - AMD APU integrated-graphics characteristics
related: []
tags: ["rocm", "amd-apu", "tts", "docker", "inference", "gfx1103", "troubleshooting"]
bibliography: ""
code_repo: "https://github.com/novicezk/Qwen3-TTS-Openai-Fastapi"
sources_used: []
---


# Qwen3-TTS on AMD Radeon 780M (gfx1103): A Docker + ROCm Deployment Autopsy

**Abstract.** This document is a complete post-mortem of deploying the
Qwen3-TTS-Openai-Fastapi service (OpenAI-compatible TTS API, Qwen3-TTS-12Hz-
1.7B-CustomVoice model, default voice Vivian) on an AMD Phoenix1 Radeon 780M
(gfx1103) APU using Docker with an ROCm container. It is written for the
author's future self and for engineers who will deploy LLM / MoE inference on
the same hardware class. The central finding: the per-token `.item()` GPU-to-
CPU synchronization inside the Hugging Face transformers `generate()` loop
permanently deadlocks on this APU because HSA completion signals are
unreliable, while short texts survive only because they trigger few syncs.
Routing around `generate()` with the project's built-in fast codebook path
eliminates the syncs and stabilizes the service at RTF ~2.32. Every command,
error message, and observation in this document comes from the actual
2026-08-02 debugging session.

## 1. Background: running TTS on a 780M

In early August 2026 I needed a self-hosted TTS service reachable over HTTP on
the local network, able to synthesize Chinese and English, running as a
persistent container whose configuration survives restarts. I chose
Qwen3-TTS-Openai-Fastapi with the Qwen3-TTS-12Hz-1.7B-CustomVoice model and
Vivian as the default voice.

The hardware was an AMD Phoenix1 Radeon 780M (gfx1103), an APU integrated GPU
that shares system memory and has no dedicated VRAM. gfx1103 does not appear
in the official ROCm supported-device list. The host passes only
`/dev/kfd` and `/dev/dri/renderD128` into the container; all ROCm userspace
lives inside the image. That design decision was the root of most subsequent
trouble.

Debugging ran from 10 pm into the early morning: four files modified, several
image rebuilds, and three distinct forms of GPU deadlock. This article records
the full chain in chronological order and closes with a distilled checklist
intended as a pitfall manual for future LLM / MoE deployments.

## 2. Environment bring-up: image, dependencies, kernel masquerade

### 2.1 The base image tag never existed

The repository's `Dockerfile.rocm` referenced
`rocm6.3.1_ubuntu22.04_py3.12_pytorch_release_2.6.0`. The build failed with
"not found" from Docker Hub. Checking the Hub's tag listing confirmed that
this exact combination was never published.

I switched to `rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0`: identical
torch 2.6.0 and Python 3.12, differing only in the Ubuntu base layer. I
deliberately avoided ROCm 7.x because the community reports SIGSEGV on the
780M; the 6.x line is comparatively stable.

```dockerfile
FROM rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
```

### 2.2 pip silently replaced torchaudio with the CUDA build

The first container start failed with
`libcudart.so.13: cannot open shared object file`. The cause was subtle: the
project's pyproject.toml declares a bare `torchaudio` dependency with no
version constraint, so pip resolved the then-latest CUDA build (torchaudio
2.11.0), overwriting the image's ROCm torch stack and pulling in CUDA-only
libcudart.

The fix was a forced reinstall of the matching ROCm build after
`pip install -e ".[api]"`, from PyTorch's official ROCm wheel index:

```bash
pip install --no-cache-dir --no-deps \
  torchaudio==2.6.0+rocm6.2.4 \
  --index-url https://download.pytorch.org/whl/rocm6.2.4
```

Lesson: treat every bare dependency inside a ROCm image with suspicion. pip
defaults to the newest CUDA build; lock versions explicitly with a `+rocm`
suffix.

### 2.3 gfx1103 has no precompiled kernels: HSA_OVERRIDE_GFX_VERSION

Model loading failed with `HIP error: invalid device function` because
gfx1103 is absent from the ROCm 6.4.3 precompiled-kernel list (which covers
gfx908/90a/1030/1100/1101/942 and others). The standard remedy is the
`HSA_OVERRIDE_GFX_VERSION` environment variable, which masquerades the chip
as a different model.

Here is the counterintuitive trap: following my ollama deployment notes I
first set `11.0.2` (the real version for gfx1103) and still got "invalid
device function", because the image contains no kernels compiled for 11.0.2.
The value that works is:

```
HSA_OVERRIDE_GFX_VERSION=11.0.0   # masquerade as gfx1100 (desktop RDNA3)
```

11.0.0 maps to gfx1100; the desktop RDNA3 kernels ship in the image and load
directly. Set this in compose `environment` (it overrides the Dockerfile ENV).

## 3. First wave of GPU hangs: flash-attn, AOTriton, and tunableop

With the model loadable, every warmup now hit the same hard error:

```
HW Exception by GPU node-1 reason: GPU Hang
```

The GPU side deadlocked completely and the driver required a reset. I
eliminated suspects one at a time.

### 3.1 flash-attn compiled, then proved useless

The Dockerfile compiles flash-attn v2.8.3 from source. It compiled
successfully and the model even loaded in flash_attention_2 mode, but warmup
always hung. After switching `attention` to `sdpa` in the config, loading
worked. Conclusion: under the gfx1103 masquerade, the flash-attn kernel path
is unstable; sdpa is the only viable choice.

### 3.2 TORCH_SDPA_ENABLE_FLASH=0 was a false positive

Logs repeatedly showed "Using AOTriton backend for Flash Attention forward".
PyTorch 2.6 on ROCm routes SDPA through an AOTriton flash implementation by
default, so I tried disabling it with `TORCH_SDPA_ENABLE_FLASH=0`. My test
script passed, which looked like success.

It was a false positive. The test used float32 inputs, and flash attention
does not support float32, so it silently fell back to the efficient backend
regardless of the environment variable. With bf16 inputs the variable had no
effect and the flash path ran anyway. The correct way is to force-disable it
in Python:

```python
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
```

To make this apply automatically at process start, I wrote a sitecustomize.py
into the image's site-packages; Python imports it at startup, so no
application code needed modification.

### 3.3 PYTORCH_TUNABLEOP_ENABLED=1 was the hidden bomb

This was the first fix that produced visible progress. The image ships with
`PYTORCH_TUNABLEOP_ENABLED=1`, which runs a kernel-tuning benchmark on first
call of every operator. On the APU that tuning process hangs the GPU. Setting
it to 0 let warmup 3/3 complete, the service came up for the first time, and
short-text synthesis worked.

At this point short text ("你好世界。" / "Hello, world.") returned reliably
in 2.6 seconds. I assumed debugging was over.

## 4. The main culprit: per-step .item() sync inside transformers generate()

### 4.1 Symptom: short text fine, medium text always hangs

The service answered short requests, but any request of moderate length
(about 30 characters) hung forever. curl timed out at 120 seconds, and the
container health went from healthy to starting (warmup wedged). The symptom
was highly reproducible: short text 100% success, medium text 100% hang.

GPU monitoring showed an odd pattern:

```
t=1s   busy=74%    # generation in progress
t=9s   busy=96%    # GPU under load
t=10s  busy=0      # GPU suddenly completely idle
t=11s+ busy=0      # stays idle, but the request never returns
```

The GPU finished its work and went idle, while two CPU threads spun at 99.9%
each. This was not a GPU computation problem; the CPU was waiting for
something that never arrived.

### 4.2 Double evidence from py-spy and gdb

After granting the container SYS_PTRACE, py-spy showed the Python stack:

```
_has_unfinished_sequences (transformers/generation/utils.py:2597)
_sample (transformers/generation/utils.py:2779)
generate (transformers/generation/utils.py:2564)
forward (qwen_tts/core/models/modeling_qwen3_tts.py:1891)
```

Wedged inside the transformers generation loop's termination check. gdb then
showed the C stack:

```
_local_scalar_dense_cuda  ->  .item()  GPU-to-CPU sync copy
-> libhsa-runtime64.so.1 signal wait
```

The chain was complete: `_has_unfinished_sequences` calls `tensor.item()`
after every generated token, copying the result from GPU to CPU to decide
whether to stop. `.item()` is a D2H copy that depends on an HSA completion
signal. On the 780M that signal is unreliable: the kernel actually finishes,
but the signal is lost and the CPU blocks forever.

Short text survived because it performs few syncs and happened to receive
every signal. Long text performs many syncs, and the probability of hitting a
lost signal approaches one. This is a probabilistic failure, not a length
threshold.

### 4.3 One change that made things worse: HSA_ENABLE_INTERRUPT=0

At one point I set `HSA_ENABLE_INTERRUPT=0` to make HSA poll instead of using
interrupts. That converted "waiting for a signal" into CPU spinning: the
symptom changed from GPU Hang to two threads at 99.9% CPU, which was harder
to diagnose. Looking back at the timeline, the first stable run (warmup
passed, short text worked) happened under the default interrupt mode, before
this variable was introduced. I removed it and went back to defaults.

## 5. The fix: non_streaming_mode and fast codebook

Once the `.item()` sync was identified, the strategy became: bypass
transformers' `generate()` loop so no per-token synchronization happens. The
project code happens to provide two routes.

### 5.1 First route: non_streaming_mode=True

`generate_custom_voice` defaults to `non_streaming_mode=False`, which
simulates streaming input and runs the per-token loop (a `.item()` on every
step). Passing `True` runs one-shot generation and cuts the sync count by an
order of magnitude. I patched the optimized backend's call:

```python
wavs, sr = self.model.generate_custom_voice(
    text=text, language=language, speaker=voice, instruct=instruct,
    non_streaming_mode=True,
)
```

An isolated test confirmed the diagnosis: for identical short text, the
streaming-simulation mode deadlocked while non_streaming mode succeeded in
4.4 seconds. This proved the root cause but did not fully cure it: medium
text still hung intermittently through the API.

### 5.2 Second route: fast codebook

The real cure is `enable_streaming_optimizations(use_fast_codebook=True)`. It
replaces the inner code_predictor's Hugging Face `generate()` with the
project's own `generate_fast`: pure tensor operations (topk / multinomial /
masked_fill, all on GPU), zero `.item()` syncs, and a fixed iteration count
of num_codebooks with no CPU involvement in the stop decision.

In the original code the fast codebook switch was gated behind
`use_compile`, so `use_compile: false` skipped the entire
`_apply_optimizations` block. I changed the condition so either flag enables
the block, and made the hardcoded `use_compile=True` read from config:

```python
if (opt.get("use_compile", True) or opt.get("use_fast_codebook", True)) \
        and self.device != "cpu":
    await self._apply_optimizations(model_key, model_info, opt)
```

Fast codebook is now enabled independently, while torch.compile stays off
(compilation is both slow and risky on the APU).

### 5.3 Supporting stability parameters

Final compose environment:

```
HSA_OVERRIDE_GFX_VERSION=11.0.0
HSA_ENABLE_SDMA=0          # known iGPU hang source
PYTORCH_TUNABLEOP_ENABLED=0
TTS_AUTOCHUNK=false        # chunked continuous generation also hung; one-pass now
TTS_LAZY_LOAD=false
```

`HSA_ENABLE_SDMA=0` is a community-known iGPU stability switch.
`TTS_AUTOCHUNK=false` came from observing that multi-chunk continuous
generation also triggered hangs; with it off, long text synthesizes in one
pass.

## 6. Verification and performance

Benchmark after the fix (1.7B + sdpa + fast codebook, no torch.compile):

| test | wall time | audio | RTF |
|---|---|---|---|
| English short | 6.8s | 2.9s | 2.32 |
| English medium | 27.4s | 11.9s | 2.32 |
| Chinese medium | 20.0s | 8.6s | 2.32 |
| overall p95 | — | — | 2.34 |

RTF 2.32 means 2.3 seconds of compute per second of audio. For a 780M
integrated GPU that is a reasonable level: batch synthesis is fully usable;
real-time conversation (RTF < 1) is not.

The long-text validation was the real relief: synthesizing the full
"蜀道难" (Hard Is the Road to Shu, a 500+ character classical Chinese poem)
in one pass produced 119.8 seconds of audio (24 kHz / 16-bit WAV) in 5
minutes 38 seconds with no hangs and steady GPU load. At this length the
failure rate used to be 100%.

One data trap surfaced along the way: the local directory labeled "0.6B"
contained a byte-for-byte copy of the 1.7B model (identical MD5, config
`tts_model_size: "1b7"`). The real 0.6B (905M parameters) lived in another
legacy directory. Never trust a directory name when switching models.

## 7. Checklist for future LLM / MoE deployments

1. Verify the base image tag exists before writing the Dockerfile. The
   Docker Hub tag listing is paginated and searchable; five minutes of
   checking saves a failed build.
2. Pin every torch-adjacent dependency inside ROCm images. Bare dependencies
   get resolved by pip to the newest CUDA build. Use `+rocm<ver>` suffixes
   and the official ROCm wheel index uniformly.
3. For gfx1103, set `HSA_OVERRIDE_GFX_VERSION=11.0.0` (masquerade as
   gfx1100). 11.0.2 / 11.0.3 are the true gfx1103 version numbers but no
   kernels exist for them in the image, producing "invalid device function".
   Set it in compose environment so it overrides the Dockerfile ENV.
4. Disable `PYTORCH_TUNABLEOP_ENABLED` on APUs. The kernel-tuning benchmark
   hangs the integrated GPU; it is the easiest hidden bomb to miss.
5. Disable flash-attn and the AOTriton flash SDPA. flash-attn compiles but
   hangs at runtime; for SDPA, force `enable_flash_sdp(False)` via a
   sitecustomize.py so it applies at Python startup. The
   `TORCH_SDPA_ENABLE_FLASH=0` environment variable is unreliable on ROCm,
   and any test of it must use bf16 inputs or it is a false positive.
6. Watch for per-token `.item()` syncs inside transformers `generate()`.
   Any GPU-to-CPU synchronization in the generation loop is a time bomb on
   the 780M. Prefer sync-free generation paths (here: fast codebook), or
   move the termination check onto the GPU.
7. Leave `HSA_ENABLE_INTERRUPT` at its default. Setting it to 0 turns lost
   signals into CPU spinning, which is much harder to diagnose.
8. Diagnostic toolkit: py-spy for Python stacks, gdb for C stacks, and
   `gpu_busy_percent` for GPU load. Cross-referencing them quickly
   distinguishes "GPU cannot compute" from "CPU waits forever".
9. Test across a length gradient. Short-text success does not imply
   long-text success; signal loss is probabilistic and scales with sync
   count.
10. Persistence trio: `restart: unless-stopped`, bind-mounted model and
    config, and bind-mounted cache directories (`~/.cache/miopen`,
    torch_extensions). Otherwise a container recreate forces kernel
    re-search/recompilation and can reproduce the hang.
11. Locking the GPU frequency to high improves stability:
    `/sys/class/drm/card0/device/power_dpm_force_performance_level` set to
    "high" requires root and resets on reboot; persist it with a systemd
    oneshot service.
12. Verify model authenticity: cross-check the directory name, the config's
    model_size field, and file MD5s. Do not trust a copied directory.

## 8. Boundary conditions

- The findings hold for ROCm 6.4.3 + torch 2.6.0 + Ubuntu 24.04. With other
  ROCm versions (especially 7.x), kernel coverage and AOTriton behavior
  change, and the HSA override value may need retesting.
- Masquerading gfx1103 as gfx1100 is a compatibility shim; performance is
  whatever it measures to be. Other APUs (gfx1101/1102, Strix Point) may
  need different stable parameter sets.
- The fast codebook path bypasses the HF `generate()` of this project's
  code_predictor only. LLM / MoE inference that must use transformers
  `generate()` (complex sampling, stopping criteria) cannot adopt this fix
  directly; a sync-free path must be found separately.
- RTF 2.32 is "usable", not "good". Streaming endpoints, voice cloning, and
  Base (non-CustomVoice) models were not validated in this deployment and
  may still trigger hangs.
- Short-text success in early versions (2.6s) was luck: few syncs. It must
  not be taken as evidence that a configuration works; always re-test with
  medium or long text.

## 9. Open questions

- What RTF does the real 0.6B model (905M parameters) achieve under the
  same configuration? If it approaches 1.0, does real-time become viable?
- Is there a usable torch.compile configuration on the 780M (e.g.
  reduce-overhead plus manual warmup)? If so, how far below RTF 2.32 can
  we go?
- Has the ROCm 7.x SIGSEGV on gfx1103 been fixed? If yes, is migrating
  worth it for better kernel coverage?
- Is the streaming endpoint (`/v1/audio/speech` streaming mode) stable
  under fast codebook? This decides voice-agent viability.
- For LLM / MoE inference (e.g. vLLM's ROCm support), do the HSA override
  and tunableop lessons transfer directly, or does vLLM have its own APU
  parameter system?

[[Q]] Six months from now: is there a fully sync-free generation path
(including a custom stopping criterion) that makes transformers
`generate()` run stable long sequences on the 780M?

## 10. References

1. Qwen3-TTS-Openai-Fastapi repository (Dockerfile.rocm, config.yaml,
   api/backends/optimized_backend.py), local copy, August 2026.
2. Docker Hub rocm/pytorch image tag listing, queried August 2026.
3. Qwen3-TTS-12Hz-1.7B-CustomVoice model config.json (local model
   directory).
4. PyTorch ROCm wheel index, download.pytorch.org/whl/rocm6.2.4, accessed
   August 2026.
5. AMD ROCm community discussions on gfx1103 and
   HSA_OVERRIDE_GFX_VERSION (web search, August 2026).
