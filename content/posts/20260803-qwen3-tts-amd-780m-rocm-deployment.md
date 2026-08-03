---
id: 20260803-qwen3-tts-amd-780m-rocm-deployment
title: Qwen3-TTS 在 AMD Radeon 780M 上的 Docker + ROCm 部署排障记录
slug: qwen3-tts-amd-780m-rocm-deployment
date: 2026-08-03
lastmod: 2026-08-03
draft: false
format: "long"
domain: deep-learning
subdomain: inference-deployment
summary: >-
  在 AMD Radeon 780M (gfx1103) 集成 GPU 上用 Docker + ROCm 部署 Qwen3-TTS 的完整
  排障记录。核心结论：transformers generate() 每步 .item() 的 GPU→CPU 同步在 APU
  上会因 HSA 信号丢失而永久卡死，绕过它的 fast codebook 方案使服务稳定运行。
confidence: confident
prerequisites:
  - Docker 与容器基本概念
  - PyTorch 推理基础
  - ROCm / HIP 基本概念
  - AMD APU 集成显卡特性
related: []
tags: ["rocm", "amd-apu", "tts", "docker", "inference", "gfx1103", "troubleshooting"]
bibliography: ""
code_repo: "https://github.com/novicezk/Qwen3-TTS-Openai-Fastapi"
sources_used: []
---


# Qwen3-TTS 在 AMD Radeon 780M 上的 Docker + ROCm 部署排障记录

## 背景：在 780M 上跑 TTS 的目标与约束

2026 年 8 月初，我要在本地部署一个自托管的 TTS 服务，选型是
Qwen3-TTS-Openai-Fastapi（OpenAI 兼容接口）加 Qwen3-TTS-12Hz-1.7B-CustomVoice 模型，
默认音色 Vivian。目标很朴素：局域网内能调 HTTP API 合成中英文语音，容器持久运行，
重启不丢配置。

硬件是一个我此前没在 PyTorch 场景认真用过的家伙：AMD Phoenix1 Radeon 780M
(gfx1103)，APU 集成显卡。它共享系统内存，没有独立显存，官方 ROCm 支持列表里
没有 gfx1103 这个名字。宿主机只透传 `/dev/kfd` 和 `/dev/dri/renderD128` 进容器，
ROCm 用户态全部住在镜像里，这是后来所有麻烦的根源。

排障从晚上 10 点持续到凌晨，前后改了 4 个文件、重建镜像多次、经历了三种不同
形态的 GPU 卡死。这篇文章按时间顺序记录完整链路，末尾是提炼出的经验清单，
目标是给之后部署 LLM / MoE 模型时当避坑手册用。

## 环境搭建：镜像、依赖与内核伪装

### 第一步就踩空：镜像 tag 不存在

仓库自带的 `Dockerfile.rocm` 引用
`rocm6.3.1_ubuntu22.04_py3.12_pytorch_release_2.6.0`，构建时 Docker Hub 直接
返回 not found。查了 Docker Hub 的 tag 列表，发现 6.3.1 这个组合从未发布过。

改用 `rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0`：torch 2.6.0 相同、
Python 3.12 相同，只差 Ubuntu 基础层。特意避开 ROCm 7.x，因为社区在 780M 上
有 SIGSEGV 报告，6.x 系列相对稳。

```dockerfile
FROM rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
```

### torchaudio 被 pip 覆盖成 CUDA 版

第一次启动容器，日志报 `libcudart.so.13: cannot open shared object file`。
根因很隐蔽：项目的 pyproject.toml 里裸依赖 `torchaudio`（没有版本约束），
pip 解析时装到了当时最新的 CUDA 版 torchaudio 2.11.0，把镜像里 ROCm 的
torch 栈覆盖掉了，还引入了 CUDA 专属的 libcudart。

修复是在 `pip install -e ".[api]"` 之后强制重装 ROCm 配套版本，并从
PyTorch 官方 ROCm wheel 源拉取：

```bash
pip install --no-cache-dir --no-deps \
  torchaudio==2.6.0+rocm6.2.4 \
  --index-url https://download.pytorch.org/whl/rocm6.2.4
```

经验：ROCm 镜像里任何裸依赖都要警惕，pip 默认解析到 CUDA 最新版，
必须显式锁定 `+rocm` 后缀的版本。

### gfx1103 没有预编译 kernel：HSA_OVERRIDE_GFX_VERSION

模型加载时报 `HIP error: invalid device function`。原因是 gfx1103 不在
ROCm 6.4.3 预编译 kernel 列表里（列表覆盖 gfx908/90a/1030/1100/1101/942 等）。
解决办法是环境变量 `HSA_OVERRIDE_GFX_VERSION` 把芯片伪装成其他型号。

这里有个反直觉的坑：我最初按 ollama 部署经验设了 `11.0.2`（gfx1103 的
真实版本号），结果反而报 invalid device function，因为镜像里根本没有为
11.0.2 编译的 kernel。最终生效值是：

```
HSA_OVERRIDE_GFX_VERSION=11.0.0   # 伪装成 gfx1100（桌面 RDNA3）
```

11.0.0 对应 gfx1100，桌面 RDNA3 kernel 在镜像里自带，加载后直接可用。
这个值必须写在 compose 的 environment 里（会覆盖 Dockerfile 的 ENV）。

## 第一波 GPU Hang：flash-attn、AOTriton 与 tunableop

模型能加载了，但每次 warmup 都触发同一个硬错误：

```
HW Exception by GPU node-1 reason: GPU Hang
```

这条错误意味着 GPU 侧彻底死锁，驱动需要重置。我按嫌疑顺序一个个排除：

### flash-attn 编译成功但没有意义

Dockerfile 里从源码编译了 flash-attn v2.8.3，居然编译成功，模型也确实以
flash_attention_2 模式加载了。但 warmup 必挂。把 config 里 `attention` 改成
`sdpa` 之后，加载正常了。结论：在 gfx1103 伪装环境下，flash-attn 的 kernel
路径不稳定，用 sdpa 是唯一选择。

### TORCH_SDPA_ENABLE_FLASH=0 是假阳性

日志里反复出现 "Using AOTriton backend for Flash Attention forward"。
PyTorch 2.6 ROCm 的 SDPA 默认走 AOTriton flash 实现，我试着用环境变量
`TORCH_SDPA_ENABLE_FLASH=0` 关掉它。测试脚本跑通了，看起来有效。

但后来发现这是假阳性：测试用的是 float32 输入，flash attention 本来就不
支持 fp32，自动降级到 efficient backend，跟环境变量无关。换成 bf16 输入
后，环境变量完全失效，flash 路径照走。正确做法是在 Python 里强制关：

```python
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
```

为了让服务进程启动时自动执行，我在 Dockerfile 里写了一个 sitecustomize.py
放进 site-packages，Python 启动时自动加载，不需要改业务代码。

### PYTORCH_TUNABLEOP_ENABLED=1 是隐藏炸弹

这是第一个真正让我看到转机的修复。镜像自带的
`PYTORCH_TUNABLEOP_ENABLED=1` 会在首次调用每个算子时做 kernel 调优基准，
在 APU 上这个调优过程会挂 GPU。设成 0 之后，warmup 3/3 全部通过，服务
第一次真正起来了，短文本合成成功。

到这里，短文本（"你好世界。"）稳定 2.6 秒出结果。我一度以为排障结束了。

## 主犯：transformers generate() 的 .item() 同步

### 症状：短文本好，长文本必挂

服务能响应短文本，但任意中等长度文本（约 30 个字符）请求就会永久挂起。
curl 等 120 秒超时，容器 health 从 healthy 变 starting（warmup 挂住）。
这个症状高度稳定：短文本 100% 成功，中文本 100% 挂。

监控 GPU 时发现一个反常现象：

```
t=1s   busy=74%    # 生成进行中
t=9s   busy=96%    # GPU 高负载
t=10s  busy=0      # GPU 突然完全空闲
t=11s+ busy=0      # 一直空闲，但请求永不返回
```

GPU 干完活就停了，但 CPU 侧有两个线程各占 99.9%，在空转。这不是 GPU 算不动，
是 CPU 在等一个永远不来的东西。

### py-spy 和 gdb 的双重证据

给容器加了 SYS_PTRACE 权限后，用 py-spy 抓 Python 栈：

```
_has_unfinished_sequences (transformers/generation/utils.py:2597)
_sample (transformers/generation/utils.py:2779)
generate (transformers/generation/utils.py:2564)
forward (qwen_tts/core/models/modeling_qwen3_tts.py:1891)
```

卡在 transformers 生成循环的结束条件判断里。再用 gdb 抓 C 栈：

```
_local_scalar_dense_cuda  ->  .item()  GPU→CPU 同步拷贝
-> libhsa-runtime64.so.1 信号等待
```

链条完整了：transformers 的 `_has_unfinished_sequences` 每生成一个 token
就调用 `tensor.item()`，把 GPU 上的结果同步回 CPU 判断是否该停。`.item()`
本质是 GPU→CPU 的 D2H 拷贝，依赖 HSA 信号通知完成。在 780M 上这个信号
不可靠，kernel 实际跑完了，但信号丢失，CPU 永久阻塞在等待里。

短文本为什么能过？因为同步次数少，碰巧每次都拿到信号；长文本同步次数
多了，撞上信号丢失的概率接近 1。这是概率性问题，不是长度阈值问题。

### 一个让事情更糟的尝试：HSA_ENABLE_INTERRUPT=0

我一度设置 `HSA_ENABLE_INTERRUPT=0` 让 HSA 用轮询代替中断，结果把
"等信号"变成了 CPU 自旋，症状从 GPU Hang 变成双线程 99.9% CPU 空转，
更加难排查。回看时间线，服务第一次稳定跑通时（warmup 全过、短文本成功）
恰恰是在没设置这个变量的默认中断模式下。撤掉它，恢复默认。

## 修复组合：non_streaming_mode 与 fast codebook

定位到 `.item()` 同步之后，思路变成：绕过 transformers 的 generate() 循环，
不逐 token 同步。项目代码里恰好有两条路。

### 第一条路：non_streaming_mode=True

`generate_custom_voice` 默认 `non_streaming_mode=False`，这个模式会模拟
流式输入，走逐 token 的生成循环（每步 `.item()`）。传 `True` 走一次性
生成，同步次数少一个量级。我改了 optimized_backend 的调用：

```python
wavs, sr = self.model.generate_custom_voice(
    text=text, language=language, speaker=voice, instruct=instruct,
    non_streaming_mode=True,
)
```

隔离测试确认：同样的短文本，流式模拟模式卡死，non_streaming 模式 4.4 秒
成功。这证明根因判断正确，但还没根治：中等文本在 API 里仍偶发卡住。

### 第二条路：fast codebook

真正的解药是 `enable_streaming_optimizations(use_fast_codebook=True)`。
它把内层 code_predictor 的 HF generate() 换成项目自带的 `generate_fast`：
纯 tensor 操作（topk / multinomial / masked_fill 全在 GPU 上），零 `.item()`
同步，固定迭代 num_codebooks 次，不需要任何 CPU 参与判断。

原代码里 fast codebook 的开关被挂在 `use_compile` 条件后面：`use_compile: false`
时整个 `_apply_optimizations` 被跳过。我把条件改成两者任一为真就执行，
并把硬编码的 `use_compile=True` 改成从 config 读取：

```python
if (opt.get("use_compile", True) or opt.get("use_fast_codebook", True)) \
        and self.device != "cpu":
    await self._apply_optimizations(model_key, model_info, opt)
```

这样 fast codebook 独立启用，torch.compile 保持关闭（APU 上编译又慢又危险）。

### 配套稳定化参数

最终 compose 环境变量组合：

```
HSA_OVERRIDE_GFX_VERSION=11.0.0
HSA_ENABLE_SDMA=0          # 已知 iGPU hang 源
PYTORCH_TUNABLEOP_ENABLED=0
TTS_AUTOCHUNK=false        # 分块连续生成也会挂，一次性合成
TTS_LAZY_LOAD=false
```

`HSA_ENABLE_SDMA=0` 是社区已知的 iGPU 稳定性开关，`TTS_AUTOCHUNK=false`
是因为实测自动分块的多段连续生成也会触发 hang，关掉后长文本一次合成。

## 性能与稳定性验证

修复后的 benchmark（1.7B + sdpa + fast codebook，无 torch.compile）：

| 测试 | 生成耗时 | 音频时长 | RTF |
|---|---|---|---|
| 英文短句 | 6.8s | 2.9s | 2.32 |
| 英文中长句 | 27.4s | 11.9s | 2.32 |
| 中文中长句 | 20.0s | 8.6s | 2.32 |
| 整体 p95 | — | — | 2.34 |

RTF 2.32 意味着生成 1 秒音频要花 2.3 秒。对 780M 集成 GPU 这是合理水平：
批量合成完全可用，实时对话（需要 RTF < 1）不行。

真正让人松一口气的是长文本验证：合成《蜀道难》全文 500+ 字，一次性
生成 119.8 秒音频（24kHz/16bit WAV），耗时 5 分 38 秒，全程无 hang，
GPU 稳定满载。之前这种长度是 100% 必挂的。

顺带发现一个数据陷阱：本地标着 "0.6B" 的模型目录其实是 1.7B 的完整拷贝
（两个文件 MD5 完全相同，config 里 `tts_model_size: "1b7"`）。真 0.6B
（905M 参数）在另一处旧部署目录里。换模型时不能只看目录名。

## 对未来 LLM / MoE 部署的经验清单

1. **先验证镜像 tag 存在**，再写 Dockerfile。Docker Hub 的 tag 列表可以
   直接翻页查询，5 分钟能省掉一次构建失败。
2. **ROCm 镜像里锁死所有 torch 相关依赖版本**，裸依赖必被 pip 解析成
   CUDA 最新版。统一用 `+rocm<ver>` 后缀 + 官方 ROCm wheel 源。
3. **gfx1103 一律 `HSA_OVERRIDE_GFX_VERSION=11.0.0`**（伪装 gfx1100）。
   11.0.2/11.0.3 是 gfx1103 真实版本号，但镜像没有对应 kernel，会报
   invalid device function。这个值在 compose 里设置，覆盖 Dockerfile ENV。
4. **在 APU 上禁用 PYTORCH_TUNABLEOP_ENABLED**。kernel 调优基准会挂
   集成 GPU，这是最容易忽略的隐藏炸弹。
5. **禁用 flash-attn 与 AOTriton flash SDPA**。flash-attn 编译能成功但
   运行必挂；SDPA 用 sitecustomize.py 强制 `enable_flash_sdp(False)`，
   Python 启动即生效。环境变量 `TORCH_SDPA_ENABLE_FLASH=0` 在 ROCm 上
   不可靠，且测试要用 bf16 输入否则是假阳性。
6. **警惕 transformers generate() 的逐 token `.item()` 同步**。任何在
   生成循环里做 GPU→CPU 同步的代码，在 780M 上都是定时炸弹。优先选
   无同步的生成路径（本项目是 fast codebook），或把结束判断放到 GPU 侧。
7. **HSA_ENABLE_INTERRUPT 保持默认**，不要设 0。轮询模式把信号丢失变成
   CPU 自旋，症状更难判断。
8. **诊断三件套**：py-spy 抓 Python 栈、gdb 抓 C 栈、`gpu_busy_percent`
   盯 GPU 负载。三者交叉能快速区分"GPU 算不动"和"CPU 等不到"。
9. **测试用例要覆盖长度梯度**：短文本成功不代表长文本能过，信号丢失是
   概率问题，同步次数越多越容易触发。
10. **持久化三件套**：`restart: unless-stopped`、模型与配置 bind mount、
    缓存目录（~/.cache/miopen、torch_extensions）bind mount，否则容器
    recreate 后 kernel 重新搜索编译，可能重现 hang。
11. **GPU 频率锁 high 能提升稳定性**：`/sys/class/drm/card0/device/
    power_dpm_force_performance_level` 写 high 需 root 且重启失效，
    用 systemd oneshot 服务持久化。
12. **验证模型真实性**：目录名、config 的 model_size 字段、文件 MD5 三者
    对照，别被复制品误导。

## Boundary conditions

- 结论基于 ROCm 6.4.3 + torch 2.6.0 + Ubuntu 24.04 这个组合。换 ROCm
  版本（尤其 7.x）后，kernel 覆盖和 AOTriton 行为都会变，HSA override
  值可能需要重测。
- gfx1103 伪装 gfx1100 只是兼容性手段，性能以实测为准；不同 APU
  （如 gfx1101/1102 或 Strix Point）可能有不同的稳定参数组合。
- fast codebook 绕过的是本项目 code_predictor 的 HF generate()。LLM /
  MoE 推理如果必须用 transformers generate()（如采样逻辑复杂、需要
  stopping criteria），这个方案不能直接平移，需要另找无同步路径。
- RTF 2.32 是"能用"不是"好用"。流式端点、voice clone、Base 模型
  （非 CustomVoice）在本部署中未验证，可能仍会触发 hang。
- 短文本 2.6s 的成功在早期版本里是侥幸（同步次数少），不能作为
  "某个配置有效"的证据，必须以中长文本复测为准。

## Open questions

- 0.6B 真模型（905M 参数）在相同配置下 RTF 能到多少？如果逼近 1.0，
  实时场景是否可行？
- torch.compile 在 780M 上是否有可用配置（例如 reduce-overhead +
  手动 warmup）？如果能用，RTF 2.32 还有多少下降空间？
- ROCm 7.x 在 gfx1103 上的 SIGSEGV 是否已被修复？如果修复，是否值得
  迁移以获得更好的 kernel 覆盖？
- 流式输出（/v1/audio/speech 的 streaming 模式）在 fast codebook 下
  是否稳定？这决定 voice agent 场景的可行性。
- 对 LLM / MoE 推理（如 vLLM 的 ROCm 支持），HSA override 和 tunableop
  的经验能否直接复用，还是 vLLM 有自己的 APU 参数体系？

[[Q]] 半年后重读：780M 上有没有一个完全无 .item() 同步的生成路径（包括
自定义 stopping criteria），能让 transformers generate() 在 APU 上稳定
跑长序列？

## References

1. Qwen3-TTS-Openai-Fastapi 项目仓库（Dockerfile.rocm / config.yaml /
   api/backends/optimized_backend.py），2026-08 本地副本。
2. Docker Hub rocm/pytorch 镜像 tag 列表，2026-08 查询。
3. Qwen3-TTS-12Hz-1.7B-CustomVoice 模型 config.json（本地模型目录）。
4. PyTorch ROCm wheel 索引 download.pytorch.org/whl/rocm6.2.4，2026-08 访问。
5. AMD ROCm 社区对 gfx1103 与 HSA_OVERRIDE_GFX_VERSION 的讨论（网络搜索，
   2026-08）。
