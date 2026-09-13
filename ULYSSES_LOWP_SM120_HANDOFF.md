# Ulysses Lowp-Sage2：代码交付与 SM120 验证手册

日期：2026-09-13。本文汇总实现、提交分支、运行脚本、最新 SM90 视频与 timeline 结果，以及 SM120 验证步骤。SM120 GPU 验收由使用者执行；下文的性能数字全部来自 SM90，不能当作 SM120 的预期收益。

## 1. 分支与可复现版本

| 仓库 | 分支 | 本次执行代码快照 |
|---|---|---|
| [DwenGu/flashinfer](https://github.com/DwenGu/flashinfer/tree/feat/ulysses-lowp-sage2) | `feat/ulysses-lowp-sage2` | `1c8283228b97c1ce575cf3815d822f640eb40e4f` |
| [DwenGu/sglang-minimax](https://github.com/DwenGu/sglang-minimax/tree/feat/minimax-h3-ulysses-lowp-boundary-first) | `feat/minimax-h3-ulysses-lowp-boundary-first` | `d6654000274cb20ba6c1a29cbbb31c94d01f4fd1` |
| [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5) | 固定版本，未修改 | `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`，包版本 2.2.0 |

SGLang 分支在执行快照之后只更新本交接文档；执行代码以表中的 SHA 为准。此前按 isort 规范补充的一个 import 分组空行已确认不改变 Python AST。两个功能分支保留已有功能历史，不覆盖 main，也没有合入 main 后续无关更新。FlashInfer 基于此前 `481cb83a`，SGLang 基于此前 `6215b598e`。

| 新提交 | 内容 |
|---|---|
| FlashInfer `865b2c10` | 统一 boundary-first 统计、量化、通用 unpack；有效长度与上下文校验、零 V 保护、共享架构绑定；独立边界回归测试 |
| FlashInfer `1c828322` | torchrun 真实 collective / Sage 验收器，更新 benchmark 与数值契约文档 |
| SGLang `3b5450de6` | consumer 的 scale 宽度统一从 FlashInfer layout 获取 |
| SGLang `72e138983` | 可移植视频请求、50/3-step 采集、SM90/SM120 NVTX 路由与功能分类分析脚本 |
| SGLang `05f52857d` | 中文交接手册与 import 分组空行 |
| SGLang `d66540002` | 预检放到子进程，避免协调进程持有 CUDA 上下文、阻塞 GPU 清理 |

提交前确认：FlashInfer 完整提交及 SGLang `3b5450de6` 相对于原 HEAD 的既有 tracked diff，与最新 SM90 视频报告记录的 SHA-256 一致。FlashInfer 为 `e3757e774aede56e561a1d833a4d0ce91525cc803cc32638f6d70d1064952094`，SGLang 为 `463dd2e8857b82a0644e21769a16869661879e67abfefe21abad217a49fadbc2`。该比较排除了当时尚未跟踪的两个新增测试/验收文件；生产修改与视频批次相同。新增的移植采集脚本另做了路由与解析验证，未宣称已在 SM120 跑通。

## 2. 方案与执行链

常规 Sage2 路径先交换 BF16 Q/K/V，再量化并计算 attention。本方案把量化移到输入 All2All 之前，以 INT8 Q/K、FP8 V 和少量 FP32 scales 组成 payload，减少输入通信字节及独立布局搬运。Attention 继续调用 Sage2 的预量化入口，输出仍走原 BF16 All2All。

```text
QKV 投影 / RoPE
  → local_stats：K sum、V amax、Q grouped-amax、Q/K 边界描述
  → 一次 FP32 stats AllGather
  → finalize_stats：全局 K mean / V scale、边界合并、K grouped-amax / 尾组修复
  → quant_and_pack：INT8 Q/K、FP8 V、scales，直接写发送 payload
  → uint8 payload All2All
  → 通用 unpack：Sage 布局、scale 宽度与尾部清零
  → Sage2 预量化 attention，仅消费有效 U 行
  → 原 BF16 输出 All2All
```

FlashInfer 提供统计、量化和 payload 操作；SGLang 负责真实 collective、缓存分配、架构选择和 Sage 调用。所有合法 shard 都执行 `BOUNDARY_MERGE`。`ALIGNED` 仅保留为旧 padding 策略，不再选择另一套 fused fast path。

| 契约 | SM90 | SM120 |
|---|---|---|
| Layout | `UlyssesLowpSageLayoutSM90` | `UlyssesLowpSageLayout` |
| Q / K 全局分组 | 16 / 128 tokens | 32 / 64 tokens |
| consumer Q scale 宽度 | `ceil(U/64)*4` | `ceil(U/128)*4` |
| consumer K scale 宽度 | `ceil(U/128)` | `ceil(U/64)` |
| V 存储长度对齐 | 128 | 64 |
| Sage 扩展 | `_qattn_sm90` | `_qattn_sm89`，编译目标为 SM120 |
| Sage 入口后缀 | `accum_f32_fuse_v_scale_attn_inst_buf` | `accum_f16_fuse_v_scale_attn_inst_buf` |
| Lowp V scale_max | 2.25 | 2.25 |
| 本批 GPU 状态 | H20-3e 已验证 | 待实际 GPU 验证 |

共同条件：head_dim=128，输入 FP16/BF16，P=2/4/8，head 数可被 P 整除，S=P×L，0<U≤S。输入只在全局尾部 `[U,S)` 填零，然后分片。量化分组始终锚定全局 token 网格，跨多个 rank 的同一组必须共享 scale。

K mean 先用 FP32 累加、除以 U，再舍入到输入 dtype 后用于中心化；Q/K amax floor 为 1e-7，量化使用 `x*(127/amax)` 与 round-to-nearest-even。V 全局 amax=0 的通道输出正 FP8 零和零 scale，该保护在现有 kernel 内执行。非零 V 算术没有改变。

payload 每个目的 rank 一段：Q INT8、K INT8、V FP8、Q/K FP32 scales，再将 chunk 总长补齐至 128 bytes；未使用 slots 和 alignment tail 清零。接收端完整分配并初始化 Sage 所需 scale 宽度。额外 Q scale slots 对应无效 Q 行，其有限非零值不得影响有效输出；验收器专门检查这一点。padding 规则、输入均值舍入和零通道行为不能随优化随意改变。

两套编译模块保留各自的 Q/K 常量，但共享 `csrc/ulysses_lowp.cu` host binding；SM90 的小入口设置宏后包含它。Python 的 K 边界和尾组修复共用辅助函数。本次代码整理的四个生产文件净减少 586 行，整理本身不作为新增性能提升证据。

详细 API 和数值约定见 [FlashInfer 设计文档](https://github.com/DwenGu/flashinfer/blob/1c8283228b97c1ce575cf3815d822f640eb40e4f/docs/design_docs/ulysses_lowp.md)。

## 3. 最新视频基线与已验证结果

当前参考基线为 2026-09-11 重构后的 UP8 批次，Sage2 视频和 trace 都使用 **CUDA per-warp**。不要使用父目录中历史 Triton per-thread Sage2 视频作为当前基线。

原 SM90 主机目录：

```text
/raid/jungu/H3-A2A/deliverables/h3-up4-up8-seed2101/up8-validation/
  REPORT.html / REPORT.md
  videos/case{1,2,3}_up8_{bf16,sage2,lowp}_50steps.mp4
  timelines/case1_up8_{bf16,sage2,lowp}_3steps.nsys-rep
```

共 9 个视频、3 份 timeline。原始媒体没有放入 Git；Git 分支提供代码、脚本和本手册。迁移时将上述完整目录复制为新机器的 `/workspace/reference-sm90/`，即可保留离线 HTML 的全部视频与 trace 链接。例如在 SM120 主机执行，替换实际主机名和目标工作区路径：

```bash
rsync -av SM90_HOST:/raid/jungu/H3-A2A/deliverables/h3-up4-up8-seed2101/up8-validation/ /raid/h3-sm120/reference-sm90/
```

固定配置：8×H20-3e、UP8/TP1、ring=1、FSDP OFF、AdaLN online/offload ON、compile OFF；MiniMax-H3 FL2VA、speed、704p、16:9、请求 5 秒、seed=2101、flow shift=12、audio flow shift=3。实际视频为 1248×704、24fps、124 帧，约 5.17 秒。视频请求 **50 timesteps，实际 49 次 denoise 更新**；性能请求 **3 timesteps，实际 2 次更新**。不要把参数改成 4 来凑 3 次更新。

| 场景 | Prompt |
|---|---|
| case1 | A hummingbird hovering beside a bright red hibiscus flower, wings blurred in slow motion, macro close-up, sunlit garden background |
| case2 | A street musician playing an acoustic guitar on a rainy evening sidewalk, warm streetlight reflections on wet pavement, close-up on hands and face |
| case3 | Waves crashing against dark volcanic rocks at dusk, sea spray backlit by the setting sun, distant seabirds circling. |

| Case | Lowp / BF16 SSIM | Sage2 CUDA / BF16 SSIM | 整理前后 Lowp 全帧解码 |
|---|---:|---:|---|
| case1 | 0.898476 | 0.864213 | 逐字节相同，SSIM=1 |
| case2 | 0.837581 | 0.916122 | 逐字节相同，SSIM=1 |
| case3 | 0.965289 | 0.914842 | 逐字节相同，SSIM=1 |

抽帧覆盖 0.5、2、4 秒：case1 中 Lowp 与 BF16 的蜂鸟轨迹及花朵位置更接近；case2 中 Lowp 的吉他角度、手指、脸部姿态和构图差异更明显，Sage2 CUDA 更接近 BF16；case3 中 Lowp 与 BF16 的太阳、岩石和浪花阶段更接近，Sage2 CUDA 的太阳主体移至画外。这些是相似度与构图观察，不是主观质量排名；仍需完整播放、检查时间连续性和手部细节。此批未独立听评音频。

| Case | BF16 pipeline / denoise s | Sage2 CUDA pipeline / denoise s | Lowp pipeline / denoise s | Lowp pipeline 耗时减少（对 Sage2） |
|---|---:|---:|---:|---:|
| case1 | 127.73 / 124.5739 | 97.73 / 94.5338 | 96.28 / 93.0296 | 1.48% |
| case2 | 127.79 / 124.5225 | 97.75 / 94.4834 | 96.26 / 93.0071 | 1.52% |
| case3 | 127.78 / 124.5441 | 97.70 / 94.4627 | 96.39 / 93.0213 | 1.34% |

每个 case 各测一次，无重复实验置信区间。整理前 Lowp 同三组 pipeline 为 96.23 / 96.29 / 96.29 秒，仅供历史回归参考。

## 4. Timeline 与单模块分析

三份新 trace 都有 8 GPU 的 CUDA/NVTX，每 GPU 2 次更新 × 50 层 = 100 次 DiT attention。

| 后端 | 3-step pipeline / denoise s | GPU0 loop ms | GPU0 输入 A2A 投影累计 ms |
|---|---:|---:|---:|
| BF16 | 8.53 / 5.2312 | 5223.450 | 144.542 |
| Sage2 CUDA | 7.15 / 3.9001 | 3890.883 | 70.054 |
| Lowp | 7.10 / 3.8180 | 3808.909 | 31.920 |

输入 A2A 投影包含其范围内的 pack 和间隙，NCCL kernel 也包含 rank 等待，因此不能解释为纯链路带宽。完整前处理从 Lowp local_stats 或 Sage2 输入 pack 首个 GPU 操作开始，到 Sage attention kernel 开始为止。

| 完整前处理 | GPU0 100 次累计 ms | 每次平均 μs | 8 GPU 各自累计范围 ms |
|---|---:|---:|---:|
| Sage2 CUDA | 167.017 | 1670.169 | 151.930–181.272 |
| Lowp | 94.942 | 949.424 | 90.026–112.906 |

本批 GPU0 前处理累计耗时减少 **43.15%**；这个局部收益与 50-step pipeline 的 **1.34%–1.52%** 是不同口径。

固定模块：GPU0、`denoising_step_1`、`transformer.blocks.25.attn`，即第二次更新、第 26 层。每个 kernel 只归入一行：

| 功能归属 | 对应 kernel / 操作 | Sage2 CUDA μs | Lowp μs |
|---|---|---:|---:|
| Q amax + INT8 | Sage `QuantInt8Kernel`；Lowp `GroupedAmaxKernel` + `QuantInt8GroupScalePackKernel` | 38.368 | 68.544 |
| K 独立统计 + INT8 | Sage mean reduction + `QuantInt8Kernel`；Lowp grouped-amax + quant-pack | 107.360 | 70.240 |
| V 独立 FP8 | Sage `MeanScaleKernel`；Lowp `QuantVFP8WithScalePackKernel` | 31.808 | 56.928 |
| K/V 融合统计 | Lowp `KSumVAmax` | 0 | 62.944 |
| 统计辅助 / 收尾 | boundary min/max、merge、tail repair 等 | 0 | 96.224 |
| 统计 AllGather | NCCL AllGather | 0 | 61.823 |
| 输入 All2All | NCCL SendRecv | 518.175 | 315.071 |
| 发送前独立 QKV pack | `_pack_qkv_destination_major_kernel` | 97.568 | 0 |
| 接收后 BF16 连续化拷贝 | 3 次 ATen direct copy | 325.407 | 0 |
| V 独立 padding + 转置 | fill / cat + `TransposePadPermuteKernel` | 256.287 | 0 |
| 接收后 Lowp unpack / 清零 | `UnpackForSage` 等 | 0 | 92.448 |
| **完整前处理窗口** | 首个 GPU 操作至 Sage kernel 开始 | **1399.325** | **902.942** |

“独立布局转换”指只改变数据排布或连续性、不完成量化的单独 kernel。Lowp 将发送 pack 融入量化，并用一次接收 unpack 生成 Sage 布局；这些融合成本已记入对应行，不能当作免费。K/V 共享统计也不能漏算。表中 Q 量化并没有更快，主要收益来自通信和减少数据搬运。kernel 时间之和与包含间隙、重叠的窗口不同。完整实例化 kernel 名、stream 和 NVTX 明细保留在最新 `REPORT.html` 中。

## 5. SM120 环境与安装

E2E 脚本固定使用 8 张同构可见 GPU：UP8/TP1 为默认，UP4/TP2 为可选。先确认 GPU 空闲、显存能承载 FSDP OFF 的 MiniMax-H3，保存 `nvidia-smi` 和 `nvidia-smi topo -m` 到最终报告。SM120 的互联拓扑与 H20 不同，不能要求相同通信收益。不要为跑通而静默开启 FSDP、关闭 AdaLN online 或改分辨率。

宿主机准备独立工作区和已有模型目录，例如 `/raid/h3-sm120`、`/raid/models/MiniMax-H3`。以下容器基线与 SM90 相同；模型权重需自行准备：

```bash
mkdir -p /raid/h3-sm120
H3_IMAGE=lmsysorg/sglang@sha256:9e148f5ac788e856a06166bd6347a831831eb9fcfab4d1770874823a7c29a1a1
docker pull "$H3_IMAGE"
docker run -it --name h3-sm120 --gpus all --ipc=host --network=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /raid/h3-sm120:/workspace \
  -v /raid/models/MiniMax-H3:/models/MiniMax-H3:ro \
  -w /workspace "$H3_IMAGE" bash
```

以下在容器内执行。代码快照固定，避免安装时隐式升级 Torch/CUDA：

```bash
set -euo pipefail
cd /workspace
git clone --branch feat/ulysses-lowp-sage2 https://github.com/DwenGu/flashinfer.git flashinfer
git -C flashinfer checkout 1c8283228b97c1ce575cf3815d822f640eb40e4f
git -C flashinfer submodule update --init --recursive

git clone --branch feat/minimax-h3-ulysses-lowp-boundary-first https://github.com/DwenGu/sglang-minimax.git sglang-lowp-fi
# 保留分支最新手册，同时核验其他文件与固定快照相同。
git -C sglang-lowp-fi diff --exit-code d6654000274cb20ba6c1a29cbbb31c94d01f4fd1 HEAD -- . ':!ULYSSES_LOWP_SM120_HANDOFF.md'

git clone https://github.com/thu-ml/SageAttention.git SageAttention-stock
git -C SageAttention-stock checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5

export TORCH_CUDA_ARCH_LIST="12.0"
export FLASHINFER_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=4
export FLASHINFER_NVCC_THREADS=4
python -m pip install --no-deps --no-build-isolation -e /workspace/flashinfer
python -m pip install --no-deps --no-build-isolation -e /workspace/SageAttention-stock
python -m pip install --no-deps --no-build-isolation -e /workspace/sglang-lowp-fi/python
python -m pip install pytest
apt-get update
apt-get install -y ffmpeg
```

SM90 已测工具链是 PyTorch 2.13.0+cu130、CUDA 13.0、driver 580.95.05、FlashInfer 0.6.18、SageAttention 2.2.0。新机器必须打印实际版本，不把镜像标签当作已验证环境。Sage 的这版 setup 支持 `TORCH_CUDA_ARCH_LIST=12.0`；SM120 编译要求 CUDA ≥12.8。不要复制 H20 生成的 `.so` 或 JIT cache。

Nsight Systems 是本次在容器内额外安装的工具，不能假设基础镜像已经包含。准备 **2026.4.1 Linux x86_64 CLI** 安装包到 `/workspace/tools/nsys.deb`，按 [NVIDIA 下载入口](https://developer.nvidia.com/nsight-systems/get-started)和[安装说明](https://docs.nvidia.com/nsight-systems/InstallationGuide/)获取，然后执行：

```bash
apt-get install -y /workspace/tools/nsys.deb
export PATH=/opt/nvidia/nsight-systems-cli/2026.4.1/target-linux-x64:$PATH
nsys --version
ffmpeg -version
nvcc --version
nvidia-smi
nvidia-smi topo -m
```

解析脚本按 2026.4.1 的 SQLite/CSV schema 验证；使用其他版本需重新确认导出字段。最终 `.nsys-rep` 可以复制到装有兼容版本 Nsight Systems GUI 的工作站打开。[NVIDIA 兼容说明](https://developer.nvidia.com/nsight-systems/get-started)

检查导入路径与 SM120 layout，必须命中三个源码 checkout：

```bash
python - <<'PY'
import torch, flashinfer, sageattention, sglang
import flashinfer.comm.ulysses_lowp as lowp
from sglang.multimodal_gen.runtime.layers.attention.backends.ulysses_lowp_v2g import _arch_ops
print(torch.__version__, torch.version.cuda)
print(flashinfer.__file__, sageattention.__file__, sglang.__file__)
assert torch.cuda.device_count() == 8
assert all(torch.cuda.get_device_capability(i) == (12, 0) for i in range(8))
cap = lowp.capability('cuda')
print(cap)
assert cap['supported'] and cap['layout_class'] == 'UlyssesLowpSageLayout'
assert (cap['compiled_q_group'], cap['compiled_k_group'], cap['compiled_head_dim']) == (32, 64, 128)
ops = _arch_ops()
assert ops is not None and ops.name == 'sm120'
print(type(ops.layout).__name__, ops.layout.scale_widths(129))
PY
```

## 6. 先验证算子与真实 collective

```bash
cd /workspace/flashinfer
python -m pytest tests/comm/test_ulysses_lowp.py tests/comm/test_ulysses_lowp_boundary.py -q -ra
python -m pytest tests/trace/test_template_init.py tests/trace/test_fi_trace_template_consistency.py -k ulysses -q -ra
mkdir -p /workspace/sm120-gate-work

torchrun --standalone --nproc-per-node=2 benchmarks/comm/validate_ulysses_lowp.py \
  --local-sequence 129 --used 257 --batch 2 --heads 8 --dtype bfloat16 \
  --output /workspace/sm120-gate-work/smoke-p2.json
```

`validate_ulysses_lowp.py` 检查真实 AllGather/AllToAll 字节、接收布局、scales、有效输出边界、额外 Q slots 不影响输出、Sage 预量化入口和 FP32 SDPA 质量。非零通道要求 cosine≥0.999，relative L1≤stock Sage2 的 1.10 倍；stock 比较排除构造的零通道，Lowp 完整输出必须有限。全零 V 要求输出精确零，不能以 stock 零 amax 行为作为 oracle。失败将写 rank failure 信息并返回非零。

之后分别验证 P=2/4/8、FP16/BF16、tiny shard / 非对齐 / 对齐 / 尾部 padding。下面是可执行的扩展矩阵；每次使用新的结果目录，脚本不会覆盖已有 JSON：

```bash
set -euo pipefail
cd /workspace/flashinfer
for p in 2 4 8; do
  for dtype in float16 bfloat16; do
    for l in 1 65 128 129; do
      total=$((p*l))
      for used in 1 $((total-1)) "$total"; do
        # l=1,p=2 时 used=1 重复，跳过已经成功的同一项。
        result=/workspace/sm120-gate-work/p${p}-${dtype}-l${l}-u${used}.json
        if test -f "$result"; then continue; fi
        torchrun --standalone --nproc-per-node="$p" benchmarks/comm/validate_ulysses_lowp.py \
          --local-sequence "$l" --used "$used" --batch 2 --heads 8 --dtype "$dtype" --output "$result"
      done
    done
    torchrun --standalone --nproc-per-node="$p" benchmarks/comm/validate_ulysses_lowp.py \
      --local-sequence 65 --used 1 --batch 2 --heads 8 --dtype "$dtype" --zero-v \
      --output /workspace/sm120-gate-work/p${p}-${dtype}-zero-v.json
  done
done
```

GPU 单测还覆盖固定 stats 分片不变性、平滑 K / outlier / small V、生产 heads 等。SM120 机器上 SM90 专属项被跳过是正常的；目标 SM120 项被跳过不能算通过。遇到断言失败保留失败条目，不先放宽阈值。先完成这一层再跑昂贵视频。

## 7. 视频与性能采集

脚本全部随 SGLang 分支提供，入口目录：

```text
python/sglang/multimodal_gen/benchmarks/ulysses_lowp/
  run.py                   参数化启动、预热、视频与 Nsight 控制
  request.py               HTTP 请求、超时与失败检查
  nvtx/sitecustomize.py    Sage CUDA per-warp 与 NVTX 插桩
  analyze_results.py       全帧解码 / SSIM / pipeline / 前处理分类报告
  timeline.py              Nsight SQLite / NVTX GPU 投影解析
  pre_attention.py         每 GPU 100 次完整前处理窗口
  attention_module.py      固定单模块与 kernel 功能归类
  quality.py               解码 SHA-256 与 SSIM
  test_instrumentation.py  仅装单架构 Sage 扩展时的路由检查
```

```bash
cd /workspace
H3_BENCH=/workspace/sglang-lowp-fi/python/sglang/multimodal_gen/benchmarks/ulysses_lowp
python "$H3_BENCH/test_instrumentation.py"
python "$H3_BENCH/run.py" \
  --model-path /models/MiniMax-H3 --up 8 --port 30041 \
  --output /workspace/results-sm120-up8 \
  --scratch /workspace/work-sm120-up8
```

脚本顺序运行 Lowp、Sage2 CUDA、BF16：每种后端独立视频服务预热后生成三组 50-step 视频，再独立 profile 服务预热后采集 case1 的 3-step trace。视频关闭 NVTX/profiler；profile 开启 CUDA/NVTX 和 layerwise markers。`--mode videos` / `--mode timelines` 可分开运行，使用同一批 output/scratch 配对。默认每 UP 三份 trace，不为三个 prompt 各采三份。

SM120 的 stock Sage 本来就走 CUDA per-warp，但插桩仍显式强制并标记此路线；只导入当前已编译扩展，避免只装 SM120 时强制导入 `_qattn_sm90`。Lowp 使用 SM120 Sage f16 accumulation 入口。不要将 SM90 的 Q16/K128 kernel 名或 V padding 宽度用于判断 SM120。

`run.py` 在短生命周期子进程中预检并记录 GPU、Torch/CUDA、源码 import path、git HEAD/status 到 scratch 的 `environment.json`；子进程退出后再开始采集，协调进程不持有 CUDA 上下文，并拒绝覆盖已有最终媒体。已成功组有 `done.json` 可跳过；失败组不会自动当作完成。重新测量请使用新的 output/scratch，不复用参考视频目录。脚本端口须空闲；不要并行启动多个占满 8 GPU 的批次。

## 8. 分析、结果检查与保留

```bash
H3_BENCH=/workspace/sglang-lowp-fi/python/sglang/multimodal_gen/benchmarks/ulysses_lowp
python "$H3_BENCH/analyze_results.py" \
  --up 8 --output /workspace/results-sm120-up8 \
  --scratch /workspace/work-sm120-up8 \
  --baseline-videos /workspace/reference-sm90/videos
```

未复制参考视频时省略 `--baseline-videos`。新报告首先用 **本机新生成 BF16** 对照本机 Sage2 CUDA / Lowp；指定的旧参考只增加 Lowp 的跨批次 SSIM 与解码比较。SM90/SM120 量化粒度及 attention 累加方式不同，跨架构不要求视频逐字节相同；不能把跨架构画面变化直接归因于回归。

输出为 `results-sm120-up8/REPORT.md`，包括全部 9 个视频链接、三份 trace、全帧解码 SHA-256、SSIM、环境记录、pipeline / denoise、8 GPU 前处理范围，以及固定模块分类和真实 kernel 名。人工观察明确留为待填写，不自动声称视频质量验收通过。用播放器完整查看三组视频，并记录 0.5 / 2 / 4 秒的蜂鸟翅膀、手指与面部、浪花和海鸟情况。

在 Nsight 中核验：每份有 8 GPU；每 GPU 2 次更新 / 100 次 DiT attention；Sage2 有 CUDA `QuantInt8Kernel` 和 per-warp attention；Lowp 有 `lowp_local_stats` → stats AllGather → finalize → quant_pack → input_a2a → unpack → Sage。若出现 fallback、漏 marker、少 rank、少更新或新的未分类 kernel，先解释原因，不能直接填性能表。

SM120 Sage 主 kernel 名为 `qk_int_sv_f8_attn_kernel`，SM90 为 `qk_int8_sv_f8_attn_kernel`，移植分析器已支持两者。功能分类是对当前源版本的映射；未知 kernel 会报错，避免静默记入“其他”导致漏算。必要时人工按 raw kernel 校对。

将算子验收摘要、失败/skip 原因、GPU/互联、代码版本和人工视频观察补充到最终报告。若要评价稳定收益，应使用独立目录重复测量并报告中位数和范围；当前 SM90 单次结果只作参考。完成检查后只保留：

```text
results-sm120-up8/
  videos/       9 个 50-step 视频
  timelines/    3 份 3-step .nsys-rep
  REPORT.md     数字、环境、验收摘要与人工观察
```

SQLite、CSV、请求状态、临时日志和 gate JSON 都位于独立 scratch；在有效证据汇总进报告、失败问题解决后可删除。不要删除尚未分析的失败证据。UP4 如需补测，另用 `--up 4` 和另一组 output/scratch；UP4 对应 TP2，同样使用 8 GPU。

## 9. 本次交付已完成的验证与边界

| 验证 | 结果 |
|---|---|
| FlashInfer 当前两个 comm 测试文件，SM90 | 252 passed / 139 skipped |
| 修改涉及的 Ulysses trace init / consistency | 24 passed / 3 skipped |
| 两个仓库修改文件的 pre-commit | 已通过 |
| 最新 SM90 E2E 视频 | 三组新旧 Lowp 全帧逐字节一致；本机 BF16 / Sage2 CUDA / Lowp 各三组已采集 |
| 最新 SM90 profile | 三份 3-step trace，8 GPU，100 次 attention / GPU |
| 新的可移植工具 | SM120-only 扩展 mock 路由通过；SM90 实际 Sage 调用通过；现有三份 trace 重新解析得到相同耗时与分类 |
| 预检进程隔离 | 8×H20 实测通过：子进程记录全部设备后退出，协调进程未导入 Torch，GPU 进程列表为空 |
| 新工具的完整报表流程 | 复用已发布 SM90 测量与原始媒体验证，未将该检查记为新的性能实验 |
| SM120 实际 GPU | 尚未执行，按本手册分层验收 |

此次不覆盖训练、多节点、SGLang CUDA graph / torch.compile；也未宣称 V scale_max=2.25 最优。GPU 验收与后续架构性能判断由使用者把关。
