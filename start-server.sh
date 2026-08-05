#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="${MINIMAX_H3_DEPLOY_ROOT:-${SCRIPT_DIR}}"
VENV="${MINIMAX_H3_VENV:-/workspace/.venvs/sglang-h3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30010}"
NUM_GPUS="${NUM_GPUS:-8}"
TP_SIZE="${TP_SIZE:-1}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-8}"
ATTENTION_MODE="${ATTENTION_MODE:-baseline}"
PROFILER_DIR="${SGLANG_DIFFUSION_TORCH_PROFILER_DIR:-${DEPLOY_ROOT}/timelines/${ATTENTION_MODE}}"

attention_args=()
case "${ATTENTION_MODE}" in
  baseline)
    unset SGLANG_SAGEATTENTION_VARIANT
    unset SGLANG_SAGEATTENTION_SM90_CUDA
    unset SGLANG_SAGEATTENTION_SM120_CUDA
    ;;
  sageattention1)
    export PYTHONPATH="${DEPLOY_ROOT}/sageattention/1.0.6${PYTHONPATH:+:${PYTHONPATH}}"
    export SGLANG_SAGEATTENTION_VARIANT=1
    unset SGLANG_SAGEATTENTION_SM90_CUDA
    unset SGLANG_SAGEATTENTION_SM120_CUDA
    attention_args=(--attention-backend sage_attn)
    ;;
  sageattention2)
    export PYTHONPATH="${DEPLOY_ROOT}/sageattention/2.2.0${PYTHONPATH:+:${PYTHONPATH}}"
    export SGLANG_SAGEATTENTION_VARIANT=2
    # This mode is the H3 Triton path even if the parent shell previously
    # exported the CUDA opt-in. Use sageattention2-cuda explicitly for SM90.
    export SGLANG_SAGEATTENTION_SM90_CUDA=0
    export SGLANG_SAGEATTENTION_SM120_CUDA=0
    # Enable the opt-in H20/MiniMax-H3 fused varlen path. Set this to 0 to
    # retain the upstream SageAttention2 implementation for A/B validation.
    export SGLANG_SAGEATTENTION_FUSED_VARLEN="${SGLANG_SAGEATTENTION_FUSED_VARLEN:-1}"
    attention_args=(--attention-backend sage_attn)
    ;;
  sageattention2-cuda)
    export PYTHONPATH="${DEPLOY_ROOT}/sageattention/2.2.0${PYTHONPATH:+:${PYTHONPATH}}"
    export SGLANG_SAGEATTENTION_VARIANT=2
    export SGLANG_SAGEATTENTION_SM90_CUDA=1
    export SGLANG_SAGEATTENTION_SM120_CUDA=0
    # Keep the fused Triton path enabled as a safe fallback for any input that
    # does not satisfy the H3 SM90 CUDA adapter contract.
    export SGLANG_SAGEATTENTION_FUSED_VARLEN=1
    attention_args=(--attention-backend sage_attn)
    ;;
  sageattention2-sm120)
    export PYTHONPATH="${DEPLOY_ROOT}/sageattention/2.2.0${PYTHONPATH:+:${PYTHONPATH}}"
    export SGLANG_SAGEATTENTION_VARIANT=2
    export SGLANG_SAGEATTENTION_SM90_CUDA=0
    export SGLANG_SAGEATTENTION_SM120_CUDA=1
    # Both H3 fused and upstream varlen use Triton. Do not use either as an
    # implicit SM120 fallback; the adapter fails closed on contract mismatch.
    export SGLANG_SAGEATTENTION_FUSED_VARLEN=0
    attention_args=(--attention-backend sage_attn)
    ;;
  *)
    echo "ATTENTION_MODE must be baseline, sageattention1, sageattention2, sageattention2-cuda, or sageattention2-sm120; got ${ATTENTION_MODE}" >&2
    exit 1
    ;;
esac

if [[ ! -x "${VENV}/bin/sglang" ]]; then
  echo "SGLang executable not found: ${VENV}/bin/sglang" >&2
  exit 1
fi

for media_tool in ffmpeg ffprobe; do
  if ! command -v "${media_tool}" >/dev/null 2>&1; then
    echo "Required media tool is missing: ${media_tool}" >&2
    exit 1
  fi
done

mkdir -p \
  "${DEPLOY_ROOT}/hf-cache" \
  "${DEPLOY_ROOT}/logs" \
  "${DEPLOY_ROOT}/run" \
  "${DEPLOY_ROOT}/server-outputs" \
  "${PROFILER_DIR}" \
  "${DEPLOY_ROOT}/tmp"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HOME="${HF_HOME:-${DEPLOY_ROOT}/hf-cache}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-${DEPLOY_ROOT}/tmp}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export SGLANG_DIFFUSION_TORCH_PROFILER_DIR="${PROFILER_DIR}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,0.0.0.0"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,0.0.0.0"

echo "MiniMax-H3 attention mode: ${ATTENTION_MODE}"

exec "${VENV}/bin/sglang" serve \
  --model-path MiniMaxAI/MiniMax-H3 \
  --model-variant fl2va \
  --num-gpus "${NUM_GPUS}" \
  --tp-size "${TP_SIZE}" \
  --ulysses-degree "${ULYSSES_DEGREE}" \
  --performance-mode speed \
  --enable-torch-compile false \
  --warmup-resolutions 1344x768 \
  --output-path "${DEPLOY_ROOT}/server-outputs" \
  --host "${HOST}" \
  --port "${PORT}" \
  "${attention_args[@]}" \
  "$@"
