#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="${MINIMAX_H3_DEPLOY_ROOT:-${SCRIPT_DIR}}"
VENV="${MINIMAX_H3_VENV:-/workspace/.venvs/sglang-h3}"
NSYS_BIN="${NSYS_BIN:-/usr/local/cuda/bin/nsys}"
ATTENTION_MODE="${ATTENTION_MODE:-baseline}"
RUN_TAG="${RUN_TAG:-${ATTENTION_MODE}-nsys}"
NSYS_CAPTURE_RANGE="${NSYS_CAPTURE_RANGE:-minimax_h3_nsys_request}"
NSYS_OUTPUT="${NSYS_OUTPUT:-${DEPLOY_ROOT}/timelines/nsys/${RUN_TAG}/minimax-h3-${RUN_TAG}}"

if [[ ! -x "${NSYS_BIN}" ]]; then
  echo "Nsight Systems CLI not found: ${NSYS_BIN}" >&2
  exit 1
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Python environment not found: ${VENV}" >&2
  exit 1
fi

mkdir -p "$(dirname "${NSYS_OUTPUT}")" "${DEPLOY_ROOT}/logs" "${DEPLOY_ROOT}/outputs"

exec "${NSYS_BIN}" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace-fork-before-exec=true \
  --wait=all \
  --force-overwrite=true \
  --stats=false \
  --output="${NSYS_OUTPUT}" \
  "${VENV}/bin/python" "${DEPLOY_ROOT}/nsys-capture-driver.py"
