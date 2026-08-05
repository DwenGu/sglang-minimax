#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SGLANG_REPO_ROOT:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
DEPLOY_ROOT="${MINIMAX_H3_DEPLOY_ROOT:-${REPO_ROOT}/artifacts/minimax_h3_sageattention}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"
ATTENTION_MODE="${ATTENTION_MODE:-baseline}"
RUN_TAG="${RUN_TAG:-${ATTENTION_MODE}-nsys}"
NSYS_CAPTURE_RANGE="${NSYS_CAPTURE_RANGE:-minimax_h3_nsys_request}"
NSYS_OUTPUT="${NSYS_OUTPUT:-${DEPLOY_ROOT}/timelines/nsys/${RUN_TAG}/minimax-h3-${RUN_TAG}}"

if [[ -z "${NSYS_BIN}" || ! -x "${NSYS_BIN}" ]]; then
  echo "Nsight Systems CLI not found; set NSYS_BIN." >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
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
  "${PYTHON_BIN}" "${SCRIPT_DIR}/nsys_capture_driver.py"
