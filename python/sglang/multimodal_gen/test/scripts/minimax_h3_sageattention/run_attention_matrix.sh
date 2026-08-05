#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SGLANG_REPO_ROOT:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
DEPLOY_ROOT="${MINIMAX_H3_DEPLOY_ROOT:-${REPO_ROOT}/artifacts/minimax_h3_sageattention}"

cleanup() {
  "${SCRIPT_DIR}/stop_server.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for mode in baseline sageattention1 sageattention2; do
  echo "Running ${mode} correctness test"
  ATTENTION_MODE="${mode}" "${SCRIPT_DIR}/launch_server.sh"
  ATTENTION_MODE="${mode}" \
    OUTPUT_FILE="${DEPLOY_ROOT}/outputs/minimax-h3-${mode}.mp4" \
    "${SCRIPT_DIR}/validate_t2va.sh"

  echo "Capturing ${mode} timeline"
  ATTENTION_MODE="${mode}" \
    PROFILE_TIMELINE=1 \
    NUM_PROFILED_TIMESTEPS="${NUM_PROFILED_TIMESTEPS:-3}" \
    OUTPUT_FILE="${DEPLOY_ROOT}/outputs/minimax-h3-${mode}-profiled.mp4" \
    "${SCRIPT_DIR}/validate_t2va.sh"
  "${SCRIPT_DIR}/stop_server.sh"
done

trap - EXIT
