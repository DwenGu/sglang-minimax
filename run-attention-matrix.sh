#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="${MINIMAX_H3_DEPLOY_ROOT:-${SCRIPT_DIR}}"

cleanup() {
  "${DEPLOY_ROOT}/stop-server.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for mode in baseline sageattention1 sageattention2; do
  echo "Running ${mode} correctness test"
  ATTENTION_MODE="${mode}" "${DEPLOY_ROOT}/launch-server.sh"
  ATTENTION_MODE="${mode}" \
    OUTPUT_FILE="${DEPLOY_ROOT}/outputs/minimax-h3-${mode}.mp4" \
    "${DEPLOY_ROOT}/validate-t2va.sh"

  echo "Capturing ${mode} timeline"
  ATTENTION_MODE="${mode}" \
    PROFILE_TIMELINE=1 \
    NUM_PROFILED_TIMESTEPS="${NUM_PROFILED_TIMESTEPS:-3}" \
    OUTPUT_FILE="${DEPLOY_ROOT}/outputs/minimax-h3-${mode}-profiled.mp4" \
    "${DEPLOY_ROOT}/validate-t2va.sh"
  "${DEPLOY_ROOT}/stop-server.sh"
done

trap - EXIT
