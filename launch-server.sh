#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="${MINIMAX_H3_DEPLOY_ROOT:-${SCRIPT_DIR}}"
PID_FILE="${DEPLOY_ROOT}/run/server.pid"
ATTENTION_MODE="${ATTENTION_MODE:-baseline}"
RUN_TAG="${RUN_TAG:-${ATTENTION_MODE}}"
LOG_FILE="${DEPLOY_ROOT}/logs/server-${RUN_TAG}.log"
MODE_FILE="${DEPLOY_ROOT}/run/server.mode"

mkdir -p "${DEPLOY_ROOT}/run" "${DEPLOY_ROOT}/logs"

if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(<"${PID_FILE}")"
  if [[ "${old_pid}" =~ ^[0-9]+$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "MiniMax-H3 server is already running (PID ${old_pid})."
    exit 0
  fi
fi

nohup setsid "${DEPLOY_ROOT}/start-server.sh" >"${LOG_FILE}" 2>&1 </dev/null &
server_pid=$!
printf '%s\n' "${server_pid}" >"${PID_FILE}"
printf '%s\n' "${ATTENTION_MODE}" >"${MODE_FILE}"
sleep 2

if ! kill -0 "${server_pid}" 2>/dev/null; then
  echo "Server exited during startup; inspect ${LOG_FILE}" >&2
  tail -100 "${LOG_FILE}" >&2 || true
  exit 1
fi

echo "MiniMax-H3 ${ATTENTION_MODE} startup initiated (PID ${server_pid}, run=${RUN_TAG})."
echo "Log: ${LOG_FILE}"
