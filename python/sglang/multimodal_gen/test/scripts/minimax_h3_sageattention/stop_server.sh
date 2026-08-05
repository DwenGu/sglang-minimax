#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SGLANG_REPO_ROOT:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
DEPLOY_ROOT="${MINIMAX_H3_DEPLOY_ROOT:-${REPO_ROOT}/artifacts/minimax_h3_sageattention}"
PID_FILE="${DEPLOY_ROOT}/run/server.pid"
MODE_FILE="${DEPLOY_ROOT}/run/server.mode"

if [[ ! -s "${PID_FILE}" ]]; then
  echo "No server PID file found."
  exit 0
fi

server_pid="$(<"${PID_FILE}")"
if [[ ! "${server_pid}" =~ ^[0-9]+$ ]] || ! kill -0 "${server_pid}" 2>/dev/null; then
  echo "Stale PID file; no live server found."
  : >"${PID_FILE}"
  : >"${MODE_FILE}"
  exit 0
fi

server_cmd="$(ps -p "${server_pid}" -o args=)"
if [[ "${server_cmd}" != *"sglang"* ]]; then
  echo "Refusing to stop PID ${server_pid}; command does not look like this deployment: ${server_cmd}" >&2
  exit 1
fi

server_sid="$(ps -p "${server_pid}" -o sid= | tr -d ' ')"
if [[ "${server_sid}" == "${server_pid}" ]]; then
  kill -- "-${server_pid}"
else
  kill "${server_pid}"
fi

for _ in $(seq 1 30); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    : >"${PID_FILE}"
    : >"${MODE_FILE}"
    echo "MiniMax-H3 server stopped."
    exit 0
  fi
  sleep 1
done

echo "Server is still shutting down (PID ${server_pid}); no SIGKILL was sent." >&2
exit 1
