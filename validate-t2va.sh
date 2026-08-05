#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="${MINIMAX_H3_DEPLOY_ROOT:-${SCRIPT_DIR}}"
VENV="${MINIMAX_H3_VENV:-/workspace/.venvs/sglang-h3}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30010}"
ATTENTION_MODE="${ATTENTION_MODE:-baseline}"
RUN_TAG="${RUN_TAG:-${ATTENTION_MODE}}"
PROFILE_TIMELINE="${PROFILE_TIMELINE:-0}"
NUM_PROFILED_TIMESTEPS="${NUM_PROFILED_TIMESTEPS:-3}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
OUTPUT_FILE="${OUTPUT_FILE:-${DEPLOY_ROOT}/outputs/minimax-h3-${RUN_TAG}.mp4}"
PAYLOAD_FILE="${DEPLOY_ROOT}/run/t2va-payload-${RUN_TAG}.json"
REQUEST_FILE="${DEPLOY_ROOT}/run/t2va-request-${RUN_TAG}.json"
STATUS_FILE="${DEPLOY_ROOT}/run/t2va-status-${RUN_TAG}.json"

export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,0.0.0.0"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,0.0.0.0"

mkdir -p "${DEPLOY_ROOT}/outputs" "${DEPLOY_ROOT}/run"

profile_json=false
if [[ "${PROFILE_TIMELINE}" == "1" ]]; then
  profile_json=true
fi
jq -n \
  --argjson profile "${profile_json}" \
  --argjson profiled_steps "${NUM_PROFILED_TIMESTEPS}" \
  --argjson inference_steps "${NUM_INFERENCE_STEPS}" \
  '{
    model: "MiniMaxAI/MiniMax-H3",
    prompt: "At night, a small orange cat wearing round aviator goggles pilots a red biplane above a glowing coastal city. The camera follows smoothly from behind; wind and engine sounds match the scene.",
    seconds: 5,
    task: "t2va",
    conditions: [],
    target: {
      short_edge: 768,
      aspect_ratio: "16:9",
      duration_seconds: 5.0
    },
    num_outputs_per_prompt: 1,
    num_inference_steps: $inference_steps,
    flow_shift: 12.0,
    audio_flow_shift: 3.0,
    seed: 1101
  } + if $profile then {
    profile: true,
    num_profiled_timesteps: $profiled_steps,
    profile_all_stages: false
  } else {} end' >"${PAYLOAD_FILE}"

for _ in $(seq 1 180); do
  if curl -fsS "${BASE_URL}/health" >/dev/null; then
    break
  fi
  sleep 5
done
curl -fsS "${BASE_URL}/health" >/dev/null || {
  echo "Server did not become healthy at ${BASE_URL} within 15 minutes." >&2
  exit 1
}

http_code="$(curl -sS -o "${REQUEST_FILE}" -w '%{http_code}' \
  -X POST "${BASE_URL}/v1/videos" \
  -H 'Content-Type: application/json' \
  --data-binary "@${PAYLOAD_FILE}")"

if [[ "${http_code}" != "200" ]]; then
  echo "Video submission failed with HTTP ${http_code}:" >&2
  jq . "${REQUEST_FILE}" >&2 || sed -n '1,100p' "${REQUEST_FILE}" >&2
  exit 1
fi

video_id="$(jq -er '.id' "${REQUEST_FILE}")"
echo "Submitted video job: ${video_id}"

deadline=$((SECONDS + 1800))
while (( SECONDS < deadline )); do
  curl -fsS "${BASE_URL}/v1/videos/${video_id}" -o "${STATUS_FILE}"
  status="$(jq -r '.status' "${STATUS_FILE}")"
  echo "$(date -u +%FT%TZ) status=${status}"
  case "${status}" in
    completed)
      break
      ;;
    failed|cancelled|deleted)
      jq . "${STATUS_FILE}" >&2
      exit 1
      ;;
  esac
  sleep 5
done

if [[ "$(jq -r '.status' "${STATUS_FILE}")" != "completed" ]]; then
  echo "Video job timed out after 30 minutes." >&2
  jq . "${STATUS_FILE}" >&2
  exit 1
fi

curl -fL "${BASE_URL}/v1/videos/${video_id}/content" -o "${OUTPUT_FILE}"
"${VENV}/bin/python" "${DEPLOY_ROOT}/inspect-video.py" "${OUTPUT_FILE}"
