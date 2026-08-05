#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV="${MINIMAX_H3_VENV:-/workspace/.venvs/sglang-h3}"
if [[ -x "${DEFAULT_VENV}/bin/python" ]]; then
  default_python="${DEFAULT_VENV}/bin/python"
else
  default_python="python3"
fi
PYTHON_BIN="${PYTHON_BIN:-${default_python}}"
DEVICE="${DEVICE:-0}"
SAGEATTN_REPOSITORY="${SAGEATTN_REPOSITORY:-https://github.com/thu-ml/SageAttention.git}"
SAGEATTN_REF="${SAGEATTN_REF:-v2.2.0}"
SAGEATTN_SOURCE="${SAGEATTN_SOURCE:-${SCRIPT_DIR}/sageattention/2.2.0}"
MAX_JOBS="${MAX_JOBS:-8}"

usage() {
  command_name="$(basename -- "$0")"
  echo "Build SageAttention2 for SM120 and run the MiniMax-H3 exact-shape check."
  echo
  echo "Usage: ${command_name} [arguments forwarded to validate-sage2-sm120-h3.py]"
  echo
  echo "Environment overrides:"
  echo "  PYTHON_BIN          Python from the target SGLang environment"
  echo "  DEVICE              SM120 CUDA device index (default: 0)"
  echo "  SAGEATTN_SOURCE     clone/source directory"
  echo "  SAGEATTN_REPOSITORY upstream repository URL"
  echo "  SAGEATTN_REF        pinned ref (default: v2.2.0)"
  echo "  MAX_JOBS            extension build parallelism (default: 8)"
  echo
  echo "Example:"
  echo "  PYTHON_BIN=/workspace/.venvs/sglang-h3/bin/python ${command_name} --repeats 10"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  echo
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate-sage2-sm120-h3.py" --help
  exit 0
fi

for required_command in git "${PYTHON_BIN}"; do
  if ! command -v "${required_command}" >/dev/null 2>&1; then
    echo "Required command not found: ${required_command}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" - "${DEVICE}" <<'PY'
import re
import sys

import torch

device = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the selected Python environment")
torch.cuda.set_device(device)
capability = torch.cuda.get_device_capability(device)
if capability != (12, 0):
    raise SystemExit(f"SM120 is required, got compute capability {capability}")
match = re.match(r"^(\d+)\.(\d+)", torch.version.cuda or "")
if match is None or tuple(map(int, match.groups())) < (12, 8):
    raise SystemExit(f"CUDA >= 12.8 is required, got {torch.version.cuda!r}")
print(f"GPU preflight: {torch.cuda.get_device_name(device)}, capability={capability}")
print(f"PyTorch/CUDA: {torch.__version__} / {torch.version.cuda}")
print(f"Torch arch list: {torch.cuda.get_arch_list()}")
PY

if [[ ! -d "${SAGEATTN_SOURCE}/.git" ]]; then
  if [[ -e "${SAGEATTN_SOURCE}" ]]; then
    echo "SAGEATTN_SOURCE exists but is not a git checkout: ${SAGEATTN_SOURCE}" >&2
    exit 1
  fi
  mkdir -p "$(dirname -- "${SAGEATTN_SOURCE}")"
  git clone --depth 1 --branch "${SAGEATTN_REF}" \
    "${SAGEATTN_REPOSITORY}" "${SAGEATTN_SOURCE}"
else
  echo "Using existing SageAttention checkout: ${SAGEATTN_SOURCE}"
fi

if [[ ! -f "${SAGEATTN_SOURCE}/setup.py" ]]; then
  echo "SageAttention setup.py not found: ${SAGEATTN_SOURCE}" >&2
  exit 1
fi
if ! grep -q 'HAS_SM120' "${SAGEATTN_SOURCE}/setup.py"; then
  echo "Selected SageAttention source does not contain SM120 build support." >&2
  exit 1
fi
if ! grep -q 'arch == "sm120"' "${SAGEATTN_SOURCE}/sageattention/core.py"; then
  echo "Selected SageAttention source does not contain the SM120 dispatcher." >&2
  exit 1
fi

cuda_home="${CUDA_HOME:-}"
if [[ -z "${cuda_home}" ]]; then
  nvcc_path="$(command -v nvcc || true)"
  if [[ -z "${nvcc_path}" ]]; then
    echo "nvcc is required; set CUDA_HOME to a CUDA >= 12.8 toolkit." >&2
    exit 1
  fi
  cuda_home="$(cd -- "$(dirname -- "${nvcc_path}")/.." && pwd)"
fi
if [[ ! -x "${cuda_home}/bin/nvcc" ]]; then
  echo "nvcc not found under CUDA_HOME=${cuda_home}" >&2
  exit 1
fi

echo "Building SageAttention2 for sm_120 from ${SAGEATTN_SOURCE}"
CUDA_HOME="${cuda_home}" \
TORCH_CUDA_ARCH_LIST="12.0" \
MAX_JOBS="${MAX_JOBS}" \
  "${PYTHON_BIN}" -m pip install \
    --no-build-isolation \
    --no-cache-dir \
    --no-deps \
    --force-reinstall \
    --editable "${SAGEATTN_SOURCE}"

mapfile -t extension_paths < <(
  PYTHONPATH="${SAGEATTN_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -c \
      'import sageattention._fused as f; import sageattention._qattn_sm89 as q; print(f.__file__); print(q.__file__)'
)
cuobjdump_path="${cuda_home}/bin/cuobjdump"
if [[ ! -x "${cuobjdump_path}" ]]; then
  echo "cuobjdump not found: ${cuobjdump_path}" >&2
  exit 1
fi

for extension_path in "${extension_paths[@]}"; do
  echo "Inspecting CUDA extension: ${extension_path}"
  cubin_listing="$("${cuobjdump_path}" --list-elf "${extension_path}")"
  echo "${cubin_listing}"
  if ! grep -q 'sm_120.cubin' <<<"${cubin_listing}"; then
    echo "The SageAttention extension does not contain an sm_120 cubin." >&2
    exit 1
  fi
done

PYTHONPATH="${SAGEATTN_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/validate-sage2-sm120-h3.py" \
    --device "${DEVICE}" \
    --sageattention-root "${SAGEATTN_SOURCE}" \
    "$@"

echo
echo "SM120 preflight passed. Start MiniMax-H3 with:"
echo "  ATTENTION_MODE=sageattention2-sm120 NUM_GPUS=8 ULYSSES_DEGREE=8 ./launch-server.sh"
