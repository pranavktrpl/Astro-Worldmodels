#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_PYTHON="$(cd "${REPO_ROOT}/.." && pwd)/conda/envs/lejepa-og/bin/python"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [[ -x "${PROJECT_PYTHON}" ]]; then
  PYTHON_BIN="${PROJECT_PYTHON}"
else
  PYTHON_BIN="$(command -v python)"
fi

GPU_LIST="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
WORKER_COUNT="${#GPUS[@]}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results}"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cd "${REPO_ROOT}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/galaxy10_probe_evolution.py" prepare \
  --output-dir "${OUTPUT_DIR}"

pids=()
for worker_index in "${!GPUS[@]}"; do
  gpu="${GPUS[$worker_index]}"
  log="${LOG_DIR}/worker_${worker_index}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    "${SCRIPT_DIR}/galaxy10_probe_evolution.py" worker \
    --output-dir "${OUTPUT_DIR}" \
    --worker-index "${worker_index}" \
    --worker-count "${WORKER_COUNT}" \
    --device cuda \
    >"${log}" 2>&1 &
  pids+=("$!")
  echo "Started worker ${worker_index} on GPU ${gpu}; log=${log}"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/galaxy10_probe_evolution.py" plot \
  --output-dir "${OUTPUT_DIR}"

if [[ "${failed}" -ne 0 ]]; then
  echo "One or more workers failed; inspect ${LOG_DIR} and rerun to resume." >&2
  exit 1
fi
