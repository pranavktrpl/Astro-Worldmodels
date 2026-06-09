#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ENV_PYTHON="$(cd "${REPO_ROOT}/.." && pwd)/conda/envs/lejepa-og/bin/python"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [[ -x "${PROJECT_ENV_PYTHON}" ]]; then
  PYTHON_BIN="${PROJECT_ENV_PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "No Python executable found. Run this from the existing project conda environment." >&2
  exit 1
fi

TORCHRUN_BIN="${TORCHRUN:-$(dirname "${PYTHON_BIN}")/torchrun}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints}"
TRAIN_GPUS="${TRAIN_GPUS:-4}"
TRAIN_BATCHES="${TRAIN_BATCHES:-64}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-64}"
TEST_BATCHES="${TEST_BATCHES:-64}"
SEED="${SEED:-42}"

cd "${REPO_ROOT}"

# Validation and test are intentionally single-process. The evaluator hardcodes one
# DataLoader worker for these splits and rejects distributed launches.
"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_loss_curves.py" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --splits validation \
  --max-batches "${VALIDATION_BATCHES}" \
  --seed "${SEED}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_loss_curves.py" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --splits test \
  --max-batches "${TEST_BATCHES}" \
  --seed "${SEED}"

# Train evaluation preserves each checkpoint's saved batch size and worker count.
# Four GPUs matches the current best training launch; override TRAIN_GPUS if needed.
"${TORCHRUN_BIN}" --standalone --nproc_per_node="${TRAIN_GPUS}" \
  "${SCRIPT_DIR}/evaluate_loss_curves.py" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --splits train \
  --max-batches "${TRAIN_BATCHES}" \
  --seed "${SEED}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_loss_curves.py" \
  --checkpoint-root "${CHECKPOINT_ROOT}"
