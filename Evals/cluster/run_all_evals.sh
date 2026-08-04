#!/bin/bash
# Run the matched-heads probe, the redshift probe, and the competitor
# comparison, in that order (the first run extracts the shared embedding
# cache; the later steps reuse it). Configuration comes from evals.env —
# see evals.env.example. Usually launched via tmux_launch.sh.
#
#   bash Evals/cluster/run_all_evals.sh [path/to/evals.env]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${1:-$SCRIPT_DIR/evals.env}"
[ -f "$ENV_FILE" ] || { echo "No env file at $ENV_FILE — cp evals.env.example evals.env and edit it."; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"

# --- interpreter -----------------------------------------------------------
if [ -z "${PYTHON:-}" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
    PYTHON="$(command -v python)"
fi
echo "python: $PYTHON"
"$PYTHON" - <<'EOF'
import h5py, timm, torch  # noqa: F401
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
EOF

# --- preflight -------------------------------------------------------------
[ -f "$GALAXY10_H5" ] || { echo "Missing Galaxy10 HDF5: $GALAXY10_H5"; exit 1; }

CKPT_LARGE="$REPO_DIR/checkpoints/VitLargePatch14_OfficialTrain5_Epoch5_2504/step_52000.pt"
CKPT_SMALL="$REPO_DIR/checkpoints/VitSmallPatch14_2204/step_21000.pt"
require() { [ -f "$1" ] || { echo "Missing checkpoint: $1 (or set MODELS in $ENV_FILE)"; exit 1; }; }
case "$MODELS" in
    all)   require "$CKPT_LARGE"; require "$CKPT_SMALL" ;;
    large) require "$CKPT_LARGE" ;;
    small) require "$CKPT_SMALL" ;;
    *) echo "MODELS must be all, large, or small (got: $MODELS)"; exit 1 ;;
esac

H5_LINK="$REPO_DIR/Evals/DeCals_linearProbing/galaxy10/Galaxy10_DECals.h5"
mkdir -p "$(dirname "$H5_LINK")"
[ -e "$H5_LINK" ] || ln -s "$GALAXY10_H5" "$H5_LINK"

mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
cd "$REPO_DIR"

# --- evals -----------------------------------------------------------------
run() {
    local name="$1"; shift
    echo; echo "=== $name ==="
    "$PYTHON" "$@" 2>&1 | tee "$LOG_DIR/${name}_${STAMP}.log"
}

run matched_heads Evals/galaxy10_matched_heads/matched_heads_probe.py \
    --model "$MODELS" --device "$DEVICE"
run redshift Evals/redshift_regression/redshift_probe.py \
    --model "$MODELS" --device "$DEVICE"
run compare Evals/competitor_comparison/compare_competitors.py

echo
echo "All done. Results:"
echo "  Evals/galaxy10_matched_heads/results/"
echo "  Evals/redshift_regression/results/"
echo "  Evals/competitor_comparison/results/  (tables + charts refreshed)"
