#!/bin/bash
# Train fixed-head GZD-5 MLPs (CE on logits + inverse-frequency class weights)
# for every label that already has extracted embeddings. Minutes per label —
# reuses the cached embeddings, no GPU image pass.
#
#   bash Evals/gzd5_morphology_probe/run_fixedhead_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for dir in "$SCRIPT_DIR"/results/*/; do
    label="$(basename "$dir")"
    if [ ! -f "$dir/train_embeddings.npy" ]; then
        echo "skip $label (no embeddings)"
        continue
    fi
    echo "=== $label (fixed head, 5 seeds) ==="
    python "$SCRIPT_DIR/train_gzd5_mlp.py" --label "$label" --fixed-head \
        --seeds 42 43 44 45 46
done
