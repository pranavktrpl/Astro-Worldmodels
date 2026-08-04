#!/bin/bash
# Download the AstroCLIP DESI-LS x DESI EDR cross-matched dataset (~65 GB of
# parquet shards: image (152,152,3) grz fluxes, spectrum (7781,1), redshift,
# targetid; train + test splits preserved from AstroCLIP's 80/20).
#
# Source: mhsotoudeh/astroclip on Hugging Face — a complete parquet conversion
# of the original astroclip_desi.1.1.5.h5. The original Flatiron URL now
# returns 403 and the author's own HF upload (EiffL/AstroCLIP) is truncated
# (128/138 train shards, no test split), so this mirror is the usable source.
#
#   bash download_astroclip_desi.sh [target-dir]   # default: ./data/astroclip
#
# Resumable — hf skips completed shards on re-run. Run inside tmux.
set -euo pipefail

TARGET_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data/astroclip}"

command -v hf >/dev/null 2>&1 || { echo "hf CLI missing: pip install -U huggingface_hub"; exit 1; }

echo "Downloading ~65 GB to $TARGET_DIR (resumable)"
hf download mhsotoudeh/astroclip --repo-type dataset --local-dir "$TARGET_DIR"

TRAIN=$(ls "$TARGET_DIR"/data/train-*.parquet 2>/dev/null | wc -l)
TEST=$(ls "$TARGET_DIR"/data/test-*.parquet 2>/dev/null | wc -l)
echo "shards present: $TRAIN/120 train, $TEST/26 test"
[ "$TRAIN" -eq 120 ] && [ "$TEST" -eq 26 ] && echo "COMPLETE" || { echo "INCOMPLETE — re-run to resume"; exit 1; }
