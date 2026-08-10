#!/bin/bash
# Download the PROVABGS DESI EDR catalog (~1.4 GB of HATS-partitioned parquet:
# object_id (= DESI targetid), LOG_MSTAR, Z_MW, TAGE_MW, AVG_SFR, ...).
#
# Source: UniverseTBD/mmu_desi_provabgs on Hugging Face (Multimodal Universe
# conversion of the PROVABGS BGS EDR posterior catalog). Joined to the
# AstroCLIP cross-match sample by targetid in spectra_properties_probe.py.
#
#   bash download_provabgs.sh [target-dir]   # default: ./data/provabgs
#
# Resumable — hf skips completed shards on re-run.
set -euo pipefail

TARGET_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data/provabgs}"

command -v hf >/dev/null 2>&1 || { echo "hf CLI missing: pip install -U huggingface_hub"; exit 1; }

echo "Downloading ~1.4 GB to $TARGET_DIR (resumable)"
hf download UniverseTBD/mmu_desi_provabgs --repo-type dataset --local-dir "$TARGET_DIR"

SHARDS=$(find "$TARGET_DIR" -name 'Npix=*.parquet' | wc -l)
echo "catalog shards present: $SHARDS/148"
[ "$SHARDS" -eq 148 ] && echo "COMPLETE" || { echo "INCOMPLETE — re-run to resume"; exit 1; }
