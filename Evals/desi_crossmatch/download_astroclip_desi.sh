#!/bin/bash
# Download AstroCLIP's DESI-LS x DESI EDR cross-matched dataset (~60 GB):
# a single HDF5 with 152x152 grz flux images, spectra, redshifts, and
# targetids for ~150k galaxies, in 10 groups with their 80/20 split.
#
#   bash download_astroclip_desi.sh [target-dir]     # default: ./data
#
# Resumable — just re-run if interrupted. Run inside tmux; this is a long one.
set -euo pipefail

TARGET_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data}"
URL="https://users.flatironinstitute.org/~flanusse/astroclip_desi.1.1.5.h5"
TARGET="$TARGET_DIR/astroclip_desi.1.1.5.h5"

mkdir -p "$TARGET_DIR"
echo "Downloading ~60 GB to $TARGET (curl -C - resume enabled)"
curl -L -C - --retry 10 --retry-delay 15 -o "$TARGET" "$URL"
ls -lh "$TARGET"
