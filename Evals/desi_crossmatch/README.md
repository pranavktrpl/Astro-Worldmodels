# DESI-LS × DESI spectra cross-match eval

Apples-to-apples redshift evaluation: our frozen backbones probed on the
**exact sample and split** behind AstroCLIP's published image-encoder numbers
(zero-shot kNN R²=0.79, few-shot MLP R²=0.78). This removes the last
measurement confound — the Galaxy10 probe's R² is computed on a much narrower
redshift distribution (median z≈0.07), so its denominator is not comparable
to AstroCLIP's.

## The cross-match already exists — don't rebuild it

AstroCLIP built their cross-matched dataset on Stein et al. 2022's Legacy
Survey cutouts × DESI EDR spectra. Sourcing it in 2026 is a small saga:

- The original `astroclip_desi.1.1.5.h5` URL (Flatiron, referenced by their
  loader script) now returns **403 Forbidden**.
- The author's own HF upload (`EiffL/AstroCLIP`) is **truncated**: 128/138
  train shards and no test split at all.
- **`mhsotoudeh/astroclip`** on HF is a complete parquet conversion — same
  schema (`image` (152,152,3) float32 grz fluxes, `spectrum` (7781,1),
  `redshift`, `targetid`), all 120 train + 26 test shards (~139k/~30k rows),
  with AstroCLIP's train/test split preserved as the dataset splits. This is
  what the download script fetches. Caveat: it's a third-party mirror; if a
  result ever hinges on it, spot-check targetid overlap against the
  `EiffL/AstroCLIP` train shards.

```bash
bash download_astroclip_desi.sh            # ~65 GB, resumable, run in tmux
```

The from-scratch alternative (only needed for DR10 or a custom footprint):
MultimodalUniverse's cross-match utilities over `MultimodalUniverse/legacysurvey`
+ the DESI spectra set — their docs reproduce AstroCLIP's cross-match in a few
lines, but the image download is TB-scale.

## Run

```bash
python astroclip_redshift_probe.py --model all
```

Reuses the redshift_regression heads (ridge / zero-shot kNN / MLP) and metric
definitions. The train/test split is fixed (theirs), so seeds (42/43/44) vary
only the 10% validation carve-out used for hyperparameter selection and the
MLP initialization. Results land in `results/<model>/metrics.json`;
`compare_competitors.py` picks them up as "(AstroCLIP sample)" rows in the
redshift table.

## Preprocessing choice (the one judgment call)

The HDF5 stores raw grz fluxes; our backbones were trained on RGB-like images
in [0,1]. We map fluxes → RGB with the Legacy Survey **dr2-style arcsinh
mapping** (legacypipe's `dr2_rgb`, scales g:6.0/r:3.4/z:2.2, m=0.03) — the
same mapping Stein et al. and AstroCLIP use to feed their own image model —
then bicubic-resize 152→140. This matches both AstroCLIP's pipeline and our
backbones' training domain. The constants are recorded in
`embedding_metadata.json`; if results look pathological, verify them against
`astroclip/astroclip/data/augmentations.py` (`dr2_rgb`) before suspecting the
backbone.

## Caveats that remain even on this sample

- Our backbones were pretrained on a different image distribution
  (Smith42/galaxies JPEG-like cutouts) than this flux-derived RGB — a domain
  gap AstroCLIP's encoder does not have on its own eval set.
- AstroCLIP's kNN/MLP hyperparameters are not identical to our grids; ours
  are selected on a validation carve-out and recorded per run.
- The `spectra` in the same file are the natural input for the future
  spectra-backbone redshift probe (AstroCLIP spectrum encoder: R²=0.98) —
  same sample, same split, zero extra downloads.
