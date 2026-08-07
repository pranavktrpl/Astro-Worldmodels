# Spectra-backbone redshift probe

First-ever evaluation of the spectra backbone
(`SPECTRA_run4_bs16_2806_Epoch5_utbd_desi`): frozen
`SpectrumTransformerEncoder` embeddings of the AstroCLIP cross-match spectra,
probed with the same ridge / zero-shot kNN / MLP heads and the same 80/20
split as the image probes — image and spectra modalities measured on
identical data with identical protocols.

**Reference target: AstroCLIP's spectrum encoder reaches test R² = 0.98.**
Redshift is nearly fully determined by a spectrum, so anything far below that
indicates the backbone (or its pretraining) needs work before it can serve as
an alignment partner for the image encoder.

## Usage — two steps, selection before test

```bash
# 1. rank all checkpoints in the family by ridge VALIDATION R2
#    (20k-spectra subsample; test split never touched)
python Evals/spectra_redshift/spectra_probe.py --scan

# 2. full battery (3 seeds, all heads, test split) on the winner, once
python Evals/spectra_redshift/spectra_probe.py \
  --checkpoint checkpoints/SPECTRA_run4_bs16_2806_Epoch5_utbd_desi/step_<best>.pt
```

`--scan` defaults to the SPECTRA_run4 family dir; pass another directory to
scan a different family. The scan writes `results/scan.json`; the full run
writes `results/<label>/metrics.json` (+ a cached `embeddings.npy`, ignored
by git).

## Protocol notes

- Inference preprocessing matches training exactly: raw flux (no
  normalization — none is applied in training either), drop the final value,
  389 ordered patches of 20, all-real mask, CLS pooling. The encoder is
  rebuilt from the checkpoint's saved cfg.
- Checkpoint selection uses ridge validation R² on a train-split subsample —
  the test split is evaluated once, on the selected checkpoint only.
- The spectra come from the same parquet shards as the image cross-match
  eval (`Evals/desi_crossmatch/data/astroclip`) — no new downloads.

## What comes after this

If the selected checkpoint probes well (R² ≳ 0.9), it becomes the alignment
partner for the CLIP-style image↔spectrum stage on the 139k cross-match
train pairs — the mechanism by which AstroCLIP's image encoder inherits
spectral information, and our best candidate for closing the remaining
image-side gap.
