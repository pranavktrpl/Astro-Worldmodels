# Spectra embedding-health diagnostics

Two scripts: `embedding_health.py` (is it collapsed / is information lost /
is the eval broken — run first) and `collapse_anatomy.py` (where and why —
run once partial collapse is confirmed).

## collapse_anatomy.py — where does rank/information die?

```bash
python Evals/spectra_diagnostics/collapse_anatomy.py --checkpoint checkpoints/spectra.pt
```

Embeds the scan-protocol subsample (10k spectra by default) and reports
effective rank + ridge redshift validation R² for every representation the
encoder exposes: each transformer layer under CLS and masked-mean pooling,
the final post-norm embedding (both poolings), the 64-d projection-head
output (the space LeJEPA's SIGReg actually regularizes — the probes read
the 768-d pre-projection embedding it never touched), and the final
embedding under a training-style contiguous PAD-masked crop instead of the
full all-real spectrum. Also correlates the probe embedding's top PCs with
redshift and per-spectrum brightness (median flux / flux std).

Writes `results/<label>/anatomy.json`, `layer_anatomy.png`,
`pc_covariates.png`, and prints verdicts for the four candidate stories:
CLS-only collapse (mean pooling rescues), SIGReg-kept-proj-healthy-but-not-
the-backbone, brightness domination from raw un-normalized flux, and
train/eval input mismatch.

## embedding_health.py — first-pass triage

Answers, before touching pretraining, why the spectra backbone probes far
below the AstroCLIP references (redshift R² 0.43 vs 0.98): is the embedding
collapsed, is the encoder destroying information, or is the eval itself
broken?

```bash
python Evals/spectra_diagnostics/embedding_health.py --label checkpoints_spectra
```

Needs the cached `embeddings.npy` from a `spectra_redshift` run for that
label plus the cross-match parquet shards. Writes to `results/<label>/`:

- **`diagnostics.json` + console verdicts** — the summary numbers below plus
  an explicit interpretation.
- **Collapse check**: effective rank (eigenvalue-entropy), RankMe
  (singular-value entropy), participation ratio, dead dimensions, and
  variance concentration in the top PCs, with `eigenspectrum.png`. A healthy
  768-d embedding has effective rank well into the hundreds; tens mean
  (partial) JEPA collapse.
- **Information-floor control**: ridge validation R² on the identical
  seed-42 subsample and carve-out (same protocol as `spectra_probe --scan`)
  for raw flux (7780 features), PCA of raw flux at embedding
  dimensionality, and the embedding itself. If the raw-flux controls beat
  the embedding, the encoder is losing information the probes needed — a
  pretraining problem, not a probe problem.
- **`preprocessing_check.png`**: first spectra with the exact patchified
  tensor the encoder sees overlaid — the two traces must coincide; any
  visible divergence means an eval-side preprocessing mismatch.

Interpretation guide:

| Effective rank | Embedding vs raw-flux ridge | Diagnosis |
|---|---|---|
| low (≲ 0.1 × dim) | any | (partial) collapse — revisit JEPA regularization / EMA / masking |
| healthy | embedding well below controls | encoder discards redshift-bearing information — revisit objective/preprocessing |
| healthy | embedding ≥ controls | information is there — the gap vs AstroCLIP is head capacity or protocol, not the backbone |
