# Spectra embedding-health diagnostics

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
