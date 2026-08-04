# Competitor comparison

Merges this repo's eval results with the published AION-1 and AstroCLIP
downstream numbers and produces comparison tables and charts.

## Usage

```bash
python Evals/competitor_comparison/compare_competitors.py
```

No GPU, no datasets — it only reads result JSONs already produced by the other
eval suites:

| Task | Local source | Published baseline |
|---|---|---|
| Galaxy10 / GZ-10 morphology | `galaxy10_checkpoint_evolution/results/best_checkpoints.json` + `galaxy10_matched_heads/results/*/metrics.json` | AION-1-B/L/XL test accuracy (2-layer MLP head) |
| GZD-5 question-wise morphology | `gzd5_morphology_probe/results/<model>/metrics.json` | AstroCLIP per-question accuracy/F1 (4-layer MLP head) |
| Redshift regression | `redshift_regression/results/<model>/metrics.json` | AstroCLIP image kNN/MLP R², AION-1 photometry R² |
| Physical properties | none yet (needs PROVABGS) | AION-1 and AstroCLIP R² reference targets |

Missing local results are reported as *pending*, not errors — re-run this
script after each new eval (e.g. after the redshift probe's first cluster run,
or the GZD-5 ViT-S run) to refresh everything.

## Outputs (`results/`)

- `comparison.json` — merged local + published numbers, machine-readable.
- `comparison_tables.md` — one markdown table per task with per-row source and
  protocol caveats.
- `galaxy10_accuracy_vs_competitors.png`, `gzd5_questions_vs_astroclip.png`,
  `redshift_r2_vs_competitors.png` — a chart per task with local results.

## Editing baselines

All published numbers live in `competitor_baselines.json` (transcribed from
`Evals/README.md`), never in code. Add new competitors or corrected numbers
there.

## Caveats

None of these comparisons are perfectly apples-to-apples; each table states
the mismatch explicitly:

- **Galaxy10**: AION evaluates on a Galaxy Zoo 10 / Legacy Survey DR10
  cross-match with a 2-layer MLP head; our number is a linear probe on the
  Galaxy10 DECaLS HDF5. The planned MLP-head rerun closes the head-capacity
  gap (`Evals/README.md`, "What This Means For Our Next Experiments").
- **GZD-5**: closest to like-for-like — same questions, debiased labels, and
  4-layer MLP head as AstroCLIP.
- **Redshift**: our probe uses Galaxy10 DECaLS metadata redshifts (z mostly
  below 0.25); AstroCLIP/AION use different cross-matched samples, and the
  AION rows include photometry inputs.
