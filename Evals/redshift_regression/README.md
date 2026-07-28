# Redshift Regression Probe

First non-classification downstream eval for the recommended frozen image
backbones. It tests whether the LeJEPA representations encode distance-related
information (apparent size, surface brightness, color), not just morphology.

Targets are spectroscopic redshifts from the `redshift` column of the local
Galaxy10 DECaLS HDF5 (`Evals/DeCals_linearProbing/galaxy10/Galaxy10_DECals.h5`),
so no new data download is required.

## Checkpoints Evaluated

The two repository-recommended checkpoints:

| Key | Label | Checkpoint |
|---|---|---|
| `large` | `astro_vit_large_step_52000` | `checkpoints/VitLargePatch14_OfficialTrain5_Epoch5_2504/step_52000.pt` |
| `small` | `astro_vit_small_step_21000` | `checkpoints/VitSmallPatch14_2204/step_21000.pt` |

## Protocol

Embedding extraction matches `Evals/galaxy10_checkpoint_evolution` exactly:
bicubic resize to `140 x 140`, RGB floats in `[0, 1]`, no ImageNet
normalization, frozen backbone, bf16 autocast on CUDA. Embeddings are cached to
`results/<label>/embeddings.npy` so head-only reruns are cheap.

Targets are filtered to finite, positive redshifts; kept/dropped counts are
recorded in `metrics.json` under `target_stats`.

Splits are 80/10/10, stratified over 10 redshift quantile bins, repeated with
seeds 42, 43, 44. Features are standardized using train statistics only. All
hyperparameters are selected on validation R^2 only; test metrics are reported
after selection.

Three heads run on the same embeddings per checkpoint:

| Head | Details | Selection grid | Mirrors |
|---|---|---|---|
| `ridge` | Deterministic closed-form linear ridge | L2 in `{1e-4, 1e-2, 1, 1e2, 1e4}` | Repo linear-probe convention |
| `knn` | k-nearest-neighbour mean in embedding space | k in `{4, 16, 64}` | AstroCLIP zero-shot kNN |
| `mlp` | `dim -> 256 -> 256 -> 1`, GELU, dropout 0.2, Adam lr 1e-3, best epoch by validation R^2 | epochs fixed (default 100) | AION-style MLP head |

## Metrics

Reported per head as mean +/- std over the three split seeds:

- **R2** on raw redshift;
- **MAE** and **RMSE**;
- **NMAD**: `1.4826 * median(|dz/(1+z) - median|)`, the standard
  photometric-redshift scatter statistic;
- **outlier fraction**: share of `|dz|/(1+z) > 0.05`.

## Run

From the repo root on the cluster:

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/ssd-cluster/pranav/conda/envs/lejepa-og/bin/python \
  Evals/redshift_regression/redshift_probe.py --model all
```

Useful flags: `--model large|small|all`, `--batch-size`, `--mlp-epochs`,
`--overwrite` (re-extract embeddings), `--galaxy10-h5` (alternate HDF5 path).

## Outputs

- `results/<label>/embeddings.npy`, `embedding_metadata.json`: cached frozen
  embeddings and provenance.
- `results/<label>/metrics.json`: per-seed and summarized metrics for all
  heads, plus target filtering stats and full config.
- `results/<label>/predicted_vs_true.png`: hexbin of predicted vs
  spectroscopic redshift per head on the seed-42 test split.
- `results/summary.csv`: one row per model/head with test metric means/stds.

## Published Reference Points

Not apples-to-apples (different datasets, matched sets, and sample sizes), but
they set expectations for image-only redshift regression:

| Model | Setup | R2 |
|---|---|---:|
| AstroCLIP Image | zero-shot kNN, DESI-LS cross-match | 0.79 |
| AstroCLIP Image | few-shot MLP | 0.78 |
| AION-1-B/L | photometry + image, PROVABGS | 0.93 - 0.94 |
| AstroCLIP Spectrum | few-shot MLP | 0.98 |

Caveats for interpreting local numbers against these: Galaxy10 has only
~17.7k examples versus AstroCLIP's ~198k cross-matched set, and Galaxy10's
redshift range is narrow (mostly z < 0.25), which deflates R2 relative to
wider-range benchmarks. NMAD and outlier fraction are more range-robust.

## Natural Extensions

- **Spectra backbone redshift probe**: same heads on frozen
  `SpectrumTransformerEncoder` embeddings once a spectra checkpoint is
  selected; needs redshift labels cross-matched to `UniverseTBD/mmu_desi_edr_sv3`.
- **Physical properties**: stellar mass, age, metallicity, sSFR via a
  PROVABGS-style matched catalog, reusing the same script structure.
