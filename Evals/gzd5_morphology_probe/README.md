# GZD-5 Morphology Probe

AstroCLIP-style Galaxy Zoo DECaLS DR5 morphology evaluation for frozen
Astro-Worldmodels image backbones.

This eval mirrors the public AstroCLIP downstream morphology code as closely as
possible:

- Galaxy Zoo DECaLS/GZD-5 question-wise labels.
- Official `*_debiased` soft-label columns from `gz_decals_volunteers_5.csv`.
- Train/test image split from the `galaxy-datasets` `GZDecals5` release.
- Frozen image encoder embeddings.
- One 4-layer MLP per morphology question.
- MLP hidden size `256`, dropout `0.2`, Adam lr `1e-3`, `25` epochs.
- Internal 90/10 train/validation split with `random_state=42`.
- WeightedRandomSampler with sample weight equal to each target row's largest
  vote probability, matching AstroCLIP's code.

Important caveat: AstroCLIP's notebook uses a DESI-LS cross-matched HDF5 with
222,929 galaxies. The public `galaxy-datasets` GZD-5 train/test split contains
228,059 galaxies before the AstroCLIP `smooth-or-featured_total-votes >= 3`
filter. This eval uses the reproducible public split and records exact counts
in `results/metadata.json`.

## Run

Use the project conda environment from the repo root:

```bash
/mnt/ssd-cluster/pranav/conda/envs/lejepa-og/bin/python \
  Evals/gzd5_morphology_probe/prepare_gzd5.py

CUDA_VISIBLE_DEVICES=0,1 \
/mnt/ssd-cluster/pranav/conda/envs/lejepa-og/bin/python \
  Evals/gzd5_morphology_probe/extract_embeddings.py \
  --model large \
  --batch-size 512 \
  --workers 8

/mnt/ssd-cluster/pranav/conda/envs/lejepa-og/bin/python \
  Evals/gzd5_morphology_probe/train_gzd5_mlp.py \
  --model large
```

Repeat `extract_embeddings.py` and `train_gzd5_mlp.py` with `--model small` for
the selected ViT-S checkpoint.

## Outputs

- `data/gz_decals_5/`: catalogs, image tarball, extracted images.
- `data/merged_train.parquet`, `data/merged_test.parquet`: image paths plus
  debiased morphology labels.
- `results/{model}/train_embeddings.npy`: frozen train embeddings.
- `results/{model}/test_embeddings.npy`: frozen test embeddings.
- `results/{model}/metrics.json`: per-question accuracy/F1.
- `results/{model}/metrics.csv`: table version of metrics.
- `results/{model}/radar_accuracy.png`, `radar_f1.png`: summary plots.

## Completed ViT-L Result

Finished for `astro_vit_large_step_52000` using the single AstroCLIP-style
cheap run: one fixed train/validation split, one 4-layer MLP per question,
`MLP_dim=256`, `epochs=25`, `dropout=0.2`.

| Question | Train Samples | Test Samples | Accuracy | F1 |
|---|---:|---:|---:|---:|
| smooth | 46,452 | 11,121 | 0.7742 | 0.6757 |
| disk-edge-on | 46,112 | 1,543 | 0.8846 | 0.8305 |
| spiral-arms | 44,226 | 1,155 | 0.9351 | 0.9460 |
| bar | 44,226 | 1,155 | 0.5377 | 0.3760 |
| bulge-size | 44,226 | 1,155 | 0.7758 | 0.7625 |
| how-rounded | 45,658 | 1,700 | 0.8247 | 0.8248 |
| edge-on-bulge | 28,842 | 140 | 0.8000 | 0.7111 |
| spiral-winding | 28,249 | 804 | 0.7525 | 0.6462 |
| spiral-arm-count | 28,249 | 804 | 0.4216 | 0.3865 |
| merging | 45,289 | 8,597 | 0.8145 | 0.7312 |
| **Mean** | - | - | **0.7521** | **0.6890** |

Main files:

- `results/astro_vit_large_step_52000/metrics.json`
- `results/astro_vit_large_step_52000/metrics.csv`
- `results/astro_vit_large_step_52000/radar_accuracy.png`
- `results/astro_vit_large_step_52000/radar_f1.png`
- `results/astro_vit_large_step_52000/astroclip_vs_ours_gzd5_radar.png`
