# Experiments And External Benchmark Comparison

This document collects the repo's current evaluation results and places them
beside the most relevant published astronomy foundation-model benchmarks:

- **AION-1**: omnimodal astronomy foundation model family.
- **AstroCLIP**: cross-modal galaxy image/spectrum foundation model.
- **Galaxy Zoo / Galaxy10**: the morphology label family behind the local
  Galaxy10 DECaLS probe.

The main local result available right now is **Galaxy10 morphology
classification**. Redshift and physical-property regression are included below
as published-reference targets, but they have not yet been run for this repo's
backbones.

## Key Comparison Caveat

The morphology numbers below are useful, but not all are directly
apples-to-apples.

| Benchmark | Label Structure | Directly Comparable To Our Galaxy10 Probe? |
|---|---|---|
| Galaxy10 DECaLS | One 10-way morphology class per image | Yes |
| AION-1 Galaxy Zoo 10 | One 10-way morphology class per image | Mostly yes |
| AstroCLIP GZD-5 | Ten Galaxy Zoo DECaLS morphology questions | No, related but different |

For a fair paper-style comparison, report both:

1. **Galaxy10 / Galaxy Zoo 10 accuracy and macro-F1**, for AION-like
   10-class morphology comparison.
2. **Galaxy Zoo DECaLS question-wise accuracy/F1**, for AstroCLIP-like
   morphology comparison.

## Local Morphology Protocol

Local results are from:

- `Evals/galaxy10_checkpoint_evolution/results/best_checkpoints.json`
- `Evals/galaxy10_checkpoint_evolution/results/dataset_metadata.json`
- `Evals/galaxy10_checkpoint_evolution/results/best_model_galaxy10_per_class_metrics.json`

Protocol:

| Item | Value |
|---|---|
| Dataset | Galaxy10 DECaLS |
| Images | 17,736 RGB images |
| Image size | 256 x 256 x 3 in the HDF5; resized to 140 x 140 for probing |
| Classes | 10 morphology classes |
| Label source | Galaxy Zoo labels on DESI Legacy Imaging Surveys images |
| Split protocol | Stratified 80/10/10 repeated over seeds 42, 43, 44 |
| Backbone | Frozen |
| Probe | Deterministic full-batch multinomial linear classifier |
| Optimizer | LBFGS from zero initialization |
| Loss | Class-balanced cross-entropy |
| Selection rule | Highest mean validation macro-F1; validation accuracy tie-break |
| Reported metrics | Test accuracy, macro-F1, balanced accuracy, per-class recall/F1 |

Local Galaxy10 class counts:

| Class | Count |
|---|---:|
| disturbed | 1,081 |
| merging | 1,853 |
| round_smooth | 2,645 |
| in_between_round_smooth | 2,027 |
| cigar_shaped_smooth | 334 |
| barred_spiral | 2,043 |
| unbarred_tight_spiral | 1,829 |
| unbarred_loose_spiral | 2,628 |
| edge_on_without_bulge | 1,423 |
| edge_on_with_bulge | 1,873 |

## Backbone Scale And Pretraining Data

This table separates **backbone pretraining data** from **downstream labeled
evaluation data**. These are different quantities.

| Model | Backbone / Model Params | Pretraining Data Size | Pretraining Data Type | Morphology Eval Dataset | Morphology Labeled Samples |
|---|---:|---:|---|---|---:|
| AION-1-B | 300M full model | 200M+ observations | Multimodal: Legacy Survey, HSC, SDSS, DESI, Gaia | Galaxy Zoo 10 cross-matched with Legacy Survey DR10 | Not clearly stated in paper summary |
| AION-1-L | 800M full model | 200M+ observations | Same as above | Same | Not clearly stated in paper summary |
| AION-1-XL | 3.1B full model | 200M+ observations | Same as above | Same | Not clearly stated in paper summary |
| AstroCLIP Image ViT-L | ~307M trainable image params | 76,446,849 images plus 197,632 image-spectrum pairs for CLIP alignment | DESI Legacy Survey images; DESI spectra pairs | Galaxy Zoo DECaLS / GZD-5 | 222,929 |
| Our ViT-L/14 | 304.37M backbone; 310.81M incl. projector | Configured HF stream target: 80M; selected checkpoint saw ~39.94M image draws assuming 4 GPUs | `Smith42/galaxies`, image crops | Galaxy10 DECaLS | 17,736 |
| Our ViT-S/14 | 22.06M backbone; 23.05M incl. projector | Configured HF stream target: 80M; selected checkpoint saw ~8.06M image draws assuming 4 GPUs | `Smith42/galaxies`, image crops | Galaxy10 DECaLS | 17,736 |

Notes:

- The local checkpoint parameter counts are exact tensor counts from the
  checkpoint state dict.
- The local "image draws" estimate is `global_step x per-device batch size x 4`
  for the selected checkpoint. It is not necessarily the number of unique
  galaxies, because the training data are streamed and augmented.
- AION-1 reports a multimodal observation count, not a clean "image-only sample
  count" for the vision component.

## Morphology Results

### Headline Comparison

| Model | Eval Dataset | Eval Head | Selection / Evaluation | Accuracy | Macro-F1 / F1 | Notes |
|---|---|---|---|---:|---:|---|
| AION-1-L | Galaxy Zoo 10 | 2-layer MLP on frozen embeddings | Published downstream eval | 87.2% | Not reported in headline table | Best AION morphology score |
| AION-1-XL | Galaxy Zoo 10 | 2-layer MLP on frozen embeddings | Published downstream eval | 86.5% | Not reported in headline table | Larger model, slightly lower than AION-L |
| AION-1-B | Galaxy Zoo 10 | 2-layer MLP on frozen embeddings | Published downstream eval | 84.0% | Not reported in headline table | Smallest AION variant |
| AstroCLIP Image ViT-L | GZD-5 question-wise morphology | 4-layer MLP on frozen image embeddings | Published downstream eval | 76.1% average across questions | 74.3 average F1 across questions | Not 10-way Galaxy10 |
| Our ViT-L/14 | Galaxy10 DECaLS | Linear probe on frozen backbone | Best validation macro-F1 checkpoint | 71.83 +/- 0.84% | 70.23 +/- 0.90 macro-F1 | `step_52000.pt` |
| Our ViT-S/14 | Galaxy10 DECaLS | Linear probe on frozen backbone | Best validation macro-F1 checkpoint | 61.19 +/- 1.42% | 59.34 +/- 1.16 macro-F1 | `step_21000.pt` |

### AION-1 Published Galaxy Zoo 10 Baselines

| Model / Baseline | Accuracy |
|---|---:|
| ZooBot | 89.6% |
| AION-1-L | 87.2% |
| AION-1-XL | 86.5% |
| AION-1-B | 84.0% |
| EfficientNet-B3 | 80.0% |
| DINOv2 ViT-g/14 | 71.4% |

AION's comparison is especially relevant because DINOv2, ZooBot, and AION use
frozen embeddings plus an MLP head, while EfficientNet-B3 is trained end-to-end
from scratch in their table.

### AstroCLIP GZD-5 Question-Wise Morphology

AstroCLIP's morphology evaluation is not a single 10-way Galaxy10 task. It is
ten Galaxy Zoo DECaLS questions.

| GZD-5 Question | Accuracy | F1 |
|---|---:|---:|
| smooth | 0.83 | 0.83 |
| disk-edge-on | 0.97 | 0.97 |
| spiral-arms | 0.92 | 0.94 |
| bar | 0.56 | 0.54 |
| bulge-size | 0.79 | 0.78 |
| how-rounded | 0.74 | 0.74 |
| edge-on-bulge | 0.82 | 0.81 |
| spiral-winding | 0.74 | 0.68 |
| spiral-arm-count | 0.44 | 0.41 |
| merging | 0.80 | 0.73 |
| **Mean** | **0.761** | **0.743** |

### Our Best Galaxy10 Model: Per-Class Test Metrics

These are for the selected local ViT-L/14 checkpoint, `step_52000.pt`.

| Class | Recall / Class Accuracy | F1 |
|---|---:|---:|
| disturbed | 0.486 +/- 0.016 | 0.456 +/- 0.019 |
| merging | 0.771 +/- 0.036 | 0.762 +/- 0.028 |
| round_smooth | 0.863 +/- 0.013 | 0.847 +/- 0.018 |
| in_between_round_smooth | 0.786 +/- 0.011 | 0.794 +/- 0.001 |
| cigar_shaped_smooth | 0.824 +/- 0.106 | 0.609 +/- 0.081 |
| barred_spiral | 0.628 +/- 0.042 | 0.643 +/- 0.022 |
| unbarred_tight_spiral | 0.652 +/- 0.033 | 0.634 +/- 0.034 |
| unbarred_loose_spiral | 0.486 +/- 0.033 | 0.546 +/- 0.030 |
| edge_on_without_bulge | 0.893 +/- 0.004 | 0.884 +/- 0.018 |
| edge_on_with_bulge | 0.862 +/- 0.018 | 0.847 +/- 0.008 |

## Redshift Regression Reference Targets

These results are included as targets for future local experiments. We have not
yet run redshift regression for this repo's backbones.

| Model | Dataset / Setup | Input | Metric | Score |
|---|---|---|---|---:|
| AION-1-B | PROVABGS + Legacy Survey DR10 + DESI EDR | Photometry | R2 | 0.75 |
| AION-1-B | Same | Photometry + Image | R2 | 0.93 |
| AION-1-B | Same | Photometry + Image + Spectrum | R2 | 1.00 |
| AION-1-L | Same | Photometry | R2 | 0.76 |
| AION-1-L | Same | Photometry + Image | R2 | 0.94 |
| AION-1-L | Same | Photometry + Image + Spectrum | R2 | 1.00 |
| AION-1-XL | Same | Photometry | R2 | 0.79 |
| AION-1-XL | Same | Photometry + Image | R2 | 0.94 |
| AION-1-XL | Same | Photometry + Image + Spectrum | R2 | 0.99 |
| AstroCLIP Image | DESI-LS + DESI cross-match | Image, zero-shot kNN | R2 | 0.79 |
| AstroCLIP Image | Same | Image, few-shot MLP | R2 | 0.78 |
| AstroCLIP Spectrum | Same | Spectrum, few-shot MLP | R2 | 0.98 |

## Galaxy Physical Property Regression Reference Targets

These are also reference targets for future local experiments. The most natural
local version would freeze the backbone, extract image embeddings, and train a
small regressor for each property.

### AION-1 R2

| Model | Input | Redshift | Stellar Mass | Age | Metallicity | SFR / sSFR-like |
|---|---|---:|---:|---:|---:|---:|
| AION-1-B | Photometry | 0.75 | 0.72 | 0.35 | 0.41 | 0.38 |
| AION-1-B | Photometry + Image | 0.93 | 0.89 | 0.45 | 0.49 | 0.64 |
| AION-1-B | Photometry + Image + Spectrum | 1.00 | 0.96 | 0.53 | 0.61 | 0.72 |
| AION-1-L | Photometry | 0.76 | 0.73 | 0.36 | 0.41 | 0.39 |
| AION-1-L | Photometry + Image | 0.94 | 0.89 | 0.45 | 0.50 | 0.64 |
| AION-1-L | Photometry + Image + Spectrum | 1.00 | 0.96 | 0.53 | 0.62 | 0.73 |
| AION-1-XL | Photometry | 0.79 | 0.76 | 0.31 | 0.38 | 0.48 |
| AION-1-XL | Photometry + Image | 0.94 | 0.89 | 0.45 | 0.49 | 0.64 |
| AION-1-XL | Photometry + Image + Spectrum | 0.99 | 0.95 | 0.53 | 0.62 | 0.73 |

### AstroCLIP R2

| Encoder / Mode | Stellar Mass | Metallicity | Age | sSFR |
|---|---:|---:|---:|---:|
| AstroCLIP Image, zero-shot | 0.74 | 0.44 | 0.27 | 0.44 |
| AstroCLIP Image, few-shot | 0.73 | 0.43 | 0.26 | 0.42 |
| AstroCLIP Spectrum, zero-shot | 0.87 | 0.57 | 0.43 | 0.63 |
| AstroCLIP Spectrum, few-shot | 0.88 | 0.58 | 0.43 | 0.64 |

## What This Means For Our Next Experiments

The cleanest next comparisons are:

1. **Galaxy10 morphology, stronger head.**
   - Current local result uses a linear probe.
   - AION uses a 2-layer MLP.
   - AstroCLIP uses a 4-layer MLP.
   - A fairer comparison should include both linear and MLP heads.

2. **Galaxy Zoo DECaLS question-wise morphology.**
   - Needed for a direct AstroCLIP-style comparison.
   - Report per-question accuracy and F1.

3. **Redshift regression on Galaxy10 DECaLS metadata or a larger matched set.**
   - Galaxy10 DECaLS includes `redshift` in the HDF5 according to the dataset
     documentation.
   - Report R2, MAE, RMSE, and normalized error.
   - Implemented in `Evals/redshift_regression/` for the recommended ViT-L
     `step_52000` and ViT-S `step_21000` checkpoints, with ridge, kNN, and MLP
     heads; awaiting a cluster run.

4. **Physical property regression.**
   - Requires a matched catalog such as PROVABGS-style labels.
   - Report R2 and MAE/RMSE for stellar mass, age, metallicity, and SFR/sSFR.

## Local Files To Inspect

| File | Purpose |
|---|---|
| `Evals/galaxy10_checkpoint_evolution/README.md` | Local Galaxy10 protocol and completed result summary |
| `Evals/galaxy10_checkpoint_evolution/results/best_checkpoints.json` | Selected checkpoints and repeated-holdout scores |
| `Evals/galaxy10_checkpoint_evolution/results/dataset_metadata.json` | Galaxy10 class names, counts, and HDF5 path |
| `Evals/galaxy10_checkpoint_evolution/results/best_model_galaxy10_per_class_metrics.json` | Best local model per-class recall/F1 |
| `Evals/vit_patch_attention_diagnostics/README.md` | PCA and attention diagnostics for selected checkpoints |
| `Evals/redshift_regression/README.md` | Redshift regression probe (ridge/kNN/MLP) for the recommended checkpoints |

## Sources

- AION-1 paper: <https://arxiv.org/html/2510.17960v1>
- AION-1 arXiv abstract page: <https://arxiv.org/abs/2510.17960>
- AstroCLIP paper: <https://arxiv.org/html/2310.03024v2>
- AstroCLIP MNRAS page: <https://academic.oup.com/mnras/article/531/4/4990/7697182>
- Galaxy10 DECaLS dataset docs: <https://astronn.readthedocs.io/en/latest/galaxy10.html>
- Galaxy10 DECaLS dataset card: <https://huggingface.co/datasets/matthieulel/galaxy10_decals>

