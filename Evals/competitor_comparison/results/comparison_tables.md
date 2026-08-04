# Competitor comparison

Published numbers are transcribed in `competitor_baselines.json`; local numbers are read from each eval suite's results. Protocol caveats per task are listed under each table — none of these are perfectly apples-to-apples.

## Galaxy10 / Galaxy Zoo 10 morphology

| Model | Head | Test accuracy | Test macro-F1 | Source |
|---|---|---:|---:|---|
| Ours ViT-L/14 (step_52000) | linear probe (LBFGS) on frozen embeddings | 0.7183 ± 0.0084 | 0.7023 ± 0.0090 | local |
| Ours ViT-S/14 (step_21000) | linear probe (LBFGS) on frozen embeddings | 0.6119 ± 0.0142 | 0.5934 ± 0.0116 | local |
| Ours ResNet9 (step_8000) | linear probe (LBFGS) on frozen embeddings | 0.4480 ± 0.0175 | 0.4269 ± 0.0170 | local |
| AION-1-B | 2-layer MLP on frozen embeddings | 0.840 | n/r | published |
| AION-1-L | 2-layer MLP on frozen embeddings | 0.872 | n/r | published |
| AION-1-XL | 2-layer MLP on frozen embeddings | 0.865 | n/r | published |

Caveats: AION evaluates on a Galaxy Zoo 10 / Legacy Survey DR10 cross-match (~8k galaxies), not the Galaxy10 DECaLS HDF5. Our linear-probe rows additionally differ in head capacity; the galaxy10_matched_heads rows reproduce AION's 2-layer MLP head (and AstroCLIP's 4-layer MLP) exactly, leaving only the dataset difference.

## GZD-5 question-wise morphology

| Question | Ours ViT-L/14 (step 52000) acc / F1 | AstroCLIP Image ViT-L (published) acc / F1 |
|---|---:|---:|
| smooth | 0.774 / 0.676 | 0.83 / 0.83 |
| disk-edge-on | 0.885 / 0.830 | 0.97 / 0.97 |
| spiral-arms | 0.935 / 0.946 | 0.92 / 0.94 |
| bar | 0.538 / 0.376 | 0.56 / 0.54 |
| bulge-size | 0.776 / 0.762 | 0.79 / 0.78 |
| how-rounded | 0.825 / 0.825 | 0.74 / 0.74 |
| edge-on-bulge | 0.800 / 0.711 | 0.82 / 0.81 |
| spiral-winding | 0.752 / 0.646 | 0.74 / 0.68 |
| spiral-arm-count | 0.422 / 0.386 | 0.44 / 0.41 |
| merging | 0.814 / 0.731 | 0.80 / 0.73 |
| **mean** | 0.752 / 0.689 | 0.761 / 0.743 |

Caveats: Same question set and debiased-label protocol as our gzd5_morphology_probe, so this is the closest to a like-for-like comparison we have.

## Redshift regression

| Model | Head / input | Test R² | Test R² (z < 0.25) | Source |
|---|---|---:|---:|---|
| _pending — run `redshift_regression/redshift_probe.py` on the cluster_ | | | | local |
| AstroCLIP Image, zero-shot kNN | image | 0.79 | n/r | published |
| AstroCLIP Image, few-shot MLP | image | 0.78 | n/r | published |
| AstroCLIP Spectrum, few-shot MLP | spectrum | 0.98 | n/r | published |
| AION-1-B | photometry | 0.75 | n/r | published |
| AION-1-L | photometry | 0.76 | n/r | published |
| AION-1-XL | photometry | 0.79 | n/r | published |
| AION-1-B | photometry+image | 0.93 | n/r | published |
| AION-1-L | photometry+image | 0.94 | n/r | published |
| AION-1-XL | photometry+image | 0.94 | n/r | published |

Caveats: Our probe regresses Galaxy10 DECaLS metadata redshifts (z mostly < 0.25) from images alone; AstroCLIP and AION use different cross-matched samples and, for AION, photometry inputs. The AstroCLIP image rows are the closest protocol match (image-only, kNN and MLP heads).

## Physical property regression (reference targets)

No local eval exists yet (requires PROVABGS labels). Published R² targets to beat once it does:

| Model | Input | Stellar mass | Age | Metallicity | sSFR-like |
|---|---|---:|---:|---:|---:|
| AION-1-B | photometry+image | 0.89 | 0.45 | 0.49 | 0.64 |
| AION-1-L | photometry+image | 0.89 | 0.45 | 0.50 | 0.64 |
| AION-1-XL | photometry+image | 0.89 | 0.45 | 0.49 | 0.64 |
| AstroCLIP Image, zero-shot | image | 0.74 | 0.27 | 0.44 | 0.44 |
| AstroCLIP Image, few-shot | image | 0.73 | 0.26 | 0.43 | 0.42 |
| AstroCLIP Spectrum, few-shot | spectrum | 0.88 | 0.43 | 0.58 | 0.64 |

