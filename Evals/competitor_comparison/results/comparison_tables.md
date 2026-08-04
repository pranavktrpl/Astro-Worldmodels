# Competitor comparison

Published numbers are transcribed in `competitor_baselines.json`; local numbers are read from each eval suite's results. Protocol caveats per task are listed under each table — none of these are perfectly apples-to-apples.

## Galaxy10 / Galaxy Zoo 10 morphology

| Model | Head | Test accuracy | Test macro-F1 | Source |
|---|---|---:|---:|---|
| Ours ViT-L/14 (step 52000) | 2-layer MLP (AION head repro) | 0.7245 ± 0.0133 | 0.7091 ± 0.0085 | local |
| Ours ViT-L/14 (step_52000) | linear probe (LBFGS) on frozen embeddings | 0.7183 ± 0.0084 | 0.7023 ± 0.0090 | local |
| Ours ViT-L/14 (step 52000) | 4-layer MLP (AstroCLIP head repro) | 0.6644 ± 0.0085 | 0.5566 ± 0.0069 | local |
| Ours ViT-S/14 (step 21000) | 2-layer MLP (AION head repro) | 0.6257 ± 0.0105 | 0.5889 ± 0.0063 | local |
| Ours ViT-S/14 (step_21000) | linear probe (LBFGS) on frozen embeddings | 0.6119 ± 0.0142 | 0.5934 ± 0.0116 | local |
| Ours ViT-S/14 (step 21000) | 4-layer MLP (AstroCLIP head repro) | 0.5750 ± 0.0143 | 0.4906 ± 0.0144 | local |
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

| Model | Head / input | Sample | Test R² | Test R² (z < 0.25) | Source |
|---|---|---|---:|---:|---|
| Ours ViT-L/14 (step 52000) | ridge on frozen image embeddings | Galaxy10 DECaLS | 0.6230 ± 0.1780 | 0.7393 ± 0.0144 | local |
| Ours ViT-L/14 (step 52000) | knn on frozen image embeddings | Galaxy10 DECaLS | 0.5574 ± 0.1604 | 0.6630 ± 0.0199 | local |
| Ours ViT-L/14 (step 52000) | mlp on frozen image embeddings | Galaxy10 DECaLS | 0.6051 ± 0.1816 | 0.7215 ± 0.0770 | local |
| Ours ViT-S/14 (step 21000) | ridge on frozen image embeddings | Galaxy10 DECaLS | 0.5648 ± 0.1638 | 0.6740 ± 0.0196 | local |
| Ours ViT-S/14 (step 21000) | knn on frozen image embeddings | Galaxy10 DECaLS | 0.5046 ± 0.1405 | 0.5973 ± 0.0189 | local |
| Ours ViT-S/14 (step 21000) | mlp on frozen image embeddings | Galaxy10 DECaLS | 0.5773 ± 0.1712 | 0.6863 ± 0.0255 | local |
| AstroCLIP Image, zero-shot kNN | image | DESI-LS images cross-matched with DESI spectra | 0.79 | n/r | published |
| AstroCLIP Image, few-shot MLP | image | DESI-LS images cross-matched with DESI spectra | 0.78 | n/r | published |
| AstroCLIP Spectrum, few-shot MLP | spectrum | DESI-LS images cross-matched with DESI spectra | 0.98 | n/r | published |
| AION-1-B | photometry | PROVABGS + Legacy Survey DR10 + DESI EDR | 0.75 | n/r | published |
| AION-1-L | photometry | PROVABGS + Legacy Survey DR10 + DESI EDR | 0.76 | n/r | published |
| AION-1-XL | photometry | PROVABGS + Legacy Survey DR10 + DESI EDR | 0.79 | n/r | published |
| AION-1-B | photometry+image | PROVABGS + Legacy Survey DR10 + DESI EDR | 0.93 | n/r | published |
| AION-1-L | photometry+image | PROVABGS + Legacy Survey DR10 + DESI EDR | 0.94 | n/r | published |
| AION-1-XL | photometry+image | PROVABGS + Legacy Survey DR10 + DESI EDR | 0.94 | n/r | published |

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

