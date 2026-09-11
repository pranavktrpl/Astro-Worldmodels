# Final 307K Cross-Modal Frozen-Backbone Evaluation

Date: 2026-08-30

> **Probe-comparison note:** the local scores below use ridge regression, while
> AstroCLIP's published scores use kNN or nonlinear MLP probes. A matched
> nonlinear diagnostic and causal analysis are recorded in
> [`ASTROCLIP_GAP_DEEP_DIVE.md`](ASTROCLIP_GAP_DEEP_DIVE.md). In particular,
> scratch raw stellar-age R2 rises from 0.3658 to 0.4101 with the released-code
> MLP shape, whereas spectrum redshift rises only from 0.6756 to 0.7016.

## Scope

This report evaluates only the final `last.pt` checkpoint from each completed
307,428-pair experiment:

- scratch: `checkpoints/CrossModalScratch_DESI307K_AllPairs_CrossOnly_SIGReg_4GPU/last.pt`;
- post-trained: `checkpoints/CrossModalPostTrain_DESI307K_AstroCLIPPool_LeJEPA_SIGReg/last.pt`.

The scratch checkpoint completed 50 epochs and 60,000 optimizer steps. The
post-trained checkpoint completed 10 epochs and 24,000 optimizer steps. The
post-trained file contains alignment heads only; evaluation reconstructs and
strictly validates its recorded frozen image and spectrum source checkpoints.
Neither source checkpoint was modified.

## Protocol

The protocol is unchanged from the earlier frozen-backbone evaluations:

- exact AstroCLIP DESI-LS x DESI EDR parquet mirror;
- shipped split: 138,583 train and 29,697 test rows;
- PROVABGS subset: 73,341 train and 15,993 test rows;
- deterministic DR2 RGB image transform and bicubic resize to 140 pixels;
- deterministic, unmasked, noise-free mean/std spectrum view;
- frozen embeddings, train-split feature standardization, and ridge probes;
- ridge L2 selected from `1e-4, 1e-2, 1, 1e2, 1e4` on a seed-specific 10% train carve-out;
- 10 seeds for image redshift and 3 seeds for spectrum probes.

Uncertainty below is the standard deviation over validation-carve-out seeds.
It measures ridge-selection sensitivity, not uncertainty from retraining the
backbone.

## Clean Summary

Primary metric: frozen raw-backbone ridge test R2. `Previous best` means the
strongest local result available before the two completed 307K runs.

| Target | Original AstroJEPA baseline | Previous best | 307K post-trained | **307K scratch** | Gain vs original | AstroCLIP best | Remaining gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Image redshift | 0.55062 | 0.55062 | 0.52998 | **0.62741** | **+0.07679** | 0.79 | 0.16259 |
| Spectrum redshift | 0.43200 | 0.64237 | 0.55506 | **0.67556** | **+0.24356** | 0.98 | 0.30444 |
| Stellar mass | 0.50600 | 0.82309 | 0.69755 | **0.84260** | **+0.33660** | 0.88 | 0.03740 |
| sSFR | 0.51000 | 0.62860 | 0.49642 | **0.63919** | **+0.12919** | 0.64 | 0.00081 |
| Metallicity | 0.23500 | 0.53313 | 0.40879 | **0.54944** | **+0.31444** | 0.58 | 0.03056 |
| Stellar age | 0.18500 | 0.32063 | 0.23220 | **0.36579** | **+0.18079** | 0.43 | 0.06421 |

The strongest new result is the 307K from-scratch model on every target. The
post-trained raw column is expected to reproduce its frozen source backbones:
step-52K image for image redshift and spectra-v2 for spectrum targets. The
original image baseline at 0.55062 is the separately cross-match-adapted image
checkpoint, not the frozen image source used by post-training.

## Headline R2

### Image redshift

| Model / representation | Dim | Test R2 | Delta vs prior counterpart |
| --- | ---: | ---: | ---: |
| Original image backbone, step 52K, historical raw | 1,024 | 0.53012 +/- 0.00022 | reference |
| Separate cross-match-adapted ViT-L, historical raw | 1,024 | 0.55062 +/- 0.00018 | reference only |
| Prior 95K scratch, raw | 1,024 | 0.52334 +/- 0.00055 | reference |
| Prior 95K scratch, aligned | 256 | 0.51729 +/- 0.00026 | reference |
| **307K post-trained, frozen raw** | 1,024 | **0.52998 +/- 0.00033** | -0.00014 vs source historical |
| **307K post-trained, aligned pooler** | 256 | **0.51175 +/- 0.00015** | -0.01824 vs its raw |
| **307K scratch, raw** | 1,024 | **0.62741 +/- 0.00041** | +0.10407 vs 95K scratch raw |
| **307K scratch, aligned** | 256 | **0.61368 +/- 0.00033** | +0.09639 vs 95K scratch aligned |
| AstroCLIP image, published | - | 0.79 | +0.16259 vs best current raw |
| AION-1-L, published | - | 0.94 | not like-for-like; includes photometry and another sample |

The direct historical source result used three seeds; the new result uses ten.
Its first three per-seed values match the historical output, validating source
reconstruction. The 0.55062 model was a separate checkpoint after 3,240 steps
of cross-match-specific image adaptation and was not the source of the
post-trained run.

### Spectrum raw backbone

| Target | V1 | V2 final | Prior 95K scratch | 307K post-trained raw | **307K scratch raw** | AstroCLIP spectrum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Redshift | 0.432 | 0.55506 | 0.64237 | 0.55506 | **0.67556 +/- 0.00049** | 0.98 |
| Stellar mass | 0.506 | 0.69755 | 0.82309 | 0.69755 | **0.84260 +/- 0.00018** | 0.88 |
| sSFR | 0.510 | 0.49642 | 0.62860 | 0.49642 | **0.63919 +/- 0.00028** | 0.64 |
| Metallicity | 0.235 | 0.40879 | 0.53313 | 0.40879 | **0.54944 +/- 0.00029** | 0.58 |
| Stellar age | 0.185 | 0.23220 | 0.32063 | 0.23220 | **0.36579 +/- 0.00033** | 0.43 |

Post-trained raw values exactly reproduce v2 because that backbone is frozen.
Relative to the prior 95K scratch checkpoint, the 307K scratch raw backbone
gains +0.03319 redshift, +0.01951 mass, +0.01059 sSFR, +0.01632 metallicity,
and +0.04516 age R2.

Remaining 307K scratch gaps to AstroCLIP are 0.30444 redshift, 0.03740 mass,
0.00081 sSFR, 0.03056 metallicity, and 0.06421 age. Thus sSFR is effectively at
the published reference under this probe; redshift remains the dominant gap.

### Objective-facing representations

| Target | V2 projector, 64d | Prior 95K scratch, 256d | **307K post-trained pooler, 256d** | **307K scratch aligned, 256d** |
| --- | ---: | ---: | ---: | ---: |
| Redshift | 0.38801 | 0.58819 | **0.57393 +/- 0.00072** | **0.63222 +/- 0.00051** |
| Stellar mass | 0.54051 | 0.77530 | **0.71761 +/- 0.00032** | **0.78151 +/- 0.00005** |
| sSFR | 0.42978 | 0.58828 | **0.53298 +/- 0.00009** | **0.57007 +/- 0.00002** |
| Metallicity | 0.25930 | 0.44331 | **0.42818 +/- 0.00006** | **0.44820 +/- 0.00018** |
| Stellar age | 0.10441 | 0.24863 | **0.25840 +/- 0.00012** | **0.27576 +/- 0.00038** |

The learned-query post-trained spectrum pooler beats the old v2 projector on
all five tasks and even beats frozen raw CLS on all five: +0.01888 redshift,
+0.02006 mass, +0.03655 sSFR, +0.01939 metallicity, and +0.02620 age. Learned
token pooling therefore adds value for spectra. The image pooler does not show
the same benefit on redshift.

## Complete Current Metrics

### Image redshift

| Model / space | R2 | MAE | RMSE | NMAD | Outlier fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 307K post-trained raw | 0.52998 +/- 0.00033 | 0.05557 +/- 0.00009 | 0.10469 +/- 0.00004 | 0.04395 +/- 0.00020 | 0.28361 +/- 0.00119 |
| 307K post-trained aligned | 0.51175 +/- 0.00015 | 0.05676 +/- 0.00006 | 0.10670 +/- 0.00002 | 0.04388 +/- 0.00008 | 0.29632 +/- 0.00050 |
| 307K scratch raw | 0.62741 +/- 0.00041 | 0.04210 +/- 0.00008 | 0.09321 +/- 0.00005 | 0.03077 +/- 0.00011 | 0.16344 +/- 0.00057 |
| 307K scratch aligned | 0.61368 +/- 0.00033 | 0.04368 +/- 0.00008 | 0.09491 +/- 0.00004 | 0.03204 +/- 0.00008 | 0.17667 +/- 0.00072 |

### Spectrum raw

| Model | Target | R2 | MAE | RMSE |
| --- | --- | ---: | ---: | ---: |
| 307K post-trained | Redshift | 0.55506 +/- 0.00025 | 0.05267 +/- 0.00008 | 0.10186 +/- 0.00003 |
| 307K post-trained | Stellar mass | 0.69755 +/- 0.00038 | 0.26849 +/- 0.00018 | 0.36358 +/- 0.00023 |
| 307K post-trained | sSFR | 0.49642 +/- 0.00013 | 0.55351 +/- 0.00025 | 0.78985 +/- 0.00011 |
| 307K post-trained | Metallicity | 0.40879 +/- 0.00076 | 0.25048 +/- 0.00016 | 0.33247 +/- 0.00021 |
| 307K post-trained | Stellar age | 0.23220 +/- 0.00041 | 1.04477 +/- 0.00083 | 1.55009 +/- 0.00041 |
| 307K scratch | Redshift | 0.67556 +/- 0.00049 | 0.03818 +/- 0.00008 | 0.08698 +/- 0.00007 |
| 307K scratch | Stellar mass | 0.84260 +/- 0.00018 | 0.19632 +/- 0.00018 | 0.26229 +/- 0.00015 |
| 307K scratch | sSFR | 0.63919 +/- 0.00028 | 0.44805 +/- 0.00055 | 0.66858 +/- 0.00026 |
| 307K scratch | Metallicity | 0.54944 +/- 0.00029 | 0.21859 +/- 0.00006 | 0.29024 +/- 0.00009 |
| 307K scratch | Stellar age | 0.36579 +/- 0.00033 | 0.94968 +/- 0.00041 | 1.40881 +/- 0.00037 |

Spectrum raw redshift NMAD/outlier fraction are 0.03974/0.25700 for
post-training and 0.02813/0.12837 for scratch.

### Spectrum aligned

| Model | Target | R2 | MAE | RMSE |
| --- | --- | ---: | ---: | ---: |
| 307K post-trained | Redshift | 0.57393 +/- 0.00072 | 0.05003 +/- 0.00005 | 0.09968 +/- 0.00008 |
| 307K post-trained | Stellar mass | 0.71761 +/- 0.00032 | 0.25729 +/- 0.00007 | 0.35132 +/- 0.00020 |
| 307K post-trained | sSFR | 0.53298 +/- 0.00009 | 0.51891 +/- 0.00011 | 0.76065 +/- 0.00007 |
| 307K post-trained | Metallicity | 0.42818 +/- 0.00006 | 0.24571 +/- 0.00004 | 0.32697 +/- 0.00002 |
| 307K post-trained | Stellar age | 0.25840 +/- 0.00012 | 1.01954 +/- 0.00035 | 1.52342 +/- 0.00013 |
| 307K scratch | Redshift | 0.63222 +/- 0.00051 | 0.04189 +/- 0.00011 | 0.09261 +/- 0.00006 |
| 307K scratch | Stellar mass | 0.78151 +/- 0.00005 | 0.23115 +/- 0.00002 | 0.30902 +/- 0.00003 |
| 307K scratch | sSFR | 0.57007 +/- 0.00002 | 0.50060 +/- 0.00013 | 0.72982 +/- 0.00001 |
| 307K scratch | Metallicity | 0.44820 +/- 0.00018 | 0.24602 +/- 0.00004 | 0.32120 +/- 0.00005 |
| 307K scratch | Stellar age | 0.27576 +/- 0.00038 | 1.01739 +/- 0.00009 | 1.50549 +/- 0.00039 |

Spectrum aligned redshift NMAD/outlier fraction are 0.03708/0.23259 for
post-training and 0.03093/0.16052 for scratch.

## Geometry

| Encoder space | Dim | Effective rank | Participation ratio | Largest eigenvalue share |
| --- | ---: | ---: | ---: | ---: |
| 307K post-trained image raw | 1,024 | 37.03 | 15.71 | 0.183 |
| 307K post-trained image aligned | 256 | 8.83 | 7.28 | 0.229 |
| 307K post-trained spectrum raw | 768 | 13.67 | 9.59 | 0.208 |
| 307K post-trained spectrum aligned | 256 | 8.60 | 7.30 | 0.224 |
| 307K scratch image raw | 1,024 | 28.26 | 12.77 | 0.192 |
| 307K scratch image aligned | 256 | 20.18 | 17.14 | 0.099 |
| 307K scratch spectrum raw | 768 | 26.07 | 10.08 | 0.225 |
| 307K scratch spectrum aligned | 256 | 17.93 | 14.68 | 0.111 |

Compared with the 95K scratch checkpoint, raw effective rank rises from 9.64
to 28.26 for images and from 7.04 to 26.07 for spectra. Aligned rank rises from
8.58 to 20.18 for images and from 10.07 to 17.93 for spectra. The final scratch
spaces are clearly non-collapsed and substantially healthier geometrically.

## Interpretation

The result supports the practical feasibility hypothesis: cross-modal-only
LeJEPA alignment with per-modality distributed SIGReg can train both encoders
from scratch into useful, non-collapsed representations. On this frozen-ridge
battery, the final scratch model beats this post-training setup on every image
and spectrum target in both raw and aligned spaces.

It does not yet prove that SIGReg removes the value of pretraining. The scratch
model updated the full approximately 392M-parameter system for 60,000 steps;
the post-trained model updated only 2.23M pooler parameters for 24,000 steps.
The source backbones also had different pretraining histories. A causal claim
needs matched optimizer steps or samples, several training seeds, and the
planned retrieval and modality-retention controls.

There is also a transductive-evaluation limitation. Both new trainers consumed
all 307,428 self-supervised pairs while ignoring historical split labels, so
objects from the shipped AstroCLIP test split were available without downstream
labels during representation learning. These frozen probes remain useful for
model comparison, but they are not a strict unseen-object generalization test.
A definitive result needs an object-disjoint held-out pair split created before
self-supervised training.

## Artifacts

Machine-readable metrics:

- `results/crossmodal_scratch_307k_last_image/metrics.json`;
- `results/crossmodal_scratch_307k_last_spectrum/metrics.json`;
- `results/crossmodal_posttrained_307k_last_image/metrics.json`;
- `results/crossmodal_posttrained_307k_last_spectrum/metrics.json`.

Large embedding caches remain under
`/mnt/datasets/pranav/astrojepa_eval_cache`; no dataset was copied into the
repository.
