# AstroCLIP InfoNCE Post-Training Ablation

Date: 2026-08-31

## Experiment

This ablation freezes the same AstroJEPA image and spectra-v2 backbones used by
the earlier LeJEPA plus SIGReg post-training run. Only AstroCLIP-style
learned-query attention adapters are trained on the same 307,428 paired DESI
objects for 10 epochs.

The alignment objective follows AstroCLIP's released implementation:

- L2-normalized 512-dimensional image and spectrum embeddings;
- fixed logit scale 15.5;
- symmetric image-to-spectrum and spectrum-to-image InfoNCE;
- global batch 256, providing 255 in-batch negatives per anchor;
- AdamW at `1e-4`, weight decay `0.05`, 1,000-step warmup, then cosine decay.

SIGReg is deliberately absent so this isolates contrastive alignment from the
previous noncontrastive objective. The run completed 12,000 optimizer steps.

## Evaluation Protocol

The comparison uses the unchanged frozen-embedding ridge protocol:

- AstroCLIP parquet mirror: 138,583 train and 29,697 test objects;
- PROVABGS subset: 73,341 train and 15,993 test objects;
- 10 seeds for image redshift and 3 seeds for spectrum properties;
- train-split standardization and validation-selected ridge regularization.

Uncertainty is standard deviation over probe split seeds. All values are test
R2. The raw source is identical between post-training methods because both keep
the unimodal backbones frozen.

## Results

| Target | Raw source | Prior JEPA+SIGReg | **New InfoNCE** | Delta | Scratch aligned | AstroCLIP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image redshift | 0.52998 | 0.51175 | **0.54174 +/- 0.00025** | **+0.02999** | 0.61368 | 0.79 |
| Spectrum redshift | 0.55506 | 0.57393 | **0.58684 +/- 0.00053** | **+0.01291** | 0.63222 | 0.98 |
| Stellar mass | 0.69755 | 0.71761 | **0.74821 +/- 0.00007** | **+0.03060** | 0.78151 | 0.88 |
| sSFR | 0.49642 | 0.53298 | **0.56640 +/- 0.00009** | **+0.03343** | 0.57007 | 0.64 |
| Metallicity | 0.40879 | 0.42818 | **0.44243 +/- 0.00026** | **+0.01425** | 0.44820 | 0.58 |
| Stellar age | 0.23220 | 0.25840 | **0.26426 +/- 0.00052** | **+0.00586** | 0.27576 | 0.43 |

## Geometry

| Aligned space | Dim | Effective rank | Participation ratio | Largest eigenvalue share |
| --- | ---: | ---: | ---: | ---: |
| Prior JEPA+SIGReg image | 256 | 8.83 | 7.28 | 0.229 |
| **InfoNCE image** | 512 | **12.01** | **8.88** | **0.212** |
| Prior JEPA+SIGReg spectrum | 256 | 8.60 | 7.30 | 0.224 |
| **InfoNCE spectrum** | 512 | **12.85** | **9.61** | **0.164** |

The dimensions differ, so effective rank is not perfectly controlled. Even so,
the InfoNCE adapters retain a broader and less top-heavy shared space.

## Interpretation

AstroCLIP-style instance discrimination is clearly better for frozen-backbone
post-training than direct similarity plus SIGReg. It improves every aligned
probe, reverses the image adapter's prior redshift degradation, and produces
healthier shared-space geometry.

This resolves the earlier apparent contradiction. Cross-only similarity plus
SIGReg worked better from scratch because gradients could reshape every
backbone token. With frozen token spaces, small adapters face a harder
one-to-one alignment problem, and positive-pair MSE does not explicitly
separate mismatched objects. InfoNCE supplies that missing discrimination.

The new adapters still trail the 307K scratch model on every target, although
the spectrum sSFR and metallicity gaps are small. The largest deficits remain
redshift and stellar age. Likely contributors include weaker source backbones,
only 307K paired objects, different pretraining/data curation, and our linear
ridge comparison to AstroCLIP's published probe results.

This is an AstroCLIP-method ablation, not a bit-identical reproduction. It uses
AstroCLIP's released adapter topology and symmetric InfoNCE recipe with our
backbones and data. The paper describes a 1,024-entry queue, while released
code uses current-batch logits with batch size 256; this run follows the
released executable path.

The same transductive limitation as the preceding 307K runs applies: all pairs
were used for self-supervised alignment, including objects later present in the
probe test split. A definitive result requires an object-disjoint split before
representation training.

## Artifacts

- Adapter-only checkpoint:
  `checkpoints/CrossModalPostTrain_DESI307K_AstroCLIP_InfoNCE/last.pt`
- Checkpoint size: 28,372,437 bytes; no backbone or optimizer tensors.
- W&B run: `cm307kpostclipv1`
- Image metrics:
  `results/crossmodal_posttrained_clip_307k_last_image/metrics.json`
- Spectrum metrics:
  `results/crossmodal_posttrained_clip_307k_last_spectrum/metrics.json`

