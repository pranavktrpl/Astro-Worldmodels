# Spectrum V2 Overlapping-Patch Experiment

Date: 2026-08-30

## Preservation boundary

The completed spectrum-v2 implementation and its checkpoints remain unchanged:

- `train-spectra-v2.py`
- `data/SpectraTransforms.py`
- `data/dataloaders.py`
- `tests/test_spectra_v2.py`
- `checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs/`

The overlap experiment is isolated in:

- `train-spectra-v2-overlap.py`
- `data/SpectraTransformsOverlap.py`
- `data/dataloaders_overlap.py`
- `tests/test_spectra_v2_overlap.py`

Its default W&B run name is
`SpectraV2_DESI_GlobalLocalLeJEPA_OverlapStride10`, and its checkpoints go to
`checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_OverlapStride10/`.

## Geometry change

The spectrum still has 7,781 pixels and each token still contains 20 adjacent
pixels. Patch starts now advance by 10 pixels:

```text
patch 0: pixels 0..19
patch 1: pixels 10..29
patch 2: pixels 20..39
...
patch 776: pixels 7760..7779
```

This gives 777 patches with 50% overlap. Pixel 7,780 remains unused, exactly as
the final pixel was unused by the 389-patch v2 representation.

The overlap doubles sequence length from 389 to 777 patch tokens and changes
the positional embedding from `[1, 390, 768]` to `[1, 778, 768]`. Old v2
checkpoints are intentionally incompatible, and the overlap run starts from
scratch. Attention compute and activation memory increase substantially because
self-attention scales approximately quadratically with token count.

## Objective deliberately unchanged

The overlap trainer is copied from spectrum v2. Its
`compute_lejepa_loss` implementation is unchanged:

```text
L = L_global + L_local

L_global = 0.95 * global_view_invariance + 0.05 * global_SIGReg
L_local  = 0.95 * local_view_invariance  + 0.05 * local_SIGReg
```

Global invariance aligns the two projected CLS representations for the same
spectrum. Local invariance aligns projected patch representations at the same
absolute patch index across the two noise/mask views. Local comparisons remain
restricted to scientifically valid target patches, and local SIGReg remains
sampled over 32 wavelength positions.

This is latent representation matching, not flux reconstruction. There is no
decoder, no target flux MSE, no EMA teacher, no stop-gradient branch, no
predictor, and no contrastive negatives. Mask ratio, mask spans, noise scale,
loss weights, SIGReg settings, optimizer, and training duration are unchanged
so the experiment isolates patch overlap as closely as possible.

## Cross-modal objectives already used

### From-scratch cross-modal training

Both encoders and both MLP projection heads were initialized from scratch and
trained end to end. Two image views and two spectrum views produced projected
embeddings. All four positive cross-modal combinations were aligned:

```text
L_cross = mean over v,w of ||z_image[v] - z_spectrum[w]||^2

L = 0.95 * L_cross
    + 0.05/2 * (SIGReg(z_image) + SIGReg(z_spectrum))
```

There were no image-image, spectrum-spectrum, local-token, reconstruction, EMA,
teacher-student, or negative-pair losses. SIGReg prevented marginal collapse,
but did not require different galaxies to be separated or any particular
physical variable to be retained.

### Post-trained cross-modal alignment

The image and spectrum backbones were loaded from their unimodal checkpoints
and frozen. Their original pretraining projectors were discarded. Each frozen
token sequence was pooled by a trainable AstroCLIP-style learned query with
four-head cross-attention and a residual MLP.

The resulting image and spectrum vectors used the exact same cross-modal
LeJEPA + SIGReg equation as the scratch run. Only the two attention poolers were
trained. The source backbone weights were neither updated nor overwritten.

The two experiments therefore differed in initialization and trainable
parameters, not in their alignment loss:

- scratch: optimize both full backbones plus projection heads;
- post-trained: optimize only learned-query pooling/alignment heads over frozen
  unimodal backbones.
