# Cross-Modal JEPA Decision Log

Last updated: 2026-08-24

## Purpose

This is the living design record for image-spectrum representation learning in
Astro-Worldmodels. It records the scientific question, architecture options,
loss definitions, data requirements, risks, evaluation plan, and decisions made
before implementation.

A draft cross-modal model, objective, loader, and trainer have been implemented,
but the data source is being revised before any training run. The immediate goal
remains a controlled experiment in which the image and spectrum encoders are
trained jointly from random initialization rather than aligned after pretraining.

## Scientific Question

The central question is:

> Can cross-modal agreement plus separate SIGReg regularization train useful
> image and spectrum backbones from random initialization, matching or beating
> backbones that first require unimodal pretraining and are aligned afterward?

This differs from the AstroCLIP recipe. AstroCLIP first pretrained image and
spectrum encoders separately, then aligned them with a contrastive objective.
That is an important baseline, but it cannot tell us whether cross-modal signal
would have changed the features learned during early backbone training.

The proposed from-scratch experiment initializes every backbone and projection
head randomly. The selected first experiment optimizes only a global
cross-modal objective. Image-only, spectrum-only, and local objectives remain
important ablations, but are deliberately excluded from this run.

### Primary hypothesis and claim boundary

The primary hypothesis is that cross-modal invariance supplies the learning
signal while SIGReg removes the collapse failure that would otherwise make
teacher-free training from random initialization degenerate. If successful,
the pair can learn the astrophysical factors shared by images and spectra
without first constructing two independently pretrained representation spaces.

SIGReg by itself is not claimed to create semantic content. It constrains the
population geometry; paired cross-modal prediction determines which information
is useful. The defensible claim is therefore not that SIGReg universally
eliminates pretraining, but that it may eliminate the collapse-driven need for
pretraining in this paired image-spectrum setting.

The strong result requires all three of the following:

1. Scratch training without SIGReg collapses or performs materially worse.
2. Scratch training with SIGReg remains geometrically healthy and learns useful
   cross-modal retrieval and downstream representations.
3. Scratch plus SIGReg matches or beats post-trained alignment on the same paired
   data and test split, while using no unimodal checkpoint initialization.

Matching only training loss or embedding variance is insufficient. Retrieval,
shared-property probes, unimodal backbone probes, and held-out geometry must all
be reported.

## Relevant Prior Work

- [AstroCLIP](https://arxiv.org/abs/2310.03024) demonstrates the value of a
  shared image-spectrum embedding, using separate self-supervised pretraining
  followed by contrastive alignment.
- [I-JEPA](https://arxiv.org/abs/2301.08243) motivates prediction in latent space
  instead of reconstructing pixels.
- [LeJEPA](https://github.com/galilai-group/lejepa) provides the teacher-free
  invariance plus SIGReg formulation already used by this project.
- [MJEPA](https://arxiv.org/abs/2606.25225) jointly trains audio-visual
  representations and reports that explicit cross-modal prediction is critical;
  a naive shared multimodal encoder degraded unimodal performance.

These papers support the direction, but none validates this exact astronomical
objective. The proposed model remains an experiment that must be tested against
strong unimodal and post-training-alignment controls.

## Definitions

### Post-trained alignment

1. Train an image encoder independently.
2. Train a spectrum encoder independently.
3. Load both checkpoints.
4. Align or fine-tune their representations using paired objects.

This remains a necessary baseline and is close to the AstroCLIP structure.

### Joint training from scratch

1. Initialize both encoders randomly.
2. Present only scientifically matched image-spectrum pairs and their views.
3. Optimize the selected cross-modal invariance and separate SIGReg terms.
4. Allow cross-modal gradients to shape both backbones.

A gradual increase in cross-modal loss weight is still from-scratch joint
training. It is an optimization curriculum, not post-trained alignment, because
neither encoder begins from a pretrained checkpoint and both belong to the same
run.

## Information Shared And Not Shared

Images and spectra observe the same astrophysical object but are not redundant.

Likely shared factors include:

- redshift;
- broad stellar-population properties;
- dust and continuum color;
- star-formation and emission activity;
- broad galaxy type and mass-related structure.

Important image-private factors include:

- orientation and projected morphology;
- bars, spiral arms, mergers, and neighbors;
- spatially resolved color structure;
- foreground and background image context.

Important spectrum-private factors include:

- detailed line ratios and equivalent widths;
- narrow line profiles and velocity information;
- wavelength-local calibration and uncertainty structure;
- features too subtle to infer from broadband imaging.

The cross-modal objective should therefore operate on a dedicated shared
representation. It should not force every backbone dimension or every local
token to match across modalities.

## Recommended First Architecture

Use two modality-specific encoders with their existing architectural families:

```text
image views -> image encoder -> image CLS and image patch tokens
                                  |
                                  +-> image cross-modal head -> z_image

spectrum views -> spectrum encoder -> spectrum CLS and wavelength tokens
                                        |
                                        +-> spectrum cross-modal head -> z_spectrum
```

Both encoders start from random weights and are optimized together. Their
backbone weights are not shared.

Each encoder has one modality-specific cross-modal head mapping its CLS output
into the shared space. There are no local heads in the selected first run.

The initial candidate shared dimension is 256. This is deliberately larger than
the current 64-dimensional unimodal training projector because the cross-modal
embedding is a deployed representation for retrieval and joint downstream use,
not merely a disposable anti-collapse head. The dimension remains an open
ablation against 64, 128, and 512.

The full 768-dimensional backbone output remains available for unimodal tasks.
The shared cross-modal output is used for image-spectrum retrieval and shared
physical-property probes.

### Why not share the transformer immediately?

A more radical model could use modality-specific input stems and positional
embeddings followed by one shared transformer trunk. This is closer to MJEPA.
It is scientifically interesting, especially because the current spectrum model
already uses a ViT-like 12-layer transformer.

It should not be the first experiment. A shared trunk changes both the training
regime and the architecture, making it difficult to determine whether gains or
failures came from joint training or parameter sharing. It also increases the
risk of modality interference. It should be a separately named follow-up after
the two-encoder joint baseline is healthy.

## Input Views

For paired object `b`, construct two image views and two spectrum views:

```text
hI[1,b], hI[2,b] = image encoder outputs
hS[1,b], hS[2,b] = spectrum encoder outputs

zI[v,b] = image cross-modal head(hI_cls[v,b])
zS[v,b] = spectrum cross-modal head(hS_cls[v,b])
```

Image views should use the accepted image masking and augmentation policy.
Spectrum views should use the DESI v2 validity, mean/std normalization,
uncertainty noise, and artificial wavelength masking contract.

The views must preserve object identity. Augmentations that erase the shared
astrophysical signal would make cross-modal agreement impossible or encourage
shortcuts.

## Unimodal Objectives

Unimodal global and local losses were considered for the first joint run. The
selected experiment does **not** include them. This makes the scientific
question cleaner: can paired cross-modal consistency plus SIGReg alone shape
useful image and spectrum backbones from random initialization?

This choice intentionally gives up use of unpaired examples and does not
directly protect modality-private information. Those are properties to measure,
not assumptions to hide with auxiliary losses. The independently pretrained
then aligned model will test the alternative in which unimodal structure is
learned before cross-modal alignment.

## Direct Cross-Modal LeJEPA Objective

The selected objective uses only cross-modality pairings. For two image views
and two spectrum views:

```text
L_cross_invariance = mean over b,v,w of
    ||z_image[v,b] - z_spectrum[w,b]||^2

(v, w) in {(1,1), (1,2), (2,1), (2,2)}
```

There is no explicit image-to-image or spectrum-to-spectrum term. Both image
views must agree with both spectrum views, so within-modality consistency is
induced transitively through the paired modality. Gradients flow through both
sides of every pair; there is no stop-gradient, EMA teacher, or target encoder.

### Cross-modal SIGReg

Apply SIGReg to each modality separately:

```text
L_cross_sigreg = 0.5 * SIGReg(zI) + 0.5 * SIGReg(zS)

L = (1 - lambda_cross) * L_cross_invariance
    + lambda_cross/2 * (SIGReg(z_image) + SIGReg(z_spectrum))
```

Separate regularization matters. A SIGReg term computed only on a concatenated
image-spectrum population could look healthy while one modality collapses or
the modalities occupy different parts of the distribution.

The initial candidate is `lambda_cross = 0.05`, matching the existing LeJEPA
setting. This is a starting point, not an accepted full-run value.

Do not L2-normalize the cross embeddings during SIGReg training. SIGReg is
trying to shape an isotropic Gaussian distribution, whereas unit normalization
forces embeddings onto a hypersphere. L2 normalization can be applied at
retrieval time if cosine similarity is evaluated.

## Total Joint Objective

For the selected first experiment, the equation above is the complete training
objective. There are no `alpha_image`, `alpha_spectrum`, local-loss, or
cross-loss curriculum coefficients. Optimization still uses learning-rate
warmup, but the full cross-modal objective is active from step one.

## What Bidirectional Prediction Means

With direct Euclidean alignment:

```text
||z_image - z_spectrum||^2
```

"image to spectrum" and "spectrum to image" are the same mathematical loss.
Writing it twice changes only its scalar weight because both representations
already receive gradients.

Genuinely directional prediction requires separate predictors:

```text
predicted_spectrum = P_image_to_spectrum(z_image)
predicted_image = P_spectrum_to_image(z_spectrum)

L_directional = ||predicted_spectrum - z_spectrum||^2
                + ||predicted_image - z_image||^2
```

In a teacher-free version, both targets remain trainable and SIGReg is applied
to the target embedding distributions. There is no stop-gradient or EMA target.

Directional predictors are a valuable second experiment because each mapping
can represent modality asymmetry. They should not be part of the first direct
alignment run: they add parameters, can absorb coordinate mismatches that the
backbones fail to resolve, and complicate comparison with LeJEPA.

## Local Cross-Modal Learning

There is no natural one-to-one correspondence between an image patch and a
wavelength patch. Directly aligning image token `(x,y)` with spectrum token
`lambda_j` would be physically unjustified.

The first model should therefore use:

- local image-to-image LeJEPA;
- local spectrum-to-spectrum LeJEPA;
- global image-spectrum LeJEPA.

A later model can introduce `K` shared latent query slots. Image tokens and
spectrum tokens would each predict the same indexed slots, while modality-local
tokens remain private. The slots might learn shared factors such as stellar
population, dust, emission activity, or morphology, but those interpretations
must be tested rather than assumed.

Another later option is conditional target-token prediction:

- spectrum context predicts selected image target embeddings conditioned on 2D
  position;
- image context predicts selected spectrum target embeddings conditioned on
  wavelength position.

This is more JEPA-like in the I-JEPA predictive sense, but it confronts real
conditional ambiguity. A spectrum does not determine galaxy orientation or
background pixels, and an image does not determine every narrow spectral line.
The direct global experiment should establish a healthy shared latent before
attempting this.

## Paired Data Audit And Contract

The selected objective can use only paired examples.

### Local DR8-DESI audit, not selected for training

The initially audited local sources were:

- images: `/mnt/datasets/utbd_pranav/galaxies/with_crops/*.parquet`;
- image coordinates: `/mnt/datasets/utbd_pranav/galaxies/metadata.parquet`;
- spectra: `/mnt/datasets/utbd_pranav/desi_edr_sv3/mmu_desi_edr_sv3/dataset/**/*.parquet`.

The image and spectrum rows are not distributed as pre-paired records. Image
rows use a Legacy Surveys DR8 `dr8_id`. DESI `object_id` is a packed TARGETID;
its `RELEASE` field is DR9 for the audited examples. Decoding TARGETID to
`BRICKID_OBJID` therefore does not produce a valid DR8 join key.

The candidate join was a nearest-neighbor sky-coordinate match using RA/Dec
with a maximum separation of 1 arcsec. The 2026-08-23 audit found:

- 8,689,370 rows in the image metadata catalog;
- 1,126,441 DESI EDR SV3 spectrum rows;
- 79,768 spectrum observations with an image within 1 arcsec;
- 78,826 unique matched images;
- 942 additional spectrum observations assigned to an already matched image;
- 116 matches with a second image candidate within 1 arcsec.

Ambiguous matches are excluded. Repeated spectra may be retained, but all rows
sharing one image identity must be assigned to the same split. Splits are made
by a stable hash of `dr8_id`, never independently by row, to prevent physical
object leakage.

A materialization attempt found that 153 matched image IDs, covering 154 pair
rows, were not present in the physical `with_crops` shards. It was stopped at
the user's direction before any paired cache was completed. The empty partial
output was removed. This custom DR8-DESI path is retained only as an audit and
is not the selected training source.

### Multimodal Universe native crossmatching

The original Multimodal Universe paper does not claim that every survey is
distributed as one pre-paired table. Each survey remains primarily unimodal.
The framework creates multimodal intersections by matching sky coordinates. Its
appendix demonstrates `cross_match_datasets(left, right, matching_radius=1.0)`
using locally downloaded MMU dataset builders and returning a new Hugging Face
dataset. MMU explicitly notes that cross-survey intersections can be much
smaller than the parent datasets.

MMU v1.5 now provides a preferable implementation through HATS and LSDB.
Surveys are partitioned into adaptive HEALPix Parquet tiles. LSDB identifies
overlapping tile pairs, runs a KD-tree angular match only inside those tiles,
and can stream result partitions lazily instead of materializing both parent
catalogs in memory.

A ready-made `EiffL/desi_legacysurvey_xmatch` dataset is also available. Its
dataset card reports 95,895 same-object galaxies and approximately 72 GB of
sharded data. It was built at 1 arcsec with LSDB by joining MMU DESI PROVABGS,
DESI EDR SV3 spectra, and Legacy Survey DR10 south imaging. It contains the
DESI spectrum struct, four-band 160x160 image products, an RGB PNG, match
distances, and PROVABGS properties. The median reported separation is about
0.01 arcsec.

On 2026-08-24, direct training on the ready-made MMU crossmatch was selected.
The downloaded and verified snapshot is:

- path: `/mnt/datasets/pranav/desi_legacysurvey_xmatch`;
- Hugging Face revision: `b18e891d67208fb81dd02d2a6ea0f5d759b6a11e`;
- 95,895 rows in 477 Parquet shards;
- 76.737 GB on disk.

The trainer streams native records directly and does not materialize another
copy. It decodes the 160x160 `rgb` PNG for the existing three-channel image
encoder and passes the native `spectrum` struct to the DESI v2 transform. The
spectrum arrays have the expected 7,781-pixel fixed grid.

Splits use a stable hash of `object_id_ls`, rather than row order, because this
is the physical image identity. The audit found 95,879 unique image IDs and 16
repeated rows; grouping on image identity prevents those repeats from crossing
splits. Seed 42 produced:

- train: 93,986 rows;
- validation: 945 rows;
- test: 964 rows.

A small per-file split index beside the dataset lets the iterable loader assign
files without overlap and balance ranks by actual training rows. With four
ranks and four workers per rank, row counts are `[23497, 23497, 23496, 23496]`.
At batch size 8, every rank therefore performs exactly 2,937 steps per epoch;
only two surplus training rows are dropped each epoch.

A ten-step production-sized smoke test passed on four A100 80 GB GPUs. It loaded
native records through all 16 workers, trained the 400,526,080-parameter model,
completed forward, distributed SIGReg, backward, clipping, optimizer, scheduler,
logging, and clean DDP shutdown without OOMs, NaNs, or data exhaustion. The final
checkpoint was written successfully and measured 4.5 GB.

A four-rank resume smoke test then restored model, optimizer, scheduler, scaler,
SIGReg state, RNG state, global step, and in-epoch position successfully.

## Training Curriculum

Both encoders start randomly. A provisional curriculum is:

1. Initialize both encoders and both shared-space projectors randomly.
2. Apply the complete cross-modal invariance plus SIGReg objective from step one.
3. Warm up the optimizer learning rate, not the objective weight.
4. Continue joint optimization of both encoders and heads for the full run.

No pretrained backbone, frozen teacher, EMA target, or post-training stage is
used in the primary experiment.

## Main Failure Modes

### Constant collapse

All objects map to one vector. Separate image and spectrum SIGReg, effective
rank, per-dimension standard deviation, and covariance diagnostics address this.

### Shortcut representation

SIGReg prevents constant collapse but does not guarantee rich astrophysical
content. The shared embedding could spread objects using only redshift, color,
signal-to-noise, calibration, or survey artifacts.

Monitor shared-neighbor consistency for morphology and physical properties,
probe each property separately, and train a modality classifier on the shared
embeddings. A powerful modality classifier indicates incomplete alignment.

### Loss domination

Cross-modal gradients could erase useful unimodal structure, or one encoder
could move much faster than the other. Track per-branch losses, gradient norms,
and frozen unimodal probes throughout training.

### Pair noise

Incorrect joins, duplicated objects, aperture mismatch, poor spectra, or image
contamination create false positives. A non-contrastive loss will still force
these pairs together, so data auditing is essential.

### Shared-information bottleneck

The shared head may improve redshift and retrieval while harming morphology or
detailed spectral tasks. Keep the full modality-specific backbone outputs and
judge them separately from the shared head.

## Evaluation Plan

### Cross-modal alignment

- image-to-spectrum and spectrum-to-image Recall@K;
- median retrieval rank;
- paired distance versus shuffled-pair distance;
- retrieval conditioned on redshift bins and object class;
- physical-property consistency among cross-modal nearest neighbors.

### Unimodal representation quality

- frozen image redshift, morphology, and property probes;
- frozen spectrum redshift, classification, and property probes;
- image and spectrum local-feature diagnostics;
- comparison of full backbone CLS against the shared cross-modal head.

### Collapse and geometry

- feature standard deviation and covariance;
- effective rank;
- SIGReg values for each modality separately;
- modality classification accuracy in the shared space;
- CKA or canonical-correlation diagnostics between paired representations.

### Missing-modality behavior

- image-only inference;
- spectrum-only inference;
- optional fused inference if a later fusion model is introduced.

## Required Experimental Controls

1. From-scratch cross-modal LeJEPA with separate image and spectrum SIGReg.
2. The identical scratch run with SIGReg disabled, testing whether collapse
   prevention is causally necessary.
3. The pretrained image and spectrum checkpoints followed by alignment with the
   same shared heads, paired data, and cross-modal objective.
4. The independently pretrained checkpoints before alignment, measuring what
   alignment adds and what it erases.
5. A scratch run with a different explicit anti-collapse regularizer, testing
   whether the result is specific to SIGReg rather than regularization generally.
6. From-scratch directional predictors as a later objective ablation.
7. A shared-transformer model as a later architecture ablation.

The comparison between controls 1 and 3 is the main scientific result; control
2 is the causal SIGReg ablation. Paired-stage data, split, shared dimension,
augmentations, optimizer budget, and evaluation must match. Total unimodal
pretraining data and compute must also be reported separately rather than hidden
from the comparison.

## Selected First Experiment

- Train two separate modality encoders jointly from random initialization.
- Use two image views and two spectrum views.
- Align only the four global image-spectrum combinations.
- Apply SIGReg separately to image and spectrum shared embeddings.
- Use no local, image-image, or spectrum-spectrum objective.
- Use no EMA teacher, stop-gradient, negatives, or directional predictor.
- Keep both full backbone CLS outputs and the shared-space projections in
  checkpoints.
- Train only on scientifically matched pairs and split by physical image ID.
- Treat local objectives, unimodal objectives, directional predictors, shared
  slots, and shared transformers as named follow-up ablations.

## Checkpoint Policy

- Write a full resumable `step_<global_step>.pt` every 1,000 optimizer steps and
  retain the latest two periodic snapshots.
- Write a full resumable `epoch_<epoch_number>.pt` after every completed epoch
  and retain the latest five epoch snapshots by default.
- Overwrite `last.pt` at every epoch boundary and controlled early stop so it
  remains the normal recovery checkpoint.
- Do not label a partial epoch caused by `--max-steps` as an epoch checkpoint.
- Make epoch retention configurable with `--keep-last-epoch-checkpoints` to
  bound local storage use.

## Remaining Decisions For A Full Run

1. Choose the existing 95,895-row paired dataset or a broader lazy LSDB
   crossmatch as the training source.
2. Choose the shared embedding dimension after the smoke test; implementation
   default is 256.
3. Choose the full training step or epoch budget.
4. Choose the primary checkpoint-selection metric.
5. Define compute-matched post-trained and no-cross controls.
6. Decide whether cross-modal retrieval uses Euclidean distance, cosine
   similarity, or both during evaluation.

## Decision History

### 2026-08-23

- Established joint image-spectrum training from random initialization as the
  principal research direction.
- Distinguished joint-from-scratch training from post-trained alignment.
- Proposed separate modality backbones with a shared latent space.
- Proposed global symmetric cross-modal LeJEPA with separate SIGReg terms.
- Kept local objectives within modality for the first experiment.
- Reserved directional predictors, shared latent slots, and a shared transformer
  for controlled follow-up studies.
- Superseded the initial recommendation to retain unimodal/local losses: the
  selected first experiment is cross-modal global alignment plus SIGReg only.
- Selected the all-four-pair cross-invariance equation rather than the four-view
  center equation.
- Completed the local data audit and selected a 1 arcsec positional join because
  DR8 image IDs cannot be joined directly to DR9 DESI TARGETIDs.
- Authorized implementation and a 1,000-step smoke test, but not a full run.
- Implemented and unit-tested the exact cross-only objective and two-encoder
  training draft; no smoke training was launched.
- Stopped the custom local pair-cache build at the user's direction. No cache
  was completed, and its empty partial output was removed.
- Reviewed the MMU paper's coordinate-crossmatch contract and the newer HATS /
  LSDB route.
- Identified `EiffL/desi_legacysurvey_xmatch` as an existing 95,895-row DESI
  spectrum plus Legacy DR10 image candidate, pending data-source approval.

### 2026-08-25

- Added full resumable snapshots after every completed cross-modal epoch.
- Selected five retained epoch snapshots alongside two retained periodic step
  snapshots and the rolling `last.pt` checkpoint.

### 2026-08-29

- Audited all paired and coordinate-matchable data physically present under the
  local dataset mounts.
- Confirmed that the current native MMU DESI plus Legacy Survey DR10 cross-match
  was built with a 1.0 arcsec LSDB radius and contains 95,895 rows.
- Confirmed that the locally loaded AstroCLIP mirror contains 197,976 pairs:
  138,583 train, 29,696 validation, and 29,697 test. No DESI target IDs overlap
  the native MMU paired set.
- The benchmark-safe union of the two existing training splits is therefore
  232,569 distinct paired observations, subject to implementing a loader that
  harmonizes their different schemas.
- Re-ran a nearest-neighbor coordinate audit against 8,689,370 local DR8 image
  coordinates. Acceptance requires the nearest image within 1.0 arcsec and no
  second image within that radius.
- DESI yields 79,652 accepted spectrum rows over 78,718 image identities; SDSS
  yields 563,267 over 563,267 identities; VIPERS W1 yields only 34.
- The combined coordinate manifest can contain 642,953 spectrum-image rows over
  631,878 unique image identities. DESI and SDSS share 10,141 image identities,
  while DESI contributes 934 repeated spectrum observations.
- These are pre-materialization ceilings. Missing physical image rows, quality
  cuts, duplicate policy, and schema validation can reduce the usable count.
- Retained 1.0 arcsec as the default radius. Moving from 1.0 to 2.0 arcsec adds
  only 676 accepted DESI and 1,988 accepted SDSS rows while increasing ambiguous
  matches from 116 to 204 and 1,686 to 2,788, respectively.
- Target-ID de-duplication showed that the 79,652 manual DESI matches overlap
  31,436 native MMU pairs and 34,637 AstroCLIP pairs. Only 13,579 are new beyond
  every loaded pre-matched split.
- The strict training-safe DESI union is 246,148 distinct objects: 232,569 from
  the existing MMU and AstroCLIP training splits plus 13,579 new manual matches.
  Including held-out splits produces 307,450 unique DESI target IDs, not the
  naive sum of approximately 374k.
- Adding all 563,267 accepted SDSS rows gives a naive training ceiling of
  809,415 spectrum-image observations before physical-object de-duplication.
  At least 10,141 image identities have both DESI and SDSS observations.
- The SDSS expansion is scientifically valuable but not drop-in compatible with
  the fixed-grid DESI encoder; wavelength, resolution, masks, and survey
  selection must be harmonized for a multi-survey run.

### 2026-08-29: combined DESI scratch run

The next scratch experiment intentionally uses every locally runnable DESI-image
pair, without preserving the source datasets' historical train, validation, or
test labels. This is a representation-pretraining corpus; downstream evaluation
must use separately controlled object-level splits.

The final materialized collection contains 307,428 pair rows:

| Source | Rows | Spectrum contract | Image contract |
| --- | ---: | --- | --- |
| AstroCLIP mirror, all source splits | 197,976 | DESI flux only | raw g/r/z nanomaggies |
| MMU DESI x Legacy DR10 | 95,895 | flux, ivar, wavelength, LSF, mask | rendered RGB PNG |
| New manual DESI x DR8 matches | 13,557 | flux, ivar, wavelength, LSF, mask | DR8 crop |
| **Total** | **307,428** | fixed 7,781-pixel DESI grid | two 140-pixel views |

The coordinate audit originally found 13,579 new target IDs after excluding
66,073 targets already present in AstroCLIP or MMU. During payload
materialization, 21 DR8 image IDs were unavailable, removing 22 pair rows. No
spectrum payload was missing. The runnable total is therefore 307,428 rather
than the pre-materialization ceiling of 307,450.

Source harmonization is explicit:

- AstroCLIP raw nanomaggy images are converted with AstroCLIP's Legacy Survey
  g/r/z arcsinh RGB mapping before the common astronomy image augmentations.
- All spectra use per-spectrum valid-pixel mean/std normalization and the same
  7,781-pixel DESI token grid.
- MMU and manual spectra retain pipeline masks and ivar-based noise views.
- AstroCLIP has flux only. Finite pixels are valid, and uncertainty noise is
  disabled rather than creating synthetic ivar.
- The loader logs the source mixture and the fraction of examples carrying
  physical uncertainty information.

The old Hugging Face streaming loader had a material epoch-accounting bug. It
manually assigned files to workers and Hugging Face then sharded those files
again internally. The prior 95k run therefore stopped each nominal epoch after
only a fraction of its indexed rows. The new collection loader streams Parquet
directly, balances files once across ranks and workers, and computes epoch length
from worker-usable batches. On eight GPUs with batch size 32 per GPU, every rank
has exactly 1,200 usable batches per epoch. Each epoch consumes 307,200 of
307,428 rows; the sub-0.1% file-balance remainder rotates because file and
row-group order are reshuffled each epoch.

The official LeJEPA distributed SIGReg behavior was checked against the authors'
repository. Our vendored implementation already matches it: random slice seeds
are synchronized, Epps-Pulley cosine and sine moments use autograd-aware
distributed averaging, and the statistic applies the required local-N times
world-size scaling. Full embedding all-gather is neither needed nor added.

Selected training configuration:

- two image views and two spectrum views;
- all four cross-modal global MSE pairings;
- separate image and spectrum SIGReg terms;
- no image-image, spectrum-spectrum, or local-patch objective;
- ViT-L image encoder and 12-layer, 768-dimensional DESI transformer from
  scratch;
- batch size 32 per GPU, global batch 256 on eight GPUs;
- 50 true epochs, 1,200 steps per epoch, 60,000 optimizer steps;
- AdamW, learning rate 1e-4, weight decay 0.05, 1,000-step warmup, cosine decay;
- W&B run name
  `CrossModalScratch_DESI307K_AllPairs_CrossOnly_SIGReg`;
- periodic checkpoints every 2,500 steps, retaining two;
- rolling `last.pt` after every epoch;
- named epoch checkpoints every five epochs, retaining five.

Validation completed before release:

- Python compilation passed for the trainer, loaders, transforms, and builder.
- Cross-modal unit/regression tests passed.
- Real samples from all three source formats produced identical model-facing
  image and spectrum tensor shapes.
- A two-step, eight-A100 distributed smoke test completed forward, distributed
  SIGReg, backward, gradient clipping, optimizer update, scheduler update, and
  final checkpoint writing without error.

### 2026-08-29: AstroCLIP post-training alignment and our LeJEPA analogue

This section records what AstroCLIP actually does after unimodal pretraining and
defines possible adaptations for our independently pretrained image and
spectrum encoders. These are discussion proposals only. No alignment trainer
has been implemented and no option has yet been selected.

Primary sources:

- [AstroCLIP paper](https://academic.oup.com/mnras/article/531/4/4990/7697182)
- [official AstroCLIP repository](https://github.com/PolymathicAI/AstroCLIP)
- [official alignment configuration](https://github.com/PolymathicAI/AstroCLIP/blob/main/configs/astroclip.yaml)
- [official model and CLIP-loss implementation](https://github.com/PolymathicAI/AstroCLIP/blob/main/astroclip/models/astroclip.py)

#### What AstroCLIP aligns

AstroCLIP first pretrains its two modalities separately. Its image transformer
uses a DINOv2-style objective and its spectrum transformer uses masked spectral
reconstruction. At alignment time, it removes the old unimodal projection heads
and uses the final token sequences from both pretrained transformers.

Each modality receives its own learned pooling head:

```text
final image tokens    -> learned-query cross-attention -> MLP -> z_image
final spectrum tokens -> learned-query cross-attention -> MLP -> z_spectrum
```

The learned query attends over every final-layer token and reduces a token
sequence to one fixed-dimensional vector. The paper describes four attention
heads, two linear layers, LayerNorm, GeLU, and a 512-dimensional shared
representation. The current released code defaults to a 1,024-dimensional head,
so the paper and current repository differ on this detail.

The most important implementation detail is that the released `ImageHead` and
`SpectrumHead` both default to `freeze_backbone=True`, and the released alignment
configuration does not override that setting. The closest description of the
public recipe is therefore: freeze the two pretrained transformers and train new
cross-attention/MLP alignment heads. It is not primarily end-to-end post-training
of both backbones.

AstroCLIP L2-normalizes the two head outputs and applies symmetric InfoNCE:

```text
L_AstroCLIP = 0.5 * CE(image-to-spectrum logits, paired indices)
            + 0.5 * CE(spectrum-to-image logits, paired indices)
```

The matching image and spectrum are positives; other objects in the similarity
matrix are negatives. The logit scale is fixed at 15.5 in the released model.
The paper motivates 1,024 image-spectrum pairs for the contrastive set. The
released configuration instead specifies batch size 256 while the README
describes four-GPU training. Its loss code contains no explicit distributed
feature all-gather, so this batch/negative-count detail should not be copied
without an explicit implementation decision.

Published artifacts also disagree on alignment duration: the repository README
reports 25,000 steps on four A100s, the current config specifies 100 epochs, and
the journal appendix reports a different iteration count. Our experiments must
state exact data exposures and optimizer steps from our own logs.

#### Direct LeJEPA adaptation

The closest teacher-free, non-contrastive analogue keeps AstroCLIP's two-stage
structure while replacing InfoNCE and its negatives with our cross-modal
invariance plus SIGReg objective:

```text
pretrained image backbone    -> new image shared head    -> z_image
pretrained spectrum backbone -> new spectrum shared head -> z_spectrum
```

Load the selected ViT-L image checkpoint and completed spectrum-v2 checkpoint.
Discard the old unimodal training projectors because they were objective-facing
heads and raw backbone CLS is the retained downstream representation. Initialize
two new 256-dimensional cross-modal heads.

For two views of each modality, use the exact objective used by the scratch run:

```text
L_cross = mean over v,w in {1,2} of
          ||z_image[v] - z_spectrum[w]||^2

L = (1 - lambda) * L_cross
    + lambda/2 * (SIGReg(z_image) + SIGReg(z_spectrum))
```

SIGReg should remain separate by modality. A single SIGReg term over pooled
modalities could look healthy while one modality is low-rank or carried by the
other. No negatives, temperature, queue, EMA teacher, stop-gradient, local loss,
image-image loss, or spectrum-spectrum loss is required for the primary
post-trained LeJEPA control.

Do not L2-normalize embeddings before the training MSE or SIGReg. SIGReg shapes
the projected distribution toward a non-collapsed Gaussian-like target, which
is incompatible with constraining every vector to a unit sphere. L2
normalization can be applied after training for cosine retrieval.

#### Pooling-head options

**CLS-to-MLP heads:** This is the recommended first comparison. Both of our
encoders were explicitly trained to put global information in CLS, raw CLS has
been the strongest spectrum-v2 downstream representation, and using the same
head as the scratch model isolates initialization from architecture.

**AstroCLIP-style learned-query attention heads:** Attend over all final image
or wavelength tokens before the shared MLP. This could recover spectral-line or
image-patch information missed by CLS, but it changes the architecture. It must
be applied to both scratch and pretrained models before attributing a gain to
pretraining.

**Directional JEPA predictors:** Predict spectrum shared features from image
features and image shared features from spectrum features. This permits
modality-specific coordinate systems and may handle asymmetric information
better, but adds predictor capacity and is a later objective ablation rather
than the clean first comparison.

#### Backbone-trainability options

1. **Frozen backbones:** Train only the two new shared heads. This is the closest
   AstroCLIP analogue and measures how alignable the existing representations
   already are.
2. **Fully trainable from step zero:** Initialize from both pretrained
   checkpoints but optimize the same parameters as the scratch model. This is
   the cleanest test of whether pretrained initialization itself is necessary.
3. **Head warmup then partial unfreeze:** Train heads first, then unfreeze the
   last few blocks with a smaller backbone learning rate. This is likely the
   strongest practical post-trained model, but it is not the cleanest causal
   comparator.
4. **Adapters or LoRA-style residual updates:** Keep backbone weights fixed and
   learn small updates inside the last blocks. This is a parameter-efficient
   middle ground, but it introduces a new architecture and should not replace
   the simpler controls.

The primary scientific comparison should be scratch versus fully trainable
pretrained initialization. Frozen-head AstroCLIP-style alignment is a useful
practical baseline, but not a complete initialization control because its
trainable parameter count and optimization problem differ.

All variants should use the same 307,428 paired rows, views, shared dimension,
loss, batch size, optimizer budget, and evaluation protocol as the scratch run.
Reports must separately state the unpaired data and compute consumed to produce
the pretrained checkpoints; sharing the paired set does not make total training
data or total compute equal.

#### Evaluation needed before making the claim

- Bidirectional image-to-spectrum and spectrum-to-image Recall@1, Recall@5,
  Recall@10, median rank, and mean reciprocal rank on object-disjoint pairs.
- Frozen probes from raw image CLS, raw spectrum CLS, and each shared head for
  redshift, stellar mass, sSFR, metallicity, age, and image morphology.
- Before-versus-after alignment representation drift and unimodal downstream
  retention.
- Per-modality feature standard deviation, effective rank, covariance, and
  SIGReg, plus paired versus mismatched embedding distances.
- Identical checkpoint selection and evaluation splits across scratch and every
  pretrained control.

Provisional recommendation for later implementation: derive a post-trained
alignment entry point from the current scratch trainer, load both backbone
checkpoints strictly, initialize fresh shared heads, and expose `frozen`, `full`,
and `partial` trainability modes. Use `full` for the primary causal comparison
and `frozen` for the closest AstroCLIP-style baseline. This remains subject to
review before any code is changed.

#### Clarification and selected next direction: token pooling during post-training

The selected next design to investigate is AstroCLIP-style learned-query token
pooling for the post-trained alignment model. This means loading the pretrained
image and spectrum backbones, removing their old objective projectors, exposing
their final CLS and patch/wavelength tokens, and attaching one new
modality-specific attention-pooling head to each backbone. No implementation is
authorized by this decision alone.

Token pooling and pretrained initialization are two separate experimental
variables. The current scratch run uses CLS-to-MLP heads. A post-trained model
using token-attention heads therefore differs from it in both initialization and
pooling architecture. Comparing those two models is a valid comparison of final
systems, but it cannot establish which difference caused a gain.

There is no immediate practical requirement to train the attention-pooling model
from scratch. A scratch attention-pooling run would serve only as an architecture
control:

```text
scratch CLS       vs scratch attention pooling    -> effect of pooling
pretrained CLS    vs pretrained attention pooling -> effect of pooling
scratch attention vs pretrained attention         -> effect of initialization
```

The reason to train any model from scratch remains the project's principal
scientific hypothesis: determine whether cross-modal invariance plus SIGReg can
learn useful image and spectrum backbones without separate unimodal pretraining.
The scratch run is unnecessary if the sole goal is to produce the strongest
post-trained model, but essential if claiming that SIGReg makes unimodal
pretraining unnecessary.

The current scratch CLS run should continue and is not wasted. The next practical
model can use pretrained backbones plus attention pooling. A scratch
attention-pooling run can be deferred until results show that the pooling choice
is important enough to warrant the extra control. For the cleanest immediate
initialization comparison using the already-running scratch model, a separate
pretrained CLS-to-MLP run will eventually still be required.

### 2026-08-29: implemented AstroCLIP-style pooling with LeJEPA alignment

The attention-pooling post-training setup is now implemented in separate files:

- `models/cross_modal_posttrained.py`
- `train-cross-modal-posttrained.py`
- `tests/test_cross_modal_posttrained.py`

Neither the scratch trainer nor either unimodal training script was modified for
this implementation.

The model reads these immutable source checkpoints:

```text
image:
checkpoints/VitLargePatch14_OfficialTrain5_Epoch5_2504/step_52000.pt

spectrum:
checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs/complete.pt
```

Loading is strict. The image loader accepts only `backbone.*` tensors and ignores
the old image `proj.*` objective head. The spectrum loader accepts only the
patch/validity embeddings, tokens, positional embeddings, transformer, and final
normalization; it ignores `global_proj.*` and `local_proj.*`. Architecture names,
key sets, and tensor shapes must match before training can begin.

Both source backbones are frozen and held in evaluation mode. The image backbone
emits all final image tokens. The spectrum backbone emits CLS plus all final
wavelength-patch tokens and supplies a key-padding mask so invalid spectral
patches cannot receive pooling attention.

Each modality has a separate AstroCLIP-style pooler:

```text
learned 256-dimensional query
    -> four-head cross-attention over final modality tokens
    -> LayerNorm and dropout
    -> 256 -> 1024 -> 256 residual GeLU MLP
    -> shared 256-dimensional projection
```

The poolers are modality-specific; their outputs occupy the same training space.
Two image views and two spectrum views produce four cross-modal pairings. The
training objective remains:

```text
L_cross = mean over v,w ||z_image[v] - z_spectrum[w]||^2

L = 0.95 * L_cross
    + 0.025 * SIGReg(z_image)
    + 0.025 * SIGReg(z_spectrum)
```

There are no negatives, temperature, queue, EMA teacher, stop-gradient, local
losses, or same-modality losses. SIGReg uses the existing distributed LeJEPA
implementation.

Parameter accounting from the real model is:

| Component | Total parameters | Trainable parameters |
| --- | ---: | ---: |
| Image branch | 305,550,336 | 1,182,720 |
| Spectrum branch | 86,440,192 | 1,051,648 |
| **Combined** | **391,990,528** | **2,234,368** |

The checkpoint format deliberately saves only new information:

- image attention-pooler weights;
- spectrum attention-pooler weights;
- optimizer, scheduler, scaler, and both SIGReg states;
- exact source checkpoint paths, sizes, modification times, steps, and epochs;
- epoch/step position and random-number-generator states.

Frozen image and spectrum tensors are not copied into alignment checkpoints.
Every output path is required to reside under the new alignment `save_dir`, and
the trainer rejects any configuration placing a source checkpoint inside that
directory. Resume also rejects a checkpoint whose recorded sources differ from
the currently loaded source files. The raw unimodal files are therefore read
only and cannot be overwritten through the alignment save path.

Validation completed:

- Python compilation and CLI construction passed.
- Attention masking tests proved that padded spectrum tokens do not affect the
  pooled result.
- Gradient tests proved that only the two poolers receive gradients.
- Strict synthetic source-loading and head-only state tests passed.
- All existing cross-modal scratch regression tests still pass.
- A two-step real-data DDP smoke test passed on four A100s, including loading the
  two production checkpoints, parameter synchronization, token pooling,
  distributed SIGReg, backward, clipping, optimizer/scheduler updates, and
  checkpoint writes.
- A subsequent four-GPU resume test restored the head weights, optimizer,
  scheduler, SIGReg states, RNG state, epoch position, and global step from
  `last.pt`, then completed the next optimizer step successfully.
- Smoke checkpoints were approximately 26 MB each and contained no backbone or
  transformer keys. The two source checkpoint sizes and modification times were
  unchanged after the test.

Default full-run schedule: all 307,428 paired rows, batch size 32 per GPU, four
workers per rank, 10 epochs, AdamW at 1e-4 with weight decay 0.05, 1,000-step
warmup and cosine decay, 2,400 steps per four-GPU epoch, and 24,000 total steps.


## Final 307K Evaluation Outcome

The completed scratch and post-trained `last.pt` checkpoints were evaluated on
the frozen image-redshift and five frozen spectrum probes. The primary raw R2
results favor scratch on every task. The learned-query post-trained spectrum
pooler improves every task over frozen v2 raw CLS, while its image pooler does
not improve image redshift. Scratch embedding ranks also improve strongly over
the earlier 95K checkpoint, confirming that the final spaces are non-collapsed.

This is evidence that the proposed SIGReg-supported from-scratch objective
works, not yet proof that it eliminates the value of pretraining. The required
next control is an object-disjoint pair split with matched optimization budgets
and multiple training seeds, followed by bidirectional retrieval and
modality-retention evaluations.

Complete results: `Evals/cross_modal_backbone_eval/FINAL_307K_COMPARISON.md`.

## 2026-08-31: from-scratch CLIP control

We decided to add a third controlled experiment before restarting the paused
spectrum CPT run. The image CPT run remains active on GPUs 0-3; after the image
checkpoint-series evaluation releases GPUs 4-7, those GPUs will train a joint
image-spectrum CLIP model from random initialization.

This experiment answers a narrower question than the earlier AstroCLIP-style
post-training run: given the same paired objects and the same randomly
initialized encoders, does contrastive instance discrimination outperform the
noncontrastive cross-modal LeJEPA plus SIGReg objective?

The controlled variables are inherited exactly from the completed 307K scratch
SIGReg run:

- all 307,428 pairs from AstroCLIP, MMU DESI-Legacy crossmatches, and the manual
  DESI DR8 crossmatch;
- ViT-L/14 image encoder and 12-layer, 768-dimensional DESI spectrum encoder;
- two global image views and two masked/noisy spectrum views;
- 256-dimensional modality projectors;
- batch size 32 per GPU, four GPUs, and global batch size 128;
- AdamW, learning rate 1e-4, weight decay 0.05, 1,000-step warmup, cosine decay,
  bfloat16, and 50 epochs.

Only the objective changes. For every image-view/spectrum-view combination, the
trainer computes symmetric image-to-spectrum and spectrum-to-image InfoNCE.
The four symmetric losses are averaged. Embeddings are L2 normalized, the
temperature is learned from an initial value of 0.07, and its multiplicative
logit scale is capped at 100. Differentiable all-gather makes all 128 objects in
the DDP-global batch available to each pairing, giving every anchor 127
negatives. SIGReg, same-modality losses, EMA teachers, and local token losses are
absent.

This is not the frozen-backbone AstroCLIP post-training model. Both complete
backbones and both projectors start randomly and receive gradients. Comparing it
with scratch LeJEPA plus SIGReg isolates the anti-collapse/alignment mechanism;
comparing either scratch model with post-training still confounds initialization
and the number of trainable parameters.

Implementation:

- trainer: `train-cross-modal-scratch-clip.py`;
- learned-temperature model: `models/cross_modal_scratch_clip.py`;
- local tests: `tests/test_cross_modal_scratch_clip.py`;
- distributed-gradient test: `tests/test_cross_modal_scratch_clip_distributed.py`;
- output: `checkpoints/CrossModalScratch_DESI307K_AllPairs_CLIP_InfoNCE`;
- planned W&B run ID: `cm307kscratchclipv1`.

Checkpoint retention is deliberately small. `last.pt` is overwritten every
epoch, only the newest periodic step checkpoint is retained, and only the two
newest five-epoch snapshots are retained. At the approximately 4.8 GB size of a
full scratch training state, peak retained storage should be about 19 GB.

Verification before the real-data smoke test:

- both new Python modules compile in `lejepa-og`;
- the exact four-pairing loss equals an independently computed reference;
- gradients reach image features, spectrum features, and learned temperature;
- the logit-scale cap works;
- all original scratch-JEPA regression tests still pass;
- a two-process Gloo test confirms finite forward/backward through
  differentiable global all-gather.

The subsequent one-GPU real-data smoke test discovered all 307,428 paired rows,
balanced them evenly across four loader workers, and completed two optimizer
steps. It wrote a valid 4.5 GB resumable checkpoint at global step 2 containing
both encoders, both projectors, optimizer/scheduler/scaler state, and the learned
temperature. The logit scale moved from 14.2857 to 14.2843, directly confirming
that the temperature received an update. A four-GPU launch check remains queued
until the checkpoint-series image evaluation releases GPUs 4-7.

The full four-GPU run started on 2026-08-31 in tmux session
`crossmodal-scratch-clip`. It uses physical GPUs 4-7, 2,400 steps per epoch, and
120,000 total steps. W&B run ID `cm307kscratchclipv1` is online at
<https://wandb.ai/pranavktrpl-personal/astrojepa/runs/cm307kscratchclipv1>.
Initial distributed training is healthy: all four GPUs reached full utilization
at approximately 25-26 GB each, loss fell from about 5.04 to 3.91 by step 190,
and the paired-to-shuffled distance ratio fell from 0.996 to 0.782. The separate
image CPT run continued unchanged on physical GPUs 0-3. The paused spectrum CPT
run remains deferred until this newer scratch-CLIP request releases GPUs 4-7.

The first full epoch completed successfully in about 13 minutes. Both
`step_2400.pt` and rolling `last.pt` were written at approximately 4.5 GB each,
their metadata records global step 2,400, and epoch 2 began without interruption.
Loss was approximately 2.1 shortly after the restart of the data stream. The
observed end-to-end rate implies roughly 11 hours for 50 epochs, including
checkpoint overhead.
