# Astro-Worldmodels Research Progress Report

**Prepared for discussion on 2 September 2026**  
**Project status covered:** through 1 September 2026  

## 1. Executive Summary

Astro-Worldmodels studies self-supervised representation learning for galaxy
images and optical spectra, followed by joint image-spectrum learning. The
project has progressed through four main stages:

1. **Image-only LeJEPA baselines.** A ViT-L/14 trained from scratch on millions
  of unlabeled galaxy images learned useful morphology features. It reached
   71.83% Galaxy10 accuracy and 70.23% macro-F1 with a frozen linear probe.
2. **A corrected DESI spectrum backbone.** Spectrum v2 repaired the baseline
  data representation, epoch accounting, invalid-pixel treatment, and lack of
   wavelength-local supervision. It improved four of five physical-property
   probes substantially over spectrum v1.
3. **Cross-modal training from scratch and after pretraining.** Joint image and
  spectrum encoders were trained from random initialization on 307,428 paired
   objects using either cross-modal invariance plus SIGReg or CLIP/InfoNCE.
   Both produced non-collapsed representations. Both scratch models
   substantially outperformed frozen-backbone post-training adapters on the
   current downstream battery.
4. **JEPA versus CLIP representation study.** CLIP was better at exact pair
  retrieval and spectrum-private information, while JEPA+SIGReg produced much
   stronger global cross-modal geometry and linear predictability. The result
   is a tradeoff between different notions of alignment, not a universal win
   for one objective.

The strongest current spectrum model is the 307K-pair CLIP model trained from
scratch. The strongest current image-redshift model is the 307K-pair
JEPA+SIGReg model trained from scratch. Spectrum redshift remains the largest
gap to AstroCLIP. Morphology remains the clearest image-only strength.

The project now supports a plausible paper centered on this result:

> Contrastive and regularized-invariance objectives learn different kinds of
> cross-modal alignment. CLIP emphasizes exact correspondence and local
> discriminability; cross-modal invariance plus SIGReg emphasizes globally
> shared, linearly transformable geometry.

Two controls are still essential before making a strong causal claim that
SIGReg removes the need for unimodal pretraining: a scratch run without SIGReg,
and a pretrained-initialization run that trains the same full architecture for
the same paired-data budget as the scratch model.

## 3. Project Scope And Terminology

The project is a representation-learning project, not currently a generative
world model. It learns encoders that turn an image or spectrum into reusable
features for morphology, redshift, physical-property prediction, retrieval,
clustering, and future multimodal tasks.

The local use of **LeJEPA** refers to a teacher-free objective with:

- direct agreement between augmented views or paired modalities;
- SIGReg, a sliced statistical regularizer that prevents constant collapse;
- no EMA teacher, stop-gradient target encoder, or pixel decoder.

The cross-modal JEPA experiments are therefore not canonical predictor-based
I-JEPA. In this report, **cross-modal JEPA+SIGReg** is shorthand for the exact
implemented objective: direct image-spectrum invariance plus separate SIGReg
for each modality.

## 4. Core Research Questions

The work completed so far addresses four questions suitable for organizing a
paper:

1. Can teacher-free LeJEPA learn scientifically useful galaxy image and DESI
  spectrum representations?
2. Does correcting spectrum preprocessing and adding wavelength-local learning
  materially improve a spectrum backbone without changing its architecture or
   raw dataset?
3. Can useful image and spectrum encoders be trained jointly from scratch using
  paired data, rather than requiring separate unimodal pretraining?
4. How do noncontrastive invariance plus SIGReg and contrastive InfoNCE differ
  in pair alignment, geometry, cross-modal predictability, and retained
   scientific information?



## 5. Data Used

All primary datasets and large embedding caches are stored under
`/mnt/datasets/utbd_pranav`. The old `/mnt/datasets/pranav` path is now a
symlink to this consolidated directory. No raw dataset is duplicated inside
the repository.

### 5.1 Training and evaluation populations


| Dataset                                         | Count                                            | Role in this project                                                  |
| ----------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------- |
| `Smith42/galaxies`, local `galaxies/with_crops` | 8,474,566 training objects                       | Unimodal image pretraining using the `image_crop` field               |
| MMU DESI EDR/SV3                                | 1,126,441 spectra                                | Spectrum v1 and v2 pretraining on a fixed 7,781-pixel wavelength grid |
| AstroCLIP paired mirror, all source splits      | 197,976 pairs                                    | One component of the 307K cross-modal training union                  |
| MMU DESI-Legacy DR10 cross-match                | 95,895 pairs                                     | Initial 95K pilot and one component of the 307K union                 |
| Materialized DESI-DR8 manual cross-match        | 13,557 pairs                                     | New non-overlapping runnable pairs added to the 307K union            |
| **Combined DESI cross-modal training set**      | **307,428 pairs**                                | All completed 307K scratch and post-training experiments              |
| AstroCLIP benchmark mirror                      | 168,280 pairs                                    | Frozen evaluation population: 138,583 train and 29,697 test           |
| PROVABGS matched subset                         | 89,334 valid objects                             | Physical-property probes: 73,341 train and 15,993 test                |
| Galaxy10 DECaLS                                 | 17,736 images                                    | Ten-class galaxy morphology and a historical redshift probe           |
| Public GZD-5 split                              | about 224K usable images before question filters | Galaxy Zoo question-wise morphology evaluation                        |


The ViT-L image baseline completed five streamed passes, corresponding to
about 42.2 million source-image presentations and 421.9 million augmented
views. These are repeated augmented exposures, not 42.2 million unique
galaxies.

### 5.2 Cross-matching work

The coordinate audit used a **1.0 arcsec** nearest-neighbor radius and rejected
ambiguous cases with a second candidate inside the radius. This was selected as
a reasonable completeness-purity point: reducing the radius to 0.5 arcsec lost
useful matches, while increasing it to 2 arcsec added few pairs and more
ambiguity.

The full DESI-to-DR8 coordinate audit found 79,652 accepted rows, but most were
already represented in AstroCLIP or the native MMU cross-match. After removing
overlap and rows whose image payload was not physically available, 13,557 new
pairs were materialized and used. This distinction is important: 79K candidate
matches did not become 79K new training objects.

### 5.3 Available but not yet used for completed multi-survey training


| Dataset                                | Approximate rows | Current status                                                                                           |
| -------------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------- |
| MMU SDSS spectra                       | 806,176          | Available locally; not compatible with the fixed DESI-grid encoder without a multi-survey representation |
| Coordinate-matched SDSS-DR8 candidates | 563,267          | Potential paired expansion; not materialized into a completed training run                               |
| MMU VIPERS W1                          | 60,528           | Available locally; candidate for spectrum v3                                                             |
| VIPERS W4 tree                         | 60,528 reported  | Suspected duplicate or misdownload until identifiers or hashes are audited                               |


A multi-survey spectrum model must handle different wavelength grids,
resolution, calibration, masks, and selection functions. These observations
cannot be appended safely to the current DESI-only sequence format without
changing the data representation.

## 6. Model And Training History



### 6.1 Image-only models


| Model/run                                    | Architecture and objective                                            | Status and scientific role                                                                             |
| -------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| Early ResNet9 baseline                       | Small convolutional backbone, two global views, LeJEPA+SIGReg         | Completed at 8K steps; establishes the earliest image baseline                                         |
| `FirstTrain_VitSmallPatch14_2104`            | First ViT-S/14 attempt                                                | Historical prototype; some files are corrupt and it is excluded from final model selection             |
| `VitSmallPatch14_2204`                       | ViT-S/14, two global and eight local image crops, LeJEPA+SIGReg       | Completed at 21K steps; valid small-transformer baseline                                               |
| `VitLargePatch14_OfficialTrain5_Epoch5_2504` | ViT-L/14, 1,024-dimensional backbone output, multi-crop LeJEPA+SIGReg | Completed at 54,940 steps; selected checkpoint is step 52K                                             |
| Historical cross-match-adapted ViT-L         | ViT-L further adapted to the paired image distribution                | Completed and used in the original benchmark report; not the source used in later frozen post-training |
| `ViTL14_LeJEPA_CPT10_LocalMMU_FromStep52000` | Continued image training from step 52K on locally loaded MMU images   | One clean additional epoch completed at cumulative step 62,988, then intentionally stopped             |


The main image architecture is a roughly 304M-parameter ViT-L/14 initialized
from scratch. Each galaxy is transformed into two large views and eight local
views with galaxy-preserving crop, rotation, reflection, blur, and noise
augmentations. A 64-dimensional training projector receives the LeJEPA+SIGReg
objective; the 1,024-dimensional frozen backbone representation is retained for
downstream evaluation.

An image global-plus-local token objective was discussed but has not been
trained as a completed experiment. The current image objective aligns pooled
representations from global and local crops rather than corresponding image
patch tokens.

### 6.2 Spectrum-only models


| Model/run                                                  | Architecture and objective                                                                               | Status and scientific role                                                                                                    |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `SPECTRA_run0` to `SPECTRA_run4`                           | Flux-only DESI transformer experiments with several batch/debug configurations                           | Historical v1 development; only run3/run4 retain checkpoints                                                                  |
| Spectrum v1 benchmark                                      | 12-layer, 768-dimensional transformer with non-overlapping 20-pixel patches; global pooled LeJEPA+SIGReg | Historical scientific baseline, reported from the old step-15,647 model                                                       |
| Initial spectrum v2 run                                    | Corrected data contract and global+local LeJEPA                                                          | Reached 39,047 steps, but a stream-sharding bug made nominal epochs incomplete; retained only as an invalid accounting record |
| `SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs` | Corrected spectrum v2 trained from scratch for ten true DESI passes                                      | Completed at 172,390 steps; final unimodal spectrum backbone                                                                  |
| `SpectraV2...CPT10_FromEpoch10`                            | Planned continuation of v2 for ten additional epochs                                                     | Started and then paused before a usable checkpoint; no scientific result                                                      |
| Spectrum v2 overlap variant                                | Same v2 objective with overlapping wavelength patches                                                    | Implemented and tested in separate files, but not yet trained as a full run                                                   |


Spectrum v2 keeps the original 12-layer, 768-dimensional architecture and fixed
DESI grid, but changes what the model is asked to learn:

- flux remains the physical encoder signal;
- true bad/invalid pixels are separated from artificial JEPA masks and padding;
- inverse variance is used for validity and realistic noise, not as an input
channel or an unjustified latent-loss weight;
- wavelength is implicit in fixed token position;
- mean/std normalization is computed only over valid pixels;
- two global CLS representations are aligned with global SIGReg;
- corresponding wavelength-patch representations are aligned with a local
LeJEPA loss and position-aware local SIGReg.

There is no EMA teacher, stop-gradient branch, flux reconstruction decoder, or
contrastive negative set in spectrum v2. The 768-dimensional raw CLS is the
downstream representation; the 64-dimensional projectors are training heads.

### 6.3 Cross-modal models

All completed 307K models use separate modality-specific encoders: a ViT-L/14
for images and a 12-layer, 768-dimensional transformer for spectra. The full
scratch system contains roughly 392M parameters. Two image views and two
spectrum views are created for every paired object.


| Experiment                          | Initialization and trainable components                                                                | Cross-modal objective                                                            | Training status                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| 95K scratch JEPA+SIGReg pilot       | Both full encoders random and trainable                                                                | All four positive image-spectrum view MSE pairings plus separate modality SIGReg | Completed at 8,396 steps; epoch counter was inaccurate, so it is identified by steps/examples seen |
| 307K scratch JEPA+SIGReg            | Both full encoders random and trainable                                                                | Same cross-only invariance+SIGReg objective; no unimodal or local loss           | Completed 50 true epochs, 60K steps                                                                |
| 307K post-trained JEPA+SIGReg       | Frozen step-52K image and final spectrum-v2 backbones; 2.23M learned-query pooler parameters trainable | Same cross-only invariance+SIGReg objective                                      | Completed 10 epochs, 24K steps; adapter-only checkpoints preserve source weights                   |
| 307K post-trained AstroCLIP/InfoNCE | Same frozen source backbones; AstroCLIP-style learned-query adapters trainable                         | Symmetric image-to-spectrum and spectrum-to-image InfoNCE                        | Completed 10 epochs, 12K steps; adapter-only checkpoint                                            |
| 307K scratch CLIP                   | Both full encoders random and trainable                                                                | Four-view symmetric global-batch InfoNCE, no SIGReg                              | Completed 50 epochs, 120K steps                                                                    |


The scratch JEPA objective intentionally contains no image-to-image,
spectrum-to-spectrum, or local loss. Within-modality view agreement is induced
transitively because both views must agree with the same opposite-modality
views. This isolates whether paired invariance plus anti-collapse regularization
can shape both backbones from random initialization.

The post-trained models use AstroCLIP-style learned-query token pooling so the
alignment heads can attend over all final image or wavelength tokens. Frozen
source weights are never written into their output directories. The JEPA and
InfoNCE post-training runs therefore test two alignment losses on the same
pretrained features, but they are not matched to scratch in trainable parameter
count or training duration.

## 7. Image Evaluation Results



### 7.1 Galaxy10 morphology

Frozen linear probes were trained on the ten-class Galaxy10 DECaLS dataset.
Checkpoint selection used validation macro-F1 rather than test performance.


| Backbone     | Selected step | Test accuracy       | Test macro-F1       |
| ------------ | ------------- | ------------------- | ------------------- |
| ResNet9      | 8,000         | 44.80% +/- 1.75     | 42.69% +/- 1.70     |
| ViT-S/14     | 21,000        | 61.19% +/- 1.42     | 59.34% +/- 1.16     |
| **ViT-L/14** | **52,000**    | **71.83% +/- 0.84** | **70.23% +/- 0.90** |


The ViT-L result is close to the 71.4% DINOv2 reference recorded in the project
materials, but below the 87.2% AION-1-L result. Those external values use
different representation histories and downstream heads, so they are context,
not a strict matched leaderboard.

A two-layer MLP reached about 72.4% accuracy, only a small gain over the linear
probe. The remaining morphology gap is therefore not mainly explained by the
linear head being too weak.

### 7.2 Galaxy Zoo DECaLS question-wise morphology

The public GZD-5 pipeline produced mean accuracy 75.21% and mean F1 68.90% for
the frozen step-52K ViT-L. The historical AstroCLIP reference is 76.1% accuracy
and 74.3% F1.

Several protocol variants were explored. An early AstroCLIP-style reproduction
matched 76.1% mean accuracy but had a majority-class/protocol failure and must
not be used as a clean headline result. A stricter corrected run recorded
65.5% +/- 1.3 mean F1. The safe paper practice is to report the exact public
split and protocol with the result rather than combine these variants.

### 7.3 Image redshift and training progression


| Image representation                               | Redshift R2 | Interpretation                                      |
| -------------------------------------------------- | ----------- | --------------------------------------------------- |
| Original ViT-L complete, fixed-seed matched probe  | 0.53023     | Raw pre-alignment backbone                          |
| Original ViT-L best step, step 40K                 | 0.53167     | Best point on a broad plateau                       |
| Local-data CPT after one extra epoch               | 0.53128     | Only +0.00099 over its step-52K source              |
| Official pre-alignment AstroDINO, same local probe | 0.52888     | Raw external backbone is essentially matched        |
| Historical cross-match-adapted ViT-L               | 0.55062     | Separate adapted checkpoint                         |
| 307K scratch CLIP raw image backbone               | 0.61012     | Strong joint-training gain                          |
| **307K scratch JEPA+SIGReg raw image backbone**    | **0.62741** | Strongest current local image-redshift result       |
| AstroCLIP aligned image reference                  | 0.79        | Published context, not the raw AstroDINO checkpoint |


The checkpoint series shows that the original image model gained most of its
redshift information early and plateaued after roughly 24K steps. One extra
local-data epoch did not establish a new scaling regime. The original raw
backbone is already comparable to raw AstroDINO under the same frozen probe;
the large published AstroCLIP gain is associated with paired alignment and its
evaluation setting, not simply a better unaligned image ViT.

The historical Galaxy10 redshift probe restricted to `z < 0.25` reached
R2=0.748. It uses a different sample and should not be compared numerically to
the DESI-LS cross-match values above.

### 7.4 Image diagnostics

The evaluation suite also generated:

- train/validation/test LeJEPA, invariance, and SIGReg curves across image
checkpoints;
- per-class Galaxy10 recall and F1;
- patch-token PCA visualizations;
- layer-by-layer CLS attention maps;
- final-layer attention and attention rollout maps.

These diagnostics confirm that the ViTs attend to structured image regions and
that the late ViT-L checkpoints form a broad downstream plateau. They are useful
qualitative figures but are not independent quantitative claims.

## 8. Spectrum Evaluation Results

All values below are frozen raw-backbone ridge test R2 unless stated otherwise.


| Target       | Spectrum v1 | Spectrum v2 final | 95K scratch JEPA | 307K scratch JEPA | **307K scratch CLIP** | AstroCLIP reference |
| ------------ | ----------- | ----------------- | ---------------- | ----------------- | --------------------- | ------------------- |
| Redshift     | 0.432       | 0.555             | 0.642            | 0.676             | **0.678**             | 0.98                |
| Stellar mass | 0.506       | 0.698             | 0.823            | 0.843             | **0.846**             | 0.88                |
| sSFR         | 0.510       | 0.496             | 0.629            | 0.639             | **0.663**             | 0.64                |
| Metallicity  | 0.235       | 0.409             | 0.533            | 0.549             | **0.567**             | 0.58                |
| Stellar age  | 0.185       | 0.232             | 0.321            | 0.366             | **0.393**             | 0.43                |


Spectrum v2 materially improves redshift, mass, metallicity, and age over v1.
sSFR remains slightly below v1, showing that the v2 changes were not uniformly
beneficial. The cross-modal scratch encoders are stronger than the completed
unimodal v2 encoder on all five properties, despite seeing fewer unique spectra.

The strongest current spectrum model, scratch CLIP, nearly reaches the recorded
AstroCLIP references for mass, sSFR, metallicity, and age under the local ridge
protocol. Spectrum redshift remains dramatically behind.

A matched nonlinear probe on the scratch JEPA raw spectrum representation
raised redshift from 0.676 to 0.702 and age from 0.366 to 0.410. It also reached
0.854 mass, 0.580 metallicity, and 0.668 sSFR. This shows that much of the age
gap was probe nonlinearity, while the redshift gap is a genuine representation
deficit.

The likely reason is objective coverage. AstroCLIP's spectrum pretraining uses
overlapping wavelength patches and masked spectral reconstruction, making the
model infer line positions and breaks. Our current cross-modal scratch models
use only a global paired objective. They can ignore spectrum-only line detail
that cannot be inferred from a broadband image. Spectrum v2 adds local latent
agreement but still does not reconstruct or predict stable masked line targets.

## 9. Cross-Modal Downstream Comparison



### 9.1 Raw backbone representations


| Target            | Spectrum v2 / image source | 307K scratch JEPA | 307K scratch CLIP | Best current |
| ----------------- | -------------------------- | ----------------- | ----------------- | ------------ |
| Image redshift    | 0.530                      | **0.627**         | 0.610             | JEPA scratch |
| Spectrum redshift | 0.555                      | 0.676             | **0.678**         | CLIP scratch |
| Stellar mass      | 0.698                      | 0.843             | **0.846**         | CLIP scratch |
| sSFR              | 0.496                      | 0.639             | **0.663**         | CLIP scratch |
| Metallicity       | 0.409                      | 0.549             | **0.567**         | CLIP scratch |
| Stellar age       | 0.232                      | 0.366             | **0.393**         | CLIP scratch |




### 9.2 Objective-facing shared representations


| Target            | Post JEPA+SIGReg | Post InfoNCE | Scratch JEPA+SIGReg | Scratch CLIP |
| ----------------- | ---------------- | ------------ | ------------------- | ------------ |
| Image redshift    | 0.5118           | 0.5417       | **0.6137**          | 0.6016       |
| Spectrum redshift | 0.5739           | 0.5868       | 0.6322              | **0.6515**   |
| Stellar mass      | 0.7176           | 0.7482       | 0.7815              | **0.8169**   |
| sSFR              | 0.5330           | 0.5664       | 0.5701              | **0.6279**   |
| Metallicity       | 0.4282           | 0.4424       | 0.4482              | **0.5089**   |
| Stellar age       | 0.2584           | 0.2643       | 0.2758              | **0.3555**   |


The frozen-backbone InfoNCE adapter beats the frozen-backbone JEPA+SIGReg
adapter on every aligned-space probe. InfoNCE is therefore the better tested
loss for aligning already frozen token spaces.

Both scratch systems outperform both frozen-adapter systems. This is strong
evidence that joint training from scratch is viable and that allowing paired
gradients to shape the entire backbone matters. It is not yet proof that
pretraining is unnecessary, because the scratch models trained hundreds of
millions of parameters for 50 epochs while post-training updated only small
poolers for 10 epochs.

## 10. JEPA Versus CLIP Study

The dedicated study under `jepa vs clip/` compares the two completed 307K
scratch models on the same 168,280 paired evaluation population. Both models
used the same encoders, paired objects, two views per modality, 50 epochs, and
approximately 15.36M sample presentations. They were not perfectly
optimizer-matched: JEPA used a global batch of 256 and 60K steps, while CLIP
used a global batch of 128 and 120K steps.

### 10.1 Exact pair retrieval


| Model       | Direction         | Test R@1  | Test R@10 | Median rank |
| ----------- | ----------------- | --------- | --------- | ----------- |
| JEPA+SIGReg | Image to spectrum | 0.46%     | 3.31%     | 429         |
| JEPA+SIGReg | Spectrum to image | 0.32%     | 2.63%     | 454         |
| CLIP        | Image to spectrum | **1.49%** | **8.77%** | **230**     |
| CLIP        | Spectrum to image | **1.05%** | **7.19%** | **270**     |


On the full 168,280-candidate pool, bidirectional R@10 falls to about 0.63% for
JEPA and 2.12% for CLIP. Exact identity retrieval is difficult because many
galaxies are physically similar and some observations may be duplicates, but
CLIP is consistently better at instance-level correspondence.

### 10.2 Global geometry and cross-modal predictability


| Diagnostic                             | JEPA+SIGReg | CLIP       | Stronger model |
| -------------------------------------- | ----------- | ---------- | -------------- |
| Projected image-spectrum linear CKA    | **0.771**   | 0.634      | JEPA           |
| Pairwise-geometry Spearman correlation | **0.607**   | 0.485      | JEPA           |
| Held-out image-to-spectrum linear R2   | **0.749**   | 0.551      | JEPA           |
| Held-out spectrum-to-image linear R2   | **0.736**   | 0.549      | JEPA           |
| Cross-modal neighbor overlap, k=10     | 1.70%       | **2.62%**  | CLIP           |
| Cross-modal neighbor overlap, k=100    | 10.49%      | **10.97%** | CLIP, narrowly |


JEPA is much more globally shared and linearly transformable across modalities.
CLIP is stronger at exact and local-neighborhood discrimination. An orthogonal
Procrustes rotation improves both but does not make their spaces equivalent;
the difference also involves scaling, subspace selection, and modality-private
variation.

### 10.3 Information retained in each modality

When the same physical-property labels are probed from each raw backbone:

- JEPA wins all five image-side probes: redshift, mass, metallicity, age, and
sSFR.
- CLIP wins all five spectrum-side probes.
- CLIP has much higher effective rank in raw and projected spaces.
- Both models are decisively non-collapsed.

Representative effective ranks are 28.3 image and 26.1 spectrum for JEPA,
versus 64.3 image and 58.1 spectrum for CLIP. SIGReg prevents constant collapse
but does not force maximal rank or uniformity.

The clean interpretation is that CLIP retains a broader set of discriminative
and spectrum-private directions, while JEPA concentrates more strongly on
smooth factors shared between modalities. This explains how CLIP can retrieve
pairs better while JEPA has higher CKA and cross-modal linear predictability.

## 11. Complete Evaluation Inventory


| Evaluation family                     | Models covered                                                              | Main output or conclusion                                                              |
| ------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Streaming LeJEPA loss curves          | ResNet9, ViT-S, ViT-L checkpoints                                           | Training/validation/test loss, invariance, SIGReg, and generalization-gap trajectories |
| Galaxy10 checkpoint evolution         | ResNet9, every valid ViT-S and ViT-L checkpoint                             | Selected ViT-L step 52K by validation macro-F1                                         |
| Galaxy10 per-class analysis           | Selected ViT-L                                                              | Strong edge-on/smooth classes; disturbed and loose-spiral classes remain difficult     |
| Galaxy10 redshift                     | Historical image models                                                     | R2=0.748 on the restricted `z < 0.25` sample                                           |
| GZD-5 morphology                      | Selected ViT-L                                                              | Question-wise MLP probes with protocol caveats described above                         |
| ViT patch and attention diagnostics   | Selected ViT-S and ViT-L                                                    | Patch PCA, layer attention, final attention, and rollout figures                       |
| Image pre-alignment redshift series   | Every ViT-L 4K-step checkpoint, completion, CPT epoch 1, official AstroDINO | Shows early gains followed by a broad redshift plateau                                 |
| Spectrum frozen probes                | v1, v2 step 104K, v2 final                                                  | Redshift, mass, sSFR, metallicity, age, and geometry                                   |
| 95K cross-modal health evaluation     | Scratch JEPA pilot                                                          | Image redshift, five spectrum probes, and rank diagnostics                             |
| Final 307K frozen-backbone battery    | Scratch JEPA and post-trained JEPA                                          | Raw/shared probes, error metrics, and embedding geometry                               |
| Frozen InfoNCE post-training ablation | Post JEPA versus post InfoNCE                                               | InfoNCE improves every aligned-space downstream target                                 |
| Scratch CLIP battery                  | Scratch CLIP versus scratch JEPA                                            | CLIP wins spectrum tasks; JEPA wins image redshift                                     |
| Matched nonlinear probes              | Scratch JEPA raw/projected spaces                                           | Age gap mostly narrows; spectrum-redshift gap remains large                            |
| Exact bidirectional retrieval         | Scratch JEPA and scratch CLIP                                               | CLIP has substantially stronger pair retrieval                                         |
| CKA and pairwise geometry             | Scratch JEPA and scratch CLIP                                               | JEPA has stronger global cross-modal agreement                                         |
| Neighborhood overlap                  | Scratch JEPA and scratch CLIP                                               | CLIP has stronger local cross-modal neighborhoods                                      |
| Cross-modal ridge mapping and CCA     | Scratch JEPA and scratch CLIP                                               | JEPA modalities are much more linearly predictable                                     |
| Procrustes analysis                   | Scratch JEPA and scratch CLIP                                               | Differences are not explained by a rotation alone                                      |
| Decoder transfer across modalities    | Scratch JEPA and scratch CLIP                                               | Physical directions transfer after alignment, with property-specific tradeoffs         |
| Effective-rank/collapse audit         | v2, 95K, post-trained, scratch JEPA, scratch CLIP                           | Every final model is non-collapsed; CLIP uses more effective dimensions                |




## 12. Negative, Null, And Inconclusive Results

These outcomes should be retained because they clarify the paper narrative:

1. **Spectrum v1 was not a fair long-run baseline.** It used flux alone,
  conflated mask meanings, lacked true local wavelength supervision, and was
   severely undertrained.
2. **The first spectrum-v2 full run had invalid epoch accounting.** Its 39,047
  steps are not ten full DESI passes and must not be reported as the final v2
   model.
3. **The 95K scratch pilot had the same nominal-epoch problem.** It remains a
  useful 8,396-step proof of feasibility, but not a true 50-epoch model.
4. **More identical image training gave little redshift improvement.** The
  original checkpoint series plateaued, and one continuation epoch improved
   only about 0.001 R2 over its source.
5. **Frozen JEPA+SIGReg post-training hurt image redshift.** The image pooler
  fell from about 0.530 raw to 0.512 aligned, although the spectrum pooler
   improved all five spectrum targets.
6. **InfoNCE fixed only part of frozen post-training.** It improved every
  aligned probe over JEPA post-training but did not approach the scratch
   models, showing that loss choice alone does not overcome limited frozen
   source features and adapter capacity.
7. **Overlap has not yet been tested scientifically.** The spectrum overlap
  script exists, but no completed checkpoint or downstream result supports a
   claim about its benefit.
8. **Spectrum continuation has no result.** The CPT process was interrupted
  before a usable checkpoint and should not appear as a completed model.



## 13. Scientific Interpretation



### 13.1 What is already well supported

- The image LeJEPA pipeline learns useful morphology representations and is
competitive with raw AstroDINO on a matched pre-alignment redshift probe.
- Correcting spectrum preprocessing and adding global plus local latent losses
materially improves a weak flux-only baseline.
- Paired image-spectrum training from random initialization can produce useful,
non-collapsed 392M-parameter encoders.
- On the present data and probe battery, joint scratch training is stronger
than frozen-backbone adapter alignment.
- CLIP and JEPA+SIGReg optimize measurably different notions of alignment.



### 13.2 What is not yet supported

- It is not yet proven that SIGReg is the reason scratch training succeeds,
because there is no otherwise-identical scratch run without SIGReg.
- It is not proven that pretraining has no value. The existing post-training
models freeze the backbones and train far fewer parameters for fewer epochs.
- It is not proven that JEPA is universally better for images or CLIP is
universally better for spectra beyond this dataset and single training seed.
- Published AstroCLIP and AION values are not all measured with identical
probe heads, splits, or pretraining populations.
- The current results are not strict unseen-object representation results,
because the 307K self-supervised union includes identities later present in
the historical evaluation split.



## 14. Main Limitations

1. **Transductive representation training.** Historical evaluation identities
  were present without labels in the 307K training union. No downstream labels
   leaked, but a final paper needs an object-disjoint pair split before
   self-supervised training.
2. **One representation-training seed per final objective.** Probe-seed
  stability does not measure variance from retraining a 392M-parameter model.
3. **JEPA and CLIP are not fully compute-controlled.** They have the same epochs
  and sample exposure but different global batches and optimizer-step counts.
4. **No SIGReg ablation.** This is the most important missing test for the
  original anti-collapse hypothesis.
5. **No matched full-backbone pretrained control.** Frozen attention adapters
  are practical AstroCLIP analogues, not a clean test of pretrained
   initialization versus random initialization.
6. **Fixed DESI spectrum representation.** SDSS and VIPERS cannot yet be used
  without a common-grid or wavelength-aware multi-survey model.
7. **Spectrum-local objective remains weaker than masked line prediction.** The
  dominant redshift gap is unlikely to close through more global-only paired
   training alone.
8. **External comparison mismatch.** Ridge, kNN, linear, and MLP values should
  be reported in separate columns rather than treated as interchangeable.



## 19. Bottom Line

The project has moved beyond a single baseline into a coherent experimental
program. The image encoder is a credible morphology backbone; spectrum v2
demonstrates that scientifically correct preprocessing and wavelength-local
learning matter; and the 307K scratch experiments show that large joint
image-spectrum models can be trained successfully without unimodal checkpoint
initialization.

The most interesting result is not simply that one loss has a higher average
score. CLIP and JEPA+SIGReg organize multimodal information differently. CLIP
is better for exact identity matching, local neighborhoods, spectrum-side
physical information, and broad feature rank. JEPA+SIGReg is better for global
cross-modal geometry, bidirectional linear predictability, and image-side
physical probes. That distinction can form the conceptual center of the paper,
provided the next experiments establish object-disjoint generalization and
separate the effects of SIGReg, initialization, and compute.