# AstroJEPA Project Conversation and Experiment Log

Last updated: 2026-08-25

This document is the durable record of the project discussion to date. It is
not a verbatim chat transcript. It records the questions, technical options,
decisions, implementations, experiment provenance, and unresolved issues that
matter for reproducing and interpreting the work.

## 1. Project Direction

AstroJEPA is exploring LeJEPA-style representation learning for astronomical
images and spectra, followed by paired image-spectrum learning. The central
research question for the paired model is whether separate pretrained
backbones are necessary, or whether a jointly trained cross-modal model can
learn useful representations from scratch when SIGReg prevents collapse.

The intended experimental ladder is:

1. Establish honest image-only and spectrum-only baselines.
2. Improve each unimodal backbone without changing its basic architecture.
3. Compare post-trained alignment of pretrained encoders against simultaneous
   image-spectrum training from scratch on exactly the same paired set.
4. Test whether cross-modal SIGReg can replace the need for unimodal
   pretraining, while preserving useful modality-specific information.
5. Scale the successful recipe to the complete Multimodal Universe data.

## 2. Baseline Checkpoint

The original image trainer, spectrum trainer, and associated data-loading code
were archived under `baseline_train_runs_2026-08-10/`. That directory is the
project checkpoint for the first completed baseline runs. New development is
kept outside the archive.

The first benchmark report found:

| Evaluation | Baseline result | Report comparator |
| --- | ---: | ---: |
| Cross-match image redshift R2 | 0.551 | AstroCLIP image 0.79 |
| Galaxy10 redshift R2, z < 0.25 | 0.748 | No external value reported |
| Galaxy10 morphology accuracy | 0.724 | AION-1-L 0.872 |
| GZD-5 reproduced accuracy | 0.761 | AstroCLIP 0.761 |
| Spectrum redshift R2 | 0.432 | AstroCLIP spectrum 0.98 |
| Spectrum stellar-mass R2 | 0.506 | AstroCLIP spectrum 0.88 |
| Spectrum sSFR R2 | 0.510 | AstroCLIP spectrum 0.64 |
| Spectrum metallicity R2 | 0.235 | AstroCLIP spectrum 0.58 |
| Spectrum age R2 | 0.185 | AstroCLIP spectrum 0.43 |

These are frozen-backbone probes. The report did not contain image-to-spectrum
retrieval or any other direct cross-modal evaluation.

## 3. Image-Backbone Discussion

The image architecture was judged fundamentally usable, but the original run
was data-light and undertrained relative to AstroCLIP and AION. The main
training-level ideas discussed were:

- global and local LeJEPA losses instead of supervising only the CLS summary;
- SIGReg on global and local projected representations;
- tuned AdamW weight decay, with no decay on biases, normalization parameters,
  positional embeddings, or special tokens;
- longer cosine training with warmup and a lower final learning rate;
- layer-wise learning-rate decay and careful gradient clipping;
- stochastic depth, modest dropout, and physically appropriate image views;
- validation-based checkpoint selection using downstream probes and embedding
  geometry, rather than selecting only by pretraining loss;
- training on substantially more unique images when the full MMU image corpus
  becomes available.

The key conclusion was that more epochs on the original image sample could
help, but could not replace the missing faint and diverse image population.
The report attributed much of the cross-match redshift deficit to magnitude
20-21 objects underrepresented by the original image pretraining distribution.

## 4. Spectrum-Backbone V2 Decision

The first spectrum baseline used only flux and had seen less than one complete
pass through its intended data. Its representation had low effective rank and
partial collapse. Spectrum v2 was therefore designed as a controlled DESI-only
experiment before introducing multi-survey complications.

The selected v2 data contract is:

- `flux` is the transformer signal;
- `mask` marks invalid or pipeline-flagged pixels and remains distinct from
  JEPA masks and padding;
- `ivar` controls uncertainty-aware noise and can support loss weighting;
- `lambda` is retained and validated but is implicit in token position on the
  fixed DESI grid;
- `lsf_sigma` is retained for future line-width and resolution experiments;
- redshift is never used during normalization.

### Normalization decision

Mean/std normalization over scientifically valid pixels was selected for v2.
The per-spectrum location and log scale are retained by the data interface,
although they are not concatenated to the transformer input.

Median/MAD remains the recommended robustness ablation. It uses the median as
location and the median absolute deviation from that median as scale. Compared
with mean/std, it is less sensitive to cosmic rays, residual sky lines, extreme
emission lines, and isolated calibration failures. Its risk is that aggressive
robust scaling can alter the relative prominence of physically meaningful
features for unusual objects. That is why mean/std was chosen as the controlled
first run and median/MAD was documented rather than made the default.

Absolute flux or luminosity was not made a primary encoder input. Raw amplitude
mixes intrinsic luminosity with distance, aperture loss, calibration, dust,
and observing conditions. The immediate goals prioritize continuum shape,
line information, redshift, classification, and cross-modal identity. Scale
statistics remain available for downstream tasks that need luminosity.

### Spectrum v2 objective

The selected objective uses direct LeJEPA-style alignment without an EMA
teacher:

- global invariance aligns projected CLS representations across two spectral
  views;
- global SIGReg prevents the shared CLS solution from collapsing;
- local invariance aligns corresponding wavelength-patch projections;
- local SIGReg prevents patch representations from collapsing;
- valid-pixel and valid-patch masks keep artificial JEPA masks separate from
  scientific data quality.

The global term is necessary because a CLS token can learn an excellent
whole-spectrum representation when the views preserve object identity and
SIGReg keeps the batch distribution informative. The local term is not a
replacement for the global term. It gives direct learning pressure to lines,
breaks, and continuum regions that a single CLS loss can otherwise
undersupervise.

Spectrum v2 is implemented in `train-spectra-v2.py`. The corrected training
run is named `SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs` in both
the checkpoint path and Weights & Biases.

## 5. Cross-Modal Alternatives

Two paired-training paths were discussed.

### Post-trained alignment

Start with the selected pretrained image and spectrum encoders. Add modality
projectors into a shared space and align paired CLS representations. This is
the lower-risk reference because each encoder already contains a useful
unimodal representation before seeing paired data.

### Simultaneous training from scratch

Initialize both encoders randomly and train them together only from paired
image-spectrum observations. The selected clean objective uses two image views
and two spectrum views:

```text
L_cross = mean over v,w of ||z_image[v] - z_spectrum[w]||^2

L = (1 - lambda) * L_cross
    + lambda / 2 * (SIGReg(z_image) + SIGReg(z_spectrum))
```

All four cross-modal view pairings are included. No explicit image-to-image or
spectrum-to-spectrum objective is used. Consistency within each modality is
induced transitively because both views must match the same opposite-modality
views. Local losses were deliberately omitted from this first scratch study:
the immediate question is whether SIGReg-supported global cross-modal learning
can replace pretraining, not whether the model optimizes every private detail.

The decisive comparison must use the same paired set X, split, encoder sizes,
training budget, frozen-probe code, and downstream labels for both scratch
training and post-trained alignment. Pair count alone cannot guarantee parity.
The 95,895-pair dataset is suitable for the first controlled proof-of-concept,
but conclusions must be labeled data-limited until tested at larger scale.

## 6. Paired Dataset

The selected scratch-training source is the ready-made MMU HATS cross-match at:

```text
/mnt/datasets/pranav/desi_legacysurvey_xmatch
```

It contains 95,895 matched objects built from DESI PROVABGS, DESI EDR spectra,
and Legacy Survey DR10 South images. Training uses an identity-hashed split
with seed 42. The indexed counts used by the run are 93,986 train pairs, with
the remainder held out for validation and test.

The data are scientifically matched upstream. The trainer does not construct
new coordinate matches during training. Datasets remain under `/mnt/datasets`;
they are not copied into the repository.

## 7. Simultaneous Scratch Run Provenance

Trainer: `train-cross-modal-scratch.py`

Run name: `CrossModalScratch_MMU95K_DR10_DESI_CrossOnly_SIGReg`

Latest completed checkpoint at the time of the 2026-08-25 evaluation request:

```text
checkpoints/CrossModalScratch_MMU95K_DR10_DESI_CrossOnly_SIGReg/last.pt
global_step = 8396
saved epoch = 49
saved step_in_epoch = 164
per-GPU batch size = 32
world size = 4
global batch size = 128
```

The epoch field is misleading. The scheduler assumed 734 steps per epoch, but
the distributed iterable stream repeatedly exhausted after approximately
164-170 steps. The outer epoch loop then advanced anyway. The model therefore
completed 8,396 optimizer steps and saw approximately 1,074,688 paired
examples, equivalent to about 11.4 nominal passes over 93,986 training pairs,
not 50 complete passes. All analyses must identify this checkpoint by global
step. The stream-exhaustion accounting needs correction before a rerun.

## 8. Benchmark Interpretation

The benchmark discussion established the following interpretation:

- the image architecture is sound, but the image training sample is too small
  and insufficiently representative of faint objects;
- spectrum v1 was severely undertrained and its representation partially
  collapsed;
- spectrum preprocessing and local learning were higher-priority fixes than
  simply extending the old v1 run;
- spectrum v2 is expected to provide the highest immediate downstream return;
- cross-modal training can improve shared physical information and embedding
  geometry, but it cannot manufacture morphology or populations absent from
  the raw data;
- the complete MMU corpus should improve unique-object coverage, survey and
  instrument diversity, redshift range, astrophysical population coverage,
  and the scale available for scratch cross-modal training.

Cross-modal learning may narrow the image-redshift gap by transferring spectral
information into the image representation, but downstream evaluation must
also check whether image morphology and spectrum-private information survive.

## 9. Evaluation Requested on 2026-08-25

The current experiment has two primary tasks:

1. Evaluate cross-match image redshift using the frozen image encoder trained
   simultaneously with the spectrum encoder.
2. Evaluate redshift, stellar mass, sSFR, metallicity, and age using both the
   latest spectrum-v2 checkpoint and the simultaneous model's spectrum
   encoder.

The original benchmark implementation was recovered from the remote Git branch
`redshift-regression-eval` at commit `3cd04a0`. The exact evaluation sample is
the `mhsotoudeh/astroclip` parquet mirror with 138,583 train rows and 29,697
test rows. It preserves AstroCLIP's shipped 80/20 split. It was downloaded to:

```text
/mnt/datasets/pranav/astroclip
```

The physical-property labels come from the local MMU PROVABGS catalog and are
joined by DESI `targetid`. The property definitions are:

- stellar mass: `LOG_MSTAR`;
- metallicity: `log10(Z_MW)`;
- age: `TAGE_MW`;
- sSFR: `log10(AVG_SFR) - LOG_MSTAR`.

The primary representations are raw backbone CLS embeddings. For the
simultaneous model, its shared 256-dimensional projector output is evaluated as
an additional diagnostic. This distinguishes what the backbone contains from
what the cross-modal objective exposes in its shared space.

### Checkpoints selected

| Model | Checkpoint | Selection rule |
| --- | --- | --- |
| Spectrum v2 | `checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs/step_104000.pt` | Newest stable periodic checkpoint available when the evaluation finished |
| Scratch simultaneous | `checkpoints/CrossModalScratch_MMU95K_DR10_DESI_CrossOnly_SIGReg/last.pt` | Final and most recent checkpoint, global step 8,396 |

Spectrum v2 training was still active on GPUs 0-3 when evaluation preparation
started. Its step-104,000 result is therefore an intermediate checkpoint, not
the final ten-epoch result.

### Protocol

- Backbones are frozen.
- Images use the original benchmark's DR2-style grz arcsinh-to-RGB transform,
  bicubic resize to 140 pixels, and no ImageNet normalization.
- Spectra use per-spectrum mean/std normalization and an unmasked,
  noise-free evaluation view.
- The AstroCLIP parquet mirror contains flux but not ivar or pipeline masks.
  Evaluation therefore treats finite flux values as valid. This is a necessary
  limitation of the exact comparison sample and is logged rather than hidden.
- Features are standardized using training-split statistics only.
- Ridge regularization is selected on a seed-specific 10% validation carve-out
  from the shipped training split.
- The shipped test split is never used for hyperparameter selection.
- Image redshift uses ten validation seeds for a headline-grade estimate.
- Spectrum probes use three seeds, matching the baseline report.

### Results

All requested frozen ridge evaluations completed successfully. Values below
are mean test R2; uncertainty is the standard deviation over validation
carve-out seeds. Image results use ten seeds and spectrum results use three.

#### Cross-match image redshift

| Representation | Dimension | Test R2 | Effective rank | Reference delta |
| --- | ---: | ---: | ---: | ---: |
| Simultaneous image raw CLS | 1,024 | **0.52334 +/- 0.00055** | 9.64 | -0.02728 vs adapted ViT-L |
| Simultaneous image shared projection | 256 | **0.51729 +/- 0.00026** | 8.58 | -0.03333 vs adapted ViT-L |
| Previous adapted ViT-L raw CLS | 1,024 | **0.55062 +/- 0.00018** | Not recomputed | report baseline |
| AstroCLIP image | - | **0.79** | - | published reference |

The scratch image encoder does not beat the adapted image-only backbone. It is
nevertheless close to the original, unadapted ViT-L result of 0.530 despite
training from scratch on only the 95k paired corpus. The raw CLS embedding is
slightly better than the shared projection, so the 256-dimensional alignment
bottleneck does not add redshift information for a linear probe.

#### Spectrum raw CLS results

| Target | Spectrum v1 baseline | Spectrum v2 step 104,000 | Simultaneous scratch step 8,396 | AstroCLIP spectrum |
| --- | ---: | ---: | ---: | ---: |
| Redshift | 0.432 | **0.53311 +/- 0.00022** | **0.64237 +/- 0.00048** | 0.98 |
| Stellar mass | 0.506 | **0.68311 +/- 0.00021** | **0.82309 +/- 0.00011** | 0.88 |
| sSFR | 0.510 | **0.48796 +/- 0.00035** | **0.62860 +/- 0.00010** | 0.64 |
| Metallicity | 0.235 | **0.37655 +/- 0.00047** | **0.53313 +/- 0.00027** | 0.58 |
| Age | 0.185 | **0.21069 +/- 0.00040** | **0.32063 +/- 0.00029** | 0.43 |

Relative to spectrum v1, v2 gains approximately +0.101 redshift, +0.177 mass,
+0.142 metallicity, and +0.026 age R2, while sSFR is approximately 0.022 below
the v1 result. The new representation materially improves four of the five
probes, although the step-104,000 checkpoint is still intermediate and the
sSFR regression should be watched as training continues.

The simultaneous scratch spectrum encoder is stronger than spectrum v2 on all
five tasks. Its remaining gaps to AstroCLIP are approximately 0.338 redshift,
0.057 mass, 0.011 sSFR, 0.047 metallicity, and 0.109 age. The physical-property
results are already close to AstroCLIP; redshift remains the dominant deficit.

#### Spectrum projector diagnostics

| Target | Spectrum v2 global projection, 64d | Simultaneous shared projection, 256d |
| --- | ---: | ---: |
| Redshift | 0.36117 | 0.58819 |
| Stellar mass | 0.53973 | 0.77530 |
| sSFR | 0.42820 | 0.58828 |
| Metallicity | 0.25686 | 0.44331 |
| Age | 0.10210 | 0.24863 |

Every raw CLS representation beats its corresponding projector on every
requested downstream task. This is expected because the projector is the
objective-facing bottleneck and may discard information that is unnecessary
for view or modality alignment. Downstream users should default to raw CLS,
not the pretraining projector output.

#### Geometry audit

Effective rank is computed from the train-split feature correlation matrix
after per-dimension standardization.

| Encoder space | Dimension | Effective rank | Participation ratio | Largest eigenvalue share |
| --- | ---: | ---: | ---: | ---: |
| Simultaneous image raw | 1,024 | 9.64 | 4.57 | 0.397 |
| Simultaneous image projected | 256 | 8.58 | 5.33 | 0.381 |
| Simultaneous spectrum raw | 768 | 7.04 | 4.21 | 0.414 |
| Simultaneous spectrum projected | 256 | 10.07 | 7.68 | 0.240 |
| Spectrum v2 raw | 768 | 13.59 | 9.32 | 0.213 |
| Spectrum v2 projected | 64 | 23.80 | 17.35 | 0.127 |

There are no constant embedding dimensions, so these low ranks are not an
artifact of dead scalar features. SIGReg has prevented total collapse and the
representations carry strong physical information, but it has not produced a
uniformly high-dimensional geometry. The simultaneous shared spectrum space is
better distributed than its raw CLS, while spectrum-v2's 64-dimensional
projector is the best distributed space but also the least useful for the
requested probes. Effective rank and downstream accessibility are related but
not interchangeable.

#### Interpretation and limits

The results strongly support the feasibility claim: global cross-modal
LeJEPA plus per-modality SIGReg can train useful image and spectrum encoders
from scratch. They do not yet prove that SIGReg removes the value of
pretraining. That stronger claim requires the planned post-trained-alignment
control on the same 95k pairs and an equal optimizer-step or sample budget.

The spectrum comparison is also not an equal-budget scratch-versus-unimodal
contest. Spectrum v2 has broad DESI exposure and was evaluated at an
intermediate step; the simultaneous model used fewer unique paired objects but
repeated them for approximately 11.4 nominal passes. The image and spectrum
results should therefore be read as checkpoint health measurements, not the
final ablation.

No physical-property labels were consumed by either self-supervised trainer.
The scratch loader reads only images, spectra, coordinates, and object IDs.
The high PROVABGS scores therefore reflect information learned from paired
modalities, not direct label leakage.

Raw JSON outputs and embedding provenance are stored under
`Evals/cross_modal_backbone_eval/results/`; large embedding caches remain under
`/mnt/datasets/pranav/astrojepa_eval_cache`.

## 10. Required Future Controls

The scratch-versus-alignment claim eventually requires more than frozen ridge
scores. The final controlled battery should include:

- image-to-spectrum and spectrum-to-image Recall@K;
- median and mean retrieval rank;
- matched-pair versus shuffled-pair distance distributions;
- retrieval within narrow redshift bins to detect a redshift-only shortcut;
- effective rank and covariance spectra for raw and projected embeddings;
- modality classification in the shared space;
- image morphology retention;
- spectrum object-class and line-sensitive probes;
- all five physical-property regressions from each modality and shared space;
- identical paired samples and optimizer-step budgets for scratch and
  post-trained alignment.


## 11. Final Spectrum-V2 Evaluation, 2026-08-29

The completed ten-epoch spectrum-v2 checkpoint was evaluated after the earlier
step-104,000 measurement:

```text
checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs/complete.pt
global_step = 172390
saved epoch = 10
saved step_in_epoch = 0
```

The protocol, sample, splits, normalization, ridge grid, three seeds, and target
definitions are identical to the intermediate and scratch-spectrum evaluations.
The primary numbers use the frozen 768-dimensional raw CLS representation.

| Target | V1 baseline | V2 step 104,000 | V2 final | Scratch cross-modal | AstroCLIP spectrum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Redshift | 0.432 | 0.53311 | **0.55506 +/- 0.00025** | 0.64237 | 0.98 |
| Stellar mass | 0.506 | 0.68311 | **0.69755 +/- 0.00038** | 0.82309 | 0.88 |
| sSFR | 0.510 | 0.48796 | **0.49642 +/- 0.00013** | 0.62860 | 0.64 |
| Metallicity | 0.235 | 0.37655 | **0.40879 +/- 0.00076** | 0.53313 | 0.58 |
| Age | 0.185 | 0.21069 | **0.23220 +/- 0.00041** | 0.32063 | 0.43 |

The final 64-dimensional projector scores are 0.38801 redshift, 0.54051 mass,
0.42978 sSFR, 0.25930 metallicity, and 0.10441 age. Raw CLS again wins on every
task, so downstream use should continue to default to the backbone output.

Final raw-CLS effective rank is 13.67, participation ratio is 9.59, and the
largest correlation-matrix eigenvalue holds 20.8% of the variance. Geometry is
slightly healthier than at step 104,000, but remains concentrated.

The completed v2 model improves over v1 on redshift, mass, metallicity, and age;
sSFR remains 0.014 below v1. Continued unimodal training helped every probe
relative to step 104,000, but the scratch cross-modal spectrum encoder remains
stronger on all five tasks. This is an empirical checkpoint comparison, not a
causal cross-modal ablation, because the models used different data and
optimization histories.

Machine-readable output:

```text
Evals/cross_modal_backbone_eval/results/spectra_v2_complete/metrics.json
```


## 12. Final 307K Cross-Modal Evaluation, 2026-08-30

Both completed `last.pt` checkpoints were evaluated with the same frozen-ridge
protocol used above. The 307K scratch model completed 60,000 steps; the
post-trained learned-query model completed 24,000 steps and reconstructed its
recorded frozen sources without modifying them.

Headline raw R2 values are 0.62741 image redshift and 0.67556 spectrum
redshift for scratch, versus 0.52998 and 0.55506 for post-training. Scratch
also reaches 0.84260 stellar mass, 0.63919 sSFR, 0.54944 metallicity, and
0.36579 age. It beats the post-trained setup on every requested target. The
post-trained spectrum pooler nevertheless improves all five tasks over both
the frozen v2 raw CLS and old v2 projector.

The result establishes feasibility of non-collapsed from-scratch cross-modal
training with SIGReg, but does not yet establish that pretraining has no value:
the two arms have unequal update budgets and trainable parameter counts, and
the all-pairs training set includes objects from the historical evaluation
split. Full scores, errors, geometry, deltas, provenance, and this limitation
are recorded in:

```text
Evals/cross_modal_backbone_eval/FINAL_307K_COMPARISON.md
```
