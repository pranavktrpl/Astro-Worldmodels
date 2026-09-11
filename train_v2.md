# AstroJEPA After the Baseline Snapshot

Status: 2026-08-29

This is the short operational record of what changed after the baseline code
was frozen in `baseline_train_runs_2026-08-10/`. It summarizes the implemented
training paths, completed runs, data, evaluations, and the caveats that matter
before the next experiment. The longer rationale remains in the decision logs
under `docs/`.

## 1. Preserved Baseline

`baseline_train_runs_2026-08-10/` is an immutable code snapshot containing the
original image trainer, spectrum trainer, loaders, transforms, and configs. It
does not duplicate checkpoints or datasets. All later development happened in
the live repository files.

The original frozen spectrum backbone scored:

| Probe | Baseline R2 | AstroCLIP spectrum R2 |
| --- | ---: | ---: |
| Redshift | 0.432 | 0.98 |
| Stellar mass | 0.506 | 0.88 |
| sSFR | 0.510 | 0.64 |
| Metallicity | 0.235 | 0.58 |
| Stellar age | 0.185 | 0.43 |

The main diagnosis was that spectrum v1 had seen less than one effective pass
over DESI, used flux alone, confused artificial crop masks with scientific
validity, had no wavelength-corresponding local objective, and had low
effective rank.

## 2. Spectrum Backbone V2

The live spectrum entry point is now `train-spectra-v2.py`. The archived
`baseline_train_runs_2026-08-10/train-spectra.py` remains unchanged.

### Data and preprocessing

V2 is a controlled DESI-only run over 1,126,441 MMU DESI EDR/SV3 spectra at:

```text
/mnt/datasets/utbd_pranav/desi_edr_sv3/mmu_desi_edr_sv3/dataset
```

Each record provides `flux`, `ivar`, `lambda`, `lsf_sigma`, and `mask`.

- Flux is the only physical encoder signal.
- Validity is `finite(flux) AND finite(ivar) AND ivar > 0 AND NOT mask`.
- Invalid normalized flux is zero-imputed, but a separate learned validity
  embedding tells the encoder that those values are missing.
- Pipeline validity, artificial JEPA masks, and transformer padding masks are
  kept as separate concepts.
- Positive inverse variance determines realistic Gaussian noise augmentation;
  it is not an encoder channel or a token-loss weight in this first run.
- Wavelength is validated but remains implicit in fixed token position.
- LSF is retained and validated but is not yet an encoder input.
- No labels or redshift-dependent preprocessing are used.

Mean/std normalization is computed per spectrum over valid pixels. The removed
mean and log-standard-deviation are retained as metadata but are not fed into
the transformer. Median/MAD remains the main robustness ablation for a later
run. Absolute observed flux was deliberately not made a primary feature because
it mixes intrinsic luminosity with distance, aperture losses, dust, and
calibration.

The 7,781-pixel spectrum is represented by 389 non-overlapping 20-pixel patches;
the final unmatched pixel is omitted to preserve the baseline architecture.

### Architecture and objective

The deployed architecture remains a 12-layer, 768-dimensional transformer with
12 attention heads and a CLS token. Each object produces two independently
masked and noise-perturbed views.

V2 uses teacher-free LeJEPA objectives:

```text
L = L_global + L_local

L_global = 0.95 * CLS view invariance + 0.05 * global SIGReg
L_local  = 0.95 * corresponding-patch invariance + 0.05 * local SIGReg
```

Local comparisons use matching wavelength positions and exclude scientifically
invalid targets. There is no EMA teacher, stop-gradient target network,
predictor, reconstruction target, or contrastive negative set.

The backbone outputs 768-dimensional CLS and patch features. The 64-dimensional
global and local projectors are objective-facing training heads; downstream
work should normally use raw backbone features.

### Split, optimization, and accounting

The deterministic object-ID split is:

| Split | Rows |
| --- | ---: |
| Train | 1,103,874 |
| Validation | 11,280 |
| Test | 11,287 |

The corrected four-GPU topology uses 17,239 synchronized steps per epoch,
global batch size 64, and 172,390 optimizer steps over ten epochs. The epoch
logic was repaired so reported epochs correspond to complete synchronized
passes rather than outer-loop counters.

The full run used AdamW, base learning rate `5e-4`, weight decay `0.05`, 1,000
warmup steps, cosine decay to `1e-6`, BF16, and gradient clipping at `1.0`.
Training metrics were logged to W&B every ten optimizer steps.

Periodic checkpoints are written every 4,000 steps and only the latest two are
retained. Epoch-boundary recovery checkpoints and a final completion checkpoint
are also written.

### Completed run

Run name:

```text
SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs
```

The ten-epoch run completed on 2026-08-25. Important files are:

```text
checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs/complete.pt
checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs/last_epoch_10.pt
checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs/step_172000.pt
```

The first frozen evaluation performed during training used `step_104000.pt`.
The final `complete.pt` has now been evaluated with the same frozen-probe
battery; its results are reported below.

## 3. Simultaneous Cross-Modal Training From Scratch

The new path is implemented by:

- `train-cross-modal-scratch.py`;
- `models/cross_modal.py`;
- `data/cross_modal.py`.

Both the ViT-L image encoder and 12-layer spectrum encoder start from random
initialization. Separate projectors map their CLS features into a shared
256-dimensional space.

For two image views and two spectrum views, the selected first objective is:

```text
L_cross = mean over all four (v, w) pairs of ||z_image[v] - z_spectrum[w]||^2

L = 0.95 * L_cross
    + 0.05/2 * (SIGReg(z_image) + SIGReg(z_spectrum))
```

SIGReg is applied separately to each modality. The run deliberately contains
no image-to-image, spectrum-to-spectrum, or local loss, and no pretrained
weights, EMA teacher, stop-gradient, predictor, or negatives. This isolates the
question of whether cross-modal invariance plus SIGReg can train useful
backbones from scratch.

### Paired data

Training streams the ready-made MMU/LSDB cross-match directly from:

```text
/mnt/datasets/pranav/desi_legacysurvey_xmatch
```

It contains 95,895 DESI spectrum plus Legacy Survey DR10 image pairs in 477
Parquet shards. Splitting by stable physical image identity gives 93,986 train,
945 validation, and 964 test rows. Sixteen repeated image identities remain in
one split each, preventing leakage. No dataset is copied into the repository.

### Completed pilot and accounting caveat

Run/checkpoint:

```text
CrossModalScratch_MMU95K_DR10_DESI_CrossOnly_SIGReg
checkpoints/CrossModalScratch_MMU95K_DR10_DESI_CrossOnly_SIGReg/last.pt
```

The completed checkpoint is identified by `global_step=8396`, per-GPU batch 32,
four GPUs, and global batch 128. It processed about 1,074,688 paired examples,
equivalent to roughly 11.4 nominal passes over the training set.

Its saved `epoch=49` is not 50 complete data passes. The original iterable
stream exhausted after roughly 164-170 steps while the scheduler assumed 734
steps per epoch, then advanced the epoch counter. This accounting error is
documented and the live loader/trainer now derives steps from indexed rows, but
the existing checkpoint must always be described by global steps and examples
seen.

## 4. Frozen Evaluations Added

`Evals/cross_modal_backbone_eval/evaluate_frozen_backbones.py` reproduces the
original AstroCLIP cross-match split and ridge protocol. It tests raw CLS as the
primary representation and projector output as a diagnostic. The backbone is
always frozen.

The spectrum probes are redshift, PROVABGS stellar mass, sSFR, metallicity, and
stellar age. Redshift uses 138,583 train and 29,697 test rows. The PROVABGS join
contains 73,341 train and 15,993 test rows. Ridge strength is selected on a
validation carve-out; the shipped test split is untouched during selection.

### Results available now

| Target | Spectrum v1 | Spectrum v2 step 104k | Spectrum v2 final | Scratch cross-modal spectrum | AstroCLIP spectrum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Redshift R2 | 0.432 | 0.533 | **0.555** | 0.642 | 0.98 |
| Stellar mass R2 | 0.506 | 0.683 | **0.698** | 0.823 | 0.88 |
| sSFR R2 | 0.510 | 0.488 | **0.496** | 0.629 | 0.64 |
| Metallicity R2 | 0.235 | 0.377 | **0.409** | 0.533 | 0.58 |
| Stellar age R2 | 0.185 | 0.211 | **0.232** | 0.321 | 0.43 |

From step 104k to the completed run, raw-CLS R2 improves by approximately
`+0.022` redshift, `+0.014` mass, `+0.008` sSFR, `+0.032` metallicity, and
`+0.022` age. The final v2 result improves over v1 on four probes; sSFR remains
approximately `0.014` below the v1 score.

The simultaneously trained frozen image encoder obtains cross-match redshift
`R2=0.523`, compared with `0.551` for the adapted image-only ViT-L and `0.79`
for AstroCLIP image.

Raw CLS beats the corresponding projector on every requested spectrum task.
SIGReg prevented total collapse, but the learned spaces still have low effective
rank. The scratch run proves that useful image and spectrum encoders can be
trained this way; it does not yet prove that pretraining has no value.

## 5. Spectral Data Currently On Disk

The unique primary spectral corpora found locally are:

| Dataset | Rows | Role |
| --- | ---: | --- |
| MMU DESI EDR/SV3 | 1,126,441 | Spectrum-v2 training; fixed DESI grid |
| MMU SDSS | 806,176 | Candidate multi-survey v3 corpus |
| MMU VIPERS W1 | 60,528 | Candidate multi-survey v3 corpus |
| DESI-LS paired cross-match | 95,895 | Cross-modal training subset |
| AstroCLIP mirror | 168,280 | Exact benchmark/evaluation sample |

The `*_10arcs` trees are HATS margin catalogs for spatial crossmatching, not
additional independent training examples. The second top-level DESI tree is a
duplicate copy of the same 1,126,441-object corpus. The local `vipers_w4` tree
currently has the same internal name, row count, schema, and size as W1; treat
it as a suspected duplicate/misdownload until object IDs or file hashes prove
otherwise. PROVABGS is a derived physical-property catalog, not another raw
spectrum corpus.

### Cross-modal pair capacity audit

The largest single paired dataset physically loaded is the AstroCLIP mirror:
197,976 pairs split into 138,583 train, 29,696 validation, and 29,697 test rows.
The test and validation objects must remain evaluation-only. The native MMU
DR10 cross-match contains 95,895 pairs, of which our identity-hashed split uses
93,986 for training. Its DESI target IDs do not overlap any AstroCLIP split, so
the two existing training splits contain 232,569 distinct paired observations.

The local coordinate audit matched the 8,689,370-object DR8 image metadata
catalog against every primary spectrum catalog. A pair is accepted only when
the nearest image is within 1.0 arcsec and the second-nearest image is outside
1.0 arcsec.

| Spectrum source | Nearest within 1 arcsec | Ambiguous | Accepted rows | Unique DR8 images |
| --- | ---: | ---: | ---: | ---: |
| DESI EDR/SV3 | 79,768 | 116 | 79,652 | 78,718 |
| SDSS | 564,953 | 1,686 | 563,267 | 563,267 |
| VIPERS W1 | 34 | 0 | 34 | 34 |

Together these produce 642,953 candidate pair rows over 631,878 unique image
objects. DESI and SDSS share 10,141 matched image identities, and DESI contains
934 additional repeated spectrum observations. These are coordinate-manifest
counts, not yet a materialized training dataset; physical image availability,
quality filtering, and cross-survey duplicate policy may lower the final total.

The selected radius remains 1.0 arcsec. At 0.5 arcsec the accepted counts are
79,184 DESI and 555,205 SDSS; increasing to 2.0 arcsec yields only 80,328 DESI
and 565,255 SDSS while increasing ambiguous matches. One arcsecond is therefore
a sensible completeness/purity knee for these catalogs.


The 79,652 manual DESI matches are not 79,652 new physical objects relative to
our loaded paired sets. Target-ID de-duplication finds 31,436 overlaps with the
native MMU pairs and 34,637 with AstroCLIP: 24,138 train, 5,245 validation, and
5,254 test. Only 13,579 manual DESI matches are new beyond every pre-matched
split. The strict training-safe DESI union is therefore 232,569 existing train
pairs plus 13,579 new manual pairs, or 246,148 distinct DESI objects. Counting
all held-out splits gives 307,450 distinct DESI target IDs after adding the
manual matches, not approximately 374k.

If SDSS support is added, the naive training-row ceiling becomes 246,148 DESI
plus 563,267 SDSS, or 809,415 observations. The unique physical-object count is
lower because surveys can observe the same galaxy; at least 10,141 DR8 image
identities occur in both the manual DESI and SDSS matches. Repeated observations
can be retained as additional views, but must not be described as new objects
and all views of one object must remain in one split.
The 563k SDSS pairs are not immediately compatible with the DESI-only trainer.
SDSS has a different wavelength grid, resolution, and selection function. A
multi-survey loader must resample to a common grid or use wavelength-aware
positions and must carry survey/instrument and resolution information. The
AstroCLIP mirror also has a reduced flux-only schema, unlike the native MMU
records with ivar, mask, wavelength, and LSF.

## 6. What Remains

Immediate work, in order:

1. Run image-to-spectrum and spectrum-to-image retrieval for the scratch model.
2. Train the equal-budget post-trained alignment control on the same 95,895
   pairs, split, shared dimension, views, optimizer steps, and probe code.
3. Run scratch cross-modal training without SIGReg to test whether SIGReg is
   causally responsible for avoiding collapse.
4. Rerun the corrected scratch trainer using honest indexed epoch accounting.
5. Audit and harmonize SDSS and VIPERS grids, masks, resolution, and selection
   functions before spectrum v3 multi-survey training.

The key scientific claim requires controls 2 and 3. Current results are strong
evidence that cross-modal SIGReg training from scratch is viable, but they are
not yet a controlled demonstration that it matches or replaces unimodal
pretraining.

## Longer References

- `docs/SPECTRA_V2_DECISION_LOG.md`: full spectra-v2 rationale and implementation.
- `docs/CROSS_MODAL_JEPA_DECISION_LOG.md`: cross-modal design and controls.
- `docs/ASTROJEPA_PROJECT_CONVERSATION_LOG.md`: chronological project and evaluation record.
- `Evals/cross_modal_backbone_eval/results/`: machine-readable frozen-probe outputs.

### Combined DESI cross-modal scratch training

The all-DESI scratch trainer now uses 307,428 runnable pairs: 197,976 AstroCLIP,
95,895 MMU DR10, and 13,557 newly materialized manual DR8 matches. The earlier
307,450 figure was a pre-materialization ceiling; 22 rows were removed because
21 matched DR8 image payloads are absent locally.

The loader uses every source row and does not retain the source datasets'
historical train/validation/test labels. It converts AstroCLIP nanomaggy images
to Legacy RGB, supports flux-only DESI spectra without inventing ivar, and keeps
physical masks and ivar for the other sources.

The earlier cross-modal worker path double-sharded files and shortened every
nominal epoch. Direct Parquet streaming now gives exactly 1,200 optimizer steps
per true epoch on eight GPUs at batch size 32 per GPU. Fifty epochs correspond
to 60,000 steps and approximately 50 complete passes through the 307k corpus.

The local SIGReg package was compared with the official LeJEPA implementation.
It already performs autograd-aware distributed reduction of Epps-Pulley moments
with correct world-size gradient scaling, so no embedding all-gather was added.

The finalized eight-GPU trainer passed real-row checks for all three sources,
the regression suite, and a distributed forward/backward smoke test. Full
training remains user-launched.
