# Spectrum Backbone V2 Decision Log

Last updated: 2026-08-23

## Purpose

This is the living design record for the DESI-only spectrum-backbone v2. It records:

- problems found in the baseline;
- options considered;
- the rationale behind recommendations;
- decisions that have been accepted;
- questions that remain open;
- later changes to any of those decisions.

The archived baseline in `baseline_train_runs_2026-08-10/` is a checkpoint of the original project and remains unchanged. V2 is implemented in the live `train-spectra-v2.py` and spectrum data modules; full-run hyperparameters that remain experimental are called out explicitly below.

## Scope

### V2

- Train only on MMU DESI EDR spectra.
- Keep the deployed backbone architecture at 12 transformer layers, embedding dimension 768, 12 attention heads, and non-overlapping 20-pixel patches unless a later decision explicitly changes it.
- Correct the spectrum data representation.
- Add genuine local, wavelength-resolved representation learning while retaining a strong global representation.
- Train the model from scratch.

### Later work

- V3 will expand to DESI, SDSS, VIPERS, and other suitable spectrum surveys.
- Cross-modal work will compare alignment of independently trained encoders with a separate jointly trained from-scratch cross-modal JEPA experiment. The living design record is `docs/CROSS_MODAL_JEPA_DECISION_LOG.md`.
- Decisions specific to V3 and cross-modal training will be added to this document or linked decision records when those stages begin.

## Baseline Problems Motivating V2

1. Only `spectrum.flux` is loaded. Available validity, uncertainty, wavelength, and line-spread information is ignored.
2. Artificial crop masks are called spectrum masks, but they do not describe invalid detector or pipeline pixels.
3. Invalid flux values can therefore influence patch embeddings as though they were valid measurements.
4. The eight existing "local" views are short crops whose pooled embeddings use the same global loss. There is no loss on corresponding wavelength-token representations.
5. The configured five-epoch baseline run reached only about 1.14 complete passes over DESI in its latest checkpoint.
6. There is no fixed train/validation/test partition for self-supervised model development.

Audit correction: the repository's Epps-Pulley implementation performs a differentiable distributed reduction of its empirical characteristic-function statistics. Baseline SIGReg therefore already used the effective cross-GPU distribution when DDP was initialized. The earlier claim that it operated only on 16 samples per GPU was incorrect. V2 will preserve and test this behavior rather than adding a redundant embedding gather.

## Data Contract

The DESI spectrum struct contains `flux`, `ivar`, `lambda`, `lsf_sigma`, and a boolean `mask`. The MMU version exposes a boolean mask, not the original DESI bit field, so individual pipeline mask reasons cannot be recovered from this copy.

Implemented roles:

| Field | V2 role | Encoder input? | Status |
| --- | --- | --- | --- |
| `flux` | Primary physical signal | Yes, after normalization | Accepted |
| `mask` | Identify invalid/pipeline-flagged pixels | Separate validity signal | Accepted in principle; representation described below |
| `ivar` | Additional validity check and realistic noise scale | No | Accepted |
| `lambda` | Validate the fixed DESI grid and map token positions | No | Accepted |
| `lsf_sigma` | Retain for later line-width experiments | No | Accepted for initial v2 |
| normalization location/scale | Preserve removed scalar information for audits and downstream use | No in initial v2 | Accepted |

Redshift and class labels must not enter preprocessing or self-supervised training.

## Decision 1: Absolute Flux And Luminosity

### Options considered

1. Preserve absolute flux amplitude directly in the transformer input.
2. Normalize each spectrum independently and discard the removed scale.
3. Normalize independently, but retain the removed location and scale as auxiliary metadata.

### Discussion

Absolute observed flux depends on intrinsic luminosity, distance, aperture/fiber losses, calibration, dust, and observing conditions. It is therefore not automatically a clean physical feature. For the immediate goals of redshift, spectral classification, retrieval, and cross-modal alignment, continuum shape and spectral lines are usually more useful than a raw amplitude offset.

Removing per-spectrum location and scale does discard information that may matter for luminosity, distance-sensitive tasks, and some calibrated physical inference. Retaining the two scalar statistics costs almost nothing and keeps that option open without letting amplitude dominate self-supervised training.

### Current decision

Prioritize spectral shape, lines, redshift, classification, and cross-modal representation quality. Do not make absolute flux amplitude a primary transformer input. Retain the normalization statistics in the data/model interface so later downstream models can use them if needed.

Status: accepted.

## Decision 2: Flux Normalization

### Options considered

1. Per-spectrum mean/std normalization.
2. Per-spectrum median/MAD normalization.
3. Median amplitude scaling without subtracting the continuum level.

### Mean/std

For valid pixels `x`, compute

```text
x_normalized = (x - mean(x)) / std(x)
```

This is simple, familiar, and close to AstroCLIP-style preprocessing. Its weakness is sensitivity to extreme pixels, unmasked artifacts, very strong emission lines, and occasional pathological flux values. A small number of extremes can move both the mean and standard deviation and compress the scientifically useful majority of the spectrum.

### Median/MAD

For valid pixels `x`, compute

```text
location = median(x)
MAD = median(abs(x - location))
robust_scale = 1.4826 * MAD
x_normalized = (x - location) / robust_scale
```

The factor 1.4826 makes MAD comparable to standard deviation for Gaussian data. Median/MAD is robust because isolated extreme pixels have little influence on either statistic. That is attractive for survey spectra containing artifacts, sky residuals, and strong narrow features. The concern is that MAD can become very small for unusual or nearly flat spectra, so it needs a well-defined fallback and scale floor. Its effect on stars, galaxies, and quasars should also be checked separately.

Median/MAD was recommended because it is less likely to let bad pixels determine the scale of an entire example. It remains worth testing, but it should not silently replace the simpler baseline without an ablation.

### Current decision

Use mean/std normalization for the first v2 run, computed only over scientifically valid pixels. Retain the mean and log-standard-deviation. Implement normalization as a configurable strategy so median/MAD can be evaluated later without rewriting the loader.

Required safeguards:

- finite-value checks;
- a scale floor;
- deterministic handling of spectra with too few valid pixels;
- logged distributions of mean, standard deviation, and clipping/fallback frequency;
- no use of redshift in normalization.

Status: accepted.

## Decision 3: Invalid Pixels And Mask Representation

### Validity rule

The initial proposed rule is:

```text
valid_pixel = finite(flux) AND finite(ivar) AND ivar > 0 AND NOT pipeline_mask
```

The MMU conversion exposes this field as a boolean rather than the original DESI bit field. V2 therefore treats `pipeline_mask=True` as invalid while retaining the limitation that individual mask reasons cannot be recovered from this dataset copy.

### Why zero-imputation alone is insufficient

After mean/std normalization, zero represents an ordinary pixel near the spectrum's mean. If an invalid pixel is also replaced by zero and no mask information reaches the encoder, the transformer cannot distinguish missing data from a genuine mean-level measurement. Partial invalidity inside a 20-pixel patch is especially problematic: dropping every partially affected patch wastes good measurements, while silently treating the imputed values as real creates false structure.

### Options considered

1. Zero-impute invalid pixels and do not tell the encoder.
2. Drop any patch containing invalid pixels.
3. Concatenate a binary mask as an ordinary second physical channel.
4. Zero-impute invalid values and add a learned embedding derived from the separate validity pattern.

### Current decision

Use option 4. Invalid normalized flux values will be set to zero. A small learned validity embedding will tell the patch encoder which elements were observed. The scientific flux tensor, pipeline-validity mask, artificial JEPA mask, and sequence-padding mask will remain separate tensors with separate meanings.

This does not ask the network to interpret missingness as flux. It prevents imputation from masquerading as a real measurement, preserves partially valid patches, and permits the local loss to exclude invalid targets.

Status: accepted, implemented, and covered by synthetic and streamed-data tests.

## Decision 4: Inverse Variance

### Options considered

1. Use `ivar` only for validity and uncertainty-derived augmentation.
2. Weight the local token loss using patch-level inverse variance.
3. Supply inverse uncertainty directly to the transformer as an input feature.

### Rationale

Inverse variance has a clear interpretation for pixel reconstruction errors. A JEPA token error, however, is a distance between learned representations rather than a flux residual. Treating raw ivar as its statistically correct weight is therefore not justified automatically. It could also cause high-S/N regions or bright object populations to dominate the representation objective.

### Current decision

For the first controlled v2 run:

- use `ivar <= 0` as evidence that a pixel is invalid;
- use positive `ivar` to scale physically motivated Gaussian noise augmentation;
- do not feed `ivar` to the encoder;
- do not ivar-weight the local representation loss;
- apply local loss only where the target has sufficient valid coverage.

Keep ivar weighting configurable as a later ablation after the unweighted objective is healthy.

Status: accepted.

## Decisions 5 And 6: Global And Local Objectives

### What the baseline currently does

The baseline creates two long crops and eight short crops. Every crop is encoded into one pooled representation. The two global projections define a per-object center, all view projections are pulled toward it, and SIGReg regularizes every projected distribution. Although some inputs are called local crops, the loss never compares corresponding wavelength tokens. It is a multi-view global objective, not a local JEPA objective.

### Proposed global objective

Two independently corrupted views of the same spectrum pass through the student encoder. Their CLS representations pass through a global projector. The LeJEPA invariance term makes the two representations agree, while SIGReg makes the representation distribution across different spectra resemble a well-spread reference distribution and prevents collapse.

Conceptually:

```text
L_global = (1 - lambda_sigreg) * L_CLS_agreement
           + lambda_sigreg * L_SIGReg
```

The CLS objective trains one summary vector to encode information that remains stable across masks and noise. This is exactly the representation needed for classification, retrieval, redshift probes, and later cross-modal CLS alignment. A global CLS objective can learn an excellent spectrum representation, just as it does for images, provided the views preserve the relevant physical identity and the anti-collapse regularizer is effective.

Its limitation is supervision density. One loss vector supervises an entire 7,781-pixel spectrum. The network can satisfy global agreement using broad continuum or object-class cues while failing to organize individual wavelength tokens around lines and localized features.

### Two different local-objective families

The earlier discussion accidentally moved between two distinct methods without naming the change. They must be treated as separate experiments.

#### Option A: symmetric local LeJEPA

Use the same teacher-free principle as global LeJEPA. Two independently corrupted views pass through the same trainable encoder. At every scientifically valid wavelength position, corresponding patch-token projections are pulled together. A local SIGReg term prevents the patch representations from collapsing.

Conceptually:

```text
z1_cls, z1_patch = encoder(view_1)
z2_cls, z2_patch = encoder(view_2)

L_local_invariance = distance(
    z1_patch[corresponding_valid_positions],
    z2_patch[corresponding_valid_positions]
)

L_local_LeJEPA = (1 - lambda_local) * L_local_invariance
                 + lambda_local * L_local_SIGReg
```

There is one encoder, no stop-gradient, no teacher, no EMA, and no predictor. This is the closest extension of LeJEPA's stated design to dense spectrum tokens.

Local SIGReg must be designed carefully. Applying it after flattening all wavelength positions would allow absolute positional embeddings to satisfy the distributional constraint even if every object had the same token at a given wavelength. It should therefore measure variation across objects at the same wavelength position, or across narrow position groups, using the global distributed batch. Positions can be sampled each step to keep the computation practical.

Artificially masked positions require a learned JEPA mask token, not a padding token. The token remains able to attend to visible context. Direct local alignment can then include visible-visible positions for augmentation invariance and masked-visible positions for contextual learning. Positions masked in both views or scientifically invalid in the target are excluded.

Advantages:

- remains faithful to LeJEPA's teacher-free, heuristics-free motivation;
- keeps image and spectrum objectives conceptually consistent;
- fixed DESI wavelength positions give exact local correspondences;
- has fewer moving parts and lower memory cost;
- makes the v2 experiment easier to interpret.

Risks:

- dense/local LeJEPA is not established by the original LeJEPA experiments;
- symmetric masked-visible alignment allows both sides to move, so the visible target is less stable;
- visible-visible alignment may reward local copying more than contextual prediction;
- local SIGReg can take a positional shortcut if it is not stratified correctly.

#### Option B: EMA-target masked JEPA

The student receives a spectrum with selected wavelength spans hidden. A no-gradient EMA target encoder receives the full valid spectrum. A predictor uses the student's visible context and the absolute positions of hidden patches to predict the target encoder's representations at those wavelengths.

Conceptually:

```text
student_context = student_encoder(masked_spectrum)
target_tokens = stop_gradient(EMA_encoder(full_spectrum))
predicted_tokens = predictor(student_context, hidden_positions)

L_local = mean_distance(
    predicted_tokens[valid_hidden_positions],
    target_tokens[valid_hidden_positions]
)
```

This is the established I-JEPA-style latent-prediction pattern and is closely related to iBOT's masked patch self-distillation. It is representation prediction rather than flux reconstruction.

Advantages:

- gives the masked student a stable, content-bearing target;
- forces prediction from spectral context rather than simple local copying;
- has stronger precedent for learning useful masked patch representations.

Risks:

- introduces an EMA schedule, stop-gradient branch, target encoder, and predictor;
- no longer tests a purely LeJEPA-style objective;
- costs more memory and compute;
- makes it harder to attribute v2 improvements to corrected data, local supervision, or teacher dynamics.

### Why both are proposed

Under either local-objective family, global and local supervision serve different outputs:

- global LeJEPA plus SIGReg directly trains the CLS representation;
- local alignment directly trains wavelength-token structure;
- the shared transformer lets improvements from each objective influence the other;
- global SIGReg protects the CLS distribution;
- symmetric local LeJEPA additionally requires correctly stratified local SIGReg;
- EMA-target local JEPA does not initially require a local SIGReg term, but local rank must be monitored.

The combined proposal is:

```text
L_total = L_global + beta * L_local
```

For symmetric local LeJEPA, `L_local` contains both local invariance and local SIGReg. For EMA-target JEPA, `L_local` is masked latent prediction. The two must not be mixed under one experiment name.

### Current decision

The initial controlled v2 will use option A: global LeJEPA plus global SIGReg, and symmetric local LeJEPA plus position-stratified local SIGReg, with no EMA teacher. Option B remains a clearly named follow-up ablation if the direct local objective fails to improve local features or downstream probes.

This recommendation prioritizes consistency with the project's LeJEPA premise and interpretability of the first v2 experiment. Option B is the more established masked-prediction recipe and may ultimately perform better, but it answers a different experimental question.

Status: accepted.

## Decision 7: Dataset Split

### Current decision

Create a deterministic 98/1/1 train/validation/test partition using a stable hash of `object_id`. Reuse this assignment in future spectrum and cross-modal experiments. Do not use Python's process-randomized `hash()` implementation.

Audited counts for BLAKE2b seed 42:

- train: 1,103,874 spectra;
- validation: 11,280 spectra;
- test: 11,287 spectra.

These are deterministic hash-bucket counts rather than forced row counts.

Status: accepted.

## Decision 8: Initialization, EMA, And Training Budget

### Initialization

Train v2 from scratch. The baseline used a materially different data contract and objective and was trained for only about 1.14 complete dataset passes. It is not a valuable enough initialization to compromise the controlled comparison.

Status: accepted.

### What EMA means

EMA means exponential moving average. The target encoder begins as a copy of the student encoder and is updated after every student optimization step:

```text
teacher_parameters = momentum * teacher_parameters
                     + (1 - momentum) * student_parameters
```

The teacher receives no gradients. It changes slowly and supplies stable target patch representations while the student learns to predict them. A momentum near 1 means a slow teacher. For example, at momentum 0.996, each update incorporates 0.4% of the current student parameters. The momentum is commonly increased toward 1 during training as the representation matures.

EMA is not an additional deployed model. It is a training mechanism. At the end, the student backbone is the primary exported encoder; exporting the EMA backbone as an additional evaluation candidate is also inexpensive.

### Options considered

1. Use an EMA target only for the local loss while global LeJEPA remains symmetric between student views.
2. Use EMA targets for both global and local objectives.
3. Use no EMA and make both global and local objectives symmetric LeJEPA losses with SIGReg.

Revised initial recommendation: option 3. This keeps v2 teacher-free and directly parallels the proposed image global-plus-local LeJEPA design. Option 1 remains the preferred follow-up if masked contextual prediction with stable targets is needed.

The previously proposed ten-epoch budget, batch size, learning rate, weight-decay treatment, and masking schedule have not yet been accepted. An EMA schedule is needed only if an EMA-target experiment is selected.

Status: train-from-scratch and teacher-free global/local LeJEPA accepted. EMA is not part of initial v2. The full run schedule remains open.

## Training Safeguards And Status

Implemented and smoke-tested:

- preserve LeJEPA's differentiable cross-rank reduction of Epps-Pulley statistics for both global and local SIGReg;
- exclude biases, normalization parameters, positional embeddings, CLS token, and mask token from weight decay;
- clip gradient norm at 1.0;
- use bf16 on A100 GPUs;
- log global and local invariance and SIGReg components separately;
- monitor CLS and fixed-wavelength patch-token feature standard deviations;
- audit normalization mean/scale, scale-floor rate, valid-pixel fraction, and artificial-mask coverage;
- save atomic, fully resumable checkpoints including optimizer, scheduler, scaler, RNG state, epoch, and within-epoch position;
- restore the independent global and local SIGReg slice counters from `global_step` on resume;
- cap the default epoch at the audited minimum rank length so uneven iterable tails cannot deadlock DDP.

Still pending for the full experiment:

- decide whether to add covariance or effective-rank monitoring;
- decide whether to save lightweight encoder-only milestone files in addition to resumable checkpoints;
- select checkpoints using held-out probes outside the training loop;
- evaluate an EMA encoder only if a later, separately named EMA-target experiment is selected.

## Open Questions Before The Full Run

1. Choose the full-run masking ratio and span-length distribution after the smoke-test audit.
2. Choose global batch target and learning rate after a short memory/throughput benchmark.
3. Choose the full-run budget and evaluation milestones.
4. Decide which downstream benchmark is the primary checkpoint-selection criterion.

## Change History

### 2026-08-23

- Created the decision record.
- Accepted shape-focused training with normalization statistics retained separately.
- Selected mean/std normalization for initial v2, with median/MAD retained as an ablation.
- Accepted explicit validity handling and a learned validity representation.
- Selected validity and noise augmentation as the initial uses of ivar.
- Accepted a deterministic 98/1/1 object-level split.
- Accepted training v2 from scratch.
- Distinguished symmetric dense/local LeJEPA from EMA-target masked JEPA.
- Revised and accepted the initial recommendation to global and local LeJEPA with SIGReg and no EMA.
- Retained EMA-target masked prediction as a separate follow-up ablation.
- Accepted teacher-free LeJEPA losses for both global CLS and local wavelength-token representations.
- Corrected the earlier SIGReg audit: Epps-Pulley already reduces distribution statistics across DDP ranks.
- Implemented the v2 data contract, encoder validity path, global/local objectives, optimizer safeguards, monitoring, and resumable checkpointing.
- Audited all 1,126,441 object IDs and balanced the 306 Parquet files across the 16 default loader workers.
- Completed the authorized 1,000-step, four-A100 smoke run and verified its final checkpoint.

## V2 Implementation Reference

This section describes the live implementation in enough detail to reconstruct the intended behavior during review, evaluation, or later refactoring. It describes `train-spectra-v2.py`, `data/dataloaders.py`, `data/desiSpectra_source.py`, and `data/SpectraTransforms.py`. The archived v1 files are not part of this implementation.

### File map

| File | Responsibility |
| --- | --- |
| `train-spectra-v2.py` | V2 config, transformer, projectors, global/local losses, DDP loop, logging, and checkpoints |
| `data/desiSpectra_source.py` | Open a local Parquet tree or retain the old Hugging Face fallback |
| `data/dataloaders.py` | Stable object split, streaming order, rank/worker sharding, and sample adaptation |
| `data/SpectraTransforms.py` | Scientific validity, normalization, noise, patching, and artificial span masks |
| `tests/test_spectra_v2.py` | Dependency-free contract, model, loss, real-SIGReg, and local-data tests |

The old `docs/SPECTRA_BACKBONE_TRAINING_README.md` is a historical v1 document. Its examples do not describe v2.

### Data location and access

The configured source is:

```text
/mnt/datasets/utbd_pranav/desi_edr_sv3/mmu_desi_edr_sv3/dataset
```

`DesiSpectraSource` enumerates the 306 actual `*.parquet` files under this directory and passes selected file lists to Hugging Face's streaming Parquet loader. The Parquet metadata sidecars are deliberately excluded. It does not download, materialize, symlink, or copy spectra into the repository. Checkpoints and logs remain under the repository; raw data remains under `/mnt/datasets`.

For the default four-rank, four-loader-worker topology, files are assigned to
the four DDP ranks using a deterministic greedy balance of the row counts stored
in each Parquet footer. Hugging Face then partitions each rank's files across its workers.

The source requests only the top-level `spectrum` struct and `object_id`. The struct contains flux, ivar, wavelength, LSF sigma, and pipeline mask. Redshift, class, and photometric labels are not requested.

### Deterministic split

For each `object_id`, the loader computes an eight-byte BLAKE2b digest of:

```text
42:<object_id>
```

The digest is converted to an integer modulo 100:

- buckets 0 through 97 are training;
- bucket 98 is validation;
- bucket 99 is test.

Files are assigned to ranks first for balanced I/O. The stable object-ID split
is then applied before each worker's shuffle. It is independent of Python's process-randomized `hash()` function and is stable across machines and runs.

The full local object-ID audit produced:

```text
train       1,103,874
validation     11,280
test           11,287
total       1,126,441
```

Under corrected rank-only file partitioning, the four rank-level training counts
are 276,097, 275,983, 275,901, and 275,893 spectra. Allowing for up to 15 dropped
tail samples in each of four workers per rank gives a conservative synchronized
cap of 17,239 optimizer steps. Each configured epoch therefore presents
1,103,296 spectra globally and omits at most 578 worker-tail positions. A
ten-epoch run uses 172,390 optimizer steps. This cap keeps all DDP ranks
synchronized despite the final per-worker partial batches.

### Scientific validity

For every pixel, v2 defines:

```text
valid = finite(flux)
        AND finite(ivar)
        AND ivar > 0
        AND NOT pipeline_mask
```

The three masks are never overloaded:

1. `valid_pixels` describes whether DESI supplied a scientifically usable measurement.
2. `jepa_masks` describes valid patches hidden artificially from one LeJEPA view.
3. `key_padding_mask` tells transformer attention that a patch has no valid measured pixels at all.

The sampled data audit found fixed arrays of length 7,781, finite flux values, a fixed wavelength grid, and sparse masks. Pipeline-mask and nonpositive-ivar patterns were identical for about 97.85% of the 512 audited spectra but not all of them, which is why both tests remain in the validity rule.

The transform verifies all five spectrum arrays have length 7,781. Each worker stores the first wavelength grid it sees, verifies that it is finite and monotonic, and checks subsequent grids periodically against that reference. `lsf_sigma` is length-validated but is not supplied to the encoder in initial v2.

### Mean/std normalization

Mean and population standard deviation are computed from valid pixels only:

```text
mu = mean(flux[valid])
sigma = std(flux[valid], correction=0)
normalized_flux[valid] = (flux[valid] - mu) / max(sigma, 1e-6)
normalized_flux[invalid] = 0
```

`mu` and `log(sigma)` are returned in every batch for auditing and possible downstream use. They are not transformer input features. The transform also returns whether the scale floor was used. A spectrum with fewer than 20 valid pixels or non-finite normalization statistics is skipped.

No clipping, rest-frame shifting, continuum fitting, or redshift-dependent normalization is applied.

### Uncertainty-derived views

The pipeline uncertainty in normalized coordinates is:

```text
normalized_noise_std = 1 / (sqrt(ivar) * sigma)
```

It is evaluated only for valid pixels and capped at 3.0. Each view receives independent Gaussian noise:

```text
view = normalized_flux + noise_scale * Normal(0, normalized_noise_std)
```

The current smoke-test default is `noise_scale=0.5`. This is configurable and is not yet accepted as the full-run value. Invalid pixels are reset to zero after noise generation. `ivar` is not concatenated to flux and does not weight the representation losses.

### Patching and the final pixel

The transformer geometry remains the baseline geometry:

```text
7,781 pixels
20 pixels per patch
389 complete patches
7,780 modeled pixels
```

The final unmatched pixel is deliberately omitted rather than creating a 390th patch containing one measurement and 19 padding values. This preserves the backbone sequence length and positional-embedding shape. It is documented rather than hidden and can be revisited in a separate architecture ablation.

A patch is eligible as a local target when at least 50% of its 20 pixels are scientifically valid. A patch with at least one valid pixel remains attendable by the transformer. A patch with no valid pixels is key-padded and excluded from every local target.

### Validity embedding

Invalid normalized flux is zero, but zero is also a legitimate normalized measurement. V2 therefore computes:

```text
token = flux_projection(flux_patch)
        + validity_projection(valid_pixel_pattern - 1)
```

Subtracting one means a completely valid patch has a zero validity offset. Missing pixels create a learned offset. The validity projection is initialized to zero, so it cannot perturb the initial flux representation but can learn useful missingness handling.

Validity is metadata, not a second physical measurement channel. It remains independent from artificial masking.

### Artificial LeJEPA masks

Each spectrum produces exactly two views. For each view, the transform masks contiguous spans until approximately 30% of eligible patches are hidden. Current spans contain 2 through 12 patches. The second view is forbidden from masking positions already masked in the first view, so every artificially hidden position has an unmasked corresponding representation in the other view.

The 30% ratio, 2-to-12-patch spans, and disjointness policy are smoke-test defaults pending full-run review.

An artificial mask does not zero a patch and does not key-pad it. The model replaces that patch embedding with a learned `mask_token`, adds the absolute positional embedding, and keeps the position attendable. Its output token can therefore summarize the visible spectral context at the hidden wavelength.

### Backbone and heads

The deployed backbone remains:

```text
patch projection        Linear(20, 768)
CLS token               one learned 768-vector
absolute positions      390 learned 768-vectors, including CLS
transformer depth       12
attention heads         12
MLP ratio               4
dropout                 0
final normalization     LayerNorm(768)
```

Both views are flattened through one shared encoder forward and restored to view-major tensors afterward.

The model returns:

```text
global_emb   [2, batch, 768]
global_proj  [2, batch, 64]
patch_emb    [2, batch, 389, 768]
patch_proj   [2, batch, 389, 64]
```

`global_proj` uses the baseline three-layer MLP projector. `patch_proj` uses a lighter training-only head: Linear(768, 768), GELU, LayerNorm, and Linear(768, 64). Both projection heads are training machinery; the 768-dimensional CLS and patch embeddings are the backbone representations.

### Global LeJEPA objective

For projected CLS representations `g[v,b]`, the per-object center is:

```text
global_center[b] = mean_over_views(g[:, b])
```

The invariance term is:

```text
global_similarity = mean((g - global_center)^2)
```

The global loss is:

```text
global_loss = (1 - lambda_global) * global_similarity
              + lambda_global * global_SIGReg
```

The current `lambda_global` is 0.05, inherited from the baseline LeJEPA setting. This loss directly trains the CLS representation to remain stable under uncertainty noise and artificial wavelength masking.

### Local LeJEPA objective

For projected patch representations `p[v,b,j]`, where `j` is an absolute wavelength patch, the center is:

```text
local_center[b,j] = mean_over_views(p[:, b, j])
```

Local similarity is the mean squared distance to that center over scientifically valid targets. Positions hidden in both views would be excluded, although disjoint masks currently prevent that case.

Because masked tokens remain contextual queries, this one symmetric objective contains two signals:

- visible-visible correspondence teaches invariance to independently sampled noise;
- masked-visible correspondence teaches a hidden wavelength token to align with the same wavelength's visible representation in the other view.

Both branches receive gradients. There is no stop-gradient, target encoder, EMA, tokenizer, or predictor.

The local loss is:

```text
local_loss = (1 - lambda_local) * local_similarity
             + lambda_local * local_SIGReg
```

The current `lambda_local` is 0.05.

### Position-stratified local SIGReg

Flattening all patch tokens would allow absolute wavelength position to create an apparently broad distribution even if object-specific content collapsed. V2 instead samples 32 wavelength positions per optimizer step and presents SIGReg with tensors organized as:

```text
[view_and_position_group, spectra_in_batch, projection_dimension]
```

Each statistical test therefore compares different objects at a fixed wavelength position. A sampled position must be a valid target for every object on every DDP rank. Position selection is deterministic from `global_step`, so all ranks call distributed collectives with identical shapes.

Local SIGReg uses 256 random slices rather than the global head's 1,024 because it evaluates many position groups. The number of sampled positions and slices are compute/quality trade-offs and remain provisional until the smoke test.

The local Epps-Pulley statistic already performs differentiable all-reduce operations on cosine and sine means. This gives each fixed-position test the effective global batch without gathering all token tensors. A separate autograd all-gather would be redundant and more memory intensive.

### Total objective

The final teacher-free loss is:

```text
total_loss = global_loss + local_loss_weight * local_loss
```

The initial `local_loss_weight` is 1.0. Global and local components, similarities, and SIGReg values are logged separately so this weight can be changed based on measured scale rather than intuition.

### Optimizer and schedule

V2 retains AdamW, learning rate `5e-4`, weight decay `0.05`, 1,000-step linear warmup from 1% of the peak rate, cosine decay, bf16 autocast, and no gradient accumulation.

Weight decay applies to matrix-like learned weights. It does not apply to:

- biases;
- LayerNorm or BatchNorm vectors;
- CLS and artificial mask tokens;
- absolute positional embeddings.

Gradient norm is clipped to 1.0 after unscaling and before the optimizer step. The unclipped norm returned by PyTorch is logged. Learning rate, batch size, weight decay, and full-run duration remain subject to smoke-test review.

### W&B identity and metrics

The default W&B identity is:

```text
project    astrojepa
run name   SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd
model      astrojepa_spectra_desi_v2_global_local_lejepa
objective  global_local_lejepa_no_teacher
```

The run config records the external dataset path, normalization, model geometry, view/mask/noise settings, global and local regularizer weights, and parameter counts.

Logged groups include:

- `train/loss`, global/local losses, similarities, and SIGReg terms;
- `collapse/*` object-to-object standard deviations for CLS and fixed-position patch features/projections;
- `data/*` valid fraction, artificial mask ratio, normalization statistics, and scale-floor rate;
- gradient norm, learning rate, epoch, and global step.

Patch collapse diagnostics compute standard deviation across objects at fixed wavelength positions. They do not flatten positions together and therefore cannot be made healthy by positional embeddings alone.

### Checkpoint and resume behavior

Every full checkpoint stores model, optimizer, scheduler, scaler, config, epoch, step within epoch, global step, and Python/Torch/CUDA RNG states. Saves write to a temporary file and atomically replace the final path.

Periodic recovery checkpoints are still written every 4,000 optimizer steps, but only the two numerically newest `step_<number>.pt` files are retained. Pruning happens only after the replacement checkpoint has been saved successfully. Epoch milestones and `complete.pt` do not match the periodic filename pattern and are never pruned. A ten-epoch run therefore finishes with 10 epoch checkpoints, 2 rolling periodic checkpoints, and 1 final checkpoint: 13 files, or approximately 14 GB at the measured checkpoint size instead of roughly 60 GB.

Mid-epoch checkpoints record the next step within the same epoch. Resume reconstructs the deterministic stream and consumes preceding batches before optimization resumes. This avoids the v1 behavior of repeating the whole epoch. Worker augmentation RNG is reconstructed rather than bitwise replayed, so resume preserves data position and optimizer state but does not promise identical noise/mask samples.

### Command-line controls

The script accepts focused operational overrides without editing source:

```text
--epochs
--max-steps
--batch-size
--num-workers
--run-name
--save-dir
--resume
--wandb-mode {online,offline,disabled}
```

All executable validation and training must use:

```text
/mnt/ssd-cluster/pranav/conda/envs/lejepa-og
```

The completed four-GPU smoke command was:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
/mnt/ssd-cluster/pranav/conda/envs/lejepa-og/bin/torchrun \
  --standalone \
  --nproc_per_node=4 \
  train-spectra-v2.py \
  --max-steps 1000 \
  --run-name SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_Smoke1000 \
  --save-dir /tmp/astrojepa-spectra-v2-smoke-1000 \
  --wandb-mode disabled
```

The run completed all 1,000 optimizer steps without NaNs, out-of-memory failures, data exhaustion, or distributed stalls. The trusted `complete.pt` reload audit reported:

```text
epoch          0
step_in_epoch  1000
global_step    1000
model tensors  174
optimizer states 168
```

The first batch took about five minutes while 16 Hugging Face streaming readers initialized. After startup, steady-state throughput was approximately 10.7 optimizer steps per second, or roughly 685 spectra per second at global batch 64. This startup cost should be measured across an epoch boundary before the full run because loader workers are currently recreated each epoch so the epoch-specific stream shuffle is propagated correctly.

Representative rank-averaged logged losses were:

| Step | Global loss | Local loss | Total loss | Learning rate |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.8788 | 0.6872 | 1.5660 | 9.95e-6 |
| 200 | 0.1236 | 0.2305 | 0.3541 | 1.04e-4 |
| 600 | 0.0991 | 0.2417 | 0.3409 | 3.02e-4 |
| 1000 | 0.1225 | 0.2372 | 0.3596 | 5.00e-4 |

These numbers establish numerical and systems health; they are not evidence of downstream representation quality. The 1,000 steps exactly cover the configured linear warmup and only about 5.8% of one training epoch. No complete training run was launched.

### Validation commands

Syntax validation:

```bash
/mnt/ssd-cluster/pranav/conda/envs/lejepa-og/bin/python -m py_compile \
  train-spectra-v2.py \
  data/SpectraTransforms.py \
  data/dataloaders.py \
  data/desiSpectra_source.py
```

Dependency-free tests:

```bash
/mnt/ssd-cluster/pranav/conda/envs/lejepa-og/bin/python -u \
  tests/test_spectra_v2.py
```

The suite covers stable splitting, normalization, mask separation, tensor shapes, finite outputs, gradient flow through both objectives, actual position-stratified SIGReg, and a real sample streamed directly from `/mnt/datasets`.

### Known constraints and open operational choices

- Dense local LeJEPA is a project extension, not an officially validated LeJEPA recipe.
- The final unmatched DESI pixel is omitted to preserve 389-token geometry.
- MMU provides a boolean pipeline mask, so original DESI bit reasons cannot be recovered.
- The v2 training loop does not yet run a validation objective or downstream probe inline.
- Noise scale, mask ratio/span, local loss weight, local SIGReg sampling, global batch, learning rate, and complete run length must be reviewed after the smoke test.
- The five-minute first-batch loader startup should be profiled at the second epoch boundary before committing to a long run.
- The complete run must not be launched without explicit user authorization.

## 2026-08-23 epoch-accounting postmortem

The first nominal ten-epoch v2 run exposed a data-loader bug that the 1,000-step
smoke test could not reach. The configured schedule was correct on paper:

- 10 epochs;
- 17,244 optimizer steps per epoch;
- global batch size 64 across four ranks;
- 172,440 intended optimizer steps total;
- 1,103,616 spectrum presentations per intended epoch.

The run instead ended after 39,047 optimizer steps. Its epoch checkpoints ended
at global steps 3,630, 7,397, 11,289, 15,616, 19,783, 23,767, 27,818, 31,530,
34,915, and 39,047. It therefore made only 2,499,008 sample presentations,
equivalent in count to about 2.26 passes over the 1,103,874-object training
split. Because different shuffled subsets were selected, this is not equivalent
to two clean complete passes.

The root cause is double worker sharding. `DesiSpectraDataset` first assigns a
balanced subset of Parquet files to each of the 16 rank/worker combinations.
Each worker then constructs a Hugging Face `IterableDataset` from its assigned
files. Hugging Face detects that iteration is occurring inside a four-worker
PyTorch `DataLoader` and automatically shards that already-partitioned stream by
four again. Each worker consequently reads only about one quarter of its
intended files. Dataset shuffle changes which inner shards are selected, so the
short nominal epochs vary in length.

The training loop compounds the problem by treating stream exhaustion as a
successfully completed epoch. W&B logs the zero-based loop index as
`train/epoch`, and epoch checkpoints advance their saved epoch counter, even
though the configured 17,244 steps were not reached. The scheduler still expects
172,440 steps, so this run also stops early in its cosine schedule. Its
`complete.pt` is a mechanically completed loop, not a completed ten-pass
training schedule, and must not be treated as the v2 result.

The required correction is to have exactly one owner of rank/worker sharding,
add an assertion that an ordinary epoch cannot end before
`steps_per_epoch`, and log both a fractional completed-data-epoch counter and the raw
zero-based loop index. A corrected full run must start from scratch; resuming
this run would preserve its biased subset history and partially consumed learning
rate schedule.

### Correction implemented

The live v2 trainer now uses one sharding layer at each scope:

- `DesiSpectraSource.balanced_file_shards(world_size)` creates four balanced
  file sets, one per DDP rank.
- Every DataLoader worker on a rank opens that rank's same file set.
- Hugging Face performs its normal automatic four-way worker partition within
  that rank.
- The non-local dataset fallback explicitly shards only by DDP rank and likewise
  delegates worker partitioning to Hugging Face.

The trainer now raises a distributed `RuntimeError` if any rank exhausts its
stream before the configured `steps_per_epoch`; early exhaustion can no longer
advance the epoch counter or produce a misleading completed-epoch checkpoint.
Completed epoch checkpoints use one-based names such as `last_epoch_1.pt`.

W&B retains `global_step` as its x-axis and now records:

- `train/epoch`: fractional number of complete dataset passes;
- `train/epoch_index`: zero-based loop index;
- `train/epoch_number`: human-readable epoch number from 1 through 10;
- `train/step_in_epoch`;
- `train/samples_seen`.

The corrected default run and checkpoint directory end in
`MeanStd_CorrectedEpochs`, keeping the invalid 39,047-step run untouched.
The syntax audit passed, and all eight dependency-free v2 tests passed,
including the synthetic four-rank/four-worker ownership regression and a sample
streamed from the external DESI dataset. A full training run was not launched.
