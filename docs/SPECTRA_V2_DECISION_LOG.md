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

The archived baseline in `baseline_train_runs_2026-08-10/` is a checkpoint of the original project and must remain unchanged. V2 will be implemented in a new directory after the open design questions below are settled.

## Scope

### V2

- Train only on MMU DESI EDR spectra.
- Keep the deployed backbone architecture at 12 transformer layers, embedding dimension 768, 12 attention heads, and non-overlapping 20-pixel patches unless a later decision explicitly changes it.
- Correct the spectrum data representation.
- Add genuine local, wavelength-resolved representation learning while retaining a strong global representation.
- Train the model from scratch.

### Later work

- V3 will expand to DESI, SDSS, VIPERS, and other suitable spectrum surveys.
- Cross-modal work will align independently trained image and spectrum encoders, followed by a separate from-scratch bidirectional cross-modal JEPA experiment.
- Decisions specific to V3 and cross-modal training will be added to this document or linked decision records when those stages begin.

## Baseline Problems Motivating V2

1. Only `spectrum.flux` is loaded. Available validity, uncertainty, wavelength, and line-spread information is ignored.
2. Artificial crop masks are called spectrum masks, but they do not describe invalid detector or pipeline pixels.
3. Invalid flux values can therefore influence patch embeddings as though they were valid measurements.
4. The eight existing "local" views are short crops whose pooled embeddings use the same global loss. There is no loss on corresponding wavelength-token representations.
5. SIGReg is currently evaluated on each GPU's local batch rather than the effective distributed batch. With batch size 16 per GPU, this gives weak distributional statistics.
6. The configured five-epoch baseline run reached only about 1.14 complete passes over DESI in its latest checkpoint.
7. There is no fixed train/validation/test partition for self-supervised model development.

## Data Contract

The DESI spectrum struct contains `flux`, `ivar`, `lambda`, `lsf_sigma`, and a boolean `mask`. The MMU version exposes a boolean mask, not the original DESI bit field, so individual pipeline mask reasons cannot be recovered from this copy.

Planned roles:

| Field | V2 role | Encoder input? | Status |
| --- | --- | --- | --- |
| `flux` | Primary physical signal | Yes, after normalization | Accepted |
| `mask` | Identify invalid/pipeline-flagged pixels | Separate validity signal | Accepted in principle; representation described below |
| `ivar` | Additional validity check and realistic noise scale | No | Accepted |
| `lambda` | Validate the claimed fixed DESI grid and map token positions | No, provided the grid is truly fixed | Accepted provisionally |
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

The semantics of `pipeline_mask=True` will be verified empirically and against the MMU conversion before implementation.

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

Status: accepted in principle. The exact validity-embedding implementation remains an engineering detail to validate with tests.

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

Not yet finalized. The revised recommendation for the initial controlled v2 is option A: global LeJEPA plus global SIGReg, and symmetric local LeJEPA plus position-stratified local SIGReg, with no EMA teacher. Option B should remain a clearly named follow-up ablation if the direct local objective fails to improve local features or downstream probes.

This recommendation prioritizes consistency with the project's LeJEPA premise and interpretability of the first v2 experiment. Option B is the more established masked-prediction recipe and may ultimately perform better, but it answers a different experimental question.

Status: open.

## Decision 7: Dataset Split

### Current decision

Create a deterministic 98/1/1 train/validation/test partition using a stable hash of `object_id`. Reuse this assignment in future spectrum and cross-modal experiments. Do not use Python's process-randomized `hash()` implementation.

Approximate counts:

- train: 1,103,912 spectra;
- validation: 11,264 spectra;
- test: 11,265 spectra.

Exact counts will depend on the deterministic hash buckets rather than forced row counts and will be recorded after the split is audited.

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

Status: train-from-scratch accepted; teacher-free local LeJEPA versus EMA-target local JEPA and the run schedule remain open.

## Proposed Training Safeguards

These are recommendations, not yet accepted decisions:

- compute SIGReg using representations gathered across all four GPUs;
- exclude biases, normalization parameters, and learned tokens from weight decay;
- clip gradient norm;
- use bf16 on A100 GPUs;
- benchmark the largest stable physical batch before fixing the learning rate;
- run a 500-step systems and stability test before the full experiment;
- log global and local losses separately;
- monitor CLS and patch-token standard deviation, covariance, and effective rank;
- save both full resumable checkpoints and encoder-only checkpoints;
- evaluate both student and EMA encoders at fixed milestones if an EMA-target experiment is run;
- audit normalization statistics, valid-pixel fractions, and mask coverage before training.

## Open Questions Before Implementation

1. Accept or revise the combined global plus local objective described above.
2. Choose symmetric local LeJEPA for initial v2 or choose the EMA-target masked-prediction alternative.
3. Choose the first-run masking ratio and span-length distribution after a mask/data audit.
4. Choose global batch target and learning rate after a short memory/throughput benchmark.
5. Choose the full-run budget and evaluation milestones.
6. Decide which downstream benchmark is the primary checkpoint-selection criterion.

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
- Revised the initial recommendation to global and local LeJEPA with SIGReg and no EMA, pending user approval.
- Retained EMA-target masked prediction as a separate follow-up ablation.
