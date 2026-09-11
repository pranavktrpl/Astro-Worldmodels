# AstroJEPA Evaluation Master Plan And Result Ledger

**Status date:** 2026-09-01  
**Historical benchmark source:** `AstroJEPA Baseline Benchmark report.pdf`  
**Primary current reports:** `cross_modal_backbone_eval/FINAL_307K_COMPARISON.md`,
`cross_modal_backbone_eval/ASTROCLIP_INFONCE_ABLATION_RESULTS.md`, and
`cross_modal_backbone_eval/IMAGE_PREALIGNMENT_CHECKPOINT_SERIES.md`

This is the canonical evaluation index for the project. It records what has
actually been measured, the exact protocol class behind each number, what can
be compared directly, and what is still required before making causal claims.

"External reference" means the strongest relevant value recorded in the
project materials. It is not a claim about the global state of the art.

## 1. Executive Status

### Completed

- Original image-backbone benchmark battery: cross-match and Galaxy10
  redshift, Galaxy10 morphology, GZD-5 morphology, checkpoint evolution, and
  ViT attention/patch diagnostics.
- Frozen pre-alignment image redshift probe for every original ViT-L checkpoint
  from step 4,000 through completion, plus official AstroDINO under the same
  local probe.
- Spectrum-v1 baseline and completed spectrum-v2 frozen evaluations: redshift,
  stellar mass, sSFR, metallicity, stellar age, raw/projected geometry, and
  intermediate checkpoint comparisons.
- The 95K scratch cross-modal checkpoint health evaluation.
- Final 307,428-pair scratch LeJEPA+SIGReg versus frozen-backbone post-training:
  image redshift; spectrum redshift, mass, sSFR, metallicity, and age; raw and
  aligned spaces; and embedding geometry.
- AstroCLIP-style InfoNCE post-training ablation on the same frozen source
  backbones and 307,428 pairs.
- Matched nonlinear probes on the completed scratch representations to separate
  representation deficiencies from ridge-head limitations.
- Image local-data CPT epoch 1, including its fixed-seed frozen redshift probe
  and continuation on the all-checkpoint evolution plot.

### Active but not yet a completed result

- The 307K image-spectrum CLIP-from-scratch control is still training. It is not
  included in any final comparison table until its selected checkpoint is
  evaluated with the unchanged suite.
- Spectrum CPT is paused while scratch CLIP owns GPUs 4-7.

### Still missing

- Bidirectional image-to-spectrum retrieval, including redshift-controlled
  retrieval and candidate-pool scaling.
- An object-disjoint self-supervised pair split. Existing 307K representation
  training consumed objects later used by the historical downstream split.
- Scratch without SIGReg, matched-step scratch versus post-training, and
  multiple representation-training seeds.
- Image-side PROVABGS properties, spectrum object classification, and the full
  quality/robustness suites.

## 2. Model And Experiment Inventory

| Family | Candidate | Training state | Evaluation state |
| --- | --- | --- | --- |
| Original image | ViT-L/14, steps 4K-54,940 | Complete | Full historical image suite; all-step redshift series complete |
| Adapted image | Cross-match-adapted ViT-L/14 | Complete | Historical matched redshift and morphology complete |
| Image CPT | `ViTL14_LeJEPA_CPT10_LocalMMU_FromStep52000/epoch_01.pt` | Stopped after first clean epoch checkpoint, 62,988 cumulative steps | Frozen redshift probe complete |
| Spectrum v1 | Old step-15,647 baseline | Complete | Historical ridge suite complete |
| Spectrum v2 | `SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs/complete.pt` | Complete, 172,390 steps | Frozen ridge and geometry complete |
| Scratch cross-modal SIGReg | `CrossModalScratch_DESI307K_AllPairs_CrossOnly_SIGReg_4GPU/last.pt` | Complete, 50 epochs/60K steps | Frozen ridge, nonlinear diagnostics, and geometry complete |
| Post-trained JEPA+SIGReg | `CrossModalPostTrain_DESI307K_AstroCLIPPool_LeJEPA_SIGReg/last.pt` | Complete, 10 epochs/24K steps | Raw/aligned ridge and geometry complete |
| Post-trained InfoNCE | `CrossModalPostTrain_DESI307K_AstroCLIP_InfoNCE/last.pt` | Complete, 10 epochs/12K steps | Raw/aligned ridge and geometry complete |
| Scratch CLIP | `CrossModalScratch_DESI307K_AllPairs_CLIP_InfoNCE` | Training, 50-epoch target | Not yet evaluated |

## 3. Protocol Ledger

### Frozen-ridge protocol

- Exact AstroCLIP DESI-LS x DESI EDR parquet mirror.
- Shipped cross-match split: 138,583 train and 29,697 test rows.
- PROVABGS subset: 73,341 train and 15,993 test rows from 89,334 valid matches.
- Images: deterministic DR2 grz-to-RGB rendering and native model resize.
- Spectra: deterministic, unmasked, noise-free mean/std-normalized view.
- Feature standardization uses shipped-train statistics only.
- Ridge L2 grid: `1e-4, 1e-2, 1, 1e2, 1e4`, selected on a seed-specific 10%
  carve-out of the training split.
- Image redshift normally uses ten carve-out seeds; spectrum probes use three.
- The checkpoint-series plot uses fixed probe seed 42 by user request. All ten
  completed repeats remain in its per-checkpoint JSON files.

Seed standard deviation measures probe-selection sensitivity. It is not
uncertainty over representation retraining.

### Matched nonlinear diagnostic

The completed scratch embeddings were also evaluated with the released
AstroCLIP downstream MLP shapes: Adam, dropout 0.1, batch size 64, 10 epochs;
image redshift uses one width-32 hidden layer, while spectrum tasks use the
released helper default of two width-64 hidden layers.

### Historical-report protocols

Galaxy10, GZD-5, adapted-image, and published external numbers retain their
original report protocols. They are not silently recast as frozen-ridge scores.

## 4. Current Headline Scoreboard

Primary local metric below is frozen raw-backbone ridge test R2 unless stated
otherwise.

| Target | Original baseline | Spectrum v2 / image source | 307K post-trained raw | 307K scratch raw | AstroCLIP reference | Best local gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image redshift | 0.55062 adapted image | 0.53012 original step 52K | 0.52998 | **0.62741** | 0.79 aligned image | -0.16259 |
| Spectrum redshift | 0.43200 | 0.55506 | 0.55506 | **0.67556** | 0.98 | -0.30444 |
| Stellar mass | 0.50600 | 0.69755 | 0.69755 | **0.84260** | 0.88 | -0.03740 |
| sSFR | 0.51000 | 0.49642 | 0.49642 | **0.63919** | 0.64 | -0.00081 |
| Metallicity | 0.23500 | 0.40879 | 0.40879 | **0.54944** | 0.58 | -0.03056 |
| Stellar age | 0.18500 | 0.23220 | 0.23220 | **0.36579** | 0.43 aligned spectrum | -0.06421 |

The adapted image checkpoint and original step-52K source are different
checkpoints. The 0.55062 value must not be attributed to the frozen source used
by either post-training run.

## 5. Image Results

### Original and adapted image benchmarks

| Benchmark | Head/metric | Best local | External context | Status |
| --- | --- | ---: | ---: | --- |
| DESI-LS x DESI redshift | Frozen ridge R2 | 0.55062, adapted ViT-L | AstroCLIP aligned image 0.79 | Complete |
| Galaxy10 redshift, z < 0.25 | Frozen ridge R2 | 0.748 | No recorded direct reference | Complete |
| Galaxy10 morphology | Frozen linear accuracy | 0.7183 +/- 0.0084 | DINOv2 ViT-g/14 0.714 | Complete |
| Galaxy10 morphology | Frozen linear macro-F1 | 0.7023 +/- 0.0090 | No matched published value | Complete |
| Galaxy10 morphology | 2-layer MLP accuracy | 0.724 | AION-1-L 0.872 | Complete |
| GZD-5 reproduction | Mean accuracy | 0.761 | AstroCLIP 0.761 | Complete; majority-collapse caveat |
| GZD-5 reproduction | Mean F1 | 0.695 | AstroCLIP 0.743 | Complete; broken-protocol caveat |
| GZD-5 corrected | Mean F1 | 0.655 +/- 0.013 | No valid direct reference | Complete; preferred internal metric |

Head capacity explains little of the Galaxy10 gap: the 2-layer MLP improves
accuracy by only about 0.006 over the controlled linear result.

### Original pre-alignment redshift evolution

Fixed probe seed 42:

| Step | Test R2 | Step | Test R2 |
| ---: | ---: | ---: | ---: |
| 4,000 | 0.51761 | 28,000 | 0.53033 |
| 8,000 | 0.52786 | 32,000 | 0.53018 |
| 12,000 | 0.52551 | 36,000 | 0.53117 |
| 16,000 | 0.52723 | **40,000** | **0.53167** |
| 20,000 | 0.52938 | 44,000 | 0.53137 |
| 24,000 | 0.53032 | 48,000 | 0.53089 |
|  |  | 52,000 | 0.53028 |
|  |  | 54,940 complete | 0.53023 |
|  |  | 62,988 CPT epoch 1 | 0.53128 |

Official pre-alignment AstroDINO scores 0.52888 under the same local seed-42
ridge probe. The original AstroJEPA curve plateaus around 24K steps and peaks at
40K; more identical pretraining was not producing monotonic redshift gains.
This matched raw-backbone comparison is distinct from AstroCLIP's published
approximately 0.79 aligned representation.

The first local-data CPT epoch improves R2 by 0.00099 over its step-52K source
and by 0.00105 over the original completed baseline, but remains 0.00039 below
the original step-40K peak. This is a small recovery within the plateau, not
evidence that continuation alone closes the aligned-model gap. The evaluated
`epoch_01.pt` is clean; 516 later steps were interrupted before checkpointing
after a watcher expected the wrong zero-based filename.

### Image conclusion

The raw image backbone itself is competitive with official raw AstroDINO under
the matched local probe. The larger published AstroCLIP image-redshift gain is
associated with its contrastive cross-modal alignment and evaluation protocol,
not simply a superior raw AstroDINO checkpoint. The 307K scratch SIGReg image
encoder reaches 0.62741 ridge R2, near AstroCLIP's reported roughly 0.63
unaligned-image ablation.

## 6. Spectrum Results

### Spectrum v2 final raw CLS

| Target | V1 | V2 step 104K | V2 final | Delta v2-v1 | AstroCLIP spectrum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Redshift | 0.43200 | 0.53311 | **0.55506 +/- 0.00025** | +0.12306 | 0.98 |
| Stellar mass | 0.50600 | 0.68311 | **0.69755 +/- 0.00038** | +0.19155 | 0.88 |
| sSFR | 0.51000 | 0.48796 | **0.49642 +/- 0.00013** | -0.01358 | 0.64 |
| Metallicity | 0.23500 | 0.37655 | **0.40879 +/- 0.00076** | +0.17379 | 0.58 |
| Age | 0.18500 | 0.21069 | **0.23220 +/- 0.00041** | +0.04720 | 0.43 |

V2 materially improves four of five tasks and repairs the worst baseline
failure, but it does not close the spectrum-redshift gap. sSFR remains slightly
below v1.

### Spectrum v2 objective-facing projection

| Target | Raw CLS, 768d | Projector, 64d |
| --- | ---: | ---: |
| Redshift | 0.55506 | 0.38801 |
| Stellar mass | 0.69755 | 0.54051 |
| sSFR | 0.49642 | 0.42978 |
| Metallicity | 0.40879 | 0.25930 |
| Age | 0.23220 | 0.10441 |

Raw CLS is the supported downstream surface. The projector is a training head,
not a generally stronger scientific representation.

### Spectrum v2 geometry

- Raw CLS: effective rank 13.67/768, participation ratio 9.59, largest
  eigenvalue share 0.208.
- Projector: effective rank 23.03/64, participation ratio 17.36, largest
  eigenvalue share 0.117.
- No near-constant standardized raw dimensions were found.

The representation is not constantly collapsed, but raw geometry remains
concentrated. High projector rank does not imply better downstream information.

## 7. Final 307K Scratch Versus Post-Training

### Raw spaces

| Target | Frozen-source post-training | Scratch SIGReg | Scratch gain |
| --- | ---: | ---: | ---: |
| Image redshift | 0.52998 | **0.62741** | +0.09743 |
| Spectrum redshift | 0.55506 | **0.67556** | +0.12050 |
| Stellar mass | 0.69755 | **0.84260** | +0.14505 |
| sSFR | 0.49642 | **0.63919** | +0.14277 |
| Metallicity | 0.40879 | **0.54944** | +0.14065 |
| Stellar age | 0.23220 | **0.36579** | +0.13359 |

### LeJEPA+SIGReg aligned spaces

| Target | Post-trained pooler, 256d | Scratch projection, 256d |
| --- | ---: | ---: |
| Image redshift | 0.51175 | **0.61368** |
| Spectrum redshift | 0.57393 | **0.63222** |
| Stellar mass | 0.71761 | **0.78151** |
| sSFR | 0.53298 | **0.57007** |
| Metallicity | 0.42818 | **0.44820** |
| Stellar age | 0.25840 | **0.27576** |

Scratch beats this post-training setup on every measured task. This establishes
practical feasibility of training useful, noncollapsed image and spectrum
encoders from scratch with cross-modal invariance plus per-modality SIGReg.

It does **not** establish that pretraining has no value. Scratch updated the
full model for 60K steps, while post-training updated only 2.23M pooler
parameters for 24K steps. There are no multiple representation-training seeds,
no scratch-without-SIGReg arm, and no object-disjoint retrieval result.

### Geometry

| Space | Dim | Effective rank | Participation ratio | Largest share |
| --- | ---: | ---: | ---: | ---: |
| Post-trained image raw | 1,024 | 37.03 | 15.71 | 0.183 |
| Post-trained image aligned | 256 | 8.83 | 7.28 | 0.229 |
| Post-trained spectrum raw | 768 | 13.67 | 9.59 | 0.208 |
| Post-trained spectrum aligned | 256 | 8.60 | 7.30 | 0.224 |
| Scratch image raw | 1,024 | 28.26 | 12.77 | 0.192 |
| Scratch image aligned | 256 | 20.18 | 17.14 | 0.099 |
| Scratch spectrum raw | 768 | 26.07 | 10.08 | 0.225 |
| Scratch spectrum aligned | 256 | 17.93 | 14.68 | 0.111 |

All final scratch spaces are substantially healthier than the earlier 95K
checkpoint and are clearly nonconstant.

## 8. Post-Trained InfoNCE Ablation

The InfoNCE arm freezes the same source backbones as JEPA+SIGReg post-training
and trains AstroCLIP-style learned-query adapters on the same 307,428 pairs.

| Target | JEPA+SIGReg adapter | InfoNCE adapter | Delta | Scratch SIGReg aligned |
| --- | ---: | ---: | ---: | ---: |
| Image redshift | 0.51175 | **0.54174 +/- 0.00025** | +0.02999 | 0.61368 |
| Spectrum redshift | 0.57393 | **0.58684 +/- 0.00053** | +0.01291 | 0.63222 |
| Stellar mass | 0.71761 | **0.74821 +/- 0.00007** | +0.03060 | 0.78151 |
| sSFR | 0.53298 | **0.56640 +/- 0.00009** | +0.03343 | 0.57007 |
| Metallicity | 0.42818 | **0.44243 +/- 0.00026** | +0.01425 | 0.44820 |
| Stellar age | 0.25840 | **0.26426 +/- 0.00052** | +0.00586 | 0.27576 |

InfoNCE improves every aligned post-training probe and aligned geometry. It is
the stronger tested frozen-backbone alignment objective. This falsifies the
narrow hypothesis that the previous direct-similarity post-training loss was
already optimal; it does not explain the entire AstroCLIP gap.

## 9. Nonlinear Probe Diagnostic

| Representation | Target | Ridge R2 | Released-style MLP R2 | AstroCLIP reference |
| --- | --- | ---: | ---: | ---: |
| Scratch image raw | Redshift | 0.6274 | 0.6198 +/- 0.0057 | 0.78-0.79 aligned image |
| Scratch image projected | Redshift | 0.6137 | 0.6088 +/- 0.0080 | 0.78-0.79 |
| Scratch spectrum raw | Redshift | 0.6756 | 0.7016 +/- 0.0033 | 0.98-0.99 |
| Scratch spectrum projected | Redshift | 0.6322 | 0.6402 +/- 0.0026 | 0.98-0.99 |
| Scratch spectrum raw | Age | 0.3658 | 0.4101 +/- 0.0037 | 0.43 aligned / 0.47 unaligned |
| Scratch spectrum projected | Age | 0.2758 | 0.2852 +/- 0.0034 | 0.43 / 0.47 |

The same raw-spectrum MLP reaches 0.8537 mass, 0.6679 sSFR, and 0.5803
metallicity. Thus most physical-property information is present, and much of
the apparent age gap was linear-probe mismatch. Spectrum redshift remains a
real representation failure after controlling for head shape.

## 10. Scientific Interpretation

### Image redshift

Raw AstroJEPA is already comparable with raw AstroDINO under a matched local
probe. AstroCLIP's larger aligned gain is consistent with spectrum-driven
organization of image neighborhoods plus InfoNCE instance discrimination.

### Spectrum redshift

The current cross-only scratch objective can discard spectrum-private detail
that images do not determine. Unlike AstroCLIP spectrum pretraining, it does not
reconstruct masked wavelength regions or force accurate prediction of displaced
line families. SIGReg prevents constant collapse but does not guarantee
redshift-sufficient local spectral structure.

### Stellar age

Age is nonlinear and degenerate with metallicity, dust, continuum shape, and
star-formation history. The nonlinear probe recovers much of the ridge gap.
AstroCLIP itself reports stronger age before alignment than after it, reinforcing
the need for a strong modality-private spectrum objective.

## 11. Integrity And Comparability Limitations

1. All 307K pairs were used for self-supervised representation training,
   including identities later used by the historical probe test split. No
   downstream labels were used, but this is transductive representation
   evaluation, not strict unseen-object generalization.
2. Scratch and post-training differ in trainable parameter count and paired
   optimizer budget.
3. Probe-seed uncertainty is not representation-seed uncertainty.
4. AstroCLIP published values use kNN or MLP heads and sometimes different
   labeled split sizes; local ridge comparisons are diagnostic, not bit-identical
   paper reproductions.
5. AION image references include photometry, and AION spectrum references may
   include multiple modalities. They are contextual ceilings, not direct
   unimodal comparisons.
6. The AstroCLIP mirror lacks ivar and pipeline masks, so local spectrum
   evaluation treats finite flux as valid.

## 12. Evaluation Rules Going Forward

- Split by physical object identity before representation training.
- Version split manifests and record unique objects, rows, duplicates, failed
  joins, quality cuts, and coordinate-match radius.
- Select checkpoints and probe hyperparameters using validation data only.
- Evaluate raw backbone CLS and objective-facing shared projections separately.
- Never substitute a projector score for backbone quality without labeling it.
- Regression: report R2, MAE, RMSE, bias, NMAD, and defined outlier fraction.
- Classification: report accuracy, macro-F1, balanced accuracy, per-class
  recall/F1, and confusion matrix.
- Retrieval: report Recall@1/5/10, MRR, median and normalized median rank, with
  pair-level bootstrap confidence intervals.
- Geometry: report effective rank, participation ratio, singular spectrum,
  per-dimension std, covariance concentration, and duplicate rate.
- Retrain final representations with at least three seeds before a headline
  causal claim.
- Store JSON/CSV beside every human-readable report and never overwrite prior
  result directories.

## 13. Remaining Evaluation Matrix

| Priority | Evaluation | Models | Why required | Status |
| --- | --- | --- | --- | --- |
| P0 | Bidirectional full-pool retrieval | Scratch SIGReg, post InfoNCE, scratch CLIP | Direct cross-modal task | Not run |
| P0 | Redshift-controlled retrieval | Same | Detect redshift-only shortcut | Not run |
| P0 | Object-disjoint retraining/evaluation | All causal arms | Remove transductive limitation | Not run |
| P0 | Scratch without SIGReg | Scratch architecture | Establish anti-collapse causality | Not run |
| P0 | Matched paired steps/samples | Scratch and post-training | Fair initialization comparison | Not run |
| P0 | Scratch CLIP frozen suite | Scratch CLIP final | Direct InfoNCE-vs-SIGReg scratch ablation | Training |
| P0 | Image CPT epoch-1 redshift | Image CPT | Test whether continuation moves plateau | Complete: R2 0.53128 |
| P1 | Image PROVABGS | Original, CPT, cross-modal image spaces | Physical-property image comparison | Not run |
| P1 | Spectrum galaxy/star/QSO classification | V2 and cross-modal spectrum spaces | Population health | Not run |
| P1 | S/N, ivar, mask-fraction slices | Spectrum models | Quality robustness | Not run |
| P1 | Missing-wavelength and ivar-noise stress | Spectrum models | Local spectral robustness | Not run |
| P1 | Line, equivalent-width, velocity-width probes | Spectrum models | Scientific local information | Not run |
| P1 | Magnitude/redshift/color/morphology slices | Image models | Selection-function audit | Not run |
| P1 | Label efficiency and corruptions | Image and spectrum | Robust accessibility | Not run |
| P1 | Modality classifier | Shared spaces | Test modality invariance | Not run |
| P1 | Pair corruption and candidate-pool scaling | Cross-modal models | Shortcut and scaling audit | Not run |

## 14. Revised Execution Order

1. Evaluate the completed scratch-CLIP checkpoint with the exact frozen suite.
2. Run bidirectional retrieval for scratch SIGReg, post-trained InfoNCE, and
   scratch CLIP on one frozen object-disjoint evaluation manifest.
3. Add redshift-controlled retrieval and candidate-pool scaling.
4. Build a true object-disjoint pair-training split and rerun the selected
   causal arms.
5. Run scratch without SIGReg and a matched paired-step post-training control.
6. Add image PROVABGS and spectrum object classification.
7. Add spectrum quality/local-line tests and image stratified robustness.
8. Retrain promoted final representations with at least three seeds.

## 15. Promotion Gates

### Image

Promote only if matched-sample redshift or corrected morphology improves without
a statistically meaningful regression on the other. Galaxy10-only gains do not
override degradation on the magnitude 20-21 cross-match population.

### Spectrum

Promote only if raw geometry is noncollapsed, redshift exceeds v2, all four
PROVABGS properties are non-regressing, and quality/class slices do not expose a
single-population shortcut.

### Cross-modal

Promote only if both branches remain useful in raw space, projected spaces are
noncollapsed, bidirectional retrieval beats controls, and retrieval remains
meaningful inside narrow redshift bins. Downstream R2 alone is insufficient.

The claim that SIGReg replaces unimodal pretraining additionally requires:

- scratch SIGReg to match or beat post-training under matched paired budget;
- scratch without SIGReg to collapse or materially underperform;
- object-disjoint evaluation;
- multiple representation-training seeds.

## 16. Artifact Index

### Human-readable reports

- `cross_modal_backbone_eval/FINAL_307K_COMPARISON.md`
- `cross_modal_backbone_eval/ASTROCLIP_INFONCE_ABLATION_RESULTS.md`
- `cross_modal_backbone_eval/ASTROCLIP_GAP_DEEP_DIVE.md`
- `cross_modal_backbone_eval/IMAGE_PREALIGNMENT_CHECKPOINT_SERIES.md`
- `galaxy10_checkpoint_evolution/README.md`
- `gzd5_morphology_probe/README.md`
- `vit_patch_attention_diagnostics/README.md`

### Machine-readable outputs

- `cross_modal_backbone_eval/results/spectra_v2_complete/metrics.json`
- `cross_modal_backbone_eval/results/crossmodal_scratch_307k_last_image/metrics.json`
- `cross_modal_backbone_eval/results/crossmodal_scratch_307k_last_spectrum/metrics.json`
- `cross_modal_backbone_eval/results/crossmodal_posttrained_307k_last_image/metrics.json`
- `cross_modal_backbone_eval/results/crossmodal_posttrained_307k_last_spectrum/metrics.json`
- `cross_modal_backbone_eval/results/crossmodal_posttrained_clip_307k_last_image/metrics.json`
- `cross_modal_backbone_eval/results/crossmodal_posttrained_clip_307k_last_spectrum/metrics.json`
- `cross_modal_backbone_eval/results/astroclip_matched_probe_results.json`
- `cross_modal_backbone_eval/results/image_pre_alignment_checkpoint_series/`
- `galaxy10_checkpoint_evolution/results/`
- `gzd5_morphology_probe/results/astro_vit_large_step_52000/metrics.json`

Large embedding caches are external under
`/mnt/datasets/utbd_pranav/astrojepa_eval_cache`.

## 17. External References

- AstroCLIP: <https://arxiv.org/abs/2310.03024>
- AION-1: <https://arxiv.org/abs/2510.17960>
- Galaxy10 DECaLS: <https://astronn.readthedocs.io/en/latest/galaxy10.html>

When a published value and local reproduction disagree, preserve both, label
their protocols, and use the matched local protocol for internal model choice.
