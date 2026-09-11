# Astro-Worldmodels: chronological project guide

This is a map of the work in this repository as of 2026-09-09, reconstructed from Git history, source code, decision logs, checkpoints, metrics, figures, and run logs.

## How to read this guide

- Dates are the first known commit date or, for the newer uncommitted work, the date recorded inside its report or log.
- Paths are relative to this `Astro-Worldmodels/` directory unless explicitly marked as external.
- `[complete]`, `[partial]`, `[failed]`, `[not trained]`, `[superseded]`, and `[reference]` describe the artifact's present status.
- A path beginning with `origin/redshift-regression-eval:` is stored in that Git branch, not in the currently checked-out `cluster` worktree; inspect it with `git show origin/redshift-regression-eval:<path>` rather than assuming it is a current file.
- Generated families are intentionally grouped: the 228,059 GZD-5 cutouts, checkpoint sequences, per-layer attention maps, cached arrays, LaTeX build files, and Python bytecode are covered by their owning directory rather than listed one file at a time.
- The shortest summary of the project arc is: pretrained-ViT visualization -> image LeJEPA baselines -> image evaluation -> spectrum v1 -> broad evaluation branch -> spectrum v2 -> 95K paired pilot -> 307K scratch and frozen-backbone alignment -> JEPA/CLIP and sequential/joint studies -> two paper framings.

## 2026-02-20 — Project seed

### Initial Astro-Worldmodels project description `[reference; now historical]`

Introduced the image-and-spectrum world-model goal and later accumulated the June-era status, so it is useful historical context but does not describe all August–September work.

**Path:** `README.md`

### Git history and branch record `[reference]`

Preserves the committed chronology through the August cleanup plus the separate evaluation branch; most work after 2026-08-23 is still present as uncommitted working-tree material.

**Path:** `.git/`

## 2026-02-27 — First pretrained-ViT visual experiments

### CLIP ViT attention and rollout prototype `[exploratory; deprecated]`

Probed final-layer attention and attention rollout on the Red Spider JWST image, with comparison variants and saved heatmaps, masks, overlays, NumPy arrays, and input images.

**Path:** `depreciated/earlyTests/image_probing_viT.py`; `depreciated/earlyTests/RedSpider_Webb_960_apod4feb.jpg`; `depreciated/earlyTests/attn_outputs/`; `depreciated/earlyTests/astrollava_same_arch/`; `depreciated/earlyTests/jepa-compa/`; `depreciated/earlyTests/materials_test/`

### Patch-token PCA-to-RGB prototype `[exploratory; deprecated]`

Projected CLIP patch embeddings to three principal components and saved the embeddings, PCA arrays, and false-color visualization used to see whether tokens followed visible structure.

**Path:** `depreciated/earlyTests/pca_to_rgb.py`; `depreciated/earlyTests/pca_outputs/`; `depreciated/earlyTests/testJepa.png`

## 2026-03-03 to 2026-03-05 — Data loading and minimal LeJEPA prototypes

### Galaxy image streaming source `[active, evolved]`

Wrapped the `Smith42/galaxies` Hugging Face stream and became the image source used by the iterable data loader and early image-only training.

**Path:** `data/galaxies_source.py`; `data/dataloaders.py`; `data/__init__.py`

### Rough galaxy data-loader test `[deprecated]`

Exercised the first streaming loader and transformation path while the data contract and sharding behavior were still being debugged.

**Path:** `depreciated/rough_work/dataloadTest.py`

### Minimal Imagenette LeJEPA training proof `[exploratory; deprecated]`

Implemented a small end-to-end LeJEPA experiment on Imagenette before moving the objective to astronomical images.

**Path:** `depreciated/quick-trainingTest/jepa_minimal.py`; `depreciated/quick-trainingTest/test_galaxy.png`

### First basic transform and stream tests `[deprecated]`

Captured the earliest image transformations and streaming experiments that were replaced by the live astronomy-specific data pipeline.

**Path:** `data/depreciated/transforms-basic.py`; `data/depreciated/data_streamTest.py`

## 2026-03-19 to 2026-04-06 — ResNet9 image baseline

### ResNet9 backbone and shared projection MLP `[historical model]`

Added the compact residual CNN baseline, corrected its residual/activation behavior and batch-normalization momentum, and supplied the MLP projector reused by early LeJEPA training.

**Path:** `models/resnet9.py`; `models/__init__.py`

### ResNet9 configurations and profiler setup `[historical]`

Defined the baseline hyperparameters and a profiling configuration for the first reproducible image-training runs.

**Path:** `configs/train_resnet9.py`; `configs/train_resnet9_profile.py`; `configs/__init__.py`

### Image LeJEPA trainer `[active historical entry point]`

Runs distributed image-only LeJEPA over global and local galaxy crops and was used for the ResNet9 and subsequent ViT experiments.

**Path:** `train-vision.py`

### Manual-backbone and profiling trainers `[superseded]`

Recorded intermediate ways of constructing and profiling the image backbone before the main trainer stabilized.

**Path:** `depreciated/trainManualBackbone.py`; `depreciated/train_profile.py`

### ResNet9 half-epoch run `[partial but usable]`

Completed roughly half an epoch and 8,000 optimizer steps, producing the earliest surviving trained AstroJEPA checkpoint.

**Path:** `checkpoints/checkpoints-test/step_8000.pt`

## 2026-04-20 to 2026-04-21 — Astronomy image views and multi-crop training

### AstroCLIP-inspired galaxy augmentations `[active, later evolved]`

Introduced astronomy-aware cropping, rotations, flips, blur, and noise, then expanded training to two global and eight local views.

**Path:** `data/AstroTransforms.py`; `configs/config.py`; `configs/config-vitL-galaxy-images.py`

### Preserved pre-cross-modal image-transform snapshot `[reference]`

Keeps the prior form of the image transform next to the live file for line-by-line provenance after later local-data and cross-modal edits.

**Path:** `data/AstroTransforms.py.orig`

## 2026-04-22 to 2026-04-25 — ViT-S attempts and training repair

### First ViT-S/14 attempt `[failed/corrupt]`

Reached nominal 10K and 20K checkpoints, but the run was recorded as corrupt and superseded; `step_20000.orig.pt` preserves an additional original tensor snapshot.

**Path:** `checkpoints/FirstTrain_VitSmallPatch14_2104/`

### Repaired ViT-S/14 run `[complete]`

Completed 21,000 steps after fixing the training path and retains every 1,000-step checkpoint plus later loss-curve summaries.

**Path:** `checkpoints/VitSmallPatch14_2204/`

### ViT-S debugging and epoch-sharding diagnosis `[failed/informative]`

Preserves the training error, data-loader debug folders, and a standalone epoch debugger that helped stabilize multi-worker streaming.

**Path:** `checkpoints/VitSmallPatch14_DEBUG/`; `depreciated/debug_dataloader_epoch.py`

### Early Galaxy10 linear probe `[superseded protocol]`

Trained small heads on the bundled Galaxy10 HDF5 data for selected ViT-S and ViT-L checkpoints, but used an early 518-pixel input/probe setup and should not replace the later controlled evolution suite.

**Path:** `Evals/DeCals_linearProbing/galaxy10/linear_probe_galaxy10.py`; `Evals/DeCals_linearProbing/galaxy10/Galaxy10_DECals.h5`; `Evals/DeCals_linearProbing/galaxy10/output/`; `Evals/DeCals_linearProbing/linearProbing/galaxy10/output/`

## 2026-04-26 to 2026-04-28 — Main unimodal image backbone

### ViT-L/14 five-epoch LeJEPA run `[complete; selected source]`

Trained the main image backbone for 54,940 steps; later studies usually select step 52,000, and the repository path is now a symlink to the external checkpoint archive.

**Path:** `checkpoints/VitLargePatch14_OfficialTrain5_Epoch5_2504/` -> `/mnt/datasets/pranav/Astro-Worldmodels-checkpoint-archive/VitLargePatch14_OfficialTrain5_Epoch5_2504`

### ViT-L training log `[complete run log]`

Contains the console trace for the five-epoch image run, including optimizer progress and the historical W&B run provenance.

**Path:** `logs/vitlarge_run.log`

### Vision-backbone training narrative `[reference]`

Documents the image data, LeJEPA multi-crop objective, ResNet/ViT experiments, checkpoint choice, and lessons from scaling the image model.

**Path:** `docs/VISION_BACKBONE_TRAINING_README.md`

## 2026-06-06 — Pretraining-loss checkpoint analysis

### Checkpoint loss-curve evaluator `[complete]`

Replayed saved image checkpoints on held-out batches and generated invariance, SIGReg, total-loss, and generalization-gap curves plus machine-readable metrics.

**Path:** `Evals/checkpoint_loss_curves/evaluate_loss_curves.py`; `Evals/checkpoint_loss_curves/plot_loss_curves.py`; `Evals/checkpoint_loss_curves/run_all.sh`; `Evals/checkpoint_loss_curves/README.md`; `Evals/checkpoint_loss_curves/logs/`

### ResNet9 and ViT-S loss-analysis artifacts `[complete]`

Stores the evaluator's plots, CSV/JSON metrics, and best-checkpoint summaries alongside each evaluated checkpoint family.

**Path:** `checkpoints/checkpoints-test/`; `checkpoints/VitSmallPatch14_2204/`

## 2026-06-07 to 2026-06-09 — Galaxy10 checkpoint evolution

### Controlled Galaxy10 checkpoint sweep `[complete]`

Froze every available ResNet9, ViT-S, and ViT-L checkpoint, trained a consistent morphology probe, and selected ViT-L step 52K while retaining per-checkpoint JSON and logs.

**Path:** `Evals/galaxy10_checkpoint_evolution/galaxy10_probe_evolution.py`; `Evals/galaxy10_checkpoint_evolution/run_all.sh`; `Evals/galaxy10_checkpoint_evolution/logs/`; `Evals/galaxy10_checkpoint_evolution/results/`; `Evals/galaxy10_checkpoint_evolution/README.md`

### Best-model Galaxy10 radar plot `[complete]`

Turns the selected checkpoint's per-class precision, recall, and F1 results into the paper-ready radar visualization.

**Path:** `Evals/galaxy10_checkpoint_evolution/plot_best_model_radar.py`; `Evals/galaxy10_checkpoint_evolution/results/best_model_galaxy10_per_class_metrics.csv`; `Evals/galaxy10_checkpoint_evolution/results/best_model_galaxy10_per_class_metrics.json`

### README figure-generation bundle `[complete]`

Generated the method overview, checkpoint-selection dashboard, and Galaxy10 class-distribution figures in PNG and PDF form.

**Path:** `docs/generate_readme_figures.py`; `docs/assets/`

## 2026-06-18 — ViT patch and attention diagnostics

### AstroJEPA-versus-CLIP patch diagnostics `[complete]`

Compared ViT-S step 21K, ViT-L step 52K, and CLIP references using per-layer CLS attention, rollout, patch PCA, masks, overlays, raw arrays, grids, and provenance metadata.

**Path:** `Evals/vit_patch_attention_diagnostics/run_vit_diagnostics.py`; `Evals/vit_patch_attention_diagnostics/README.md`; `Evals/vit_patch_attention_diagnostics/results/`

## 2026-06-26 to 2026-06-27 — GZD-5 morphology benchmark

### GZD-5 data preparation `[complete]`

Downloaded/materialized the five-question Galaxy Zoo DECaLS benchmark, producing metadata, merged train/test Parquet tables, labels, and 228,059 image cutouts.

**Path:** `Evals/gzd5_morphology_probe/prepare_gzd5.py`; `Evals/gzd5_morphology_probe/data/`; `Evals/gzd5_morphology_probe/data/gz_decals_5/`

### GZD-5 embedding extraction and MLP probes `[complete with protocol caveat]`

Extracted ViT-L step-52K embeddings and evaluated question-specific heads, revealing a majority-collapse issue in the reproduced protocol and motivating the later fixed-head branch experiments.

**Path:** `Evals/gzd5_morphology_probe/extract_embeddings.py`; `Evals/gzd5_morphology_probe/train_gzd5_mlp.py`; `Evals/gzd5_morphology_probe/results/astro_vit_large_step_52000/`; `Evals/gzd5_morphology_probe/README.md`

### GZD-5 AstroCLIP comparison plot `[complete]`

Builds the radar comparison between local ViT results and the recorded AstroCLIP reference metrics.

**Path:** `Evals/gzd5_morphology_probe/plot_astroclip_comparison.py`

### Vendored AstroCLIP repository `[reference dependency]`

Keeps the external AstroCLIP implementation used to understand its data preparation, head design, and comparison protocol; it is reference code rather than an AstroJEPA model.

**Path:** `Evals/gzd5_morphology_probe/external/AstroCLIP/`

## 2026-06-27 to 2026-07-13 — Spectrum backbone v1

### DESI streaming source and spectrum transformations `[historical, later evolved]`

Added DESI spectrum loading, patchification, artificial crops/masks, and the original flux-only input path that underlies the v1 runs.

**Path:** `data/desiSpectra_source.py`; `data/SpectraTransforms.py`; `data/dataloaders.py`

### Original spectrum-v1 trainer `[preserved snapshot]`

Trained the first 12-layer, 768-dimensional spectrum transformer with LeJEPA; the live `train-spectra.py` is currently deleted, but the exact baseline trainer survives in the August snapshot.

**Path:** `baseline_train_runs_2026-08-10/train-spectra.py`

### Spectrum runs 0–2 `[failed/empty placeholders]`

These named attempts left no checkpoint files and record the early batch-size and pipeline failures before a usable spectrum run was obtained.

**Path:** `checkpoints/SPECTRA_run0_2706_Epoch5/`; `checkpoints/SPECTRA_run1_bs64_2706_Epoch5/`; `checkpoints/SPECTRA_run2_bs32_2706_Epoch5/`

### Spectrum run 3 `[partial]`

Produced 4K, 8K, and 12K step checkpoints plus an epoch marker at batch size 16 while the DESI streaming setup was still being stabilized.

**Path:** `checkpoints/SPECTRA_run3_bs16_2806_Epoch5_utbd_desi/`

### Spectrum run 4 `[complete v1 family]`

Extended the workable v1 configuration through 20K steps; historical downstream reports use a separately recorded v1 checkpoint at global step 15,647, so the exact evaluated tensor should be resolved from each metric file's provenance.

**Path:** `checkpoints/SPECTRA_run4_bs16_2806_Epoch5_utbd_desi/`; `docs/SPECTRA_BACKBONE_TRAINING_README.md`

## 2026-07-28 — First systematic downstream evaluation branch

### Evaluation-branch boundary `[important reference]`

The July 28–August 11 redshift, adaptation, competitor, and spectrum-diagnostic work diverged from the current branch and was never merged, so its paths below exist in Git history rather than the live directory.

**Path:** `origin/redshift-regression-eval` (tip `3cd04a0`)

### Galaxy10 image-redshift regression `[complete; branch only]`

Compared ViT-S and ViT-L frozen embeddings with ridge, kNN, and MLP-style probes across ten seeds, including the more stable clipped `z < 0.25` metrics and prediction plots.

**Path:** `origin/redshift-regression-eval:Evals/redshift_regression/`

### Galaxy10 data-download job `[utility; branch only]`

Provided the CopyQ/PBS job used to fetch the Galaxy10 HDF5 file for cluster evaluation.

**Path:** `origin/redshift-regression-eval:Evals/redshift_regression/download_galaxy10_copyq.pbs`

## 2026-08-04 to 2026-08-06 — Matched probes and image-domain adaptation

### Galaxy10 matched-head comparison `[complete; branch only]`

Evaluated the same frozen image features with controlled AION-like two-layer and AstroCLIP-like four-layer heads to separate backbone quality from probe capacity.

**Path:** `origin/redshift-regression-eval:Evals/galaxy10_matched_heads/`

### Exact AstroCLIP-sample image-redshift probe `[complete; branch only]`

Moved image-redshift evaluation to the 168,280-object `mhsotoudeh/astroclip` Parquet mirror and stored baseline/adapted metrics plus embedding-drift checks.

**Path:** `origin/redshift-regression-eval:Evals/desi_crossmatch/`

### Cross-match-specific continued image pretraining `[complete historically; branch only]`

Continued ViT-S/ViT-L LeJEPA on the AstroCLIP cross-match training split with smoke-test and CLI safeguards; the best adapted ViT-L result reached redshift ridge R2 0.55062.

**Path:** `origin/redshift-regression-eval:train-vision-continue.py`; `origin/redshift-regression-eval:configs/config_continue_astroclip.py`; `origin/redshift-regression-eval:data/astroclip_crossmatch_source.py`

### Adapted image checkpoints `[external historical artifacts]`

The branch metrics point to complete ViT-L and ViT-S adaptation tensors under another workspace, not to checkpoints currently stored in this repository.

**Path:** `/mnt/ssd-cluster/maja/Astro-Worldmodels/checkpoints/ContinuePretrain_AstroclipXmatch_checkpoints/complete.pt`; `/mnt/ssd-cluster/maja/Astro-Worldmodels/checkpoints/ContinuePretrain_AstroclipXmatch/complete.pt`

### GZD-5 fixed-head, multi-seed, and adapted-checkpoint sweep `[complete; branch only]`

Extended the original morphology study to ViT-S, ViT-L, and their adapted checkpoints using both reproduced and corrected fixed-head protocols across multiple seeds.

**Path:** `origin/redshift-regression-eval:Evals/gzd5_morphology_probe/`

### Competitor comparison suite `[complete; branch only]`

Aggregated local Galaxy10, GZD-5, redshift, and spectrum results against recorded AION and AstroCLIP context into JSON, Markdown tables, and comparison plots.

**Path:** `origin/redshift-regression-eval:Evals/competitor_comparison/`

### Cluster evaluation launchers `[utility; branch only]`

Added environment, batch launcher, and tmux orchestration scripts for running the whole evaluation battery on the cluster.

**Path:** `origin/redshift-regression-eval:Evals/cluster/`

### Compact result printer `[utility; branch only]`

Provided a short console summary for the growing collection of redshift and morphology metric files.

**Path:** `origin/redshift-regression-eval:Evals/print_results.py`

## 2026-08-07 to 2026-08-10 — Spectrum-v1 downstream probes

### First spectrum-redshift probe `[complete; branch only]`

Froze the v1 spectrum encoder and measured redshift regression, establishing the historical raw-backbone baseline near R2 0.432.

**Path:** `origin/redshift-regression-eval:Evals/spectra_redshift/`

### PROVABGS physical-property probes `[complete; branch only]`

Measured stellar mass, sSFR, metallicity, and mass-weighted stellar age and documented that the MMU PROVABGS Parquet catalog, not the local VAC HDF5, supplies the derived labels.

**Path:** `origin/redshift-regression-eval:Evals/spectra_properties/`

## 2026-08-10 to 2026-08-16 — Baseline preservation and report

### Immutable baseline code snapshot `[reference]`

Copied the original image/spectrum trainers, configs, transforms, loaders, sources, and ResNet9 model before the v2 rewrite; it deliberately contains no datasets or checkpoint tensors.

**Path:** `baseline_train_runs_2026-08-10/README.md`; `baseline_train_runs_2026-08-10/train-vision.py`; `baseline_train_runs_2026-08-10/train-spectra.py`; `baseline_train_runs_2026-08-10/configs/config.py`; `baseline_train_runs_2026-08-10/configs/config-vitL-galaxy-images.py`; `baseline_train_runs_2026-08-10/configs/train_resnet9.py`; `baseline_train_runs_2026-08-10/configs/__init__.py`; `baseline_train_runs_2026-08-10/data/AstroTransforms.py`; `baseline_train_runs_2026-08-10/data/SpectraTransforms.py`; `baseline_train_runs_2026-08-10/data/dataloaders.py`; `baseline_train_runs_2026-08-10/data/galaxies_source.py`; `baseline_train_runs_2026-08-10/data/desiSpectra_source.py`; `baseline_train_runs_2026-08-10/data/__init__.py`; `baseline_train_runs_2026-08-10/models/resnet9.py`; `baseline_train_runs_2026-08-10/models/__init__.py`

### Baseline benchmark report `[complete reference]`

Collected the original image redshift, Galaxy10, GZD-5, spectrum redshift, and PROVABGS property results that later experiments use as their historical baseline.

**Path:** `AstroJEPA Baseline Benchmark report.pdf`

## 2026-08-11 — Spectrum-v1 collapse diagnosis and run-5 design

### Spectrum embedding-health and collapse anatomy `[complete; branch only]`

Mapped effective rank, eigenspectra, per-layer and pooling behavior, projection geometry, preprocessing controls, and covariates, including a fix for an initially misaligned target subsample.

**Path:** `origin/redshift-regression-eval:Evals/spectra_diagnostics/`

### Spectrum run-5 anti-collapse redesign `[implemented on branch; training state unresolved here]`

Introduced the next anti-collapse pretraining changes after the v1 geometry diagnosis, ending the evaluation branch at commit `3cd04a0` without a corresponding checkpoint in the current tree.

**Path:** `origin/redshift-regression-eval` at commit `3cd04a0`

## 2026-08-17 — Streaming throughput check

### Hugging Face streaming benchmark `[utility]`

Measures iteration and data-loading throughput for the v2-style streaming path and leaves only generated Python bytecode beside the script.

**Path:** `streaming_test/benchmark_stream.py`; `streaming_test/__pycache__/`

## 2026-08-23 to 2026-08-25 — Spectrum backbone v2

### Spectrum-v2 decision log `[canonical design record]`

Records why v2 uses 1,126,441 DESI EDR/SV3 spectra, valid-pixel mean/std normalization, ivar-driven noise, separate scientific and artificial masks, 389 fixed-position patches, and global-plus-local teacher-free LeJEPA with SIGReg.

**Path:** `docs/SPECTRA_V2_DECISION_LOG.md`; `docs/SPECTRA_BACKBONE_TRAINING_README.md`

### Spectrum-v2 data pipeline `[active]`

Implements local-Parquet DESI loading, deterministic 98/1/1 object splits, correct DDP/worker sharding, validity embeddings, uncertainty augmentation, patchification, and paired global/local views.

**Path:** `data/desiSpectra_source.py`; `data/dataloaders.py`; `data/SpectraTransforms.py`

### Spectrum-v2 trainer and tests `[complete implementation]`

Trains the 12-layer ViT-style spectrum encoder with global CLS and wavelength-corresponding local losses, validates checkpoint/accounting behavior, and retains the first implementation snapshot.

**Path:** `train-spectra-v2.py`; `tests/test_spectra_v2.py`; `train-spectra-v2.py.orig`

### Spectrum-v2 smoke test `[passed]`

Ran the real-data path for 1,000 steps before the production launch, confirming finite loss, gradients, distributed execution, and checkpoint writing.

**Path:** `train-spectra-v2.py`; `tests/test_spectra_v2.py`

### First nominal ten-epoch spectrum-v2 run `[invalid epoch accounting; preserved]`

Finished only 39,047 optimizer steps because data were double-sharded and outer-loop epochs were not true corpus passes, so its checkpoints are diagnostic rather than the final v2 model.

**Path:** `checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd/`

### Corrected spectrum-v2 production run `[complete; selected source]`

Completed ten real passes, global batch 64, and 172,390 optimizer steps on four GPUs; `complete.pt` is the supported downstream spectrum source and the repository path is an external-archive symlink.

**Path:** `checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs/` -> `/mnt/datasets/pranav/Astro-Worldmodels-checkpoint-archive/SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs`

### Post-baseline operational summary `[reference; status 2026-08-29]`

Condenses the v1 diagnosis, v2 implementation and scores, paired-data sources, 95K pilot, 307K plan, and caveats into a practical hand-off document.

**Path:** `train_v2.md`

## 2026-08-23 to 2026-08-25 — Cross-modal design and 95K pilot

### Cross-modal JEPA decision log `[canonical design record]`

Records the choice between frozen post-training and simultaneous scratch learning, the cross-only MSE-plus-SIGReg objective, data audits, accounting fixes, and the progression from 95K to 307K pairs.

**Path:** `docs/CROSS_MODAL_JEPA_DECISION_LOG.md`

### Original cross-modal decision-log snapshot `[reference]`

Preserves the earlier version of the decision record before the 307K union and later experimental outcomes were appended.

**Path:** `docs/CROSS_MODAL_JEPA_DECISION_LOG.md.orig`

### Local DR8–DESI coordinate-match attempt `[stopped; no materialized dataset retained]`

Audited a 1-arcsec local match that yielded 79,652 accepted rows with overlap concerns, then stopped the first materialization and moved to scientifically prepared pair sources.

**Path:** `docs/CROSS_MODAL_JEPA_DECISION_LOG.md`; later reusable builder at `scripts/build_cross_modal_pairs.py`

### 95,895-pair MMU loader `[complete]`

Indexed the ready-made DESI spectrum plus Legacy Survey DR10 cross-match with identity-hashed splits and exposed synchronized image/spectrum augmentations for training.

**Path:** `data/cross_modal.py`

### From-scratch cross-modal model and trainer `[complete implementation]`

Initializes both full backbones and their 256-dimensional projectors randomly, aligning all four image/spectrum view pairings with positive MSE plus per-modality SIGReg and no unimodal, local, reconstruction, teacher, or negative-pair loss.

**Path:** `models/cross_modal.py`; `train-cross-modal-scratch.py`; `tests/test_cross_modal_scratch.py`

### Original scratch-test snapshot `[reference]`

Keeps the pre-fix test suite used while distributed shapes and objective behavior were being brought up.

**Path:** `tests/test_cross_modal_scratch.py.orig`

### 95K scratch pilot `[complete but miscounted epochs]`

Reached global step 8,396 and about 1.07 million pair presentations (roughly 11.4 real passes); its saved epoch 49 is misleading because the iterable stream exhausted early.

**Path:** `checkpoints/CrossModalScratch_MMU95K_DR10_DESI_CrossOnly_SIGReg/`

### First unified frozen-backbone evaluator `[complete and reused]`

Evaluates spectrum-v2, scratch, and post-trained raw/aligned embeddings with frozen ridge probes on the AstroCLIP mirror and PROVABGS targets while placing large caches outside the repository.

**Path:** `Evals/cross_modal_backbone_eval/evaluate_frozen_backbones.py`; `Evals/cross_modal_backbone_eval/README.md`

### Initial 95K and spectrum-v2 evaluation outputs `[complete]`

Stores image/spectrum metrics for scratch step 8,396 and for spectrum-v2 steps 100K, 104K, and completion.

**Path:** `Evals/cross_modal_backbone_eval/results/crossmodal_scratch_step8396_image/`; `Evals/cross_modal_backbone_eval/results/crossmodal_scratch_step8396_spectrum/`; `Evals/cross_modal_backbone_eval/results/spectra_v2_step100000/`; `Evals/cross_modal_backbone_eval/results/spectra_v2_step104000/`; `Evals/cross_modal_backbone_eval/results/spectra_v2_complete/`

### Evaluator source snapshot `[reference]`

Preserves an earlier evaluator version next to the evolved multi-encoder implementation.

**Path:** `Evals/cross_modal_backbone_eval/evaluate_frozen_backbones.py.orig`

## 2026-08-29 — The 307K paired-data union

### Reproducible coordinate matcher and pair materializer `[complete utility]`

Matches local MMU galaxy metadata to DESI coordinates, resolves unique identities, copies selected payload columns into a paired Parquet dataset, and writes provenance metadata.

**Path:** `scripts/build_cross_modal_pairs.py`

### Combined 307,428-pair training corpus `[external data]`

Unifies 197,976 AstroCLIP-mirror rows, 95,895 MMU/LSDB rows, and 13,557 manual unique matches; trainers ignore source split labels for SSL, creating the documented transductive evaluation caveat.

**Path:** `/mnt/datasets/pranav/astroclip`; `/mnt/datasets/pranav/desi_legacysurvey_xmatch`; `/mnt/datasets/pranav/desi_dr8_manual_unique_xmatch`

### General combined-pair loader `[active]`

Discovers all three pair formats, reconciles image/spectrum representations, deduplicates identities, shards indexed rows correctly, and feeds both scratch and post-training JEPA runs.

**Path:** `data/cross_modal.py`; `data/AstroTransforms.py`; `data/SpectraTransforms.py`

### 307K joint-from-scratch JEPA+SIGReg run `[complete]`

Trained both backbones end to end for 50 epochs and 60,000 optimizer steps on four GPUs with global batch 256, producing the main joint noncontrastive checkpoint family.

**Path:** `train-cross-modal-scratch.py`; `models/cross_modal.py`; `checkpoints/CrossModalScratch_DESI307K_AllPairs_CrossOnly_SIGReg_4GPU/`

### Frozen-backbone post-trained JEPA+SIGReg adapters `[complete]`

Froze image step 52K and corrected spectrum-v2, discarded their pretraining projectors, and trained AstroCLIP-style learned-query poolers into a 256-dimensional shared space for ten epochs/24K steps.

**Path:** `train-cross-modal-posttrained.py`; `models/cross_modal_posttrained.py`; `tests/test_cross_modal_posttrained.py`; `checkpoints/CrossModalPostTrain_DESI307K_AstroCLIPPool_LeJEPA_SIGReg/`

### Durable project conversation and experiment record `[reference; status 2026-08-25]`

Captures the reasoning behind the baseline, spectrum-v2, paired objective, data choices, 95K pilot, early evaluations, and unresolved controls without reproducing the raw chat.

**Path:** `docs/ASTROJEPA_PROJECT_CONVERSATION_LOG.md`

## 2026-08-30 — Final 307K JEPA evaluation and follow-up design

### Scratch-versus-post-trained frozen evaluation `[complete]`

Measured raw and aligned image redshift plus spectrum redshift, mass, sSFR, metallicity, age, and geometry, finding the fully trained scratch system stronger on every local target in this confounded-budget comparison.

**Path:** `Evals/cross_modal_backbone_eval/FINAL_307K_COMPARISON.md`; `Evals/cross_modal_backbone_eval/results/crossmodal_scratch_307k_last_image/`; `Evals/cross_modal_backbone_eval/results/crossmodal_scratch_307k_last_spectrum/`; `Evals/cross_modal_backbone_eval/results/crossmodal_posttrained_307k_last_image/`; `Evals/cross_modal_backbone_eval/results/crossmodal_posttrained_307k_last_spectrum/`

### AstroCLIP gap and matched-nonlinear-probe analysis `[complete]`

Separated probe mismatch from representation failure, showing that spectrum redshift remains the exceptional gap while nonlinear heads recover much of the stellar-age information.

**Path:** `Evals/cross_modal_backbone_eval/ASTROCLIP_GAP_DEEP_DIVE.md`; `Evals/cross_modal_backbone_eval/results/astroclip_matched_probe_results.json`

### Spectrum-v2 50%-overlap variant `[implemented and tested; not trained]`

Changed patch stride from 20 to 10 and sequence length from 389 to 777 while deliberately keeping the global/local LeJEPA objective fixed; no production checkpoint directory was created.

**Path:** `train-spectra-v2-overlap.py`; `data/SpectraTransformsOverlap.py`; `data/dataloaders_overlap.py`; `tests/test_spectra_v2_overlap.py`; `docs/SPECTRA_V2_OVERLAP_EXPERIMENT.md`

## 2026-08-31 — Frozen InfoNCE adapters and continuation tests

### Frozen-backbone AstroCLIP/InfoNCE ablation design `[complete]`

Specified a controlled replacement of MSE+SIGReg with symmetric global-batch InfoNCE over the same frozen source backbones, 307K pairs, and learned-query adapter family.

**Path:** `docs/CROSS_MODAL_ASTROCLIP_INFONCE_ABLATION.md`

### Frozen InfoNCE implementation and distributed test `[complete]`

Implements 512-dimensional AstroCLIP-style asymmetric adapters, fixed logit scale 15.5, differentiable DDP all-gather, current-batch negatives, and the matching evaluation path.

**Path:** `train-cross-modal-posttrained-clip.py`; `models/cross_modal_posttrained_clip.py`; `data/cross_modal_clip.py`; `tests/test_cross_modal_posttrained_clip.py`; `Evals/cross_modal_backbone_eval/evaluate_posttrained_clip.py`

### Frozen InfoNCE real-data smoke run `[passed]`

Trained two real-data steps across two GPUs, validated 7,090,176 trainable adapter parameters, and wrote a small adapter-only checkpoint.

**Path:** `checkpoints/CrossModalPostTrain_AstroCLIP_InfoNCE_SMOKE/last.pt`

### Frozen InfoNCE production run `[complete]`

Completed ten epochs and 12,000 steps with global batch 256; its adapter-only checkpoint excludes backbone and optimizer tensors.

**Path:** `checkpoints/CrossModalPostTrain_DESI307K_AstroCLIP_InfoNCE/last.pt`

### Frozen InfoNCE evaluation `[complete]`

Showed that instance discrimination improves every aligned downstream probe and produces broader geometry than frozen JEPA+SIGReg adapters, although it still trails the full scratch systems.

**Path:** `Evals/cross_modal_backbone_eval/ASTROCLIP_INFONCE_ABLATION_RESULTS.md`; `Evals/cross_modal_backbone_eval/results/crossmodal_posttrained_clip_307k_last_image/`; `Evals/cross_modal_backbone_eval/results/crossmodal_posttrained_clip_307k_last_spectrum/`

### Image local-data continued pretraining `[partial; one clean epoch]`

Resumed the original image step-52K checkpoint on local MMU galaxy crops, saved a clean epoch-1 tensor at cumulative step 62,988, then ran 516 unsaved steps before manual interruption after a watcher expected the wrong filename.

**Path:** `train-vision-cpt.py`; `checkpoints/ViTL14_LeJEPA_CPT10_LocalMMU_FromStep52000/`; `logs/image-cpt10.log`

### Spectrum-v2 continued pretraining `[started then paused; no usable checkpoint]`

Prepared a ten-epoch continuation from corrected spectrum-v2 but stopped while compute was reassigned, leaving an empty checkpoint folder and an interruption log.

**Path:** `train-spectra-v2-cpt.py`; `checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_CPT10_FromEpoch10/`; `logs/spectra-cpt10.log`

## 2026-08-31 to 2026-09-01 — From-scratch CLIP control

### Scratch CLIP model, data path, and tests `[complete implementation]`

Replaced positive MSE+SIGReg with four-view symmetric InfoNCE while keeping both backbones trainable from random initialization, with dedicated single-process and distributed tests.

**Path:** `train-cross-modal-scratch-clip.py`; `models/cross_modal_scratch_clip.py`; `data/cross_modal_clip.py`; `tests/test_cross_modal_scratch_clip.py`; `tests/test_cross_modal_scratch_clip_distributed.py`

### Scratch CLIP 307K training run `[complete]`

Completed 50 epochs and 120,000 optimizer steps at global batch 128, retaining epoch 45, epoch 50, step 120K, and final checkpoints plus the full console log.

**Path:** `checkpoints/CrossModalScratch_DESI307K_AllPairs_CLIP_InfoNCE/`; `logs/crossmodal-scratch-clip.log`

### Scratch CLIP frozen evaluation `[complete]`

Found broader representations and stronger raw spectrum probes than scratch SIGReg on all five spectrum targets, but slightly weaker image redshift and a still-large spectrum-redshift gap.

**Path:** `Evals/cross_modal_backbone_eval/SCRATCH_CLIP_307K_RESULTS.md`; `Evals/cross_modal_backbone_eval/results/crossmodal_scratch_clip_307k_final_image/`; `Evals/cross_modal_backbone_eval/results/crossmodal_scratch_clip_307k_final_spectrum/`; `logs/scratch-clip-eval-image.log`; `logs/scratch-clip-eval-spectrum.log`

## 2026-09-01 — Image training-saturation audit

### Image pre-alignment checkpoint series `[complete]`

Evaluated every ViT-L checkpoint from 4K through 54,940 steps, the image-CPT epoch, and official AstroDINO under one fixed frozen-ridge protocol, finding a plateau after roughly 24K and a peak at 40K.

**Path:** `Evals/cross_modal_backbone_eval/evaluate_image_checkpoint_series.py`; `Evals/cross_modal_backbone_eval/IMAGE_PREALIGNMENT_CHECKPOINT_SERIES.md`; `Evals/cross_modal_backbone_eval/results/image_pre_alignment_checkpoint_series/`; `logs/image-redshift-series.log`; `logs/image-cpt-epoch1-eval.log`

### Evaluation master plan and score ledger `[reference; time-stamped]`

Indexes completed probes, protocols, caveats, and missing causal controls as of September 1; its line saying scratch CLIP was still training is superseded by the completed run and result report above.

**Path:** `Evals/EVALUATION_MASTER_PLAN.md`

### Evaluation-area landing page `[reference]`

Provides the older overview of the evaluation suites and where their results are stored.

**Path:** `Evals/README.md`

## 2026-09-01 — JEPA versus CLIP representation study

### Study driver `[complete]`

Runs paired retrieval, CKA, distance correlation, neighborhood overlap, linear mappings, downstream probes, decoder transfer, and effective-rank analysis for scratch JEPA versus scratch CLIP.

**Path:** `jepa vs clip/study.py`

### Figure builder `[complete]`

Converts the saved metrics into ten numbered comparison figure families covering retrieval, geometry, mappings, probes, transfer, and eigenspectra.

**Path:** `jepa vs clip/make_figures.py`; `jepa vs clip/figures/`

### Study artifacts, metrics, tables, and logs `[complete]`

Stores sample labels, nearest neighbors, Procrustes/PCA data, JSON results, seven CSV summary tables, provenance manifest, and execution logs.

**Path:** `jepa vs clip/artifacts/`; `jepa vs clip/results/`; `jepa vs clip/tables/`; `jepa vs clip/logs/`

### JEPA-versus-CLIP report `[complete]`

Concludes that CLIP is better at exact/local pair retrieval and yields broader spectrum-private representations, while JEPA better preserves global cross-modal geometry, linear predictability, and the measured image probes.

**Path:** `jepa vs clip/STUDY_REPORT.md`; `jepa vs clip/README.md`

## 2026-09-02 — Sequential versus joint representation study

### Sequential-versus-joint core study `[complete]`

Compares frozen unimodal-plus-adapter alignment with joint-from-scratch JEPA using retrieval, CKA, mappings, effective rank, and frozen downstream probes.

**Path:** `sequential vs joint/study.py`

### Shared/private information and retention analysis `[complete]`

Decomposes accessible shared versus modality-private information and quantifies how well the aligned spaces retain their source representations.

**Path:** `sequential vs joint/analysis.py`

### Galaxy10 morphology-retention analysis `[complete]`

Tests morphology in raw and aligned image spaces to show that the joint raw backbone retains useful morphology while the shared bottleneck filters some of it.

**Path:** `sequential vs joint/morphology.py`; `sequential vs joint/artifacts/galaxy10/`

### Study output builder `[complete]`

Builds six CSV tables, five numbered figures, and the manifest from the saved study results.

**Path:** `sequential vs joint/make_outputs.py`; `sequential vs joint/tables/`; `sequential vs joint/figures/`

### Sequential-versus-joint artifacts and source notes `[complete]`

Stores labels, neighbor arrays, PCA/Procrustes objects, JSON metrics, provenance, and the two analysis prompts that record how the follow-up interpretation was framed.

**Path:** `sequential vs joint/artifacts/`; `sequential vs joint/results/`; `sequential vs joint/source_notes/`

### Sequential-versus-joint report `[complete]`

Finds that joint training wins on the measured shared representation and that raw joint features retain more private information, while explicitly noting unmatched budgets and initialization histories prevent a clean causal claim.

**Path:** `sequential vs joint/STUDY_REPORT.md`; `sequential vs joint/README.md`

### September 2 project progress report `[reference]`

Provides the broadest narrative snapshot through the scratch CLIP and representation-comparison work, including scores, caveats, paths, and proposed next experiments.

**Path:** `report_2sep.md`

## 2026-09-07 — Redshift-distribution diagnosis

### Spectrum SSL versus cross-modal redshift audit `[complete]`

Compared training-population redshift coverage and showed that the spectrum SSL corpus is much broader at high redshift than the paired 307K corpus, with 44.58% versus 4.15% at `z >= 0.5`.

**Path:** `spectra-redshift-diagnostics/analyze_redshift_distributions.py`; `spectra-redshift-diagnostics/README.md`; `spectra-redshift-diagnostics/summary.json`; `spectra-redshift-diagnostics/redshift_band_counts.csv`; `spectra-redshift-diagnostics/redshift_values.npz`; `spectra-redshift-diagnostics/ssl_pretraining_redshift_distribution.png`; `spectra-redshift-diagnostics/crossmodal_redshift_distribution.png`

## 2026-09-09 — First NeurIPS/ML4PS paper framing

### Alignment-is-not-one-thing paper `[complete four-page main paper]`

Frames scratch JEPA versus scratch CLIP as a distinction between shared global geometry and exact paired retrieval, with references and appendices following the four-page main body.

**Path:** `AstroJepa_ML4PS/neurips_2026.tex`; `AstroJepa_ML4PS/neurips_2026.pdf`

### First-paper figure pipeline and assets `[complete]`

Generates the alignment-tradeoff figure and retains the JEPA overview image in raster/vector paper-ready forms.

**Path:** `AstroJepa_ML4PS/make_paper_figure.py`; `AstroJepa_ML4PS/Figures/`

### First-paper NeurIPS support files `[complete]`

Contains the bibliography, checklist, NeurIPS style file, and LaTeX auxiliary/build logs needed to reproduce and inspect the compiled manuscript.

**Path:** `AstroJepa_ML4PS/references.bib`; `AstroJepa_ML4PS/checklist.tex`; `AstroJepa_ML4PS/neurips_2026.sty`; `AstroJepa_ML4PS/neurips_2026.aux`; `AstroJepa_ML4PS/neurips_2026.bbl`; `AstroJepa_ML4PS/neurips_2026.blg`; `AstroJepa_ML4PS/neurips_2026.log`; `AstroJepa_ML4PS/neurips_2026.out`

## 2026-09-09 — Alternate AstroJEPA-centered paper framing

### AstroJEPA method paper `[complete alternate four-page main paper]`

Re-centers the story on cross-modal joint-embedding learning for images and spectra, led by the joint-from-scratch versus frozen-pooler comparison and followed by detailed appendices.

**Path:** `AstroJepa_ML4PS_2/astrojepa_2026.tex`; `AstroJepa_ML4PS_2/astrojepa_2026.pdf`; `AstroJepa_ML4PS_2/README_V2.md`

### Alternate-paper figure generator `[complete]`

Builds the AstroJEPA architecture and training-comparison figures used by the new framing.

**Path:** `AstroJepa_ML4PS_2/make_astrojepa_figures.py`; `AstroJepa_ML4PS_2/Figures/astrojepa_architecture.png`; `AstroJepa_ML4PS_2/Figures/astrojepa_architecture.pdf`; `AstroJepa_ML4PS_2/Figures/astrojepa_training_comparison.png`; `AstroJepa_ML4PS_2/Figures/astrojepa_training_comparison.pdf`

### Alternate-paper appendix figure archive `[complete]`

Copies the objective, sequential/joint, unimodal, redshift-distribution, and legacy JEPA figures into one self-contained manuscript directory.

**Path:** `AstroJepa_ML4PS_2/Figures/objectives/`; `AstroJepa_ML4PS_2/Figures/sequential/`; `AstroJepa_ML4PS_2/Figures/unimodal/`; `AstroJepa_ML4PS_2/Figures/JEPA.png`; `AstroJepa_ML4PS_2/Figures/alignment_tradeoffs.png`; `AstroJepa_ML4PS_2/Figures/alignment_tradeoffs.pdf`

### Alternate-paper NeurIPS support files `[complete]`

Contains the v2 checklist, bibliography, style, and LaTeX build products for the active alternate manuscript.

**Path:** `AstroJepa_ML4PS_2/checklist_v2.tex`; `AstroJepa_ML4PS_2/references.bib`; `AstroJepa_ML4PS_2/neurips_2026.sty`; `AstroJepa_ML4PS_2/astrojepa_2026.aux`; `AstroJepa_ML4PS_2/astrojepa_2026.bbl`; `AstroJepa_ML4PS_2/astrojepa_2026.blg`; `AstroJepa_ML4PS_2/astrojepa_2026.log`; `AstroJepa_ML4PS_2/astrojepa_2026.out`

### Preserved first-paper copy inside the alternate directory `[reference; do not confuse with active v2]`

Keeps the original `neurips_2026` source, PDF, checklist, figure generator, and build products untouched beside the new `astrojepa_2026` manuscript.

**Path:** `AstroJepa_ML4PS_2/neurips_2026.tex`; `AstroJepa_ML4PS_2/neurips_2026.pdf`; `AstroJepa_ML4PS_2/checklist.tex`; `AstroJepa_ML4PS_2/make_paper_figure.py`; `AstroJepa_ML4PS_2/neurips_2026.aux`; `AstroJepa_ML4PS_2/neurips_2026.bbl`; `AstroJepa_ML4PS_2/neurips_2026.blg`; `AstroJepa_ML4PS_2/neurips_2026.log`; `AstroJepa_ML4PS_2/neurips_2026.out`

## Shared infrastructure used across the chronology

### LeJEPA statistical-regularization package `[active dependency]`

Provides the multivariate BHEP/BHEP-M, Henze–Visagie, Henze–Zirkler, combination, and slicing tests plus the univariate Anderson–Darling, Cramér–von Mises, entropy, Epps–Pulley, Jarque–Bera, likelihood, moments, Shapiro–Wilk, Watson, and shared utilities used by SIGReg.

**Path:** `lejepa/__init__.py`; `lejepa/multivariate/__init__.py`; `lejepa/multivariate/base.py`; `lejepa/multivariate/bhep.py`; `lejepa/multivariate/bhep_m.py`; `lejepa/multivariate/comb.py`; `lejepa/multivariate/hv.py`; `lejepa/multivariate/hz.py`; `lejepa/multivariate/slicing.py`; `lejepa/univariate/__init__.py`; `lejepa/univariate/base.py`; `lejepa/univariate/anderson_darling.py`; `lejepa/univariate/cramer_von_mises.py`; `lejepa/univariate/entropy.py`; `lejepa/univariate/epps_pulley.py`; `lejepa/univariate/jarque_bera.py`; `lejepa/univariate/likelihood.py`; `lejepa/univariate/moments.py`; `lejepa/univariate/shapiro_wilk.py`; `lejepa/univariate/utils.py`; `lejepa/univariate/watson.py`

### Current model package `[active]`

Collects the historical ResNet9 and the four current cross-modal architectures: scratch JEPA, frozen JEPA poolers, frozen InfoNCE adapters, and scratch CLIP.

**Path:** `models/resnet9.py`; `models/cross_modal.py`; `models/cross_modal_posttrained.py`; `models/cross_modal_posttrained_clip.py`; `models/cross_modal_scratch_clip.py`; `models/__init__.py`

### Current data package `[active]`

Collects image/spectrum sources, unimodal transforms/loaders, overlapping-spectrum variants, combined paired datasets, and CLIP-specific paired views, with package markers and `.orig` provenance snapshots.

**Path:** `data/galaxies_source.py`; `data/desiSpectra_source.py`; `data/AstroTransforms.py`; `data/SpectraTransforms.py`; `data/SpectraTransformsOverlap.py`; `data/dataloaders.py`; `data/dataloaders_overlap.py`; `data/cross_modal.py`; `data/cross_modal_clip.py`; `data/__init__.py`; `data/AstroTransforms.py.orig`; `data/depreciated/`

### Current configuration package `[active/historical mix]`

Contains the live ViT-L image settings, a named ViT-L copy, the ResNet9 baseline, the profiling variant, and package marker.

**Path:** `configs/config.py`; `configs/config-vitL-galaxy-images.py`; `configs/train_resnet9.py`; `configs/train_resnet9_profile.py`; `configs/__init__.py`

### Tests `[active]`

Holds real-shape and distributed checks for spectrum v2, overlap, scratch JEPA, frozen JEPA, frozen InfoNCE, and scratch CLIP, including the preserved original scratch test.

**Path:** `tests/`

### Python dependencies `[active]`

Lists the training, data, evaluation, and plotting dependencies; it was expanded during the evaluation branch to include HDF5 support.

**Path:** `requirements.txt`

### Run logs `[active evidence]`

Stores the surviving training and evaluation console logs for ViT-L, image/spectrum continuation, image checkpoint sweeps, scratch CLIP, and its final image/spectrum probes.

**Path:** `logs/`

### Checkpoint root `[active artifact index]`

Contains every surviving local checkpoint family, two external-archive symlinks, the empty failed/paused placeholders, and no hidden deletion or cleanup performed by this guide.

**Path:** `checkpoints/`

### Generated Python bytecode `[generated; not a scientific artifact]`

The `__pycache__` directories are interpreter caches created by running scripts and can be ignored when tracing scientific decisions.

**Path:** `__pycache__/`; nested `__pycache__/` directories throughout the repository

### Ignore rules `[project hygiene]`

Prevent large datasets, evaluator downloads/caches, secrets, checkpoints, and generated material from being accidentally committed.

**Path:** `.gitignore`

### Local environment and W&B credentials `[secret configuration; do not publish]`

These files support local credentials/configuration and are intentionally documented only by name; their contents should never be copied into reports, commits, or issue threads.

**Path:** `.env`; `.wandb-apiKey`

## Documented ideas that were not completed as controlled experiments

### Spectrum median/MAD normalization `[proposed, not run]`

Was retained as the main robustness ablation after choosing valid-pixel mean/std for controlled spectrum-v2 training.

**Path:** `docs/SPECTRA_V2_DECISION_LOG.md`; `docs/ASTROJEPA_PROJECT_CONVERSATION_LOG.md`

### Spectrum reconstruction or masked contextual prediction `[proposed, not run]`

Was identified as the highest-value route for preserving line-position information and closing the large spectrum-redshift gap; current models have no decoder or masked-flux target.

**Path:** `Evals/cross_modal_backbone_eval/ASTROCLIP_GAP_DEEP_DIVE.md`; `report_2sep.md`

### Spectrum auxiliary mean/log-scale tokens `[proposed, not run]`

Would return normalization statistics to the model as controlled side information rather than relying on absolute flux as the primary signal.

**Path:** `docs/SPECTRA_V2_DECISION_LOG.md`; `Evals/cross_modal_backbone_eval/ASTROCLIP_GAP_DEEP_DIVE.md`

### Multi-survey spectrum v3 `[proposed, not run]`

Would expand beyond fixed-grid DESI to the locally discovered MMU SDSS and VIPERS corpora after the DESI-only v2 recipe was understood.

**Path:** `train_v2.md`; `docs/ASTROJEPA_PROJECT_CONVERSATION_LOG.md`

### Matched causal training controls `[missing]`

Still needed are object-disjoint SSL splits, matched optimizer/sample budgets, multiple representation-training seeds, scratch without SIGReg, pair-corruption controls, and clean retrieval scaling before making causal superiority claims.

**Path:** `Evals/EVALUATION_MASTER_PLAN.md`; `Evals/cross_modal_backbone_eval/FINAL_307K_COMPARISON.md`; `jepa vs clip/STUDY_REPORT.md`; `sequential vs joint/STUDY_REPORT.md`

### Broader downstream and robustness suite `[planned, not complete]`

Image-side physical properties, spectrum object classification, redshift-controlled retrieval, candidate-pool scaling, magnitude/SNR slices, and perturbation robustness remain outside the completed battery.

**Path:** `Evals/EVALUATION_MASTER_PLAN.md`; `report_2sep.md`

## 2026-09-09 — This map

### Chronological project guide `[current]`

Connects the repository's experiments, models, scripts, checkpoints, reports, figures, failures, branch-only work, and unfinished ideas without modifying or deleting any earlier artifact.

**Path:** `guide.md`
