# Galaxy10 Checkpoint Evolution

This is a clean, standalone linear-probe experiment across all healthy model
checkpoints. It does not modify existing training or evaluation code.

## Controlled protocol

- Dataset: Galaxy10 DECaLS, all 17,736 images.
- Preprocessing: deterministic bicubic resize to `140x140`, RGB float tensor in
  `[0, 1]`, no ImageNet normalization. This matches the backbone's pretraining
  input scale.
- Backbone: frozen; only a linear classifier is trained.
- Repeated holdout: stratified 80/10/10 splits with seeds 42, 43, and 44.
- Probe: deterministic full-batch multinomial linear classifier optimized by
  LBFGS from zero initialization.
- Imbalance handling: class-balanced cross-entropy.
- Regularization: L2 grid selected using validation macro-F1 only.
- Checkpoint selection: highest mean validation macro-F1, with validation
  accuracy as tie-breaker.
- Test metrics are plotted and reported, but never used to select a checkpoint.

The old corrupt `FirstTrain_VitSmallPatch14_2104` family is ignored.

## Run

```bash
bash Evals/galaxy10_checkpoint_evolution/run_all.sh
```

Four GPUs are used by default. Override with:

```bash
GPU_LIST=0,1,2,3,4,5,6,7 \
bash Evals/galaxy10_checkpoint_evolution/run_all.sh
```

The run is resumable. Completed per-checkpoint JSON files are reused.

## Outputs

All outputs remain in `Evals/galaxy10_checkpoint_evolution/results`:

- per-checkpoint repeated-holdout metrics;
- `summary.csv` and `summary.json`;
- `best_checkpoints.json`;
- macro-F1 and accuracy evolution plots for each family;
- cross-model validation comparison plots;
- per-class F1 heatmap for selected checkpoints.

## Completed result

The full 40-checkpoint run completed without errors. Selection uses validation
macro-F1 only:

- ViT-L/14: `step_52000.pt`, validation macro-F1 `0.6805 +/- 0.0077`.
- ViT-S/14: `step_21000.pt`, validation macro-F1 `0.5719 +/- 0.0058`.
- ResNet9 test run: `step_8000.pt`, validation macro-F1 `0.4387 +/- 0.0112`.

The held-out test macro-F1 values are `0.7023`, `0.5934`, and `0.4269`,
respectively. Test results were not used to choose checkpoints.
