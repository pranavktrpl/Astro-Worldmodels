# Streaming Checkpoint Loss Curves

This directory contains standalone evaluation code. It imports the existing
`data.dataloaders.MyDataset` HF streaming dataset and does not modify training,
model, data, config, or checkpoint files.

## What it produces

For every checkpoint family under `checkpoints/`, the evaluator writes these
new files directly into that family's checkpoint directory:

- `loss_evolution.png` and `.pdf`: train, validation, and test LeJEPA loss.
- `invariance_evolution.png` and `.pdf`: invariance component.
- `sigreg_evolution.png` and `.pdf`: SIGReg component.
- `generalization_gap.png` and `.pdf`: validation loss minus train loss.
- `loss_evolution_metrics.json` and `.csv`: resumable numeric results.
- `best_checkpoint.json`: checkpoint with minimum validation loss.
- `loss_evaluation_errors.json`: unreadable/incompatible checkpoints, if any.

`FirstTrain_VitSmallPatch14_2104` is ignored by default because its checkpoint
files are corrupt. Passing that directory explicitly with `--checkpoint-dir`
is the only way to include it.

The shaded region is one standard error across evaluated batches. Every
checkpoint sees the same split seed and dataset epoch, making the stochastic
comparison reproducible.

## Run

Run from the project's existing conda environment:

```bash
bash Evals/checkpoint_loss_curves/run_all.sh
```

The launcher also auto-detects the existing sibling
`conda/envs/lejepa-og` environment when it is not activated.

Defaults:

- Validation: one GPU process, exactly one DataLoader worker, 64 batches.
- Test: one GPU process, exactly one DataLoader worker, 64 batches.
- Train: four GPUs, checkpoint-saved batch size and worker count, 64 batches
  per rank.
- HF loading is streaming for every split.

Train evaluation disables the existing 50,000-example shuffle buffer by
default. Refilling that buffer for every checkpoint is expensive and is not
needed for a fixed comparison sample. The underlying train split is still HF
streamed, while image augmentations and SIGReg remain stochastic and
seed-controlled. Pass `--train-shuffle` to the Python evaluator to reproduce
the training loader's shuffle behavior exactly.

Environment variables adjust the launcher without editing code:

```bash
TRAIN_GPUS=4 \
TRAIN_BATCHES=128 \
VALIDATION_BATCHES=128 \
TEST_BATCHES=128 \
bash Evals/checkpoint_loss_curves/run_all.sh
```

Set a batch count to `0` to consume the full split. A full train pass contains
8.4M examples and is usually not practical at every checkpoint; the default is
a fixed stochastic sample intended for model selection and overfitting checks.

## Targeted runs

Evaluate one family on one GPU:

```bash
python Evals/checkpoint_loss_curves/evaluate_loss_curves.py \
  --checkpoint-dir checkpoints/VitSmallPatch14_2204 \
  --splits validation test \
  --max-batches 64
```

Evaluate its train stream with the current four-GPU setup:

```bash
torchrun --standalone --nproc_per_node=4 \
  Evals/checkpoint_loss_curves/evaluate_loss_curves.py \
  --checkpoint-dir checkpoints/VitSmallPatch14_2204 \
  --splits train \
  --max-batches 64
```

Use `--overwrite` to replace matching cached measurements. Without it, completed
measurements with the same checkpoint, split, seed, batch limit, batch size,
worker count, and world size are reused.

Use `--list-only` to inspect checkpoint discovery without loading models or HF
data. Duplicate aliases for the same global step are deduplicated by default;
pass `--include-aliases` to evaluate every physical file.
