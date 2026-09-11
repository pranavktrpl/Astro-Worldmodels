# Frozen Image Redshift by Pretraining Step

Last updated: 2026-09-01

## Question

How did redshift information in the frozen AstroJEPA image backbone evolve over
its original unimodal pretraining run, and how does it compare with the official
pre-alignment AstroCLIP image backbone, AstroDINO?

## Protocol

- Evaluated every saved `step_*.pt` image checkpoint plus the completed
  54,940-step baseline.
- Evaluated the first clean local-data CPT epoch, resumed from step 52,000 and
  saved after 10,988 additional optimizer steps. It is plotted at cumulative
  step 62,988 as a dashed continuation from its actual source checkpoint.
- Evaluated the official `polymathic-ai/astrodino` teacher checkpoint as the
  AstroCLIP pre-alignment reference. Its released configuration contains 200
  epochs of 1,250 optimizer steps, so it is plotted at 250,000 steps.
- Used the same AstroCLIP benchmark split and DR2 RGB rendering as the existing
  frozen-backbone evaluation: 138,583 training rows and 29,697 test rows.
- Backbones were frozen. A ridge probe selected its L2 value on an internal
  validation split and was scored on the fixed test set.
- The primary table and plot use probe seed 42 only, as requested. The worker
  JSON files retain the already-computed repeat results for auditability.
- AstroJEPA uses its native 140-pixel input; AstroDINO uses its native 144-pixel
  input. Neither backbone was cross-modally aligned for this comparison.

## Results

| Model | Optimizer step | Selected L2 | Test R2 |
| --- | ---: | ---: | ---: |
| AstroJEPA | 4,000 | 100 | 0.51761 |
| AstroJEPA | 8,000 | 1 | 0.52786 |
| AstroJEPA | 12,000 | 1 | 0.52551 |
| AstroJEPA | 16,000 | 1 | 0.52723 |
| AstroJEPA | 20,000 | 1 | 0.52938 |
| AstroJEPA | 24,000 | 1 | 0.53032 |
| AstroJEPA | 28,000 | 1 | 0.53033 |
| AstroJEPA | 32,000 | 1 | 0.53018 |
| AstroJEPA | 36,000 | 1 | 0.53117 |
| **AstroJEPA** | **40,000** | **1** | **0.53167** |
| AstroJEPA | 44,000 | 1 | 0.53137 |
| AstroJEPA | 48,000 | 1 | 0.53089 |
| AstroJEPA | 52,000 | 1 | 0.53028 |
| AstroJEPA complete | 54,940 | 1 | 0.53023 |
| AstroJEPA local-data CPT, epoch 1 | 62,988 | 1 | 0.53128 |
| Official AstroCLIP AstroDINO | 250,000 | 100 | 0.52888 |

## Interpretation

The largest gain happened early: 4,000 to 8,000 steps improved R2 by 0.01025.
After about 24,000 steps, the score stayed in a narrow 0.5302-0.5317 band. The
best checkpoint was step 40,000, only 0.00144 above the completed baseline. This
is a clear plateau for this downstream property under the existing objective,
not evidence that redshift representation continued improving steadily through
the end of training.

Under this matched frozen-feature protocol, the completed AstroJEPA backbone is
0.00135 above official pre-alignment AstroDINO, and the best AstroJEPA checkpoint
is 0.00279 above it. The much larger AstroCLIP redshift number cited elsewhere
must not be treated as the raw AstroDINO point: it involves the aligned
AstroCLIP representation and/or a different reporting protocol. This experiment
therefore localizes that earlier gap away from the basic pre-alignment image
backbone.

One local-data CPT epoch improves the fixed-seed score by 0.00099 over its
step-52,000 source and by 0.00105 over the original completed baseline. It is
still 0.00039 below the original run's best step-40,000 checkpoint. The CPT
therefore nudges the representation back toward the top of the existing
plateau; it does not yet demonstrate a new scaling regime. It remains 0.00240
above official pre-alignment AstroDINO under this matched protocol.

The trainer produced a clean atomic `epoch_01.pt` checkpoint at step 62,988.
Because the stop watcher expected the wrong zero-based filename, training ran
516 additional, unsaved steps before manual interruption. Those discarded
steps do not affect the checkpoint or score reported here.

## Artifacts

- Plot: `results/image_pre_alignment_checkpoint_series/image_redshift_step_vs_r2.png`
- Vector plot: `results/image_pre_alignment_checkpoint_series/image_redshift_step_vs_r2.pdf`
- Exact table: `results/image_pre_alignment_checkpoint_series/image_redshift_by_step.csv`
- Aggregate JSON: `results/image_pre_alignment_checkpoint_series/image_redshift_by_step.json`
- Per-checkpoint metrics and logs are in the same results directory.
