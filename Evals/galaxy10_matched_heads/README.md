# Galaxy10 matched-heads probe

Reruns Galaxy10 classification on the recommended frozen backbones with exact
reproductions of the competitor probe heads, so the headline numbers are
head-capacity-comparable to the published ones (the confound flagged in
`Evals/README.md`):

| Head | Reproduces | Architecture | Loss |
|---|---|---|---|
| `aion_mlp` | AION-1 Galaxy Zoo 10 probe ([arXiv:2510.17960](https://arxiv.org/abs/2510.17960)) | 2-layer MLP: `Linear(dim, 256) → GELU → Dropout(0.1) → Linear(256, 10)` | cross-entropy on logits |
| `astroclip_mlp` | AstroCLIP morphology MLP (via our reimplementation in `Evals/gzd5_morphology_probe/train_gzd5_mlp.py`) | 4 linear layers, hidden 256, ReLU, dropout 0.2 | cross-entropy applied to softmax outputs (the AstroCLIP code quirk, kept deliberately) |

## What is exact vs assumed

**Exact:** both head architectures and losses.

**Assumed (recorded in each `metrics.json` under `heads_metadata`):**

- AION-1 does not publish the probe's optimizer, learning rate, epochs, or
  batch size. We use this repo's convention: Adam 1e-3, batch 256, 50 epochs,
  best-validation-loss checkpoint selection, features standardized by
  train-split statistics.
- Splits are this repo's 80/10/10 class-stratified splits with seeds 42/43/44
  (identical to `galaxy10_checkpoint_evolution`, so the linear-probe numbers
  are directly comparable). AION uses 80/20 with no validation set.
- The eval set is Galaxy10 DECaLS (~17.7k images), not AION's Galaxy Zoo 10 ×
  Legacy Survey DR10 cross-match (~8k galaxies), and not GZD-5 soft labels as
  in AstroCLIP's own morphology eval.

## Running

```bash
python Evals/galaxy10_matched_heads/matched_heads_probe.py --model all
```

Needs `Galaxy10_DECals.h5` (see `Evals/redshift_regression/download_galaxy10_copyq.pbs`)
and the recommended checkpoints (`VitLargePatch14_OfficialTrain5_Epoch5_2504/step_52000.pt`,
`VitSmallPatch14_2204/step_21000.pt`). Embeddings are shared with
`Evals/redshift_regression` — an existing cache there is reused; otherwise
extraction runs once (GPU recommended) and caches under `results/`.

## Output

`results/<model>/metrics.json` with per-seed metrics (accuracy, macro-F1,
balanced accuracy, per-class F1, confusion matrix) and a mean/std summary per
head. `Evals/competitor_comparison/compare_competitors.py` picks these up
automatically and adds them to the Galaxy10 table and chart next to the
linear-probe and published numbers.
