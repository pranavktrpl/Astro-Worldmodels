# Frozen Backbones After Simultaneous Training

This evaluation compares the latest spectrum-v2 checkpoint and both encoders
from the simultaneous cross-modal scratch run against the exact samples and
ridge protocol used by `AstroJEPA Baseline Benchmark report.pdf`.

The evaluator writes only metrics and provenance to this directory. Its large
embedding caches default to `/mnt/datasets/pranav/astrojepa_eval_cache`.

Primary representations are the raw backbone CLS embeddings. Projector outputs
are included as diagnostics for the information exposed to the LeJEPA/SIGReg
objective. The evaluator also supports `posttrained-image` and
`posttrained-spectrum`; these modes reconstruct the source backbones recorded
in a head-only checkpoint and evaluate the learned-query token poolers.

The exact AstroCLIP parquet mirror contains flux but not inverse variance or
pipeline masks. Spectrum evaluation therefore applies the training-compatible
mean/std normalization to finite flux values, with one deterministic unmasked
and noise-free view. This limitation is written into every metadata file.

Example:

```bash
CUDA_VISIBLE_DEVICES=5 \
  /mnt/ssd-cluster/pranav/conda/envs/lejepa-og/bin/python \
  Evals/cross_modal_backbone_eval/evaluate_frozen_backbones.py \
  --encoder crossmodal-image --stage all --device cuda:0
```

See `docs/ASTROJEPA_PROJECT_CONVERSATION_LOG.md` for decisions, checkpoint
provenance, interpretation, and the final result tables.


Final 307K scratch-versus-post-training results are organized in
`FINAL_307K_COMPARISON.md`.
