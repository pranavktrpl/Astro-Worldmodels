# ViT Patch PCA And Attention Diagnostics

This fresh eval recreates the old diagnostics in `depreciated/earlyTests` without
editing those deprecated scripts:

- PCA-to-RGB from final patch tokens.
- Layer-wise CLS attention heatmaps for every transformer block.
- Last-layer CLS attention heatmaps.
- Attention rollout heatmaps.

The default input is the same RedSpider image used by the old CLIP ViT outputs:

`depreciated/earlyTests/RedSpider_Webb_960_apod4feb.jpg`

The default checkpoints are the finalized Galaxy10-selected models:

- ViT-L: `checkpoints/VitLargePatch14_OfficialTrain5_Epoch5_2504/step_52000.pt`
- ViT-S: `checkpoints/VitSmallPatch14_2204/step_21000.pt`

Run from the repo root with the existing conda environment:

```bash
/mnt/ssd-cluster/pranav/conda/envs/lejepa-og/bin/python \
  Evals/vit_patch_attention_diagnostics/run_vit_diagnostics.py
```

Outputs are written to `Evals/vit_patch_attention_diagnostics/results/`.

The output directory contains:

- `astro_vit_large_step_52000/`: ViT-L PCA and attention PNG/NPY outputs.
- `astro_vit_small_step_21000/`: ViT-S PCA and attention PNG/NPY outputs.
- `*/cls_attention_by_layer/`: one CLS attention heatmap per transformer layer.
- `*/cls_attention_layers_grid.png`: contact sheet of all layer-wise CLS maps.
- `clip_vit_references/`: copied historical CLIP ViT maps from the deprecated evals.
- `raw_input/`: copied original raw image.
- `comparison_grid.png`: side-by-side input, PCA-RGB, last-layer attention, and rollout.
- `provenance.json`: exact checkpoint, image, model, and preprocessing metadata.

The Astro ViT inputs use RGB floats in `[0, 1]` with no ImageNet or CLIP
normalization, matching the repo's probe/eval convention for these checkpoints.
