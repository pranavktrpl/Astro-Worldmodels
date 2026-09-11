# Frozen-Backbone AstroCLIP InfoNCE Ablation

Last updated: 2026-08-31

## Question

Does AstroCLIP's instance-discriminative alignment objective improve the shared
image-spectrum space when applied to the same pretrained AstroJEPA image and
spectrum backbones and the same 307,428 paired objects used by the LeJEPA plus
SIGReg post-training experiment?

This is an objective ablation, not a new unimodal pretraining run. The image and
spectrum source checkpoints are immutable and frozen.

## Compared Experiments

### Existing noncontrastive post-training

- Frozen image source: ViT-L/14 step 52,000.
- Frozen spectrum source: corrected-epoch spectra-v2 final checkpoint.
- Trainable modules: one learned-query attention pooler per modality.
- Shared dimension: 256.
- Objective: all cross-view image-spectrum MSE pairings plus separate SIGReg.
- Training data: the combined 307,428 DESI image-spectrum pairs.
- Duration: 10 epochs.

### New AstroCLIP alignment ablation

- The same frozen image and spectrum source checkpoints.
- Trainable modules: AstroCLIP-style learned-query cross-attention and two-layer
  MLP adapters only.
- Four attention heads, dropout 0.1, output dimension 512.
- Symmetric image-to-spectrum and spectrum-to-image InfoNCE.
- L2-normalized embeddings and fixed logit scale 15.5.
- DDP-global in-batch negatives. With 8 GPUs and local batch 32, each anchor is
  classified against 256 spectra or images, including 255 negatives.
- AdamW, learning rate 1e-4, weight decay 0.05, 1,000-step warmup, then cosine
  decay.
- The same combined 307,428 pairs and 10 epochs, yielding 12,000 optimizer
  steps and a global batch of 256.
- Image alignment views use center cropping, horizontal/vertical flips, and
  arbitrary rotation. Spectrum views preserve mean/std normalization and
  invalid-pixel masking but disable artificial JEPA masks and uncertainty noise.
- SIGReg is intentionally absent. Adding it would test a hybrid objective and
  would no longer isolate AstroCLIP's contrastive alignment from the prior
  LeJEPA objective.

## Fidelity Notes

The official AstroCLIP paper describes a queue length of 1,024, while the
released `AstroClipModel` computes symmetric cross-entropy directly over the
current batch and the released configuration uses batch size 256. This
experiment follows the executable released-code path and forms that batch
globally across DDP ranks with a differentiable all-gather.

The released code freezes both unimodal backbones. It uses a learned query,
four-head cross-attention, layer normalization, and an MLP. It applies the MLP
without a residual connection for the image head and with a residual connection
for the spectrum head; this asymmetry is preserved here.

Sources:

- AstroCLIP paper: <https://arxiv.org/html/2310.03024v2>
- Official implementation: <https://github.com/PolymathicAI/AstroCLIP>
- Source revision inspected: `e129576a16bccd25a2794be21fab34d05c608661`

## Implementation

- Trainer: `train-cross-modal-posttrained-clip.py`
- Adapter model: `models/cross_modal_posttrained_clip.py`
- Alignment data wrapper: `data/cross_modal_clip.py`
- Distributed test: `tests/test_cross_modal_posttrained_clip.py`
- Isolated evaluator: `Evals/cross_modal_backbone_eval/evaluate_posttrained_clip.py`
- Output: `checkpoints/CrossModalPostTrain_DESI307K_AstroCLIP_InfoNCE/last.pt`
- W&B run ID: `cm307kpostclipv1`

The output checkpoint format is
`cross_modal_posttrained_astroclip_adapters_v1`. It stores the two adapter state
dictionaries, source-checkpoint provenance, configuration, and progress only.
It contains no image- or spectrum-backbone tensors.

## Verification

- Python compilation passed in `lejepa-og`.
- A two-GPU test passed for adapter shapes, augmentation output, global negative
  gathering, finite bidirectional InfoNCE, and gradients through both branches.
- A two-step real-data smoke test loaded both source checkpoints, discovered all
  307,428 pairs, trained 7,090,176 adapter parameters, and wrote a valid 28.4 MB
  adapter-only checkpoint.

## Run Status

The full 8-GPU run completed on 2026-08-31 with 1,200 steps per epoch and 12,000
steps total. The final loss was 1.1263. W&B synced the run at
<https://wandb.ai/pranavktrpl-personal/astrojepa/runs/cm307kpostclipv1>.

The adapter-only final checkpoint is 28,372,437 bytes. Compared with the prior
LeJEPA plus SIGReg adapters, aligned image redshift R2 improved from 0.51175 to
0.54174. Spectrum R2 improved from 0.57393 to 0.58684 for redshift, 0.71761 to
0.74821 for mass, 0.53298 to 0.56640 for sSFR, 0.42818 to 0.44243 for
metallicity, and 0.25840 to 0.26426 for age.

The complete comparison is in
`Evals/cross_modal_backbone_eval/ASTROCLIP_INFONCE_ABLATION_RESULTS.md`.
