# AstroJEPA-centered paper (V2)

This directory contains an alternative manuscript. The deliverable is
astrojepa_2026.pdf, built from astrojepa_2026.tex. The copied
neurips_2026.tex/PDF pair remains present as a historical snapshot; the first
paper directory, AstroJepa_ML4PS, is not an input to this build.

## Title and thesis

Title: “AstroJEPA: Cross-Modal Joint-Embedding Learning for Astronomical Images
and Spectra”

One-sentence thesis: AstroJEPA learns useful, non-collapsed shared
image–spectrum representations without negative pairs; among the two completed
AstroJEPA routes, joint end-to-end training is stronger than frozen-pooler
alignment, while raw backbone states preserve modality-specific science that
the shared bottleneck removes.

## Four-page main paper

1. Introduction: astronomy motivation, AstroJEPA contribution, and study
   questions.
2. AstroJEPA: architecture, exact cross-modal invariance–SIGReg objective, and
   the Pretrain→Align versus Joint-Scratch strategies.
3. Experiments:
   - useful shared physical representations;
   - central training-strategy comparison;
   - representation change and shared/private information;
   - compact scratch-CLIP control.
4. Discussion: why a frozen pooler may find a narrow intersection and why
   scientific models should expose raw plus shared states.
5. Limitations, impact, and a compact conclusion.

Main Figure 1 is a new AstroJEPA architecture/training-strategy diagram.
Main Table 1 reports all five shared-space physical probes for both modalities
and strategies.
Main Figure 2 jointly summarizes scientific utility, CKA and cross-modal
prediction, local/pair alignment, and effective rank.

## Appendix inventory

- A — Full AstroJEPA architecture and objectives.
- B — Data and preprocessing.
- C — Training and evaluation details.
- D — Independent image and spectrum SSL.
- E — Cross-modal training: pretrained versus scratch.
- F — Representation analysis before and after alignment.
- G — Detailed contrastive baselines: frozen adapters and scratch JEPA versus
  CLIP.
- H — Additional downstream results.
- I — Quantitative and qualitative diagnostic figures.
- J — Ablations, null/incomplete runs, statistical scope, and artifact map.
- NeurIPS paper checklist — completed after the supplemental material.

## Disposition of prior material

The prior paper’s JEPA-versus-CLIP framing is no longer the thesis. Its exact
retrieval, CKA, pairwise geometry, neighborhood overlap, linear prediction,
CCA, Procrustes, decoder-transfer, probe, and eigenspectrum results are retained
in Appendix G. The earlier control-oriented schematic is retained in Appendix
I, while the main architecture figure now explains AstroJEPA and its two
training routes. The sequential-versus-joint experiment, previously supporting
material, is promoted to the central main-paper result.

## Intentional exclusions from headline claims

Appendix J records every excluded run and why: the corrupt first ViT-S
prototype; invalid first spectrum-v2 epoch accounting; the qualified 95k pilot;
paused spectrum continuation; missing no-SIGReg and full-finetuning controls;
untrained image local-token design; incomplete spectrum-overlap audit; and the
unharmonized SDSS/VIPERS extension. They remain documented and are not treated
as completed evidence.

The main paper also deliberately avoids:

- claiming an initialization-only comparison, because capacity, heads, batches,
  updates, and duration differ;
- claiming unseen-object generalization, because representation evaluation is
  transductive;
- claiming training-seed uncertainty from probe-seed error bars;
- claiming that attention/PCA plots prove semantic localization;
- treating published AstroCLIP, DINOv2, or AION numbers as matched leaderboards.

## Experiment sources audited

- Evals/cross_modal_backbone_eval: unimodal checkpoints, 95k pilot, final 307k
  models, frozen InfoNCE adapters, nonlinear probes, and full error metrics.
- sequential vs joint: all downstream, retrieval, retention, decomposition,
  morphology, and rank/geometry tables plus five summary figures.
- jepa vs clip: all machine-readable tables and ten figure families.
- Evals/galaxy10_checkpoint_evolution: model selection and class analysis.
- Evals/gzd5_morphology_probe: public-split and protocol-sensitive results.
- Evals/vit_patch_attention_diagnostics: qualitative PCA/attention diagnostics.
- spectra-redshift-diagnostics: unimodal-versus-paired redshift shift.
- Evals/checkpoint_loss_curves: optimization-health diagnostics.

## Build

From this directory:

    /tmp/tectonic-bin/tectonic astrojepa_2026.tex --keep-logs --keep-intermediates

Regenerate the two new vector figures with:

    python3 make_astrojepa_figures.py

The compiled artifact has exactly four numbered main-text pages, followed by
references, Appendices A–J, and the completed NeurIPS checklist.

## Preservation guard

The original AstroJepa_ML4PS directory was treated as read-only. Its baseline
tree checksum recorded before V2 work was:

    d11f1d8447b32c7d68adb0ed287345c7db0fbb5a1f57720d8b5f3276ef9e5805
