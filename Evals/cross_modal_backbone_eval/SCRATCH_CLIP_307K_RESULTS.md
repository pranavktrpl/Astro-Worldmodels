# From-Scratch CLIP 307K Frozen-Backbone Evaluation

Date: 2026-09-01

## Experiment

This report evaluates the final checkpoint of
`CrossModalScratch_DESI307K_AllPairs_CLIP_InfoNCE`, trained from scratch for 50
epochs and 120,000 optimizer steps on the same 307,428 paired DESI objects as
the scratch LeJEPA plus SIGReg experiment. Its objective is four-view symmetric
global-batch InfoNCE over two image and two spectrum views.

## Protocol

The frozen evaluation protocol is unchanged from `FINAL_307K_COMPARISON.md`:

- AstroCLIP DESI-LS x DESI EDR mirror with 138,583 train and 29,697 test rows;
- PROVABGS subset with 73,341 train and 15,993 test rows;
- deterministic DR2 RGB images and deterministic mean/std spectra;
- frozen raw and 256-dimensional projected embeddings;
- standardized ridge probes with validation-selected L2;
- 10 image-redshift seeds and 3 spectrum-property seeds.

Reported uncertainty is the standard deviation over probe split seeds, not
over representation-training seeds.

## Headline Results

| Target | Scratch SIGReg raw | Scratch CLIP raw | Delta CLIP-SIGReg | AstroCLIP reference |
| --- | ---: | ---: | ---: | ---: |
| Image redshift | **0.62741** | 0.61012 +/- 0.00024 | -0.01729 | 0.79 |
| Spectrum redshift | 0.67556 | **0.67818 +/- 0.00048** | +0.00262 | 0.98 |
| Stellar mass | 0.84260 | **0.84597 +/- 0.00006** | +0.00337 | 0.88 |
| sSFR | 0.63919 | **0.66281 +/- 0.00007** | +0.02362 | 0.64 |
| Metallicity | 0.54944 | **0.56656 +/- 0.00003** | +0.01712 | 0.58 |
| Stellar age | 0.36579 | **0.39302 +/- 0.00036** | +0.02723 | 0.43 |

Scratch CLIP improves all five raw spectrum probes. It exceeds the recorded
AstroCLIP sSFR reference by 0.02281 under the local ridge protocol, comes within
0.01344 on metallicity, 0.03403 on mass, and 0.03698 on age, but retains a large
0.30182 spectrum-redshift gap. Image redshift drops by 0.01729 relative to
scratch SIGReg.

## Projected Spaces

| Target | Scratch SIGReg, 256d | Scratch CLIP, 256d | Delta CLIP-SIGReg | Post-trained InfoNCE, 512d |
| --- | ---: | ---: | ---: | ---: |
| Image redshift | **0.61368** | 0.60164 +/- 0.00024 | -0.01204 | 0.54174 |
| Spectrum redshift | 0.63222 | **0.65153 +/- 0.00016** | +0.01931 | 0.58684 |
| Stellar mass | 0.78151 | **0.81694 +/- 0.00007** | +0.03543 | 0.74821 |
| sSFR | 0.57007 | **0.62794 +/- 0.00005** | +0.05787 | 0.56640 |
| Metallicity | 0.44820 | **0.50891 +/- 0.00022** | +0.06071 | 0.44243 |
| Stellar age | 0.27576 | **0.35545 +/- 0.00012** | +0.07969 | 0.26426 |

The scratch CLIP projected spectrum space is substantially more informative
than both scratch SIGReg and post-trained InfoNCE on every measured spectrum
target. Its projected image redshift is lower than scratch SIGReg but remains
0.05990 above the post-trained InfoNCE adapter.

## Complete Error Metrics

| Space | Target | R2 | MAE | RMSE | NMAD | Outlier fraction |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Image raw | Redshift | 0.61012 | 0.04466 | 0.09535 | 0.03310 | 0.18651 |
| Image projected | Redshift | 0.60164 | 0.04523 | 0.09638 | 0.03365 | 0.19169 |
| Spectrum raw | Redshift | 0.67818 | 0.03721 | 0.08663 | 0.02704 | 0.12304 |
| Spectrum projected | Redshift | 0.65153 | 0.03959 | 0.09014 | 0.02907 | 0.13766 |

| Spectrum target | Raw R2 | Raw MAE | Raw RMSE | Projected R2 | Projected MAE | Projected RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stellar mass | 0.84597 | 0.19401 | 0.25947 | 0.81694 | 0.21174 | 0.28286 |
| sSFR | 0.66281 | 0.43367 | 0.64632 | 0.62794 | 0.46015 | 0.67892 |
| Metallicity | 0.56656 | 0.21249 | 0.28467 | 0.50891 | 0.22922 | 0.30302 |
| Stellar age | 0.39302 | 0.92044 | 1.37822 | 0.35545 | 0.95537 | 1.42024 |

## Geometry

| Space | Dim | Effective rank | Participation ratio | Largest eigenvalue share |
| --- | ---: | ---: | ---: | ---: |
| Scratch CLIP image raw | 1,024 | 64.27 | 29.05 | 0.101 |
| Scratch CLIP image projected | 256 | 45.37 | 28.61 | 0.102 |
| Scratch CLIP spectrum raw | 768 | 58.12 | 20.99 | 0.139 |
| Scratch CLIP spectrum projected | 256 | 44.63 | 24.27 | 0.128 |

All four spaces are clearly non-collapsed. Their effective ranks are much
higher than the corresponding scratch SIGReg spaces (28.26, 20.18, 26.07, and
17.93), so InfoNCE produces broader representations under this geometry test.

## Interpretation

This is not a universal win for InfoNCE: scratch SIGReg remains the stronger
image-redshift encoder. However, from-scratch InfoNCE is the strongest current
spectrum model on all five frozen-ridge targets and preserves substantially
more information in its projected spectrum space. The result rejects the idea
that SIGReg is required merely to make from-scratch cross-modal optimization
possible. Both objectives train non-collapsed models; they impose different
information tradeoffs.

The comparison is not compute matched. Scratch CLIP used 120,000 optimizer
steps versus 60,000 for scratch SIGReg. Both also share the existing
transductive limitation: all 307K unlabeled pairs were available during
representation training, including objects later appearing in the historical
probe test split. A causal objective comparison requires matched steps, several
training seeds, and an object-disjoint representation-training split.

## Artifacts

- Checkpoint: `checkpoints/CrossModalScratch_DESI307K_AllPairs_CLIP_InfoNCE/last.pt`
- Image metrics: `results/crossmodal_scratch_clip_307k_final_image/metrics.json`
- Spectrum metrics: `results/crossmodal_scratch_clip_307k_final_spectrum/metrics.json`
- Embedding caches: `/mnt/datasets/pranav/astrojepa_eval_cache/crossmodal_scratch_clip_307k_final_*`
