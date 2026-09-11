# Sequential versus Joint Cross-Modal JEPA

**Completed:** 2026-09-02  
**Paired training corpus:** 307,428 DESI spectrum and Legacy Survey image pairs  
**Primary evaluation mirror:** 168,280 matched objects  
**Historical test split:** 29,697 objects  
**Physical-property test subset:** 15,993 objects

## Executive Summary

This study asks two connected questions:

1. Is it better to align independently pretrained image and spectrum encoders, or to learn both modalities jointly from scratch?
2. What information becomes shared, remains modality-private, or becomes inaccessible after alignment?

For the completed models in this repository, the jointly trained JEPA+SIGReg model is the stronger cross-modal representation. It has better exact pair retrieval, stronger cross-modal geometry, substantially higher linear cross-modal predictability, and better linear probes for every tested continuous physical property.

The result is not that joint alignment preserves everything. Both models' objective-facing 256-dimensional embeddings discard information relative to their raw backbone outputs. This is clearest for morphology:

| Image representation | Galaxy10 accuracy | Galaxy10 macro F1 |
| --- | ---: | ---: |
| Independent SSL backbone | 71.83 +/- 0.84% | 70.23 +/- 0.90% |
| Sequential aligned pooler | 59.63 +/- 1.20% | 57.53 +/- 1.14% |
| Joint raw backbone | **72.22 +/- 1.26%** | 70.19 +/- 1.30% |
| Joint aligned projection | 44.99 +/- 0.97% | 42.77 +/- 0.97% |

The raw joint image backbone retains morphology as well as the independently pretrained backbone, while the joint shared projection filters it aggressively. Thus joint training did not simply erase morphology from the encoder. It learned a shared bottleneck that emphasizes information predictable from spectra.

The strongest defensible conclusion is:

> Joint cross-modal JEPA training with SIGReg is viable and, in this implementation, learns a richer shared image-spectrum manifold than frozen-backbone sequential alignment. Pretraining is not required to obtain a strong representation here. However, this experiment does not yet prove that pretraining is unnecessary under a fully controlled budget, because the sequential model froze its backbones and trained only small poolers for fewer epochs.

## 1. Exact Experimental Definition

### Independent unimodal state

The independently pretrained backbones are:

- Image: ViT-L/14 LeJEPA checkpoint at step 52,000.
- Spectrum: corrected DESI spectrum-v2 backbone at step 172,390.
- Spectrum-v2 was trained with global and local LeJEPA-style objectives, valid-pixel handling, uncertainty-aware augmentation, and SIGReg.
- These raw outputs are denoted (U_I) and (U_S).

### Sequential cross-modal state

The sequential model starts from the two independent SSL backbones above.

- Both large backbones are frozen.
- AstroCLIP-style learned-query attention poolers are trained.
- The objective is cross-modal squared-error invariance plus separate image and spectrum SIGReg.
- The aligned output dimension is 256.
- Approximately 2.23 million alignment parameters are trainable.
- Training runs for 10 epochs, 24,000 optimizer steps.
- The aligned outputs are denoted (S_I) and (S_S).

Crucially, the raw sequential backbones are exactly the independent backbones. They cannot forget anything during this training because they are frozen. Any loss measured at the sequential stage is loss through the aligned pooler, not destructive modification of the source backbone.

### Joint cross-modal state

The joint model initializes both encoders randomly.

- Image encoder: ViT-L/14.
- Spectrum encoder: 12-layer, 768-dimensional transformer.
- Both encoders and their 256-dimensional projection heads train end-to-end.
- The objective averages squared-error invariance over four image-view/spectrum-view pairs.
- Separate distributed SIGReg terms prevent collapse in each modality.
- There are no image-image, spectrum-spectrum, local, predictor, or EMA-teacher losses.
- Approximately 392 million parameters train for 50 epochs, 60,000 optimizer steps.
- Raw backbone outputs are (J_I^{raw}) and (J_S^{raw}); objective-facing outputs are (J_I) and (J_S).

### Important comparison limitation

This is not an initialization-only controlled comparison.

| Regime | Backbone initialization | Backbone updates | Trained parameters | Epochs |
| --- | --- | --- | ---: | ---: |
| Sequential | Independent SSL checkpoints | Frozen | ~2.23M | 10 |
| Joint | Random | End-to-end | ~392M | 50 |

The models use the same paired corpus and JEPA+SIGReg family of alignment objective, but differ in trainable capacity, training length, and pooling/projector structure. The results compare the two completed strategies as implemented. A claim that joint training is universally superior requires a matched full-finetuning sequential control.

## 2. Evaluation Design

All paired evaluations preserve the exact row order of the cached representations.

- 138,583 historical train rows and 29,697 historical test rows.
- 73,341 train and 15,993 test rows have valid PROVABGS properties.
- Continuous targets: redshift, stellar mass, metallicity, stellar age, and sSFR.
- Linear ridge probes use train-only standardization, validation-selected regularization, and three split seeds.
- Pair retrieval is exact over the complete candidate pool, not approximate.
- Neighborhoods are exact top-100 cosine neighbors within the test pool.
- CKA and sampled pairwise-cosine correlations compare representation geometry.
- Galaxy10 uses 17,736 images, three stratified 80/10/10 splits, class-balanced linear probes, and identical image preprocessing for every representation.

The 307K paired representation-training corpus includes objects in the historical evaluation mirror. Labels and probe heads are held out, but the representation evaluation is transductive. Results therefore compare representations fairly to one another, but should not be presented as a fully unseen-catalog benchmark.

## 3. Downstream Information

### Linear-probe R2

| Regime and modality | Representation | Redshift | Mass | Metallicity | Age | sSFR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Independent image | Raw | 0.530 | 0.691 | 0.413 | 0.241 | 0.501 |
| Independent spectrum | Raw | 0.555 | 0.695 | 0.410 | 0.228 | 0.496 |
| Sequential image | Aligned | 0.512 | 0.673 | 0.401 | 0.210 | 0.498 |
| Sequential spectrum | Aligned | 0.574 | 0.715 | 0.427 | 0.254 | 0.533 |
| Joint image | Raw | 0.627 | 0.806 | 0.474 | 0.314 | 0.591 |
| Joint spectrum | Raw | **0.675** | **0.843** | **0.549** | **0.363** | **0.639** |
| Joint image | Aligned | 0.613 | 0.768 | 0.459 | 0.278 | 0.564 |
| Joint spectrum | Aligned | 0.632 | 0.780 | 0.448 | 0.274 | 0.570 |

Three findings matter.

First, joint raw backbones outperform the independent raw backbones on every target. On this paired corpus, a cross-only objective plus SIGReg was sufficient to learn useful unimodal encoders from scratch.

Second, the sequential pooler is asymmetric. It slightly reduces every tested image score, but improves every spectrum score. The largest spectrum gain is sSFR, from 0.496 to 0.533. The learned spectrum pooling is extracting a cleaner shared physical summary; the image pooling is filtering some image-accessible information.

Third, the aligned projection is not the whole model. Joint raw spectrum features are much stronger than joint projected features, especially for metallicity, age, and sSFR. A downstream system that uses only the 256-dimensional shared embedding leaves useful spectrum-private information behind.

## 4. Pair Alignment

### Exact retrieval in the 29,697-object test candidate pool

| Regime | Direction | R@1 | R@10 | Median rank |
| --- | --- | ---: | ---: | ---: |
| Sequential | Image to spectrum | 0.074% | 0.849% | 1,432 |
| Sequential | Spectrum to image | 0.091% | 0.892% | 1,404 |
| Joint | Image to spectrum | **0.461%** | **3.307%** | **429** |
| Joint | Spectrum to image | **0.320%** | **2.633%** | **454** |

The joint model is roughly three to four times stronger in R@10 and reduces median rank by about a factor of three. Absolute exact-instance retrieval remains low because galaxies have many astrophysically similar alternatives and exact identity is a stricter task than recovering physical similarity.

The same ranking holds against all 168,280 candidates:

- Sequential R@10: 0.20% in each direction.
- Joint R@10: 0.74% image-to-spectrum and 0.51% spectrum-to-image.

## 5. Shared Geometry

| Diagnostic | Sequential | Joint |
| --- | ---: | ---: |
| Cross-modal projected CKA | 0.720 | **0.770** |
| Cross-modal test kNN overlap at k=100 | 3.81% | **10.49%** |
| Ridge image-to-spectrum global R2 | 0.598 | **0.749** |
| Ridge spectrum-to-image global R2 | 0.627 | **0.737** |
| CCA top-10 mean correlation | 0.728 | **0.916** |
| CCA top-50 mean correlation | 0.310 | **0.675** |

Joint training does more than move paired points closer. It creates many more mutually predictable shared directions.

The independent SSL spaces already contain a few strongly corresponding astrophysical axes: their top CCA correlation is 0.930 and top-10 mean is 0.680. But an unrestricted equal-capacity linear map explains only about 2.5% of total held-out variance. In other words, a small number of shared factors are discoverable, while the full coordinate systems remain largely unrelated.

Sequential alignment modestly strengthens the leading shared axes. Joint training reorganizes a much broader fraction of the representation around cross-modal structure.

## 6. What Changed and What Was Retained?

### Geometry retention

All values below are measured on held-out test objects. Independent raw spaces are reduced with train-fitted PCA to 256 dimensions for equal-capacity comparisons.

| Comparison | Linear CKA | Pairwise geometry rho | kNN overlap@100 |
| --- | ---: | ---: | ---: |
| Independent to sequential, image | 0.443 | 0.356 | 10.78% |
| Independent to sequential, spectrum | **0.520** | **0.503** | **22.46%** |
| Independent to joint raw, image | 0.451 | 0.346 | 14.33% |
| Independent to joint raw, spectrum | 0.398 | 0.280 | 7.50% |
| Sequential to joint, image | 0.679 | 0.433 | 17.76% |
| Sequential to joint, spectrum | 0.648 | 0.459 | 7.63% |

The sequential spectrum pooler preserves more of its unimodal neighborhood structure than the sequential image pooler. The joint spectrum representation diverges most strongly from the independent spectrum geometry, yet it improves all tested physical probes. This is not simple forgetting: it is a major reorganization toward factors useful on the paired galaxy population.

### Effective rank

| Representation | Image rank | Spectrum rank |
| --- | ---: | ---: |
| Independent raw | 37.04 | 13.67 |
| Sequential aligned | 8.83 | 8.60 |
| Joint raw | 28.26 | 26.08 |
| Joint aligned | **20.19** | **17.94** |

The sequential aligned space is highly concentrated, with effective rank below 9 in both modalities despite having 256 coordinates. Joint projections use roughly twice as many effective directions. SIGReg prevents total collapse in both runs, but it does not guarantee equally rich representations; the small frozen-backbone poolers converged to a much narrower shared manifold.

## 7. Shared and Private Information

A cross-modal ridge predictor was fitted on 70% of the historical training rows, selected on 10%, and refitted on those mapping rows. Property probes were then trained on a disjoint 20% and evaluated on the final historical test set.

For a target modality:

```text
shared(target)  = prediction from the other modality
private(target) = target representation - shared(target)
```

This is an operational linear-predictability decomposition. It is not a unique information-theoretic definition of shared and private content.

### Mean accessibility over five physical targets

| Regime | Predicted target | Total R2 | Shared-component R2 | Residual-component R2 | Cross-modal global R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| Independent | Spectrum | 0.455 | 0.440 | 0.156 | 0.025 |
| Independent | Image | 0.438 | 0.459 | 0.138 | 0.025 |
| Sequential | Spectrum | 0.493 | 0.452 | 0.152 | 0.597 |
| Sequential | Image | 0.449 | 0.491 | 0.075 | 0.626 |
| Joint | Spectrum | **0.535** | **0.531** | 0.107 | **0.749** |
| Joint | Image | **0.531** | **0.534** | 0.071 | **0.736** |

The apparently high property predictability from the independent shared prediction despite low global mapping R2 is informative, not contradictory. Physical labels occupy a small set of correlated axes. A map can recover those axes while failing to explain most feature variance.

Joint training places nearly all linearly accessible tested physical information in the cross-predictable component. The residual still carries nonzero information, particularly in the spectrum direction, but much less than the total representation.

This supports a shared-physics interpretation of the joint model. It does not prove that the residual is purely modality-private astrophysics; residuals also include nonlinear shared factors, noise, and linear-map error.

## 8. Morphology as Image-Private Information

Galaxy10 provides a direct test of visual structure that spectra need not uniquely determine.

- Independent raw macro F1: 70.23%.
- Sequential aligned macro F1: 57.53%, a loss of 12.70 percentage points.
- Joint raw macro F1: 70.19%, essentially equal to independent SSL.
- Joint aligned macro F1: 42.77%, a loss of 27.42 points from the joint raw backbone.

This cleanly separates content from accessibility:

- Joint training preserves morphology in the raw image encoder.
- The joint shared projection makes much of that morphology inaccessible.
- Sequential pooling also filters morphology, but less aggressively.
- A useful multimodal architecture should expose both a shared embedding and a modality-private/raw embedding rather than forcing every downstream task through the shared bottleneck.

A corresponding spectrum-private benchmark is still missing. Line-equivalent widths, velocity dispersion, line ratios, or spectral reconstruction would be appropriate.

## 9. Modality Dominance

There is no single globally dominant modality.

- Sequential spectrum-to-image predictability is slightly stronger than image-to-spectrum (0.627 versus 0.598).
- Joint image-to-spectrum is slightly stronger than spectrum-to-image (0.749 versus 0.736).
- Joint raw spectra are strongest on all five continuous physical targets.
- Joint raw images preserve morphology as well as independent image SSL.
- Sequential spectrum pooling improves physical probes while sequential image pooling reduces them.

The best interpretation is task-dependent specialization around a strong shared core, rather than one branch universally dragging the other.

## 10. Scientific Interpretation

### What the evidence supports

1. **From-scratch joint cross-modal JEPA is successful.** SIGReg prevents collapse, and the completed joint model is stronger than the completed sequential alignment on every shared-space diagnostic.
2. **Joint training learns broad shared structure.** The gains in CKA, neighborhood overlap, CCA, and bidirectional linear predictability agree.
3. **The raw joint encoders remain useful unimodally.** They exceed the independent backbones on continuous properties, and the image branch matches independent SSL on morphology.
4. **Shared embeddings are selective bottlenecks.** They emphasize cross-predictable astrophysics and suppress modality-private content.
5. **Frozen post-training is underpowered here.** The sequential effective rank below 9, weak retrieval, and weaker shared geometry suggest the poolers did not fully reorganize the pretrained features.

### What the evidence does not support yet

1. It does not prove that SSL pretraining is generally unnecessary.
2. It does not isolate initialization from training duration, trainable capacity, or head architecture.
3. It does not show that fully fine-tuning pretrained encoders would lose to joint random initialization.
4. It does not establish literal percentages of shared and private mutual information.
5. It does not test robustness to pairing noise or smaller paired datasets.
6. It does not establish performance on a completely unseen paired catalog.

## 11. Paper-Ready Story

A strong paper structure emerging from these results is:

1. **Premise:** astronomy contains naturally paired views of the same latent physical system.
2. **Method:** direct cross-modal invariance plus modality-wise SIGReg permits stable end-to-end joint training without unimodal pretraining.
3. **Main result:** joint training learns a more cross-predictable, higher-rank shared manifold than frozen-backbone sequential alignment.
4. **Information result:** shared and raw representations serve different scientific roles. Shared embeddings concentrate common physical factors; raw branches retain modality-private information.
5. **Morphology case study:** morphology remains in the joint image backbone but is filtered from the cross-modal projection.
6. **Design implication:** retain both shared and private outputs in future multimodal foundation models.

The current evidence is strongest as a comparison of two training strategies actually used in this project. The headline should not yet be “SIGReg eliminates the need for pretraining.” A defensible headline is:

> Cross-modal JEPA with SIGReg can train large image and spectrum encoders jointly from scratch, producing stronger shared representations than frozen-backbone post-training while retaining useful private information in the raw encoders.

## 12. Next Experiments Required for the Strong Claim

These experiments require new training and were not fabricated from fixed checkpoints.

1. **Matched sequential full-finetuning control:** initialize from independent SSL, fine-tune both backbones using the same 50 epochs, optimizer-step count, architecture, projector, paired batches, and objective as joint.
2. **Initialization ablation:** random versus pretrained initialization with every other choice identical.
3. **Paired-data efficiency:** 1%, 5%, 10%, 25%, 50%, and 100% of pairs, with multiple seeds.
4. **Frozen versus gradual versus full unfreezing:** quantify when pretraining helps and when it constrains alignment.
5. **Pair corruption:** 1%, 5%, 10%, and 20% shuffled pairs to test whether joint training is more vulnerable to bad matches.
6. **Training dynamics:** save comparable checkpoints and track retrieval, rank, CKA, shared/private probes, and morphology through training.
7. **Spectrum-private tasks:** evaluate spectral lines, kinematics, and reconstruction to complement Galaxy10.
8. **Nonlinear accessibility:** compare linear and small-MLP probes to distinguish information absence from nonlinear inaccessibility.

The most decisive immediate experiment is number 1. It turns the current scientifically interesting comparison into a causal test of sequential initialization versus joint learning.

## 13. Reproducibility and Files

Core scripts:

- `study.py`: exact retrieval, CKA, rank, ridge mappings, and downstream probes.
- `analysis.py`: train-fitted PCA, exact neighbors, retention, and shared/private decomposition.
- `morphology.py`: Galaxy10 extraction and controlled probes.
- `make_outputs.py`: tables, figures, and manifest.

Machine-readable outputs:

- `results/`: complete JSON results, including repeat-level metrics.
- `tables/`: CSV tables.
- `figures/`: five report figures.
- `artifacts/manifest.json`: dataset definitions, states, and central comparison warning.
- `artifacts/labels.npz`: immutable local copy of the exact evaluation labels.
- `artifacts/galaxy10/`: local morphology embeddings for all four image states.

The large paired embedding caches and original checkpoints were read in place and were not copied, modified, or overwritten.

