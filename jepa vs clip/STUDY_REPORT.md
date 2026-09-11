# Align or Predict? JEPA versus CLIP in Cross-Modal Galaxy Representation Learning

**Completed:** 2026-09-01  
**Models:** final 307K-pair scratch JEPA+SIGReg and scratch CLIP checkpoints  
**Primary representation:** 256-dimensional objective-facing projection  
**Dataset:** 168,280 matched DESI-LS image and DESI spectrum objects

## Executive Summary

This study asks what *kind* of multimodal alignment is induced by contrastive
and noncontrastive objectives when architecture, paired objects, epochs, and
sample exposure are held nearly fixed.

The result is not that one objective is uniformly better:

1. **CLIP is stronger at pair alignment.** On the 29,697-object test candidate
   pool, bidirectional mean R@10 is 7.98% for CLIP versus 2.97% for JEPA.
2. **JEPA is stronger at global geometric alignment.** Cross-modal projected
   CKA is 0.771 for JEPA versus 0.634 for CLIP; sampled pairwise-geometry
   Spearman correlation is 0.607 versus 0.485.
3. **CLIP is stronger at local neighborhood alignment.** At k=100, image and
   spectrum neighbor overlap is 13.79% for CLIP versus 10.49% for JEPA.
4. **JEPA is much more linearly predictable across modalities.** Held-out
   image-to-spectrum global R2 is 0.749 for JEPA versus 0.551 for CLIP;
   spectrum-to-image is 0.736 versus 0.549.
5. **CLIP retains more spectrum-private physical information.** Its raw
   spectrum backbone is better on redshift, mass, metallicity, age, and sSFR,
   with the largest gain on age (+0.027 R2) and sSFR (+0.023).

The cleanest interpretation is:

> CLIP produces stronger instance correspondence and local discriminability;
> JEPA+SIGReg produces a more globally shared and linearly transformable
> cross-modal manifold. CLIP's spectrum branch retains more modality-private
> detail, while JEPA's image branch better preserves smooth image-redshift
> structure.

This is a more nuanced result than "CLIP aligns coordinates while JEPA merely
predicts." Both objectives create direct shared coordinates, but their local,
global, and information geometry differ substantially.

## 1. Experimental Definition

### Models

`JEPA` in this report means the implemented cross-modal objective:

```text
mean MSE over four image-view/spectrum-view pairs
+ per-modality distributed SIGReg
```

It is not a canonical predictor-based context-to-target JEPA with an EMA target
encoder. `CLIP` means symmetric global-batch InfoNCE over the same four view
pairings, with normalized embeddings and learned temperature.

Both models use:

- the same ViT-L/14 image encoder;
- the same 12-layer, 768-dimensional spectrum transformer;
- 256-dimensional image and spectrum projections;
- the same 307,428 paired training objects;
- two image views and two spectrum views;
- random initialization and end-to-end backbone training;
- 50 epochs, or approximately 15.36 million sample presentations.

They are sample-matched but not optimizer-granularity matched:

| Run | GPUs | Batch/GPU | Global batch | Steps | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| JEPA+SIGReg | 4 | 64 | 256 | 60,000 | 10h 13m |
| CLIP | 4 | 32 | 128 | 120,000 | 11h 5m |

Consequently, objective, global batch, number of negatives, and optimizer-update
frequency are partially confounded. Total data exposure and wall-clock compute
are closely matched.

### Evaluation population

- 168,280 paired objects in the exact AstroCLIP parquet mirror.
- Historical split: 138,583 train and 29,697 test rows.
- Physical-property subset: 73,341 train and 15,993 test rows.
- Representation training consumed all 307K unlabeled pairs, including objects
  later present in the historical test split. Mapping and probe heads are held
  out, but the representation evaluation is transductive.

Primary pair-alignment analysis uses projected embeddings because these are the
spaces acted upon by each objective. Raw image and spectrum dimensions differ,
so raw direct cosine retrieval is not defined; raw spaces are compared with
CKA, eigenspectra, and probes.

## 2. Pair Alignment

Retrieval is exact, not approximate. Every query is ranked against either all
168,280 candidates or the 29,697-object historical test pool.

### Test candidate pool

| Model | Direction | R@1 | R@5 | R@10 | Median rank | MRR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| JEPA | Image to spectrum | 0.46% | 1.81% | 3.31% | 429 | 0.0179 |
| JEPA | Spectrum to image | 0.32% | 1.43% | 2.63% | 454 | 0.0148 |
| CLIP | Image to spectrum | **1.49%** | **5.38%** | **8.77%** | **230** | **0.0424** |
| CLIP | Spectrum to image | **1.05%** | **4.15%** | **7.19%** | **270** | **0.0345** |

### Full candidate pool

| Model | Direction | R@1 | R@5 | R@10 | Median rank |
| --- | --- | ---: | ---: | ---: | ---: |
| JEPA | Image to spectrum | 0.087% | 0.402% | 0.739% | 2,461 |
| JEPA | Spectrum to image | 0.055% | 0.256% | 0.513% | 2,577 |
| CLIP | Image to spectrum | **0.358%** | **1.403%** | **2.426%** | **1,350** |
| CLIP | Spectrum to image | **0.225%** | **0.993%** | **1.811%** | **1,555** |

CLIP is unambiguously better at exact correspondence. Nevertheless, absolute
retrieval remains low for both objectives. This is scientifically plausible:
many galaxies are visually or spectroscopically similar, and exact instance
identity is much stricter than physical similarity. Duplicate observations,
augmentation-to-evaluation shift, and false-negative structure should be
audited before interpreting low R@1 as pure model failure.

JEPA has a larger matched-minus-random cosine margin (about 0.75 versus 0.30),
yet worse rank. Mean random cosine therefore hides the high-similarity impostor
tail. Retrieval rank, not average matched cosine, is the reliable pair metric.

## 3. Geometric Alignment

### Linear CKA

| Comparison | Projected CKA, all rows | Raw CKA, test rows |
| --- | ---: | ---: |
| JEPA image versus JEPA spectrum | **0.7705** | **0.8983** |
| CLIP image versus CLIP spectrum | 0.6338 | 0.8320 |
| JEPA image versus CLIP image | 0.6895 | **0.9395** |
| JEPA spectrum versus CLIP spectrum | 0.6614 | **0.8528** |

JEPA has the stronger global cross-modal geometry. At the same time, raw
same-modality CKA across objectives is extremely high, especially for images.
The two objectives learn broadly similar backbone relational structure and
reshape it more strongly in their projection heads.

### Pairwise-distance geometry

Spearman correlations were computed from 500,000 deterministic random object
pairs using projected cosine similarity:

| Comparison | Spearman rho |
| --- | ---: |
| JEPA image versus JEPA spectrum | **0.6073** |
| CLIP image versus CLIP spectrum | 0.4845 |
| JEPA image versus CLIP image | 0.3562 |
| JEPA spectrum versus CLIP spectrum | 0.3275 |

This supports the CKA result: JEPA modalities agree more strongly on broad
pairwise organization.

### Local neighborhood consistency

Exact k-nearest-neighbor sets were computed within the 29,697-object test pool.

| k | JEPA image-spectrum overlap | CLIP image-spectrum overlap |
| ---: | ---: | ---: |
| 10 | 1.70% | **2.62%** |
| 50 | 6.26% | **9.31%** |
| 100 | 10.49% | **13.79%** |

CLIP is stronger locally despite weaker global CKA and distance correlation.
This distinction is central: contrastive learning produces better exact and
near-neighbor discrimination, while JEPA preserves more global relational
structure across modalities.

## 4. Cross-Modal Predictability

An unrestricted ridge map was selected on a train validation carve-out, refit
on the full train split, and evaluated on held-out test rows.

| Model | Direction | Global R2 | Explained variance | Mean cosine after prediction |
| --- | --- | ---: | ---: | ---: |
| JEPA | Image to spectrum | **0.7493** | **0.7493** | **0.8398** |
| JEPA | Spectrum to image | **0.7365** | **0.7365** | **0.8301** |
| CLIP | Image to spectrum | 0.5507 | 0.5507 | 0.6980 |
| CLIP | Spectrum to image | 0.5491 | 0.5491 | 0.6787 |

JEPA is dramatically more linearly transformable. This is the strongest result
for the proposed "information alignment" axis: much more of either JEPA
modality's representation can be recovered from the other with a held-out
linear map.

Regularized CCA tells a compatible but subtler story:

| Model | Test top canonical corr. | Top-10 mean | Top-50 mean | All-256 mean |
| --- | ---: | ---: | ---: | ---: |
| JEPA | **0.9838** | **0.9156** | **0.6750** | 0.2118 |
| CLIP | 0.9794 | 0.9046 | 0.6579 | **0.2156** |

JEPA has stronger leading shared factors; CLIP has a marginally broader weak
tail. That is consistent with JEPA concentrating on shared latent structure and
CLIP retaining more numerous low-strength discriminative directions.

## 5. Is the Difference Just a Rotation?

An orthogonal Procrustes transform was fit on 138,583 train objects after
per-modality standardization and evaluated on 29,697 held-out objects.

| Model | Direction | R@1 before | R@1 after | Matched cosine before | After |
| --- | --- | ---: | ---: | ---: | ---: |
| JEPA | Image to spectrum | 0.44% | **0.54%** | 0.780 | **0.806** |
| JEPA | Spectrum to image | 0.30% | **0.45%** | 0.780 | **0.806** |
| CLIP | Image to spectrum | 1.55% | **2.02%** | 0.402 | **0.595** |
| CLIP | Spectrum to image | 1.49% | **1.84%** | 0.402 | **0.595** |

Rotation helps both models but does not produce the dramatic retrieval recovery
that would indicate equivalent spaces in different coordinate systems. The
unrestricted ridge map is much more successful than an orthogonal rotation.
Cross-modal differences therefore include anisotropic scaling, subspace
selection, and modality-private variation, not merely rigid rotation.

## 6. Information Retained by Each Representation

Every cell below uses the same standardized ridge protocol and three identical
probe seeds. These labels are diagnostics of representation content.

### Raw backbones

| Space | Redshift | Mass | Metallicity | Age | sSFR |
| --- | ---: | ---: | ---: | ---: | ---: |
| JEPA image | **0.6268** | **0.8060** | **0.4743** | **0.3141** | **0.5913** |
| CLIP image | 0.6099 | 0.7928 | 0.4654 | 0.3068 | 0.5795 |
| JEPA spectrum | 0.6754 | 0.8425 | 0.5491 | 0.3627 | 0.6393 |
| CLIP spectrum | **0.6784** | **0.8459** | **0.5670** | **0.3905** | **0.6626** |

JEPA wins all five image-side probes. CLIP wins all five spectrum-side probes.
This modality split is consistent and unlikely to be explained by a single
probe accident.

### Objective-facing projections

| Space | Redshift | Mass | Metallicity | Age | sSFR |
| --- | ---: | ---: | ---: | ---: | ---: |
| JEPA image | **0.6131** | 0.7680 | 0.4593 | 0.2783 | 0.5639 |
| CLIP image | 0.6015 | **0.7923** | **0.4688** | **0.3060** | **0.5827** |
| JEPA spectrum | 0.6321 | 0.7799 | 0.4483 | 0.2735 | 0.5699 |
| CLIP spectrum | **0.6513** | **0.8165** | **0.5091** | **0.3534** | **0.6275** |

CLIP's projections preserve more physical-property information, especially on
the spectrum side. JEPA retains its advantage only for projected image
redshift. This explains why CLIP has broader projected geometry while JEPA is
more directly predictable across modalities: breadth and shared predictability
are different properties.

## 7. Do Physical Decoding Directions Transfer?

A scalar decoder was trained in one projected modality and evaluated in the
other. Direct coordinate reuse was frequently poor or negative. The table shows
test R2 after mapping the target modality through train-fitted Procrustes.

| Target | JEPA image decoder on spectrum | CLIP image decoder on spectrum | JEPA spectrum decoder on image | CLIP spectrum decoder on image |
| --- | ---: | ---: | ---: | ---: |
| Redshift | **0.515** | 0.467 | **0.529** | 0.502 |
| Stellar mass | 0.621 | **0.626** | 0.630 | **0.666** |
| Metallicity | **0.369** | 0.355 | **0.373** | 0.372 |
| Stellar age | 0.166 | **0.225** | **0.219** | 0.214 |
| sSFR | **0.474** | 0.365 | **0.495** | 0.441 |

JEPA transfers redshift and sSFR directions especially well. CLIP transfers
mass better and has one stronger age direction. The large improvement from
direct to Procrustes transfer demonstrates that both objectives encode related
physical variables along rotated modality-specific axes. Neither enforces
literal decoder interchangeability.

## 8. Dimensionality and Collapse

| Space | Dim | Effective rank | Participation ratio | Largest eigenvalue share |
| --- | ---: | ---: | ---: | ---: |
| JEPA image raw | 1,024 | 28.26 | 12.77 | 0.192 |
| JEPA spectrum raw | 768 | 26.08 | 10.08 | 0.225 |
| CLIP image raw | 1,024 | **64.29** | **29.05** | **0.101** |
| CLIP spectrum raw | 768 | **58.14** | **21.00** | **0.139** |
| JEPA image projected | 256 | 20.19 | 17.14 | 0.099 |
| JEPA spectrum projected | 256 | 17.94 | 14.68 | 0.111 |
| CLIP image projected | 256 | **45.39** | **28.61** | 0.102 |
| CLIP spectrum projected | 256 | **44.65** | **24.28** | 0.128 |

Both methods are decisively non-collapsed. CLIP uses many more effective
directions, while JEPA's lower rank is paired with stronger cross-modal CKA and
ridge predictability. SIGReg prevents collapse but does not require maximal
uniformity or rank.

## 9. What the Study Supports

The evidence supports a three-level alignment taxonomy:

### Pair alignment

CLIP is better at identifying the exact corresponding object and preserving
cross-modal local neighborhoods.

### Geometric alignment

JEPA is better at preserving broad cross-modal relational geometry, as measured
by CKA and sampled distance correlations. CLIP is better only at the most local
neighborhood scale.

### Information alignment

JEPA representations are far more recoverable across modalities with an
unrestricted linear map, and their leading canonical factors are more strongly
shared. CLIP retains more modality-private spectrum information and broader
feature rank.

The proposed paper claim can therefore be sharpened to:

> Contrastive and regularized invariance objectives do not merely vary the
> strength of a single notion of alignment. They trade off exact
> correspondence, local discriminability, global relational agreement,
> cross-modal predictability, and modality-private information.

## 10. What the Study Does Not Prove

1. It does not isolate objective causality perfectly. Global batch is 256 for
   JEPA and 128 for CLIP, and optimizer steps are 60K versus 120K.
2. It does not establish generalization to unseen representation-training
   objects because the historical test identities were present unlabeled in
   the 307K training union.
3. It uses one representation-training seed per objective. Probe-seed
   stability is not model-seed stability.
4. Exact retrieval may penalize scientifically valid near-duplicates and false
   negatives. Duplicate-aware and redshift-controlled retrieval remain useful.
5. The implemented JEPA objective is direct cross-modal invariance plus SIGReg,
   not predictor-based latent target prediction.
6. CKA and CCA describe linear/global relations; they do not guarantee equal
   nonlinear information.

## 11. Reproducibility and Artifacts

Everything created for this study is under this directory:

- `study.py`: preparation and all numerical experiments;
- `make_figures.py`: deterministic figures and compact CSV tables;
- `artifacts/manifest.json`: checkpoint, cache, shape, split, and hash ledger;
- `artifacts/labels.npz`: exact row-aligned diagnostic labels;
- `artifacts/neighbors_*`: exact test-set neighbor indices;
- `results/*.json`: full machine-readable outputs;
- `tables/*.csv`: compact paper tables;
- `figures/*.png`: paper figures;
- `logs/*.log`: execution logs.

Large source embeddings remain in the existing read-only evaluation cache and
are referenced by the manifest rather than duplicated.

Methodological references supplied with the study specification:

- [Kornblith et al., Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a)
- [What Representational Similarity Measures Imply about Decodable Information](https://arxiv.org/abs/2411.08197)
- [When Embedding Models Meet: Procrustes Bounds and Applications](https://arxiv.org/abs/2510.13406)
