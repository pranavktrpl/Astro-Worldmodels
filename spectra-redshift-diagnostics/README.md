# DESI Spectrum Redshift Diagnostics

This directory measures the spectroscopic-redshift distributions of the two
corpora used by AstroJEPA:

1. the DESI spectrum-v2 self-supervised pretraining set;
2. the combined 307,428-row DESI image-spectrum cross-modal set.

## Main Result

`z >= 0.5` is the explicit high-redshift threshold used in this report.
Negative DESI `Z` values are retained and counted as `z < 0.5`; they are the
near-zero stellar/radial-velocity population, not missing values.

| Training corpus | Spectrum rows | z < 0.5 | z >= 0.5 | High-z fraction | z >= 1 | z >= 2 | z >= 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spectrum-v2 SSL train split | 1,103,874 | 611,771 | **492,103** | **44.58%** | 246,558 | 14,991 | 2,209 |
| Combined cross-modal corpus | 307,428 | 294,662 | **12,766** | **4.15%** | 891 | 40 | 22 |

Among nonnegative redshifts only, 54.64% of the SSL set and 4.17% of the
cross-modal set have `z >= 0.5`. The different denominator matters because the
SSL set contains 203,226 negative-Z stellar rows, while the paired set contains
only 938.

The central finding is therefore not subtle: **the standalone spectrum
pretraining data are high-redshift rich, but the image-spectrum training pairs
are strongly concentrated below z=0.5.**

## Redshift Bands

| Redshift interval | SSL train | Cross-modal |
| --- | ---: | ---: |
| z < 0 | 203,226 | 938 |
| 0 <= z < 0.1 | 138,054 | 34,521 |
| 0.1 <= z < 0.3 | 179,741 | 174,142 |
| 0.3 <= z < 0.5 | 90,750 | 85,061 |
| 0.5 <= z < 1.0 | 245,545 | 11,875 |
| 1.0 <= z < 2.0 | 231,567 | 851 |
| 2.0 <= z < 3.0 | 12,782 | 18 |
| z >= 3.0 | 2,209 | 22 |

## Cross-Modal Source Breakdown

| Source | Rows | z < 0.5 | z >= 0.5 | High-z fraction | z >= 1 | z >= 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AstroCLIP mirror, all splits | 197,976 | 187,277 | 10,699 | 5.40% | 812 | 40 |
| MMU DESI x Legacy DR10 | 95,895 | 93,975 | 1,920 | 2.00% | 2 | 0 |
| Manual DESI x DR8 | 13,557 | 13,410 | 147 | 1.08% | 77 | 0 |
| **Combined** | **307,428** | **294,662** | **12,766** | **4.15%** | **891** | **40** |

All 22 cross-modal rows at `z >= 3` come from the AstroCLIP mirror.

## What Was Measured

These are not neural-backbone predictions. They are DESI spectroscopic
pipeline/template-fit redshifts derived from the raw spectra and stored with
the datasets:

| Source | Redshift field | Quality field |
| --- | --- | --- |
| MMU DESI SSL | `Z` | `ZWARN` |
| AstroCLIP mirror | `redshift` | Not available in this mirror |
| MMU DESI x Legacy | `Z_spec` | `ZWARN_spec` |
| Manual DESI x DR8 | Joined by `object_id` to MMU DESI `Z` | MMU DESI `ZWARN` |

Using the released DESI pipeline result is the scientifically appropriate way
to audit the training-set distribution. Re-running a new approximate template
fitter on 1.1 million spectra would be slower and less authoritative.

The plots include every finite stored redshift because the goal is to describe
the samples actually supplied to representation training. No rows are removed
using `ZWARN`. The SSL training split contains 70,623 warned rows. Of the
109,452 cross-modal rows with an available warning flag, 138 are warned; the
197,976 AstroCLIP rows do not expose that flag.

## Dataset Scope

The SSL directory contains 1,126,441 spectra. Spectrum v2's deterministic
object-ID split assigns:

| Split | Rows |
| --- | ---: |
| Train, plotted | 1,103,874 |
| Validation | 11,280 |
| Test | 11,287 |

The scanner processed the full corpus to reconstruct and verify those split
counts. The headline SSL plot uses only the actual training split.

The cross-modal plot uses all rows consumed by the 307K representation runs:

- 197,976 AstroCLIP mirror rows across train, validation, and test source files;
- 95,895 MMU DESI x Legacy rows;
- 13,557 manually materialized DESI x DR8 rows.

## Files

| File | Contents |
| --- | --- |
| `ssl_pretraining_redshift_distribution.png` | SSL training distribution, linear low-z and logarithmic full-range panels |
| `crossmodal_redshift_distribution.png` | Combined paired distribution with source-level curves |
| `summary.json` | Exact counts, fractions, quantiles, quality flags, and provenance |
| `redshift_band_counts.csv` | Machine-readable counts for fixed redshift intervals |
| `redshift_values.npz` | Cached scalar redshifts and warning flags; no spectra or images |
| `analyze_redshift_distributions.py` | Reproducible Parquet scanner and plotting program |

## Reproduce

From the repository root:

```bash
source /mnt/ssd-cluster/pranav/.cluster/bashrc
conda activate lejepa-og
python spectra-redshift-diagnostics/analyze_redshift_distributions.py
```

To recreate tables and figures without rescanning Parquet:

```bash
python spectra-redshift-diagnostics/analyze_redshift_distributions.py --reuse-cache
```
