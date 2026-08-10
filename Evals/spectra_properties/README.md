# Spectra-backbone physical-property probe (PROVABGS)

Second spectra eval: regress the four AstroCLIP galaxy properties from frozen
`SpectrumTransformerEncoder` embeddings of the AstroCLIP cross-match spectra,
with the same ridge / zero-shot kNN / MLP heads, seeds, and 80/20 split as
the redshift probes. Labels come from the PROVABGS BGS EDR posterior catalog
(`UniverseTBD/mmu_desi_provabgs`), joined by DESI `targetid`.

**Why this matters more than redshift:** redshift is nearly fully determined
by a spectrum, so R² ≈ 0.98 mostly confirms the encoder preserved
information. Properties like age and sSFR test whether the embedding
organizes continuum shape and line ratios — the information the image
encoder is supposed to inherit through the planned CLIP-style alignment.

## Reference targets (AstroCLIP spectrum encoder, test R²)

| Property | zero-shot kNN | few-shot MLP |
|---|---:|---:|
| Stellar mass | 0.87 | 0.88 |
| Metallicity | 0.57 | 0.58 |
| Age | 0.43 | 0.43 |
| sSFR | 0.63 | 0.64 |

(Transcribed in `Evals/README.md`; also in
`competitor_comparison/competitor_baselines.json` under `physical_properties`
together with the AION-1 rows.)

## Usage

```bash
# one-time: the MMU catalog conversion (~1.4 GB, resumable)
bash Evals/spectra_properties/download_provabgs.sh

# full battery (3 seeds, all heads, all four properties)
python Evals/spectra_properties/spectra_properties_probe.py \
  --checkpoint checkpoints/SPECTRA_run4_bs16_2806_Epoch5_utbd_desi/step_<best>.pt
```

The MMU conversion is required (not just convenient): the original DESI VAC
HDF5 (`BGS_ANY_full.provabgs.sv3.v0.hdf5`, e.g. under `/mnt/datasets/provabgs/`
on the cluster) stores posterior chains and `PROVABGS_THETA_BF` but no derived
`Z_MW`/`TAGE_MW`/`AVG_SFR` columns — deriving them needs the `provabgs` SPS
package, which is the work the MMU conversion already did. `--provabgs-path`
still accepts an HDF5 file for catalogs that do carry the derived columns;
otherwise point it at (or symlink) a directory of MMU parquet shards.

Results land in `results/<label>/metrics.json`. Use the checkpoint the
redshift probe's `--scan` selected — this probe does its own head/epoch
selection on a validation carve-out but no checkpoint selection.

## Protocol notes

- **Embeddings are shared with `Evals/spectra_redshift`**: the cache under
  its `results/<label>/embeddings.npy` is reused if present and created
  there otherwise, so running both probes embeds the spectra only once.
- **Property derivations (AstroCLIP convention):** `stellar_mass =
  LOG_MSTAR`, `metallicity = log10(Z_MW)`, `age = TAGE_MW` (Gyr), `ssfr =
  log10(AVG_SFR) - LOG_MSTAR` (average SFR over the last 1 Gyr).
- **Sample:** cross-match spectra with a PROVABGS row (`object_id` =
  `targetid`) whose fit succeeded (`LOG_MSTAR > 0`) and whose `Z_MW`,
  `AVG_SFR`, `TAGE_MW` are strictly positive. One shared validity mask keeps
  all four properties on the identical sample; match counts are printed and
  saved in `metrics.json`.
- **Targets are z-scored on the train split** before fitting (the shared MLP
  head assumes redshift-scale targets). R² is unaffected; reported MAE/RMSE
  are converted back to physical units. The redshift-specific NMAD and
  outlier-fraction metrics (both built on the photo-z `(1+z)` convention)
  are dropped.
- The AstroCLIP train/test split is fixed; seeds vary only the validation
  carve-out (10% of train) and MLP initialization, exactly as in the other
  probes.
