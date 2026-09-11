# Why AstroJEPA Still Trails AstroCLIP on Redshift and Stellar Age

Date: 2026-08-30

## Executive conclusion

The three apparent gaps do not have one common cause.

1. **Image redshift is primarily an alignment-objective gap, not a weak image
   backbone.** The 307K scratch image backbone reaches ridge R2 = 0.627, almost
   exactly AstroCLIP's reported unaligned image result of about 0.63. AstroCLIP's
   gain to 0.79 appears only after contrastive image-spectrum alignment.
2. **Spectrum redshift is a genuine representation gap.** Replacing ridge with
   AstroCLIP's released nonlinear probe raises the scratch raw result only from
   0.676 to 0.702, far below 0.98-0.99. The current cross-only objective does
   not require reconstruction or prediction of wavelength-local line structure.
3. **Most of the apparent stellar-age gap was a probe mismatch.** The same
   nonlinear probe raises scratch raw age from 0.366 to 0.410. The remaining
   gap is 0.020 to AstroCLIP's aligned spectrum score of 0.43, although the
   stronger AstroCLIP reference for unimodal spectral information is its
   *unaligned* spectrum encoder at 0.47.

SIGReg successfully prevents complete collapse. It does not guarantee that a
noncollapsed representation retains every scientifically useful variable.
The current result is evidence that from-scratch cross-modal SIGReg training
works, but not that anti-collapse regularization replaces a modality-specific
predictive objective.

## Corrected comparison

### Published and local probe protocols

The original local report used standardized ridge regression for every local
representation. AstroCLIP does not report ridge probes:

- image redshift zero-shot uses distance-weighted kNN;
- image redshift few-shot uses an MLP;
- spectrum redshift uses an MLP;
- physical properties use kNN or an MLP.

The paper describes a one-hidden-layer width-32 MLP. The released notebook is
not fully consistent with that description: image redshift explicitly uses
one width-32 hidden layer, while spectrum redshift and property estimation use
the helper default of two width-64 hidden layers. The helper trains for 10
epochs with Adam, batch size 64, dropout 0.1, and MSE.

### Matched nonlinear diagnostic

The released-code MLP shapes were run for three seeds on the already cached
307K scratch embeddings. No backbone or checkpoint was modified.

| Modality / representation | Target | Ridge R2 | Nonlinear MLP R2 | AstroCLIP reference |
| --- | --- | ---: | ---: | ---: |
| Image raw | redshift | 0.6274 | 0.6198 +/- 0.0057 | 0.78 few-shot / 0.79 kNN |
| Image projected | redshift | 0.6137 | 0.6088 +/- 0.0080 | 0.78 few-shot / 0.79 kNN |
| Spectrum raw | redshift | 0.6756 | 0.7016 +/- 0.0033 | 0.98-0.99 |
| Spectrum projected | redshift | 0.6322 | 0.6402 +/- 0.0026 | 0.98-0.99 |
| Spectrum raw | age | 0.3658 | 0.4101 +/- 0.0037 | 0.43 aligned / 0.47 unaligned |
| Spectrum projected | age | 0.2758 | 0.2852 +/- 0.0034 | 0.43 aligned / 0.47 unaligned |

The same raw-spectrum nonlinear probe reaches 0.8537 stellar mass, 0.5803
metallicity, and 0.6679 sSFR. Those meet or exceed the aligned AstroCLIP
references of 0.88, 0.58, and 0.64 except for the small remaining mass gap.
This isolates spectrum redshift as the exceptional failure, rather than a
general inability of the spectrum encoder to learn astrophysics.

## Why image redshift remains below AstroCLIP

### The image backbone is already at AstroCLIP's unaligned level

AstroCLIP reports roughly 0.63 for its unaligned DINO image encoder and 0.79
after CLIP alignment. Our scratch raw image result is 0.627. A nonlinear MLP
does not improve it. The missing performance is therefore not well explained
by a decoder mismatch or by a generally defective ViT.

AstroCLIP explicitly interprets its own ablation as the spectrum organizing
the image latent neighborhood around redshift. Spectra provide a much stronger
redshift signal than three-band images, and InfoNCE rewards using any shared
factor that distinguishes one image-spectrum pair from all other pairs.
Redshift is an especially efficient factor for this purpose.

### Positive alignment plus SIGReg is not instance discrimination

Our objective minimizes all four positive-view image-spectrum squared
distances and applies SIGReg separately to each modality. It has no explicit
term comparing one galaxy with other galaxies.

SIGReg constrains marginal feature statistics and prevents every object from
mapping to the same constant. It does **not** require:

- different galaxies to be separable;
- neighboring points to have similar redshift;
- paired identity to be recoverable among batch negatives;
- the image embedding to inherit the spectrum's finest information.

AstroCLIP's symmetric InfoNCE loss does require paired-object identification
against all other objects in the batch. This produces the local metric
geometry needed by its strong kNN result. Our effective ranks show a
noncollapsed representation, but noncollapse alone says nothing about whether
redshift is the axis along which neighborhoods are organized.

### The spectrum branch cannot teach information it did not encode

The scratch spectrum redshift MLP reaches only 0.702. Cross-modal transitivity
cannot make the image branch inherit near-perfect redshift from a spectrum
branch that does not contain near-perfect redshift. Both branches can instead
agree through easier shared factors such as broad color, mass, star-formation
state, and morphology.

### Data scale is secondary, but still relevant

AstroCLIP pretrains its image ViT on 76.4 million Legacy Survey images using
DINO, iBOT, and KoLeo before paired alignment. Its training set is selected
from the same survey and magnitude regime as its paired downstream sample.
Our prior image backbone saw fewer draws from a broader stream, and the 307K
scratch model was trained only on paired objects.

That scale and local image objective can improve robustness. It is unlikely to
be the principal explanation for the current 0.16 gap, however, because our
raw image representation already matches AstroCLIP's unaligned result.

## Why spectrum redshift remains far below AstroCLIP

### AstroCLIP makes observed-wavelength line prediction unavoidable

AstroCLIP cuts spectra into length-20 patches with stride 10, masks six long
contiguous regions, and minimizes reconstruction MSE only on the missing
regions. Its appendix demonstrates that the trained model reconstructs masked
absorption and emission lines accurately.

Redshift is encoded by the common displacement of a family of known lines and
breaks on the fixed observed-wavelength grid. A model that must reconstruct a
missing line from the rest of the spectrum has a direct reason to infer that
displacement. This is much closer to a redshift-sufficient surrogate task than
global pair alignment.

Our scratch model uses non-overlapping length-20 patches, masks input patches,
and aligns only global modality projections. It has no decoder, predictor, or
target loss for the masked wavelength content. A masked feature can simply be
ignored as long as the remaining spectrum is sufficient to agree with the
image. Non-overlapping patches also make line behavior at patch boundaries
less smooth than AstroCLIP's 50% overlap.

### The shared-information bottleneck works against exact spectral redshift

Images do not determine redshift perfectly. A pure cross-modal objective is
therefore allowed to discard spectrum-only redshift detail and retain only the
coarser redshift information shared with images. The local results show this
directly: spectrum redshift falls from raw 0.702 to projected 0.640 under the
matched nonlinear probe.

AstroCLIP first builds a strong spectrum encoder with masked reconstruction,
then freezes that backbone by default while training its attention pooling
head. This prevents cross-modal alignment from erasing the underlying spectral
features. Our scratch run updates the whole spectrum backbone using only the
cross-modal objective, so precise spectral information is never independently
secured.

### Training exposure matters more for spectra than it first appeared

The paper reports 500 spectrum-pretraining epochs, and the released config has
a 500,000-step limit with batch size 64. The repository README instead reports
30,000 steps, so the public sources are inconsistent about the exact completed
run. All descriptions nevertheless agree on a substantial dedicated masked
reconstruction stage before alignment.

Our scratch run completed 50 paired epochs. More identical cross-only epochs
may improve optimization, but cannot create the missing wavelength-prediction
pressure. The highest-value fix is objective coverage first, then additional
training.

### Spectrum mean and scale are a smaller missing cue

Both pipelines normalize each spectrum by its own mean and standard deviation.
AstroCLIP writes the original mean and standard deviation into a special token;
our encoder computes and logs these quantities but does not feed them to the
transformer or frozen probe. Flux amplitude and S/N can carry selection and
distance priors, so retaining these statistics may help. They cannot plausibly
explain most of a 0.28 redshift R2 gap because line positions alone should be
nearly sufficient.

## Why stellar age is much closer than the ridge table suggested

Mass-weighted stellar age is a nonlinear, model-derived PROVABGS target. Its
information is spread across continuum shape, the 4000-Angstrom break, Balmer
absorption, metallicity-sensitive features, and star-formation history, with
substantial degeneracy among age, metallicity, and dust. A linear probe is a
particularly restrictive decoder for this target.

The nonlinear diagnostic recovers 0.410 from our raw spectrum representation,
so most of the information was already present. The remaining plausible gap is
due to AstroCLIP's overlapping local patches, masked line reconstruction, more
dedicated spectrum training, and the larger effective downstream training
split.

Crucially, AstroCLIP's own table reports age R2 = 0.47 for the unaligned
spectrum encoder and 0.43 after alignment. Cross-modal alignment did not create
the best age representation; it degraded it. The appropriate route to stronger
age performance is therefore a better spectrum-local objective, not stronger
pressure to retain only image-shared information.

## Remaining protocol caveats

The local evaluator uses 138,583 train and 29,697 test objects and omits the
29,696-row validation split. For PROVABGS it uses 73,341 train and 15,993 test
objects. AstroCLIP reports 105,159 total PROVABGS objects and describes a 90/10
downstream split. Consequently, its nonlinear probe likely receives more
labeled training examples than ours. This can explain a small part of the age
gap, but not the spectrum-redshift failure.

The 307K scratch pretraining also consumed all available paired sources without
respecting the AstroCLIP downstream test identity split. Its frozen evaluation
is therefore transductive at the self-supervised stage. That exposure favors
our result and still does not close redshift, which strengthens the conclusion
that the redshift objective is insufficient.

## Ranked interventions

1. Add a spectrum-local masked contextual prediction or reconstruction loss on
   fixed wavelength positions while retaining the global cross-modal LeJEPA +
   SIGReg objective.
2. Use overlapping spectrum patches (length 20, stride 10) or otherwise improve
   local wavelength resolution.
3. Preserve a modality-private spectrum representation for downstream use;
   align a head while ensuring the raw backbone is not forced to discard
   spectrum-only information.
4. Test a cross-modal discriminative term as an ablation: symmetric InfoNCE, or
   a JEPA-compatible instance-separation/neighborhood objective. Keep the
   current SIGReg objective as the central from-scratch baseline.
5. Feed retained normalization mean/log-scale as auxiliary tokens or features
   and measure the incremental redshift gain.
6. Retrain longer only after the local objective is present.
7. Re-run the official kNN and MLP protocols on a documented object-disjoint
   90/10 split, including the current validation rows, before claiming a final
   numerical gap.

## Sources and artifacts

- [AstroCLIP paper](https://arxiv.org/html/2310.03024)
- [Official AstroCLIP repository](https://github.com/PolymathicAI/AstroCLIP)
- Local primary ridge report: `FINAL_307K_COMPARISON.md`
- Matched-probe machine output: `results/astroclip_matched_probe_results.json`
- Local scratch checkpoint:
  `checkpoints/CrossModalScratch_DESI307K_AllPairs_CrossOnly_SIGReg_4GPU/last.pt`
