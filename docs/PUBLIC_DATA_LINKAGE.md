# Public-data route to microstructure and process optimisation

## Decision

The primary route is now **existing public data**, not a new experimental
campaign. The next scientific bottleneck is record linkage: recovering which
images, process records, compositions, and mechanical measurements describe
the same material state across archives and papers.

Fuzzy matching is useful, but it must be fuzzy over provenance and metallurgy,
not merely over filenames. An explicit conflict in alloy, heat-treatment
temperature, build strategy, specimen, sampling location, or tensile
orientation vetoes a superficially similar match.

## Current position

| source | images | independent property groups | property linkage |
|---|---:|---:|---|
| UHCSDB | 961 | 7 directly measured hardness groups | sample label |
| cited steel panels | 19 | 10 hardness groups | same study, condition and location |
| Zenodo 14163786, LPBF IN718 | 22 | 10 hardness groups | same archive, alloy, beam strategy, state and temperature |

The IN718 archive also contributes yield strength, UTS, elongation, reduction
of area, and modulus. These make 174 plausible image/property candidates, but
the BSE-SEM filenames omit H/V orientation. They are retained for review and
not auto-attached.

This is enough to prove the ingestion and linkage mechanism. It is not enough
to infer a highly optimised microstructure. The effective sample size is the
number of independent material states (7, 10, and 10), not the number of image
fields. The families cannot be pooled as though one steel treatment and one
IN718 treatment were interchangeable.

## What is implemented

`src/microhard/linkage.py` scores image/property pairs using:

- source record and explicit physical specimen identity;
- normalised alloy and process descriptions;
- material state, treatment temperature and hold time;
- build strategy, sampling location where available, and orientation;
- hard vetoes for explicit conflicts.

Every retained link reports a score, confidence tier, reason list, training
weight, whether it can be auto-attached, and whether it is eligible for
validation. Confidence changes the regression sample weight:

| tier | training weight | use |
|---|---:|---|
| exact | 1.00 | training and held-out validation |
| high | 0.85 | training; same-study condition link |
| medium | 0.55 | candidate/review by default |
| review | 0.25 | catalogue only |
| reject | 0 | excluded |

A same-study condition match is deliberately not validation data. Only an
explicitly shared physical specimen can be used as exact ground truth. Tied
best candidates are omitted rather than averaged.

The first source adapter is `godec_in718`. Download and audit it with:

```bash
python scripts/fetch_zenodo_in718.py
uv run microhard audit-public-links \
  --output data/public_in718_godec_2024/link_audit.csv
```

The result is 22 high-confidence HV1 links across ten leakage-safe condition
groups, all with weight 0.85 and none misrepresented as exact validation.

## How close are we?

There are four distinct capabilities:

| capability | status | evidence |
|---|---|---|
| recognise microstructure/process signal in an SEM image | demonstrated, narrow | UHCS temperature probe is above chance at a controlled magnification |
| link public images to bulk properties | demonstrated | steel literature adapter plus the audited IN718 integration |
| predict properties across unseen process conditions | not demonstrated | too few independent groups per material family and study effects are not yet held out |
| solve the inverse problem—target property to microstructure to producible process | not yet supported | no validated forward process–structure–property surrogate or uncertainty-calibrated search space |

So we are close to a credible **public-data forward-modelling pilot**, but not
close to a defensible claim that the system has discovered an optimum.

The next gate should be assessed per material/process family. A practical
pilot target is:

- at least 50 independent, property-linked material states from at least three
  public studies;
- composition and process variables captured explicitly rather than hidden in
  source names;
- at least 15 exact-specimen groups reserved for a source-held-out test;
- magnification/scale and SEM mode recorded, with models prevented from using
  figure lettering, scale bars, or lab-specific layout as shortcuts;
- property uncertainty or replicate scatter retained.

These are engineering gates, not a statistical guarantee. Learning curves and
leave-one-study-out performance decide whether more public ingestion is needed.

## Next ingestion sequence

The tracked queue is `data/public_sources.csv`.

1. **Zenodo 16603134, IN718 with carbide additives.** It stays within the
   newly integrated alloy family and includes process, chemistry, SEM, EBSD,
   and a tensile property map. This is the highest-value next source.
2. **Zenodo 18800251, LPBF 316L.** SEM/EBSD plus hardness across three material
   variants gives a compact second-family test.
3. **WAAM steel and 30CrMnSiA Mendeley datasets.** Both describe
   microstructure and mechanical response in the same archive.
4. **Zenodo 19813205.** Useful for exact fracture-surface/tensile linkage, but
   hydrogen charge and orientation must be explicit.
5. **Zenodo 8090777.** Thirty-seven LMD pieces with images, parameters,
   tensile, and hardness; high value but about 40 GB, so ingest selectively.

Property-only tables can help regularise composition/process response, but
they must not be pretended to have image labels. Simulated microstructures are
a separate domain and must not enter the experimental validation split.
Post-test fracture surfaces are also a separate task: using them as inputs to
predict the tensile result that created the fracture would be target leakage.

## Route to inverse design

The forward model should be conditional:

```
(composition, process parameters, SEM/EBSD representation)
                         -> property distribution
```

Training and evaluation splits must hold out whole specimens and then whole
studies. Once this model predicts unseen conditions with calibrated
uncertainty, optimisation can search the observed, physically feasible process
space for a target property vector. Candidate processes are then decoded into
their predicted microstructure representations. The answer must be a Pareto
front with uncertainty—not one synthetic “best-looking” SEM image.

The immediate work is therefore source ingestion, entity resolution,
harmonisation, and leave-one-study-out validation. No new material production
is required for that stage.
