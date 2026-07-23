# Professor discussion brief — 24 July 2026

## The project in one sentence

`microhard` is an auditable small-data pipeline that links microscopy,
processing history, composition, and mechanical properties while separating
direct measurements from weaker condition-level labels.

## What exists today

| source | microscopy | independent property groups | label status |
|---|---:|---:|---|
| UHCSDB | 961 micrographs | 7 hardness groups | directly measured |
| cited steel studies | 19 SEM panels | 10 hardness groups | same-study, condition-level |
| public LPBF IN718 archive | 22 BSE-SEM fields | 10 hardness groups | same-archive, condition-level |

The effective sample size is the number of independent material states, not
the number of fields of view. The condition-level links can support training,
with lower weight, but are not treated as exact-specimen validation data.

The existing UHCS hardness model is a feasibility result: leave-one-sample-out
MAE is 122.9 HV and R² is 0.134 at \(n=7\). This is not yet evidence of reliable
prediction across unseen processing conditions. Similarly, 180 passing
software tests establish implementation consistency, not scientific validity.

## The central scientific problem

The software can ingest more images faster than the project can establish that
an image and a property measurement describe the same material state.
Metallurgical equivalence—not model architecture—is therefore the limiting
question.

A concrete warning appears in UHCS. Two samples sharing the coarse condition
“austenitize, then unspecified quench” report 810 and 611 HV because their
austenitization temperatures differ. A join key that omits temperature would
collapse a physically important distinction and manufacture label noise.

The current linkage logic therefore treats composition, process, material
state, heat-treatment temperature and time, sampling location, build strategy,
section/orientation, and explicit specimen conflicts as evidence or vetoes.
The confidence weights (1.00, 0.85, 0.55, 0.25) are heuristics for sensitivity
analysis, not calibrated probabilities.

## Proposed near-term position

Frame the next phase as a provenance-aware IN718
process–microstructure–property pilot, not as inverse design.

1. Stay within one alloy/process family long enough to avoid pooling
   incompatible study and material effects.
2. Ingest the downloaded IN718 carbide-additive archive, preserving specimen,
   state, section plane, orientation, composition, processing, and replicate
   identity.
3. Use hardness as the first bulk-property test. Keep orientation-sensitive
   tensile properties out of automatic linkage unless image and test
   orientation can be matched.
4. Reserve exact physical specimens—and eventually whole studies or process
   regimes—for validation.
5. Report learning curves and uncertainty before attempting process
   optimisation. Property-conditioned generation waits for a trustworthy
   forward model.

Persistent homology remains attractive for fatigue, pore connectivity, and
crack-related targets. A pivot should happen only if a suitable EBSD,
tomography, or other defensible microstructure–property dataset exists; 2D SEM
hardness is not, by itself, a strong test of that topology thesis.

## Questions for the discussion

1. When is it defensible to assign batch- or condition-level bulk hardness to
   individual SEM fields for training, if it is excluded from validation?
2. Which variables must define a material-state key for UHCS and IN718? In
   particular, when are section plane, build orientation, prior condition,
   sampling location, cooling rate, indenter load, or microscopy settings
   mandatory?
3. Which target is genuinely observable from 2D SEM: hardness, yield strength,
   fatigue behaviour, or none without additional EBSD/3D information?
4. What prospective validation would be convincing: how many independent
   specimens, treatment conditions, replicates, and held-out regimes?
5. Is a narrow IN718 pilot the right next step, or is there a better
   topology-governed material/property dataset to pursue?
6. Are there relevant raw datasets, former students, collaborators, or a small
   measurement campaign that could provide exact specimen-level links?

## A useful outcome from the meeting

Leave with three decisions:

- the minimum variables required for a defensible linkage key;
- one material family and property for the next forward-model benchmark;
- an agreed standard for exact-specimen validation before making any
  optimisation or inverse-design claim.
