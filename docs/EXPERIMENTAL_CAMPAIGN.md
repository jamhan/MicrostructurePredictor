# Experimental campaign: the shortest path to inverse process design

This is the executable data plan for moving from the current UHCS proof of
concept to a prospectively validated process recommendation. The first
demonstrator stays narrow: one composition family, a controlled heat-treatment
space, and direct mechanical measurements linked to the exact imaged specimen.

The scientific object is not an SEM image by itself. It is the complete chain:

    composition + process route -> microstructure -> property distribution

The inverse-design claim becomes credible only when a model selects a previously
untested process, that process is run, and the resulting microstructure and
properties meet the target on independent specimens.

## Phase 0: define the first objective

Start with UHCS AC1 and a process space we can actually control:

- austenitizing temperature and hold time;
- cooling route and measured cooling curve;
- tempering temperature and time if tempering is introduced.

The minimum first property is directly measured Vickers hardness, because that
is compatible with the current model. Hardness alone is not "mechanically
optimized." The first multi-objective extension should add one strength measure
and one damage-tolerance measure, for example yield strength plus Charpy impact
energy or fracture toughness. Fatigue requires a larger, purpose-designed
campaign because its scatter and defect sensitivity make grade/condition
lookups invalid.

Before specimens are made, write the objective as a constrained problem, for
example:

    maximize yield strength and fatigue life
    subject to toughness >= threshold, elongation >= threshold,
    process temperature <= limit, and cost <= limit

This produces a Pareto set rather than one supposedly universal optimum.

## Phase 1: rescue the already-imaged UHCS value

Run:

    uv run microhard plan-measurements --limit 12

The current ranked handoff, including the recommended first five specimens, is
in `docs/UHCS_MEASUREMENT_PRIORITY.md`.

The command excludes samples whose processing key is unresolved and uses a
deterministic maximin design over temperature, log hold time, cooling route, and
casting. It defaults to the AC1 grade so the first campaign does not turn alloy
identity into a process variable, then weights process novelty by metadata
completeness and imaging coverage. This is diversity sampling, not
uncertainty-based active learning:
seven direct property groups cannot support reliable model uncertainty yet.

To see specimens that become useful after metadata remediation:

    uv run microhard plan-measurements --limit 12 --include-unverified

For every selected physical specimen that still exists:

1. Confirm its identity and independence. Sample ids 24 and 25 currently share
   one conservative split group because their database labels are identical.
2. Verify the process route against laboratory records, including actual
   cooling medium and time-temperature profile.
3. Make at least three direct hardness measurements with load, dwell, standard,
   location, uncertainty, and raw readings retained.
4. If material remains, reserve it for the second mechanical property rather
   than consuming all of it on additional SEM fields.

The success gate for this phase is at least 20 directly measured, independent
conditions and an image model that beats grade-only and process-only baselines
under held-out-condition validation. Twenty is a gate for the next experiment,
not a claim that the model is production-ready.

## Phase 2: controlled process-structure-property design

For a narrow process space, the planning target is:

- 50 to 100 distinct process conditions;
- three independent coupons per condition;
- 5 to 10 fields per coupon at two or three fixed physical scales;
- direct mechanical measurements for every condition;
- at least one entire heat/batch and 20% of process conditions untouched until
  final evaluation.

Use a space-filling design initially, then active learning after the calibrated
forward model has enough direct targets to estimate uncertainty. A hundred
images from one coupon are spatial repeats, not a hundred property labels.

### Imaging SOP

Record every field in `data/experimental_campaign/images.csv`:

- exact specimen, field, sampling location, and orientation;
- preparation and etchant;
- modality, detector, accelerating voltage, working distance, magnification,
  micrometres per pixel, and acquisition date;
- original file path and SHA-256;
- constituent labels only when assigned by a metallographer.

Acquire the same physical scales for every coupon. The current temperature
probe is strongest near 1964X and near chance at several other magnifications,
so uncontrolled scale mixing is a demonstrated confound, not a theoretical
concern.

Add EBSD for grain size, boundary character, and texture. Add EDS/XRD where
phase identity is ambiguous. Use tomography rather than 2D SEM alone when pore
connectivity or fatigue is the target.

### Mechanical SOP

Record every result in `mechanical_tests.csv`, linked to the exact specimen and
mechanical coupon. Preserve test standard, units, temperature, orientation,
sampling location, replicate number, reported uncertainty, test-parameter JSON
(for example indenter load and dwell, or fatigue stress ratio and frequency),
and raw-data path. Never convert multiple scales or units silently.

### Process SOP

`process_steps.csv` is long-form: one row per ordered operation. Record measured
profiles where possible. Nominal furnace set-points and labels such as "water
quench" are not substitutes for the thermal history experienced by the coupon.

## Data contract and ingestion

The four tables under `data/experimental_campaign/` are tracked empty templates:

- `specimens.csv`: material identity, chemistry source, heat/batch, controlled
  taxonomy ids, and physical split group;
- `process_steps.csv`: ordered, machine-readable process history;
- `images.csv`: specimen-linked imaging and provenance;
- `mechanical_tests.csv`: direct test replicates and uncertainty.

Images and raw files use paths relative to that directory. Run:

    uv run microhard validate-campaign

Validation checks required columns, unique ids, foreign keys, ordered process
steps, numeric domains, consistent property units, dates, path containment, and
image hashes. Enable ingestion with:

    adapters = ["uhcs", "literature_steel", "experimental_steel"]

The experimental adapter averages direct replicates per specimen/property only
when creating canonical image records. The raw replicate rows remain the source
of truth. Every attached property is tagged `measured`.

## Model and prospective-validation gates

Do not advance because training loss improved. Advance only when the following
gates pass:

1. **Representation:** constituent/morphology measurements agree with held-out
   metallographer annotations at the relevant physical scales.
2. **Forward property model:** microstructure adds predictive value beyond
   grade and process metadata on an unseen condition and unseen batch.
3. **Calibration:** prediction intervals cover their stated fraction of
   held-out direct measurements.
4. **Process model:** predicted morphology distributions match independent
   process runs, not merely representative-looking images.
5. **Inverse test:** manufacture 10 to 20 model-proposed treatments blindly;
   the measured property vector reaches the requested Pareto region within
   calibrated uncertainty.
6. **Repeatability:** repeat the closed loop on a new heat/batch before making
   a generalization claim.

Property-conditioned image generation can be explored before Gate 5, but it is
not evidence of inverse design until a feasible process produces the generated
state and its properties are measured.
