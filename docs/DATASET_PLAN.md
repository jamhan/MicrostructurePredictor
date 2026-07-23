# Growing the property labels by distant supervision

The binding constraint on this project is property labels. Seven measured
hardness values, all on one alloy, varying along essentially one processing
axis. `docs/ROADMAP.md` names that as the thing everything else waits on. This
document is the plan for attacking it without new laboratory work, and an
honest account of what the resulting labels are and are not worth.

## The idea, and why it is defensible here

Distant supervision means labelling data by joining it against an existing
table rather than annotating it directly. The version here: attach a published
hardness value to a micrograph by matching on the alloy grade and the heat
treatment.

This works because bulk hardness is a reproducible function of grade and
condition. Two labs that austenitize the same steel from the same temperature
and quench it in water measure hardnesses that agree far more closely than
either agrees with a furnace-cooled sample of the same steel. The spread within
a (grade, condition) pair is small compared to the spread across pairs, so a
handbook value for the pair is a usable, if noisy, label for any micrograph of
that pair.

It works because hardness is a bulk average over a volume much larger than any
single feature in the image. That is exactly the class of property distant
supervision is valid for, and the reason this document keeps saying hardness
and not something else. See "Where this is invalid" below.

The exact-key audit leaves 215 of 961 UHCS micrographs across 14 split groups
with a complete, defensible key. Seven UHCS groups carry measured hardness.
The first literature extraction adds 19 SEM panels across ten independent
weak-label groups. Published micrographs can grow this set, but only when the
caption, methods, and property table identify the same material state.

## The join key

The key is a pair of taxonomy node ids, one from each of two new axes in
`src/microhard/taxonomy.yaml`:

    (alloy_grade, condition)
    ("grade/ferrous/aisi_1045", "condition/austenitize/water_quench")

Both are ordinary taxonomy nodes, so the existing contract holds unchanged:
every label anywhere in the system is a registered node id, never a bare
string. A root node declares its axis with `axis:` and descendants inherit it;
roots that declare nothing are on the `microconstituent` axis, which is why the
`ferrous` and `aluminum` trees read exactly as they did before. `Taxonomy.families()`
still returns material families only, so the router is untouched.

`CanonicalRecord` carries the key in two optional fields, `alloy_grade` and
`condition`, with a `join_key` property that returns the pair or None. Either
half may be missing. A record missing either half is never joined, and there is
no fallback to a partial key: a hardness looked up for the grade alone would be
an average over treatments spanning 400 HV.

`BaseAdapter.validated_records()` checks each field against its own axis, so a
grade id used where a constituent label belongs fails at adapter time rather
than producing a key nothing matches.

### Choosing the granularity of a condition node

The condition axis has one node per physical processing route, not one per
phrase people use to describe a route. Normalizing is austenitizing followed by
a still-air cool, so it is `condition/austenitize/air_cool`, and "normalized"
is an alias for that node rather than a second node meaning the same thing. Two
nodes for one route would silently split a join key in half.

The axis allows four levels rather than the microconstituent axis's three, and
the fourth level exists for one purpose: splitting a route when the property
genuinely differs across it. The UHCS data shows why this is not hypothetical.
Three of the seven measured samples share the route "austenitize, then quench":

| sample | condition node | measured HV |
|---|---|---|
| AC1 900C 90M Q | `condition/austenitize/unspecified_quench` | 810 |
| AC1 970C 90M Q | `condition/austenitize/unspecified_quench` | 611 |
| AC1 800C 90M WQ | `condition/austenitize/water_quench` | 876 |

The two samples sharing a single node differ by 199 HV, on a dataset whose full
measured range is 466 HV. The route is not the whole story for this alloy: the
austenitizing temperature sets how much carbide dissolves, and that sets the
hardness. So `condition/austenitize/unspecified_quench` is too coarse to be a
distant-supervision key for UHCS, and using it as one would teach the model
that two visibly different microstructures have the same hardness.

The rule that follows: before a (grade, condition) pair is used as a join key,
the within-key scatter has to be small compared to the across-key range. Where
it is not, split the condition node until it is (`.../water_quench/t970c`), or
mark the entry low confidence and expect the benchmark to show it.

## The lookup table

`data/property_lookup.csv`, one row per (alloy_grade, condition,
property_name). It ships as a header with no rows. Loader, validation and join
live in `src/microhard/properties.py`.

| column | meaning |
|---|---|
| `alloy_grade` | node id on the alloy_grade axis |
| `condition` | node id on the condition axis |
| `property_name` | key used in `CanonicalRecord.properties`; must be in `PROPERTY_UNITS` |
| `value` | number, in the unit that `property_name` implies |
| `unit` | must match `PROPERTY_UNITS[property_name]`; a mismatch is a transcription error, not a conversion request |
| `scatter` | within-condition spread, same unit, blank if the source does not report one |
| `scatter_kind` | `sd`, `half_range`, `tolerance_band`, or `unreported` |
| `n_measurements` | samples behind the value, blank if unreported |
| `join_confidence` | `high`, `medium`, `low`; see the rubric below |
| `source_citation` | free text that identifies a real document |
| `source_url` | DOI or URL where the value can be checked |
| `note` | assumptions made in matching the source's wording to this key |

The registered bulk-property schema now covers Vickers hardness, yield
strength, UTS, elongation, reduction of area, and Young's modulus. Hardness is
still the only property with a fitted head. Adding anything else means adding a
line to `PROPERTY_UNITS`, which is deliberate: an unregistered property name is
rejected at load, so nobody quietly starts a fatigue column in a table whose
whole premise does not apply to fatigue. Orientation-sensitive tensile
properties also require orientation-compatible image metadata; see
`docs/PUBLIC_DATA_LINKAGE.md`.

Invalid rows are rejected at load rather than warned about. This includes an
unregistered join-key node, an empty citation, a unit mismatch, a non-finite
value, inconsistent scatter metadata, and a non-integer sample count. A
`high`-confidence row must report scatter or a sample count. Two rows for the
same key are also rejected: two published values have to be reconciled by a
person, or the condition split until they are different keys. The loader must
not pick one silently.

### Confidence rubric

`high`: the source states the grade and the full treatment, both map onto
single nodes with no interpretation, and the source reports scatter or a sample
count.

`medium`: the mapping required a judgement that is written down in `note`, for
example a source that gives a treatment temperature range where the node covers
a point, or reports a single value with no scatter.

`low`: the grade or the condition was inferred rather than stated, or the
source is a secondary compilation with no traceable measurement behind it.
These rows are still useful as a weak prior, and `join_properties(...,
min_confidence="medium")` excludes them in one argument.

### How a row becomes a record property

Exactly, from `properties.join_properties`:

1. A record whose `join_key` is None is returned unchanged.
2. Otherwise every entry for that key whose `join_confidence` meets the
   threshold sets `properties[property_name] = value` and
   `property_sources[property_name] = DISTANT`.
3. A property the record already carries is left alone. A direct measurement on
   the physical sample always beats a value looked up for its grade and
   condition, and one distant value never overwrites another.

Records are frozen, so the join returns new ones. The citation is not copied
onto the record; it stays in the table, reachable through the record's
join_key, so a wrong citation is corrected in one place.

`CanonicalRecord.property_sources` marks each value `measured` or `distant`.
Nothing downstream may treat the two as interchangeable, and
`properties.measured_properties(record)` is the accessor for code that must
not see weak labels.

### The protocol for filling it

The table is empty on purpose. Filling it is a research task with a citation
requirement, not a data-entry task, and the following is what a row costs.

1. Pick a (grade, condition) pair that micrographs in hand actually carry.
   There is no value in a row nothing joins to.
2. Find the value in a source that can be cited and checked. In descending
   order of preference: the source paper of the image dataset itself, which is
   best because the measurement and the micrograph are the same material; a
   standards or handbook value (ASM Metals Handbook, ASTM); a per-grade
   datasheet aggregator such as MatWeb, which is a secondary compilation and
   therefore rarely better than `medium` confidence.
3. Check the source's condition wording maps onto exactly one condition node.
   If it straddles two, either the wording is ambiguous, in which case do not
   add the row, or the vocabulary is too coarse, in which case add the node
   first.
4. Convert to the registered unit before entering the value. Hardness scales
   are not linearly interconvertible; use a published conversion table, cite
   it in `note`, and drop the confidence a level if the conversion was needed.
   `data/hardness_labels.csv` is the precedent: its `source_note` records the
   table, the original HRC value with its uncertainty, and every assumption
   made, and rows resting on an assumption say ASSUMED.
5. Record scatter if the source reports it. If it does not, `unreported` is the
   honest entry. Do not substitute a plausible-looking number.
6. Reject the pair entirely if step 2 turns up two sources that disagree by
   more than the across-key range would tolerate. That is a signal the
   condition node is too coarse, not a signal to average.

The measured UHCS values stay in `data/hardness_labels.csv` and are not
migrated into this table. They are per-sample measurements keyed by sample
label, tagged `measured`, and they are the only values in the project a
benchmark may be scored against.

## Panel-level literature matches

The grade/condition lookup is the right representation for a handbook value,
but not for every paper. A source can report several hardnesses for one
treatment at different sampling locations. Flattening those cells into one
lookup row would discard a real variable, while adding several rows would
violate the lookup's unique-key contract.

`data/literature_steel/manifest.csv` handles this case. It has one row per
published panel and records:

- the citation, DOI, article and PDF URLs, license, PDF page, figure and panel;
- the property table page and exact row/column locator;
- the controlled grade and condition ids, sampling location and specimen id;
- the match relation and confidence, plus whether the exact physical specimen
  is confirmed;
- the value, unit, Vickers load and dwell, reported scatter and measurement
  count;
- the redistributed image path and SHA-256 hash.

The literature adapter attaches the property directly from this manifest. It
does not insert the value into `property_lookup.csv`. Panel pairs at two
magnifications share a specimen id and therefore one split group.

The first source is Guan et al. (2026), DOI 10.3390/met16030243. Figure 3 has
18 SEM panels for three heat treatments and three through-thickness locations;
Table 3 reports one HV1 mean for each of the nine treatment/location cells.
The full mapping, processing route, composition, test method and CC BY
attribution are in
`data/literature_steel/guan_2026_metals_16_243/SOURCE.md`.

The second source is Ren et al. (2023), DOI 10.3390/met13040771. Figure 2b is
one SEM panel of 35CrMo after a documented homogenize, austenitize and oil
quench route. Section 3.1 reports 532.1 plus or minus 7.2 HV for that state at
0.5 kgf and 10 s. The source does not define the plus-or-minus statistic or a
hardness measurement count, so the manifest records both limitations. The
crop, chemistry, route and match are documented in
`data/literature_steel/ren_2023_metals_13_771/SOURCE.md`.

This match has high metadata confidence: the figure caption and table use the
same condition and location names, and all specimens came from the same plate.
It is still tagged `distant`. The paper does not establish that a panel shows
the exact coupon or indent neighbourhood used for hardness. High match
confidence and a direct physical measurement are different claims.

The source panels contain letters, scale bars and phase annotations. They are
kept unchanged as provenance copies. A raw-image model must mechanically mask
or crop those markings before training, or it may learn figure layout instead
of microstructure.

## String normalization

Source metadata is written by people: "normalised low carbon steel", "0.45C
quenched & tempered", or the bare UHCSDB cool_method code "WQ".
`src/microhard/normalize.py` maps those onto node ids.

The design decision that matters is that nothing in it infers metallurgy. The
alias tables are curated by hand, one entry per phrase a source actually uses,
and the parser only handles messiness: case, punctuation, British spellings,
inflections, filler words, and surrounding text. A string with no matching
alias returns None.

The normalizer does not turn a composition into a specific designation.
`0.45C`, for example, maps to the registered medium-carbon family, not AISI
1045. Several alloy grades contain about 0.45% carbon, and a more specific id
would claim information the source did not provide.

That is the right default because the two failure modes are not symmetric. A
missing join costs one record out of hundreds. A wrong join teaches the model
that a real microstructure has a hardness it does not have, and nothing
downstream can detect it. So `WC` and `650-1H`, two UHCSDB cool_method codes
whose meaning is not clear from the dataset, map to nothing rather than to the
nearest plausible route.

Two rules make the tables safe to extend. The longest matching alias wins, so
"water quenched" beats the "quenched" contained inside it. A tie between two
different ids raises `AmbiguousAliasError` rather than picking one. Separate
metadata fields that resolve to conflicting ids raise the same error.
`check_aliases(taxonomy)` asserts every alias target is a registered node on
the right axis, which catches an alias left pointing at a renamed node.

Applied coarsely to the real UHCSDB metadata, normalization recognizes 36 of
47 samples and 564 of 961 micrographs. That is not the property-join count.
The stricter audit in `docs/UHCS_JOIN_KEY_AUDIT.md` checks temperature, hold
time, cooling route, multi-step labels and unlinked images. Only 14 groups and
215 micrographs currently pass that exact-key audit.

## The held-out-condition split

Design, not yet implemented.

### The problem it exists to detect

Distant supervision gives every micrograph of a (grade, condition) pair the
same label. Under the current split, which is grouped by physical sample, two
samples of the same pair can land on opposite sides, carrying an identical
label. A model that learns to recognise the pair, from the etchant, the
magnification, the imaging conditions, or genuinely from grade-specific
microstructure, can then recall its label without having learned anything about
how microstructure sets hardness. Validation numbers look excellent and the
model has learned a lookup table.

That failure is invisible to a random split and it is the specific risk
distant supervision introduces. The benchmark has to be built to expose it.

### The protocol

Three split modes, reported side by side:

`by-sample`, the current behaviour. Groups are physical samples. Measures
performance on new micrographs of conditions the model has seen. Retained
because it is the right measure for the segmentation task and the honest
measure for a deployment where the conditions are known in advance.

`held-out-condition`. Choose a set of condition nodes; every record whose
condition is in that set goes to test, and no record of those conditions
appears in train or validation. The test set therefore contains (grade,
condition) pairs the model has never seen a label for. Measures whether the
model learned microstructure to property.

`held-out-grade`. The same construction over grades. The harder version, and
the one that matters for the claim that this generalises across the ferrous
family rather than within one casting.

Implementation is a generalisation of `records.split_records_by_group`: the
same deterministic assignment, but taking a key function and an explicit
held-out set instead of a random permutation over group ids. Within the train
portion, grouping by sample still applies, so the no-leakage guarantee survives
in both modes. Literature panel pairs use the manifest specimen id as that
group. Sampling location remains part of the specimen match even when the
headline holdout is by condition.

### Choosing what to hold out

Not at random. The held-out set has to leave the training range intact, or the
result measures extrapolation rather than generalization. Three constraints:

The training set must still span the property range. Holding out both extremes
of hardness leaves a model that has never seen a hard sample, and its failure
on one says nothing about whether it learned physics.

The training set must still cover the microconstituent vocabulary. If every
martensitic condition is held out, the test is whether the model recognises an
unseen constituent, which is the classifier's job and a different question.

The held-out conditions must have enough records to give the number a
confidence interval worth printing. At current scale that is a real constraint,
and where it cannot be met, say so instead of reporting the point estimate.

### The baselines that make the number mean something

A held-out-condition MAE on its own is not interpretable. It has to be read
against three references:

The join-key-only baseline. Predict from (grade, condition) alone, ignoring the
image entirely, backing off to the grade mean for a key not seen in training. A
model that does not beat this learned nothing from the images. This is the most
important of the three, and it is cheap.

The same model under `by-sample`. The gap between the two is the size of the
memorization effect. A small gap is the result worth having. A large gap means
the by-sample number was measuring label recall.

The model scored on measured labels only. Distant labels are the table's
opinion, so an evaluation against them measures agreement with the table, not
with a durometer. `measured_properties()` exists for this. With seven measured
samples this is a very small test set, which is itself worth reporting plainly.

Headline reporting is the held-out-condition number against the join-key-only
baseline. The by-sample number goes alongside it, labelled as the optimistic
one.

## Where this is invalid

Distant supervision is valid for a property when (grade, condition) determines
it. That is a strong condition and most interesting properties fail it.

It fails for anything controlled by defects rather than by average structure.
Fatigue life is the case this project cares about most, per `docs/ROADMAP.md`,
and it is the clearest failure: two specimens of the same grade and the same
heat treatment routinely differ in fatigue life by an order of magnitude,
because life is set by the largest pore or inclusion in the specimen, not by
the mean structure. A handbook fatigue limit for a grade and condition is a
statistic about a population, not a label for a specimen. Attaching it to a
micrograph would teach a model that the thing distinguishing two specimens is
noise, which is the exact opposite of the truth, and it would do so invisibly.
The same argument rules out fracture toughness, ductility near failure, and
anything else whose value is an extreme-value statistic. It is worth being
blunt about this, because the topological features in `topo.py` were written
for precisely those properties, and this technique cannot be used to feed them.

It fails when the property is anisotropic or geometry-dependent and the source
does not record the orientation or the section.

It fails when the source's condition wording hides a variable the property
depends on, which is the UHCS austenitizing-temperature case above, in the
milder form where nothing in the metadata reveals the problem.

It degrades, rather than failing, with surface condition, specimen size, and
indenter load. Those broaden the within-key scatter, which is what the
`scatter` column is for.

Hardness is the first defensible target. It is a bulk average and is measured by
a standardised method. It still needs a key fine enough to capture variables
that matter. The Guan plate varies by almost 40 HV through thickness within one
treatment, so location cannot be thrown away for that source. Indenter load is
also retained because HV1 and a different Vickers load are not assumed
interchangeable. That caution is why `PROPERTY_UNITS` still has exactly one
entry rather than opening the same machinery to defect-controlled properties.

## Current boundary

The generic lookup remains a header and protocol because no defensible
external value has yet matched an existing UHCS grade/condition key. The Guan
sources are instead a panel-level ingest: 19 images, ten weak-label groups and
ten published hardness means. They expand the data without pretending those
values belong to UHCS samples.

This extraction does not make the existing UHCS constituent segmenter valid on
the new bainitic microstructures. The present hardness head consumes that
segmenter's fractions and is scoped to the UHCS adapter, so it must not absorb
these records silently. The next modeling step is a raw-image hardness baseline
with annotation-safe crops, source-aware groups and held-out-condition
evaluation. Until that exists, the new records are data ready, not a claim of a
better calibrated model.
