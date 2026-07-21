# UHCS join-key audit

This is the first Phase 1 audit. It asks which UHCSDB records have enough
metadata to support an exact processing-condition node before any distant
hardness values are added.

Reproduce it with:

    uv run python scripts/audit_join_keys.py

The audit reads `data/microstructures.sqlite`. It does not add taxonomy nodes
or property values.

## Result

| measure | count |
|---|---:|
| sample rows | 47 |
| samples with micrographs | 43 |
| micrographs | 961 |
| micrographs linked to a sample row | 803 |
| micrographs with no sample link | 158 |
| micrographs with the current coarse join key | 564 |
| exact water-quench candidates | 215 |
| provisional air/furnace-cool records | 76 |
| coarse-key records still blocked | 273 |
| records without a complete coarse key | 397 |

The existing figure of 564 joinable images is too optimistic. It means only
that a grade alias and a cooling-route alias were found. It does not mean that
temperature, hold time, and all processing steps agree.

## Why the current key is too coarse

| grade | current condition | samples | images | structured thermal signatures |
|---|---|---:|---:|---:|
| UHCS casting AC | water quench | 1 | 21 | 1 |
| UHCS casting AC1 | air cool | 1 | 20 | 1 |
| UHCS casting AC1 | furnace cool | 3 | 73 | 2 |
| UHCS casting AC1 | unspecified quench | 14 | 224 | 14 |
| UHCS casting AC1 | water quench | 16 | 226 | 14 |

The AC1 water-quench key collapses 14 recorded temperature/hold combinations
onto one condition id. The unspecified-quench key does the same and also omits
the quench medium. The furnace-cool count understates its variation: sample 31
has a second 65 C / 4 h step in its label that is absent from the structured
columns.

No property row should be added against these coarse AC1 keys.

## Exact water-quench candidates

These rows have a recognized casting, Celsius temperature, hold time, a plain
`WQ` code, and a single thermal step in the sample label. Candidate means the
metadata can support a precise taxonomy node. It does not mean a hardness value
has been found.

| casting | proposed condition leaf | sample ids | images |
|---|---|---|---:|
| AC | `t800c_8h` | 2 | 21 |
| AC1 | `t1000c_5m` | 4 | 14 |
| AC1 | `t700c_5m` | 9 | 11 |
| AC1 | `t750c_5m` | 10 | 4 |
| AC1 | `t800c_24h` | 6 | 19 |
| AC1 | `t800c_3h` | 7 | 16 |
| AC1 | `t800c_5m` | 8 | 8 |
| AC1 | `t800c_85h` | 18 | 21 |
| AC1 | `t800c_90m` | 12 | 9 |
| AC1 | `t900c_90m` | 19 | 14 |
| AC1 | `t970c_24h` | 24, 25 | 47 |
| AC1 | `t970c_3h` | 26 | 14 |
| AC1 | `t970c_48h` | 27 | 5 |
| AC1 | `t970c_8h` | 28 | 12 |

The full node form would be, for example,
`condition/austenitize/water_quench/t800c_8h`. Sample ids 24 and 25 have the
same sample label and condition. They may be duplicate database entries for one
physical sample. Treat them as one split group until that is resolved.

## Provisional combinations

| casting | route and proposed leaf | sample ids | images | unresolved point |
|---|---|---|---:|---|
| AC1 | air cool / `t970c_90m` | 29 | 20 | confirm that `AR` means air cooled in the dataset source |
| AC1 | furnace cool / `t970c_5m` | 34 | 37 | furnace cooling rate is not recorded |
| AC1 | furnace cool / `t970c_90m` | 35 | 19 | furnace cooling rate is not recorded |

These can become exact nodes after checking the source documentation. They
should not receive `high`-confidence property rows on the current evidence.

## Blocked records

The 273 images that have a coarse key but are not candidates break down as:

| reason | images |
|---|---:|
| `Q` does not identify the quench medium | 224 |
| sample 11 lists 800 C, 900 C, and 970 C in one route | 7 |
| sample 31 contains an additional 65 C / 4 h step | 17 |
| sample 33 uses the special cooling code `WQ-2C` | 25 |

The remaining 397 images have no complete coarse key. Of these, 158 are not
linked to a sample row at all. Another 75 are linked to AC/AC1-labelled samples
whose treatment is missing or uses an unresolved code, and 164 belong to
samples whose grade is not identified by the current metadata.

## Implementation status

The exact temperature/hold leaves for these 14 water-quench combinations are
now registered. The UHCS adapter uses the structured temperature and time
columns, cross-checks them against the sample label, and emits no coarse
fallback. This produces 215 exact-key images across 15 database groups. The
duplicate `t970c_24h` rows remain separate database groups for now and must be
assigned to one conservative split group; the adapter now does this whenever
sample labels are duplicated.

## Hardness source check

Hecht's thesis Appendix Tables A.1 and A.2 were checked against the rendered
pages. They report hardness for 90-minute treatments only: 970 C under three
cooling routes, and water quenches from 800 C, 900 C, and 970 C. Those values
are already represented in `data/hardness_labels.csv` as direct measurements.
The thesis does not report hardness for the other exact hold times above, so no
new distant rows were added to `data/property_lookup.csv`.

The next data step is therefore external source discovery or new measurements,
not extrapolating the 90-minute values across hold times.
