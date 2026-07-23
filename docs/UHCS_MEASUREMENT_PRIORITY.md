# UHCS direct-measurement priority

Generated from the current database on 2026-07-23 with:

    uv run microhard plan-measurements --limit 11

The score is a deterministic acquisition heuristic, not a predicted model
improvement or probability. It combines distance from the seven directly
labelled conditions in process space with metadata completeness and existing
image coverage. The default scope is AC1, so casting identity is not confused
with heat-treatment effects. Only exact water-quench process keys are included.

| rank | sample id(s) | condition | existing images | priority score |
|---:|---|---|---:|---:|
| 1 | 18 | AC1 800C 85H WQ | 21 | 0.636 |
| 2 | 4 | AC1 1000C 5M WQ | 14 | 0.619 |
| 3 | 9 | AC1 700C 5M WQ | 11 | 0.619 |
| 4 | 24, 25 | AC1 970C 24H WQ | 47 | 0.611 |
| 5 | 26 | AC1 970C 3H WQ | 14 | 0.551 |
| 6 | 6 | AC1 800C 24H WQ | 19 | 0.547 |
| 7 | 8 | AC1 800C 5M WQ | 8 | 0.542 |
| 8 | 28 | AC1 970C 8H WQ | 12 | 0.536 |
| 9 | 7 | AC1 800C 3H WQ | 16 | 0.525 |
| 10 | 27 | AC1 970C 48H WQ | 5 | 0.476 |
| 11 | 10 | AC1 750C 5M WQ | 4 | 0.458 |

## First laboratory batch

Start with ranks 1 through 5 if material is available. Together they cover:

- the longest recorded hold at 800 C;
- both low- and high-temperature five-minute treatments;
- two long/intermediate holds at 970 C.

Before testing, resolve whether database sample ids 24 and 25 are independent
physical specimens or duplicate entries. Keep them in one validation group
until that is proven.

For every selected specimen:

1. Confirm the label, casting, physical identity, and remaining material.
2. Verify the complete thermal route and cooling medium against source records.
3. Acquire at least three Vickers measurements, preserving individual readings,
   load, dwell, location, method, and uncertainty.
4. If re-imaging is possible, acquire common-scale fields near 1964X under one
   preparation/detector protocol in addition to retaining the original images.
5. Enter the process, image, and mechanical rows in
   `data/experimental_campaign/`, then run
   `uv run microhard validate-campaign`.

After the first five conditions are ingested, rerun the planner. The greedy
ranking is conditional on what is already measured, so later ranks may change
when direct values or corrected metadata are added.
