---
name: transcribe-hardness
description: Add hand-transcribed hardness (HV) values from the Hecht papers and recalibrate the hardness head. Use when new hardness labels are available for UHCS samples.
---

# Transcribe hardness labels

The UHCS sqlite has **no hardness columns**; `data/hardness_labels.csv` is
the only source of HV values and starts empty by design. Never fabricate
values — every row must trace to a published measurement.

## CSV contract

```csv
sample_label,hardness_hv,source_note
DUM1,310,"Hecht et al. 2017, Table 2"
```

- `sample_label` must equal the `label` column of the sqlite `sample` table
  (check with `sqlite3 data/microstructures.sqlite 'SELECT label FROM sample'`).
- `hardness_hv`: Vickers hardness, numeric. Non-numeric rows are dropped
  silently at load — don't use placeholders like "TBD".
- `source_note`: paper + table/figure, so provenance survives.

## After adding rows

```bash
uv run pytest tests/test_adapters.py   # CSV parses, joins to samples
uv run microhard extract-features      # if not done since last seg training
uv run microhard fit-hardness          # LOO CV; prints MAE (HV) and R^2
```

Fitting needs ≥ 3 labeled samples that also have extracted features; below
that it skips with a message rather than fitting nonsense. Expect wide error
bars until the label count grows — report the printed LOO MAE alongside any
prediction shown to stakeholders.
