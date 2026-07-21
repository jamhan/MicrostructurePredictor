---
name: transcribe-hardness
description: Add hardness (HV) values from published sources and recalibrate the hardness head. Use when new hardness labels are available for UHCS samples.
---

# Transcribe hardness labels

The UHCS sqlite has no hardness columns; `data/hardness_labels.csv` is the
only source of HV values. Every row must trace to a published measurement or
a real test report. Do not enter estimates or placeholders.

CSV format:

```csv
sample_label,hardness_hv,source_note
AC1 800C 90M WQ,876,"Hecht 2017 PhD thesis Table A.2: 66.3±0.2 HRC; HV interpolated from ASTM E140-07 Table 1"
```

- `sample_label` must equal the `label` column of the sqlite `sample` table
  (`sqlite3 data/microstructures.sqlite 'SELECT label FROM sample'`).
- `hardness_hv` is numeric Vickers hardness. If the source reports another
  scale (the Hecht thesis uses HRC), convert via ASTM E140 and record the
  original value and scale in the note. Non-numeric rows are dropped at load.
- `source_note` records the document, the table or figure, the original
  value with uncertainty, and any assumption made in matching the source
  condition to a UHCSDB sample label. Mark assumptions with the word ASSUMED
  so they are easy to audit.

After adding rows:

```bash
uv run pytest tests/test_adapters.py   # CSV parses and joins to samples
uv run microhard extract-features      # if the segmenter changed since last run
uv run microhard fit-hardness          # leave-one-out CV, prints MAE and R²
```

Fitting needs at least three labeled samples with extracted features; below
that it prints why and does nothing. With single-digit n, quote the LOO MAE
next to any prediction you show anyone.
