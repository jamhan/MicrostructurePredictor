---
name: add-adapter
description: Add a new dataset adapter (new material family or micrograph dataset) producing canonical records. Use when integrating a dataset like EM3M, a MicroNet subset, or any new image+label source.
---

# Add a dataset adapter

Adapters translate one dataset's native layout into `CanonicalRecord`s
(src/microhard/records.py). Everything downstream — router, segmenter,
features, property heads — consumes only canonical records, so a new dataset
never touches task code.

## Steps

1. **Taxonomy first.** If the dataset introduces a new family or constituents,
   extend `src/microhard/taxonomy.yaml` (see the `edit-taxonomy` skill).
   Every label an adapter emits must be a registered taxonomy node id.

2. **Pick a base.**
   - Simple folder of images + CSV labels → subclass `ImageFolderAdapter`
     (src/microhard/adapters/folder.py); set `name`, `family`, optionally
     `modality`; drop `labels.csv` (columns: `path, taxonomy_labels[,
     group_id, scale_um_per_px]`, multiple node ids joined with `|`) under
     `data/<name>/`. See `MicroNetAlAdapter` — often zero new logic needed.
   - Anything richer (databases, masks, properties) → subclass `BaseAdapter`
     in a new `src/microhard/adapters/<name>.py`. `UHCSAdapter`
     (adapters/uhcs.py) is the reference implementation: scale computation,
     label mapping, mask attachment, property join.

3. **Decorate with `@register_adapter`** and import the module at the bottom
   of `src/microhard/adapters/__init__.py` so registration runs on import.

4. **Record contract checklist**
   - `record_id` unique, prefixed with the adapter name.
   - `group_id` = the physical sample (this is the anti-leakage split unit) —
     records of one sample must share it.
   - `scale_um_per_px` computed whenever derivable; `None` only when the
     source truly lacks it.
   - `taxonomy_labels` / `mask_class_nodes`: taxonomy node ids only, never
     bare strings. `validated_records()` enforces this — use it in tests.
   - `properties`: measured sample-level values, e.g. `{"hardness_hv": 310.0}`.

5. **Tests** (tests/test_adapters.py): add a synthetic fixture in
   tests/conftest.py (tiny generated files, never real downloads) and assert:
   record count, scale, group_id prefix, label mapping, `validated_records()`
   passes. Mirror `test_folder_stub_records`.

6. **Enable it**: add the adapter name to `Config.adapters` (or an `adapters =
   [...]` line in a run TOML). Run `uv run pytest` — the router picks up new
   families automatically on the next `train-router`.
