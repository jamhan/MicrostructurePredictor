---
name: add-adapter
description: Add a new dataset adapter (new material family or micrograph dataset) producing canonical records. Use when integrating a dataset like EM3M, a MicroNet subset, or any new image+label source.
---

# Add a dataset adapter

Adapters translate one dataset's native layout into `CanonicalRecord`s
(src/microhard/records.py). Task code only ever sees canonical records, so a
new dataset should not require touching the router, segmenter, or heads.

Steps:

1. If the dataset introduces a new family or constituents, extend
   `src/microhard/taxonomy.yaml` first (see the edit-taxonomy skill). Every
   label an adapter emits must be a registered taxonomy node id.

2. Pick a base class. For a folder of images plus a CSV of labels, subclass
   `ImageFolderAdapter` (adapters/folder.py): set `name`, `family`, optionally
   `modality`, and put `labels.csv` (columns `path, taxonomy_labels[,
   group_id, scale_um_per_px]`, multiple node ids joined with `|`) under
   `data/<name>/`. `MicroNetAlAdapter` shows this needing no new logic. For
   anything richer (databases, masks, properties), subclass `BaseAdapter` in a
   new `adapters/<name>.py`; `UHCSAdapter` (adapters/uhcs.py) is the reference
   for scale computation, label mapping, mask attachment, and property joins.

3. Decorate the class with `@register_adapter` and import the module at the
   bottom of `adapters/__init__.py` so registration runs on package import.

4. Record fields to get right: `record_id` unique and prefixed with the
   adapter name; `group_id` set to the physical sample, since it is the split
   unit that prevents leakage; `scale_um_per_px` computed whenever the source
   provides it; `taxonomy_labels` and `mask_class_nodes` as node ids
   (`validated_records()` checks them, use it in tests); `properties` as
   measured sample values like `{"hardness_hv": 310.0}`.

5. Add a synthetic fixture in tests/conftest.py (small generated files, no
   real downloads) and tests in tests/test_adapters.py covering record count,
   scale, group_id prefix, and label mapping. `test_folder_stub_records` is a
   template.

6. Enable the adapter in `Config.adapters` or a run TOML, and run
   `uv run pytest`. The router picks up new families on its next training run.
