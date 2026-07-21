---
name: edit-taxonomy
description: Extend or modify the materials taxonomy (families, constituents, morphologies). Use when adding a material family, a microconstituent class, or renaming taxonomy nodes.
---

# Edit the taxonomy

The taxonomy (src/microhard/taxonomy.yaml) is the single label registry:
family → constituent → morphology, max 3 levels. Node ids are derived from
the YAML keys as paths (`ferrous/pearlite/lamellar`). Every label in the
pipeline is a node id — bare strings are a bug.

## Rules

- **Ids are contracts.** Checkpoints (`class_nodes` in segmenter), feature
  CSV columns (`frac:ferrous/network`), fitted head pickles, and adapter
  label maps all reference node ids. Renaming a node orphans them — prefer
  adding; if you must rename, retrain/re-extract everything downstream and
  update `PRIMARY_TO_NODES` / `SEG_CLASS_NODES` in adapters/uhcs.py plus any
  folder-adapter labels.csv files.
- Keys: lowercase, hyphens over spaces. Give each node a human `name:`.
- New **family** = new level-1 key. The router only learns it after
  `train-router` runs with an adapter of that family enabled.
- Depth beyond 3 levels is rejected at load time by design.

## Workflow

1. Edit `src/microhard/taxonomy.yaml` (or point `Config.taxonomy_path` at a
   project-specific file — `.toml`/`.json` also load).
2. `uv run microhard taxonomy` — eyeball the tree.
3. `uv run pytest tests/test_taxonomy.py tests/test_adapters.py` — adapter
   label maps are validated against the taxonomy, so stale references fail
   here, loudly, before any training run.
