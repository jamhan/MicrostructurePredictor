---
name: edit-taxonomy
description: Extend or modify the materials taxonomy (families, constituents, morphologies). Use when adding a material family, a microconstituent class, or renaming taxonomy nodes.
---

# Edit the taxonomy

The taxonomy (src/microhard/taxonomy.yaml) is the single label registry:
family, constituent, morphology, at most three levels. Node ids are derived
from the YAML keys as paths (`ferrous/pearlite/lamellar`), and every label in
the pipeline must be one of them.

Things to know before editing:

- Node ids are referenced by trained artifacts: the segmenter checkpoint
  stores `class_nodes`, feature CSV columns embed ids
  (`frac:ferrous/network`), fitted head pickles store feature names, and
  adapter label maps point at ids. Renaming a node orphans all of those.
  Prefer adding nodes. If a rename is unavoidable, retrain the affected
  models, re-extract features, and update `PRIMARY_TO_NODES` and
  `SEG_CLASS_NODES` in adapters/uhcs.py plus any folder-adapter labels.csv.
- Keys are lowercase with hyphens; give each node a human-readable `name:`.
- A new family is a new top-level key. The router only learns it after
  `train-router` runs with an adapter of that family enabled.
- Nesting deeper than three levels is rejected at load time.

Workflow: edit the YAML (or point `Config.taxonomy_path` at a project file;
`.toml` and `.json` also load), check the tree with `uv run microhard
taxonomy`, then run `uv run pytest tests/test_taxonomy.py
tests/test_adapters.py`. Adapter label maps are validated against the
taxonomy, so stale references fail in tests before any training run.
