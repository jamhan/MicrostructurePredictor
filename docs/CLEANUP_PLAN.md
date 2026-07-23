# microhard repo cleanup

## Context

A review of the repo on 2026-07-23 found the code healthy (180 tests pass) but
the repo around it untidy in ways that will cost time later:

- **45 files of finished work are uncommitted** on a repo with a public GitHub
  remote — the whole IN718 linkage effort (`linkage.py`,
  `adapters/godec_in718.py`), the measurement planner, four new docs, two new
  notebooks. All of it has passing tests. None of it is in git.
- **The same concept is spelled three different ways in three modules.**
  `SCATTER_KINDS` exists in `properties.py:56`, `adapters/literature.py:68`, and
  `experimental_campaign.py:104` with three *disjoint* vocabularies, so a
  `scatter_kind` value accepted by one CSV loader is rejected by another. The
  confidence ladder is similarly triplicated, with the weights `0.25/0.55/0.85`
  hand-copied into all three.
- **Accumulated cruft**: 9 MB of stale `.ipynb_checkpoints`, a `.ruff_cache`
  with no ruff installed and no ruff config, orphaned docs, missing `SOURCE.md`
  files that `.gitignore` already expects.

Goal: get the work safely into version control, collapse the duplication that
will otherwise keep drifting, and leave a linter behind so it does not
re-accumulate. **No model behavior changes** — segmentation and hardness numbers
must be identical afterward, so any future change in them is attributable.

Decisions taken up front: strictly cleanup (classifier wiring deferred);
`topo.py` and `experimental_campaign.py` stay, marked as parked; the 6.3 GB of
raw Zenodo downloads stay on disk.

---

## Stage 1 — Get the working tree into git

Work on a new branch `repo-cleanup` off `main`; nothing gets pushed.

Run `uv run pytest` first to confirm the 180-test baseline, then four commits:

1. **Code** — everything under `src/` and `tests/`, plus the `pyproject.toml`
   description change. Self-contained and testable on its own; the one test that
   reads real repo data (`test_literature_adapter.py:17`) reads
   `data/literature_steel/manifest.csv`, which is already tracked.
2. **Data + provenance** — `data/public_in718_godec_2024/`,
   `data/experimental_campaign/`, `data/public_sources.csv`, `.gitignore`.
3. **Docs** — the four new files in `docs/` plus the `DATASET_PLAN.md` edits.
4. **Notebooks + tooling** — `notebooks/`, the build and fetch scripts, the
   `notebook` dependency group, `uv.lock`.

Re-run `uv run pytest` after commit 1 to confirm the code commit stands alone.

## Stage 2 — Collapse the duplicated vocabularies

New `src/microhard/confidence.py` holding the single source of truth:

- `CONFIDENCE_WEIGHTS` — one table covering every rung: `reject` 0.0,
  `low`/`review` 0.25, `medium` 0.55, `high` 0.85, `exact` 1.0.
- `LOOKUP_LEVELS` / `LINKAGE_LEVELS` — the subset each surface may legally
  declare, so validation stays as strict as it is today.
- `SCATTER_KINDS` — the union of the three current vocabularies. This is safe:
  `scatter_kind` is only ever validated and stored, never interpreted
  numerically (the only semantic rule is the "`unreported` exactly when scatter
  is blank" invariant, which stays with each loader).
- `weight_for(level)`.

Then delete the local tables and import from it:

| File | Removes |
|---|---|
| `properties.py:56,59-60` | `SCATTER_KINDS`, `CONFIDENCE_LEVELS`, `CONFIDENCE_WEIGHTS` |
| `adapters/literature.py:62-63,68` | `MATCH_CONFIDENCE`, `MATCH_WEIGHTS`, `SCATTER_KINDS` |
| `linkage.py:36` + `:293-314` | `CONFIDENCE_LEVELS`, five inline `training_weight` literals |
| `experimental_campaign.py:104` | `SCATTER_KINDS` |

The emitted level *names* stay exactly as they are — `low` in the two CSV
schemas, `review` in the linkage scorer — because both are part of a documented
file format and of the `audit-public-links` output. Only the weights and the
scatter vocabulary get centralized. A comment in `confidence.py` records that
`low` and `review` are the same rung, and that unifying the names is a later,
schema-breaking change.

## Stage 3 — Remove the duplicated property/weight pass

`pipeline.py:165` (`_property_by_group`) and `pipeline.py:178`
(`_property_weight_by_group`) are near-identical: both walk every adapter and
both call `load_property_lookup` + `lookup_index` + `join_properties`, so
`fit-hardness` does the entire join twice.

Replace with one
`_labels_by_group(cfg, taxonomy, property_name) -> dict[str, tuple[float, float]]`
returning `(value, weight)` from a single pass; `fit_property_head` unpacks it
into `y` and `sample_weight`.

This also closes a latent mismatch: the value function selects the first record
where the property `is not None`, while the weight function selects the first
record where the key is merely *present*. A record carrying
`{"hardness_hv": None}` therefore contributes its weight to a value taken from a
different record. One pass makes that impossible.

## Stage 4 — Lint

Add `ruff` to the `dev` dependency group and a `[tool.ruff]` block to
`pyproject.toml`: `line-length = 100`, `select = ["E", "F", "I", "B", "BLE"]`.
Those rule families are the evident intent already — the codebase carries
`# noqa: BLE001`, `E402`, and `F401` comments with no linter behind them.

Then fix what it flags:

- `heads/hardness.py:14-21` — imports unsorted (`sklearn.ensemble` before
  `sklearn.base`), plus two separate `from sklearn.pipeline import` lines.
- Three lines over 100 characters (longest is 107).

`--fix` handles the import ordering; the rest is small and manual. This is the
step that stops Stage 2's duplication from silently coming back.

## Stage 5 — Cruft on disk

- Delete `notebooks/.ipynb_checkpoints/` (9 MB of stale copies — the
  `distant_supervision` checkpoint is 3.9 MB against a 1.5 MB live file).
- Delete `__pycache__/` under `src/` and `scripts/`, plus `.pytest_cache/` and
  the now-stale `.ruff_cache/` (Stage 4 regenerates the latter legitimately).

All are already gitignored, so this is disk only, no history impact.

## Stage 6 — Provenance and docs

- Add the two `SOURCE.md` files `.gitignore` already un-ignores but which do not
  exist: `data/public_in718_carbides_2025/SOURCE.md` and
  `data/public_316l_composition_2026/SOURCE.md`, each recording the Zenodo
  record id, retrieval date, license, and that no adapter consumes it yet.
- Add `.env.example` naming `HF_TOKEN` and `HF_WRITE_TOKEN`. The real `.env` is
  correctly ignored and — verified — was never committed, but nothing currently
  tells a reader those tokens are needed.
- README: link the three orphaned docs (`UHCS_MEASUREMENT_PRIORITY.md`,
  `CRYSTAL_DISCOVERY_TO_MICROSTRUCTURE_ML.md`, `EXPERIMENTAL_CAMPAIGN.md`) and
  the newest notebook, `microstructure_ai_concepts_lab.ipynb` (134 cells,
  unmentioned while the other three are described).
- Park the two dormant modules with a one-line header comment each stating they
  are not on a live code path and what would activate them: `topo.py` (never
  called from `features.py`) and `experimental_campaign.py` (the fallback
  acquisition plan). No code moves.
- Move `demo.toml` next to its only consumer,
  `notebooks/how_the_net_works.ipynb`, or document it in the README beside the
  `myrun.toml` example — it currently sits at the root explained by nothing.

## Stage 7 — Small correctness items

- `heads/hardness.py:115` — `predict` reads features via `fv.get(name)`, which
  defaults a missing feature to `0.0`. A fit/predict feature-name mismatch
  therefore yields a plausible-looking HV number instead of failing. Raise
  instead, matching the abstain-don't-fabricate rule the rest of the pipeline
  follows. Add a test.
- Expose `join_audit.py` as a `microhard audit-join-keys` command so both audits
  are reachable the same way; `scripts/audit_join_keys.py` becomes a thin
  wrapper. Currently `audit-public-links` is a CLI command while its sibling is
  script-only.

## Deferred — not in this pass

- Wiring `classify.py` into the FeatureVector (the README's named fix for the
  123 HV hardness error). Real feature work: changes results, needs
  `extract-features` and `fit-hardness` re-run. Best as its own branch.
- Converting `region_stats` areas from px² to µm² via `scale_um_per_px`. Also
  behavior-changing, and worth doing together with the magnification
  stratification the temperature-probe result implies.
- The 2911-line `scripts/build_microstructure_ai_concepts_lab.py` notebook
  generator (jupytext would replace it). Large, and orthogonal to everything
  above.

---

## Verification

1. `uv run pytest` — 180 tests, green at the baseline, after the Stage 1 code
   commit, and after each of Stages 2, 3, 4 and 7.
2. `uv run ruff check .` — clean after Stage 4.
3. `uv run microhard --help`, `taxonomy`, `plan-measurements`, and
   `audit-public-links` — confirms Stages 2 and 3 did not break the CLI paths
   that touch confidence weights and the property join.
4. **No-behavior-change check**: keep the current `data/features.csv` and the
   fitted `checkpoints/heads/ferrous__uhcs--hardness_hv.pkl`, re-run
   `uv run microhard fit-hardness`, and confirm the reported LOO MAE is still
   122.9 HV with R² 0.13. This is the real guard on Stages 2 and 3 — the whole
   point of the pass is that these numbers do not move.
5. `git status` clean, `git log --oneline` showing the intended commits, nothing
   pushed.
