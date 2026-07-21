---
name: add-property-head
description: Add a new property regression head (e.g. tensile strength, conductivity) for a material family or adapter scope. Use when a new material property should be predicted from microstructure features.
---

# Add a property head

Property heads regress a `FeatureVector` to one property value. They never
see raw images — that contract keeps them material-agnostic and cheap to
retrain. Registry + ABC live in `src/microhard/heads/`.

## Steps

1. **Subclass `PropertyHead`** (heads/base.py) in a new
   `src/microhard/heads/<property>.py`. Implement:
   - `fit(X: pd.DataFrame, y: np.ndarray) -> dict` — X columns are feature
     names like `frac:ferrous/network`; store what `predict` needs
     (fitted model + `feature_names`) on `self`; return a metrics dict.
     With tiny label sets, use leave-one-out CV and report MAE + R²
     (copy the pattern in heads/hardness.py).
   - `predict(fv: FeatureVector) -> float` — align via
     `fv.get(name)` over the stored `feature_names`.

2. **Register it** at module bottom:
   `register("<scope>", "<property_name>", MyHead)` where scope is
   `family` or `family/adapter` (e.g. `"ferrous/uhcs"`). Lookup falls back
   from `family/adapter` to `family`, so register at family level only if the
   head is genuinely calibration-transferable across datasets.

3. **Import the module** at the bottom of `src/microhard/heads/__init__.py`
   so registration runs on package import.

4. **Property values** come from adapter records (`record.properties`), keyed
   by `property_name` — make sure some adapter emits them, then fitting is
   just `fit_property_head(cfg, scope, property_name)` (pipeline.py); no new
   orchestration code. Add a CLI command in cli.py mirroring `fit-hardness`
   if it deserves one.

5. **Tests** (tests/test_heads.py): synthetic linear relation → fit recovers
   it; `< MIN_SAMPLES` raises; save/load roundtrip via `PropertyHead.load`;
   registry lookup + family fallback behavior. `predict_image` must abstain
   (not crash) before the head is fitted — tests/test_pipeline.py shows the
   pattern.
