---
name: add-property-head
description: Add a new property regression head (e.g. tensile strength, conductivity) for a material family or adapter scope. Use when a new material property should be predicted from microstructure features.
---

# Add a property head

Property heads regress a `FeatureVector` to one property value. They read
feature vectors and nothing else, which keeps them cheap to retrain and easy
to inspect. Registry and base class live in `src/microhard/heads/`.

Steps:

1. Subclass `PropertyHead` (heads/base.py) in a new `heads/<property>.py` and
   implement two methods. `fit(X, y)` receives a DataFrame whose columns are
   feature names like `frac:ferrous/network`; store whatever `predict` needs
   (fitted model, `feature_names`) on `self` and return a metrics dict. With
   few labeled samples, use leave-one-out CV and report MAE and R² the way
   heads/hardness.py does. `predict(fv)` aligns features via `fv.get(name)`
   over the stored `feature_names`.

2. Register it at module bottom: `register("<scope>", "<property_name>",
   MyHead)`, where scope is `family` or `family/adapter` (e.g.
   `"ferrous/uhcs"`). Lookup falls back from `family/adapter` to `family`, so
   only register at family level if the calibration genuinely transfers
   across datasets.

3. Import the module at the bottom of `heads/__init__.py` so registration
   runs on package import.

4. Property values come from adapter records (`record.properties`), keyed by
   `property_name`. Once an adapter emits them, fitting is
   `fit_property_head(cfg, scope, property_name)` (pipeline.py) with no new
   orchestration code. Add a CLI command mirroring `fit-hardness` if the
   property warrants one.

5. Tests (tests/test_heads.py): a synthetic linear relation the fit should
   recover, the below-minimum-samples error, a save/load roundtrip through
   `PropertyHead.load`, and the registry lookup with family fallback.
   `predict_image` must decline, not crash, before the head is fitted;
   tests/test_pipeline.py shows the pattern.
