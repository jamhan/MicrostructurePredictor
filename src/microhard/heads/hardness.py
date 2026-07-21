"""Hardness head for ferrous/UHCS: constituent features -> Vickers HV.

Gradient boosting + a linear (scaled Ridge) baseline, evaluated with
leave-one-sample-out CV — the honest choice for the tiny hand-transcribed
label set. The better model (by LOO MAE) is refit on all data and kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features import FeatureVector
from . import register
from .base import PropertyHead

MIN_SAMPLES_FOR_CV = 3

HARDNESS_HINT = (
    "Transcribe (sample_label, hardness_hv, source_note) rows from the Hecht "
    "papers into data/hardness_labels.csv, then run `microhard fit-hardness`."
)


class HardnessHead(PropertyHead):
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.model: Any = None
        self.feature_names: list[str] = []
        self.metrics: dict[str, Any] = {}

    def _candidates(self) -> dict[str, Any]:
        return {
            "linear": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "gbrt": GradientBoostingRegressor(random_state=self.seed),
        }

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> dict[str, Any]:
        if len(X) < MIN_SAMPLES_FOR_CV:
            raise ValueError(
                f"need >= {MIN_SAMPLES_FOR_CV} labeled samples for leave-one-out CV, got {len(X)}"
            )
        self.feature_names = list(X.columns)
        values = X.to_numpy(dtype=float)

        results: dict[str, Any] = {"n_samples": len(y)}
        best_name, best_mae = "", float("inf")
        for name, model in self._candidates().items():
            preds = cross_val_predict(model, values, y, cv=LeaveOneOut())
            mae = float(mean_absolute_error(y, preds))
            r2 = float(r2_score(y, preds))
            results[name] = {"mae": mae, "r2": r2}
            print(f"[microhard] {name:>6}: LOO MAE {mae:.1f} HV   R^2 {r2:.3f}   (n={len(y)})")
            if mae < best_mae:
                best_name, best_mae = name, mae
        results["best"] = best_name

        self.model = self._candidates()[best_name].fit(values, y)
        self.metrics = results
        return results

    def predict(self, fv: FeatureVector) -> float:
        if self.model is None:
            raise RuntimeError("HardnessHead is not fitted")
        x = np.array([[fv.get(name) for name in self.feature_names]])
        return float(self.model.predict(x)[0])


register("ferrous/uhcs", "hardness_hv", HardnessHead)
