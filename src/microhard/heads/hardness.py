"""Hardness head for ferrous/UHCS: constituent features to Vickers HV.

Gradient boosting plus a scaled-Ridge linear baseline, evaluated with
leave-one-sample-out CV, which is all the label count supports. The model
with the lower LOO MAE is refit on all data and kept.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
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

    @staticmethod
    def _fit_model(
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
    ) -> Any:
        if sample_weight is None:
            return model.fit(X, y)
        if isinstance(model, Pipeline):
            final_step = model.steps[-1][0]
            return model.fit(X, y, **{f"{final_step}__sample_weight": sample_weight})
        return model.fit(X, y, sample_weight=sample_weight)

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if len(X) < MIN_SAMPLES_FOR_CV:
            raise ValueError(
                f"need >= {MIN_SAMPLES_FOR_CV} labeled samples for leave-one-out CV, got {len(X)}"
            )
        self.feature_names = list(X.columns)
        values = X.to_numpy(dtype=float)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.shape != y.shape:
                raise ValueError("sample_weight must have the same shape as y")
            if not np.isfinite(sample_weight).all() or (sample_weight <= 0).any():
                raise ValueError("sample_weight values must be finite and positive")

        results: dict[str, Any] = {
            "n_samples": len(y),
            "effective_sample_weight": float(
                sample_weight.sum() if sample_weight is not None else len(y)
            ),
        }
        best_name, best_mae = "", float("inf")
        for name, model in self._candidates().items():
            preds = np.empty_like(y, dtype=float)
            for train, test in LeaveOneOut().split(values):
                fitted = self._fit_model(
                    clone(model),
                    values[train],
                    y[train],
                    sample_weight[train] if sample_weight is not None else None,
                )
                preds[test] = fitted.predict(values[test])
            mae = float(mean_absolute_error(y, preds))
            r2 = float(r2_score(y, preds))
            results[name] = {"mae": mae, "r2": r2}
            print(f"[microhard] {name:>6}: LOO MAE {mae:.1f} HV   R^2 {r2:.3f}   (n={len(y)})")
            if mae < best_mae:
                best_name, best_mae = name, mae
        results["best"] = best_name

        self.model = self._fit_model(
            self._candidates()[best_name], values, y, sample_weight
        )
        self.metrics = results
        return results

    def predict(self, fv: FeatureVector) -> float:
        if self.model is None:
            raise RuntimeError("HardnessHead is not fitted")
        x = np.array([[fv.get(name) for name in self.feature_names]])
        return float(self.model.predict(x)[0])


register("ferrous/uhcs", "hardness_hv", HardnessHead)
