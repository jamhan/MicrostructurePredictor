"""PropertyHead ABC: FeatureVector in, property value out. Never raw images."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..features import FeatureVector


class PropertyHead(ABC):
    """One regressor for one (scope, property) pair.

    ``scope`` is a taxonomy family id, optionally refined by an adapter name
    ("ferrous" or "ferrous/uhcs"). Heads consume FeatureVector instances and
    nothing else; in particular they never see images.
    """

    scope: ClassVar[str]
    property_name: ClassVar[str]

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Fit on sample-level features; returns a metrics dict."""

    @abstractmethod
    def predict(self, fv: FeatureVector) -> float:
        """Property estimate for one feature vector."""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "PropertyHead":
        with open(path, "rb") as f:
            head = pickle.load(f)
        if not isinstance(head, PropertyHead):
            raise TypeError(f"{path} does not contain a PropertyHead")
        return head
