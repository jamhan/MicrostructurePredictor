import numpy as np
import pandas as pd
import pytest

from microhard import heads
from microhard.features import FeatureVector
from microhard.heads.base import PropertyHead
from microhard.heads.hardness import HardnessHead


def test_registry_has_hardness_head() -> None:
    assert ("ferrous/uhcs", "hardness_hv") in heads.registered()
    assert heads.head_class("ferrous/uhcs", "hardness_hv") is HardnessHead


def test_scope_fallback_to_family() -> None:
    # "ferrous/some_new_adapter" should fall back to family-level lookup;
    # nothing is registered at bare "ferrous", so the uhcs-scoped head does
    # NOT leak to other adapters.
    assert heads.head_class("ferrous/uhcs", "hardness_hv") is not None
    assert heads.head_class("ferrous/other", "hardness_hv") is None
    assert heads.head_class("aluminum", "hardness_hv") is None
    assert heads.heads_for_family("ferrous") == [("ferrous/uhcs", "hardness_hv")]
    assert heads.heads_for_family("aluminum") == []


def test_register_validates_type() -> None:
    with pytest.raises(TypeError):
        heads.register("ferrous", "density", dict)


def _linear_frame(n: int = 8) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    frac = rng.uniform(0.1, 0.9, n)
    X = pd.DataFrame({"frac:ferrous/network": frac, "n_regions:ferrous/network": 5.0 + frac})
    y = 100 + 400 * frac
    return X, y


def test_hardness_head_fit_and_predict() -> None:
    X, y = _linear_frame()
    head = HardnessHead(seed=0)
    metrics = head.fit(X, y)
    assert metrics["n_samples"] == 8
    assert metrics["linear"]["r2"] > 0.9
    assert metrics["best"] in {"linear", "gbrt"}

    fv = FeatureVector("ferrous", {"frac:ferrous/network": 0.5, "n_regions:ferrous/network": 5.5})
    assert head.predict(fv) == pytest.approx(300.0, abs=15)


def test_hardness_head_accepts_confidence_weights() -> None:
    X, y = _linear_frame()
    weights = np.array([1.0, 1.0, 0.85, 0.85, 0.55, 0.55, 0.25, 0.25])
    metrics = HardnessHead(seed=0).fit(X, y, sample_weight=weights)
    assert metrics["effective_sample_weight"] == pytest.approx(weights.sum())


def test_hardness_head_needs_min_samples() -> None:
    X, y = _linear_frame(2)
    with pytest.raises(ValueError, match="leave-one-out"):
        HardnessHead().fit(X, y)


def test_save_load_roundtrip(tmp_path) -> None:
    X, y = _linear_frame()
    head = HardnessHead(seed=0)
    head.fit(X, y)
    path = tmp_path / "h.pkl"
    head.save(path)
    loaded = PropertyHead.load(path)
    fv = FeatureVector("ferrous", {"frac:ferrous/network": 0.4, "n_regions:ferrous/network": 5.4})
    assert loaded.predict(fv) == pytest.approx(head.predict(fv))


def test_load_fitted_none_when_absent(cfg) -> None:
    assert heads.load_fitted(cfg, "ferrous/uhcs", "hardness_hv") is None
