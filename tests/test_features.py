import numpy as np
import pandas as pd
import pytest

from microhard.adapters.uhcs import SEG_CLASS_NODES
from microhard.features import (
    FeatureVector,
    aggregate_by_group,
    constituent_fractions,
    feature_names,
    image_feature_vector,
    region_stats,
)


def test_constituent_fractions_exact() -> None:
    mask = np.zeros((10, 10), dtype=np.int64)
    mask[:5, :] = 1  # 50% class 1
    mask[9, :] = 3  # 10% class 3
    fractions = constituent_fractions(mask, 4)
    assert fractions.sum() == 1.0
    np.testing.assert_allclose(fractions, [0.4, 0.5, 0.0, 0.1])


def test_region_stats_counts_blobs() -> None:
    mask = np.zeros((10, 10), dtype=np.int64)
    mask[0:2, 0:2] = 1  # blob of 4 px
    mask[6:9, 6:8] = 1  # blob of 6 px
    stats = region_stats(mask, class_idx=1)
    assert stats["n_regions"] == 2
    assert stats["mean_region_area"] == 5.0
    assert stats["max_region_area"] == 6.0
    assert region_stats(mask, class_idx=2)["n_regions"] == 0


def test_image_feature_vector_keys_by_taxonomy_id() -> None:
    mask = np.zeros((8, 8), dtype=np.int64)
    fv = image_feature_vector(mask, SEG_CLASS_NODES, family="ferrous")
    assert fv.family == "ferrous"
    assert fv.get("frac:ferrous/matrix") == 1.0
    assert fv.get("frac:ferrous/network") == 0.0
    assert "n_regions:ferrous/spheroidite" in fv.values
    # no bare strings anywhere in the namespace
    assert all(":" in key for key in fv.values)


def test_feature_vector_merge() -> None:
    fv = FeatureVector("ferrous", {"frac:ferrous/matrix": 1.0})
    merged = fv.merged_with({"topo_h0_count:ferrous/matrix": 3.0})
    assert merged.get("topo_h0_count:ferrous/matrix") == 3.0
    assert fv.get("topo_h0_count:ferrous/matrix") == 0.0  # original untouched


def test_aggregate_by_group_means_and_metadata() -> None:
    df = pd.DataFrame(
        {
            "record_id": ["r1", "r2", "r3"],
            "group_id": ["A", "A", "B"],
            "family": ["ferrous", "ferrous", "ferrous"],
            "adapter": ["uhcs", "uhcs", "uhcs"],
            "frac:ferrous/matrix": [0.2, 0.4, 0.9],
        }
    )
    agg = aggregate_by_group(df).set_index("group_id")
    assert agg.loc["A", "frac:ferrous/matrix"] == pytest.approx(0.3)
    assert agg.loc["B", "frac:ferrous/matrix"] == pytest.approx(0.9)
    assert agg.loc["A", "family"] == "ferrous"
    assert "record_id" not in agg.columns
    assert feature_names(agg.reset_index()) == ["frac:ferrous/matrix"]
