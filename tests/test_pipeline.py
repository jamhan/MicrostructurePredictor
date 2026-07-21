"""End-to-end proof over synthetic data: the pipeline runs on a steel family,
runs on a non-steel family, and abstains instead of fabricating properties."""

import sqlite3

import pytest

from microhard.pipeline import _property_by_group, fit_property_head, predict_image
from microhard.properties import PROPERTY_LOOKUP_COLUMNS
from microhard.segment import train_segmentation
from microhard.taxonomy import Taxonomy


@pytest.fixture()
def two_family_cfg(seg_benchmark, aluminum_stub):
    cfg = seg_benchmark  # seg_benchmark and aluminum_stub share the same Config
    cfg.adapters = ["uhcs", "micronet_al"]
    return cfg


def test_end_to_end_steel_and_aluminum(two_family_cfg):
    cfg = two_family_cfg

    # --- train segmenter on the 4 masked UHCS records ------------------------
    assert train_segmentation(cfg).exists()

    # --- features for every on-disk ferrous record ---------------------------
    from microhard.features import extract_features

    features = extract_features(cfg)
    assert len(features) == 15  # aluminum records skipped (wrong family)
    assert set(features["family"]) == {"ferrous"}
    assert any(c.startswith("frac:ferrous/") for c in features.columns)

    # --- hardness: empty labels -> clean skip --------------------------------
    assert fit_property_head(cfg, "ferrous/uhcs", "hardness_hv") is None

    # --- hardness: labeled samples -> fitted head ----------------------------
    cfg.hardness_csv.write_text(
        "sample_label,hardness_hv,source_note\n"
        + "\n".join(f"S{i},{200 + 30 * i},synthetic" for i in range(1, 6))
        + "\n"
    )
    metrics = fit_property_head(cfg, "ferrous/uhcs", "hardness_hv")
    assert metrics is not None
    assert metrics["n_samples"] == 5

    # --- steel image: fractions + property estimate --------------------------
    steel_image = cfg.micrographs_dir / "micrograph1.png"
    result = predict_image(cfg, steel_image, family="ferrous")
    assert result.family == "ferrous"
    assert result.fractions
    assert sum(result.fractions.values()) == pytest.approx(1.0)
    assert all(node.startswith("ferrous/") for node in result.fractions)
    assert "hardness_hv" in result.properties

    # --- non-steel image: runs end-to-end and abstains on properties ---------
    aluminum_image = cfg.data_dir / "micronet_al" / "al0.png"
    result = predict_image(cfg, aluminum_image, family="aluminum")
    assert result.family == "aluminum"
    assert result.properties == {}  # nothing fabricated
    assert "features" in result.abstentions  # segmenter is ferrous-only
    assert "ferrous" in result.abstentions["features"]


def test_predict_without_router_abstains_on_family(two_family_cfg):
    result = predict_image(two_family_cfg, two_family_cfg.micrographs_dir / "micrograph1.png")
    assert result.family is None
    assert "train-router" in result.abstentions["family"]


def test_property_lookup_is_joined_into_training_values(synthetic_db) -> None:
    with sqlite3.connect(synthetic_db.sqlite_path) as con:
        con.execute(
            "UPDATE sample SET label = 'AC1 800C 90M WQ', cool_method = 'WQ', "
            "anneal_temperature = 800, anneal_time = 90, anneal_time_unit = 'M' "
            "WHERE sample_id = 1"
        )
    row = (
        "grade/ferrous/uhcs_ac1,condition/austenitize/water_quench/t800c_90m,"
        "hardness_hv,500,HV,,unreported,,medium,Test source,https://example.invalid/,fixture"
    )
    synthetic_db.property_lookup_csv.write_text(
        ",".join(PROPERTY_LOOKUP_COLUMNS) + "\n" + row + "\n"
    )
    values = _property_by_group(synthetic_db, Taxonomy.load(None), "hardness_hv")
    assert values["uhcs-sample-1"] == 500.0


def test_predict_rejects_unknown_family_override(two_family_cfg):
    from microhard.taxonomy import UnknownNodeError

    with pytest.raises(UnknownNodeError):
        predict_image(
            two_family_cfg, two_family_cfg.micrographs_dir / "micrograph1.png", family="vibranium"
        )


def test_routed_predict_is_graceful(two_family_cfg):
    """With a trained router, prediction either routes or abstains — never crashes."""
    from microhard.router import train_router

    cfg = two_family_cfg
    cfg.router_calib_frac = 0.4
    train_router(cfg)
    result = predict_image(cfg, cfg.micrographs_dir / "micrograph2.png")
    assert result.family in {None, "ferrous", "aluminum"}
    if result.family is None:
        assert "family" in result.abstentions
    assert result.family_probabilities  # router ran and reported probabilities
