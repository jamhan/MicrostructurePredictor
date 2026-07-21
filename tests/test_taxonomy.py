from pathlib import Path

import pytest

from microhard.taxonomy import (
    CONDITION_AXIS,
    FAMILY_LEVEL,
    GRADE_AXIS,
    MICROCONSTITUENT_AXIS,
    Taxonomy,
    UnknownNodeError,
)


def test_bundled_seed_loads() -> None:
    tax = Taxonomy.load(None)
    families = {n.id for n in tax.families()}
    assert "ferrous" in families
    assert "aluminum" in families
    node = tax.node("ferrous/pearlite/lamellar")
    assert node.level == 3
    assert node.parent == "ferrous/pearlite"
    assert tax.node("ferrous").level == FAMILY_LEVEL


def test_family_of_and_children() -> None:
    tax = Taxonomy.load(None)
    assert tax.family_of("ferrous/spheroidite/particles") == "ferrous"
    child_ids = {n.id for n in tax.children("ferrous")}
    assert "ferrous/martensite" in child_ids


def test_unknown_node_raises() -> None:
    tax = Taxonomy.load(None)
    with pytest.raises(UnknownNodeError, match="unobtainium"):
        tax.node("unobtainium")
    with pytest.raises(UnknownNodeError):
        tax.require(["ferrous", "ferrous/mystery"])
    assert "ferrous" in tax
    assert "ferrous/mystery" not in tax


def test_custom_yaml_file(tmp_path: Path) -> None:
    path = tmp_path / "tax.yaml"
    path.write_text(
        "polymer:\n  name: Polymers\n  children:\n    crystalline:\n      name: Crystalline\n"
    )
    tax = Taxonomy.load(path)
    assert tax.node("polymer/crystalline").name == "Crystalline"
    assert [n.id for n in tax.families()] == ["polymer"]


def test_toml_and_json_formats(tmp_path: Path) -> None:
    toml = tmp_path / "tax.toml"
    toml.write_text('[ceramic]\nname = "Ceramics"\n')
    assert Taxonomy.load(toml).node("ceramic").name == "Ceramics"

    json_file = tmp_path / "tax.json"
    json_file.write_text('{"ceramic": {"name": "Ceramics"}}')
    assert Taxonomy.load(json_file).node("ceramic").name == "Ceramics"


def test_grade_and_condition_axes_are_registered() -> None:
    tax = Taxonomy.load(None)
    assert tax.axis_of("grade/ferrous/aisi_1045") == GRADE_AXIS
    assert tax.axis_of("condition/austenitize/water_quench") == CONDITION_AXIS
    assert tax.axis_of("ferrous/pearlite") == MICROCONSTITUENT_AXIS  # unchanged default
    assert tax.node("condition/austenitize/water_quench").parent == "condition/austenitize"
    grade_ids = {n.id for n in tax.grades()}
    assert grade_ids >= {"grade", "grade/ferrous", "grade/ferrous/aisi_1045"}
    assert all(n.id.startswith("condition") for n in tax.conditions())
    exact = tax.node("condition/austenitize/water_quench/t970c_24h")
    assert exact.axis == CONDITION_AXIS
    assert exact.level == 4
    assert tax.node("grade/ferrous/guan_2026_low_alloy_hs").axis == GRADE_AXIS
    assert tax.node("grade/ferrous/gb_35crmo").axis == GRADE_AXIS
    direct_quench = tax.node("condition/hot_roll_direct_quench/t900c_to_lt150c")
    assert direct_quench.axis == CONDITION_AXIS
    oil_quench = tax.node(
        "condition/homogenize_austenitize/oil_quench/t1050c_20h_t860c_2h"
    )
    assert oil_quench.axis == CONDITION_AXIS


def test_axes_do_not_leak_into_families() -> None:
    """The router classifies over material families only, as it always did."""
    tax = Taxonomy.load(None)
    assert {n.id for n in tax.families()} == {"ferrous", "aluminum"}


def test_require_can_enforce_an_axis() -> None:
    tax = Taxonomy.load(None)
    tax.require(["grade/ferrous/aisi_1045"], axis=GRADE_AXIS)
    with pytest.raises(UnknownNodeError, match="expected 'microconstituent'"):
        tax.require(["grade/ferrous/aisi_1045"], axis=MICROCONSTITUENT_AXIS)
    with pytest.raises(UnknownNodeError, match=f"expected {GRADE_AXIS!r}"):
        tax.require(["ferrous/pearlite"], axis=GRADE_AXIS)


def test_family_of_rejects_other_axes() -> None:
    tax = Taxonomy.load(None)
    with pytest.raises(ValueError, match="family_of"):
        tax.family_of("condition/as_cast")


def test_axis_declared_on_a_child_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tax.yaml"
    path.write_text("grade:\n  axis: alloy_grade\n  children:\n    x:\n      axis: condition\n")
    with pytest.raises(ValueError, match="only root nodes"):
        Taxonomy.load(path)


def test_unknown_axis_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tax.yaml"
    path.write_text("thing:\n  axis: vibes\n")
    with pytest.raises(ValueError, match="unknown axis"):
        Taxonomy.load(path)


def test_condition_axis_allows_a_fourth_level(tmp_path: Path) -> None:
    """Routes can be split by hold temperature when the property demands it."""
    path = tmp_path / "tax.yaml"
    path.write_text(
        "condition:\n  axis: condition\n  children:\n    austenitize:\n"
        "      children:\n        water_quench:\n          children:\n            t970c: {}\n"
    )
    tax = Taxonomy.load(path)
    assert tax.node("condition/austenitize/water_quench/t970c").level == 4


def test_too_deep_nesting_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tax.yaml"
    path.write_text(
        "a:\n  children:\n    b:\n      children:\n        c:\n          children:\n            d: {}\n"
    )
    with pytest.raises(ValueError, match="too deep"):
        Taxonomy.load(path)
