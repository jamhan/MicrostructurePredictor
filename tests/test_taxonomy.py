from pathlib import Path

import pytest

from microhard.taxonomy import FAMILY_LEVEL, Taxonomy, UnknownNodeError


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


def test_too_deep_nesting_rejected(tmp_path: Path) -> None:
    path = tmp_path / "tax.yaml"
    path.write_text(
        "a:\n  children:\n    b:\n      children:\n        c:\n          children:\n            d: {}\n"
    )
    with pytest.raises(ValueError, match="too deep"):
        Taxonomy.load(path)
