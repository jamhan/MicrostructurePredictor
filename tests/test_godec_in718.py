from pathlib import Path

import pytest

from microhard.adapters.godec_in718 import (
    GodecIN718Adapter,
    audit_godec_links,
    load_godec_observations,
    parse_godec_image,
)
from microhard.records import DISTANT
from microhard.taxonomy import Taxonomy
from tests.conftest import write_image


@pytest.fixture()
def godec_data(cfg):
    root = cfg.public_in718_dir
    (root / "raw").mkdir(parents=True)
    write_image(root / "raw" / "BEI HT 954 Gauss 1.tif")
    write_image(root / "raw" / "BEI HT 954 Gauss 2.tif")
    (root / "hardness.csv").write_text(
        "observation_id,build_strategy,state,temperature_c,hold_minutes,"
        "hardness_hv,hardness_sd_hv,n_measurements,source_locator\n"
        "h1,Gauss,heat-treated,954,60,487.1,15.3,3,workbook hardness row\n"
    )
    (root / "tensile.csv").write_text(
        "observation_id,build_strategy,state,temperature_c,hold_minutes,"
        "orientation,yield_strength_mpa,ultimate_tensile_strength_mpa,"
        "elongation_pct,reduction_area_pct,youngs_modulus_gpa,source_locator\n"
        "t-h,Gauss,heat-treated,954,60,H,1269,1556,11.7,18.2,176.5,summary H\n"
        "t-v,Gauss,heat-treated,954,60,V,1169,1408,17.5,31.5,185.2,summary V\n"
    )
    return cfg


def test_filename_parser_extracts_material_state() -> None:
    state = parse_godec_image(Path("BEI HT 1034 Ring 2.tif"))
    assert state.temperature_c == 1034
    assert state.hold_minutes == 60
    assert state.build_strategy == "Ring"
    assert state.state == "heat-treated"


def test_observation_loader_expands_tensile_columns(godec_data) -> None:
    observations = load_godec_observations(godec_data.public_in718_dir)
    assert len(observations) == 11  # 1 hardness + 2 orientations x 5 properties
    assert {item.property_name for item in observations} >= {
        "hardness_hv",
        "yield_strength_mpa",
        "ultimate_tensile_strength_mpa",
    }


def test_adapter_attaches_only_unambiguous_hardness(godec_data) -> None:
    records = GodecIN718Adapter(godec_data, Taxonomy.load(None)).validated_records()
    assert len(records) == 2
    assert {record.group_id for record in records} == {"godec-in718-ht-954c-gauss"}
    assert all(record.properties == {"hardness_hv": 487.1} for record in records)
    assert all(record.property_sources == {"hardness_hv": DISTANT} for record in records)
    assert all(record.property_weights == {"hardness_hv": 0.85} for record in records)
    assert all(record.alloy_grade == "grade/nickel/in718" for record in records)
    assert all(record.condition == "condition/heat_treat/t954c_1h" for record in records)


def test_audit_preserves_orientation_candidates(godec_data) -> None:
    audit = audit_godec_links(godec_data.public_in718_dir)
    hardness = audit[audit["property_name"] == "hardness_hv"]
    tensile = audit[audit["property_name"] != "hardness_hv"]
    assert len(hardness) == 2
    assert hardness["auto_attach"].all()
    assert not tensile["auto_attach"].any()
    assert set(tensile["property_orientation"]) == {"H", "V"}
    assert not audit["validation_eligible"].any()
