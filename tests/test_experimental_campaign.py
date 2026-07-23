from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from microhard.experimental_campaign import (
    IMAGE_COLUMNS,
    MECHANICAL_COLUMNS,
    PROCESS_COLUMNS,
    SPECIMEN_COLUMNS,
    ExperimentalSteelAdapter,
    load_campaign,
)
from microhard.records import MEASURED
from microhard.taxonomy import Taxonomy
from tests.conftest import write_image


def _write_campaign(root: Path) -> None:
    root.mkdir()
    write_image(root / "sem.png")
    image_hash = hashlib.sha256((root / "sem.png").read_bytes()).hexdigest()
    (root / "raw.csv").write_text("load,depth\n1,2\n")

    pd.DataFrame(
        [
            {
                "specimen_id": "S-001",
                "group_id": "campaign-S-001",
                "alloy_grade": "grade/ferrous/uhcs_ac1",
                "condition": "condition/austenitize/water_quench/t800c_90m",
                "heat_id": "H-01",
                "batch_id": "B-01",
                "parent_material_id": "BAR-01",
                "composition_wt_pct_json": '{"C": 2.0, "Cr": 4.0}',
                "composition_source": "certificate H-01",
                "process_route_id": "P-001",
                "sampling_location": "mid_radius",
                "orientation": "transverse",
                "notes": "",
            }
        ],
        columns=SPECIMEN_COLUMNS,
    ).to_csv(root / "specimens.csv", index=False)

    pd.DataFrame(
        [
            {
                "process_route_id": "P-001",
                "step_index": "1",
                "operation": "austenitize",
                "start_temperature_c": "20",
                "end_temperature_c": "800",
                "duration_min": "90",
                "cooling_rate_c_per_s": "",
                "atmosphere": "argon",
                "strain": "",
                "strain_rate_s_inv": "",
                "equipment_id": "FURNACE-01",
                "measured_profile_path": "",
                "notes": "",
            }
        ],
        columns=PROCESS_COLUMNS,
    ).to_csv(root / "process_steps.csv", index=False)

    pd.DataFrame(
        [
            {
                "image_id": "IMG-001",
                "specimen_id": "S-001",
                "image_path": "sem.png",
                "field_id": "F-01",
                "modality": "SEM",
                "scale_um_per_px": "0.1",
                "magnification": "1964",
                "detector": "SE",
                "accelerating_voltage_kv": "15",
                "working_distance_mm": "10",
                "preparation": "polished",
                "etchant": "2% nital",
                "sampling_location": "mid_radius",
                "orientation": "transverse",
                "acquisition_date": "2026-07-23",
                "taxonomy_labels": "ferrous/martensite",
                "sha256": image_hash,
                "notes": "",
            }
        ],
        columns=IMAGE_COLUMNS,
    ).to_csv(root / "images.csv", index=False)

    tests = []
    for replicate, value in ((1, 300), (2, 320)):
        tests.append(
            {
                "measurement_id": f"HV-{replicate}",
                "specimen_id": "S-001",
                "test_coupon_id": f"C-{replicate}",
                "property_name": "hardness_hv",
                "value": str(value),
                "unit": "HV",
                "test_method": "ASTM E384",
                "test_parameters_json": '{"load_kgf": 1, "dwell_s": 10}',
                "test_temperature_c": "20",
                "replicate_index": str(replicate),
                "sampling_location": "mid_radius",
                "orientation": "transverse",
                "uncertainty": "",
                "scatter_kind": "unreported",
                "raw_data_path": "raw.csv",
                "test_date": "2026-07-23",
                "notes": "",
            }
        )
    pd.DataFrame(tests, columns=MECHANICAL_COLUMNS).to_csv(
        root / "mechanical_tests.csv", index=False
    )


def test_campaign_validates_and_summarizes(tmp_path: Path) -> None:
    root = tmp_path / "experimental_campaign"
    _write_campaign(root)
    tables = load_campaign(root)
    assert tables.summary() == {
        "specimens": 1,
        "process_routes": 1,
        "images": 1,
        "imaged_specimens": 1,
        "mechanical_measurements": 2,
        "mechanically_tested_specimens": 1,
        "complete_specimens": 1,
    }


def test_experimental_adapter_attaches_direct_replicate_mean(cfg) -> None:
    _write_campaign(cfg.experimental_campaign_dir)
    records = ExperimentalSteelAdapter(cfg, Taxonomy.load(None)).validated_records()
    assert len(records) == 1
    assert records[0].group_id == "campaign-S-001"
    assert records[0].properties == {"hardness_hv": 310.0}
    assert records[0].property_sources == {"hardness_hv": MEASURED}
    assert records[0].scale_um_per_px == pytest.approx(0.1)
    assert records[0].taxonomy_labels == ("ferrous/martensite",)


def test_campaign_rejects_changed_image(tmp_path: Path) -> None:
    root = tmp_path / "experimental_campaign"
    _write_campaign(root)
    write_image(root / "sem.png", seed=99)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_campaign(root)


def test_campaign_rejects_unknown_specimen_foreign_key(tmp_path: Path) -> None:
    root = tmp_path / "experimental_campaign"
    _write_campaign(root)
    images = pd.read_csv(root / "images.csv", dtype=str)
    images.loc[0, "specimen_id"] = "MISSING"
    images.to_csv(root / "images.csv", index=False)
    with pytest.raises(ValueError, match="absent from specimens.csv"):
        load_campaign(root)


def test_empty_tracked_templates_are_valid() -> None:
    root = Path(__file__).parents[1] / "data" / "experimental_campaign"
    assert load_campaign(root).summary()["specimens"] == 0
