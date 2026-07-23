"""Strict tables for directly measured process-structure-property campaigns.

The literature adapter is deliberately permissive about physical linkage
because papers rarely identify the exact imaged coupon. This module represents
the opposite case: a new experiment controlled by us, where every SEM image,
process route, and mechanical test is linked through a specimen id.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .records import MEASURED, CanonicalRecord
from .adapters import register_adapter
from .adapters.base import BaseAdapter

SPECIMEN_COLUMNS = [
    "specimen_id",
    "group_id",
    "alloy_grade",
    "condition",
    "heat_id",
    "batch_id",
    "parent_material_id",
    "composition_wt_pct_json",
    "composition_source",
    "process_route_id",
    "sampling_location",
    "orientation",
    "notes",
]

PROCESS_COLUMNS = [
    "process_route_id",
    "step_index",
    "operation",
    "start_temperature_c",
    "end_temperature_c",
    "duration_min",
    "cooling_rate_c_per_s",
    "atmosphere",
    "strain",
    "strain_rate_s_inv",
    "equipment_id",
    "measured_profile_path",
    "notes",
]

IMAGE_COLUMNS = [
    "image_id",
    "specimen_id",
    "image_path",
    "field_id",
    "modality",
    "scale_um_per_px",
    "magnification",
    "detector",
    "accelerating_voltage_kv",
    "working_distance_mm",
    "preparation",
    "etchant",
    "sampling_location",
    "orientation",
    "acquisition_date",
    "taxonomy_labels",
    "sha256",
    "notes",
]

MECHANICAL_COLUMNS = [
    "measurement_id",
    "specimen_id",
    "test_coupon_id",
    "property_name",
    "value",
    "unit",
    "test_method",
    "test_parameters_json",
    "test_temperature_c",
    "replicate_index",
    "sampling_location",
    "orientation",
    "uncertainty",
    "scatter_kind",
    "raw_data_path",
    "test_date",
    "notes",
]

CAMPAIGN_FILES = {
    "specimens": ("specimens.csv", SPECIMEN_COLUMNS),
    "process_steps": ("process_steps.csv", PROCESS_COLUMNS),
    "images": ("images.csv", IMAGE_COLUMNS),
    "mechanical_tests": ("mechanical_tests.csv", MECHANICAL_COLUMNS),
}

SCATTER_KINDS = ("sd", "se", "ci95_half_width", "range_half_width", "unreported")
_HASH_LENGTH = 64


@dataclass(frozen=True)
class CampaignTables:
    specimens: pd.DataFrame
    process_steps: pd.DataFrame
    images: pd.DataFrame
    mechanical_tests: pd.DataFrame

    def summary(self) -> dict[str, int]:
        measured = set(self.mechanical_tests["specimen_id"])
        imaged = set(self.images["specimen_id"])
        return {
            "specimens": len(self.specimens),
            "process_routes": self.process_steps["process_route_id"].nunique(),
            "images": len(self.images),
            "imaged_specimens": len(imaged),
            "mechanical_measurements": len(self.mechanical_tests),
            "mechanically_tested_specimens": len(measured),
            "complete_specimens": len(imaged & measured),
        }


def load_campaign(root: Path) -> CampaignTables:
    """Load and validate all four campaign tables under ``root``."""
    root = Path(root)
    frames: dict[str, pd.DataFrame] = {}
    for name, (filename, columns) in CAMPAIGN_FILES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"campaign table not found: {path}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        frames[name] = frame[columns].copy()

    tables = CampaignTables(**frames)
    _validate_campaign(tables, root.resolve())
    return tables


def _validate_campaign(tables: CampaignTables, root: Path) -> None:
    _unique(tables.specimens, "specimen_id", "specimens.csv")
    _unique(tables.specimens, "group_id", "specimens.csv")
    _unique(tables.images, "image_id", "images.csv")
    _unique(tables.mechanical_tests, "measurement_id", "mechanical_tests.csv")

    specimens = set(tables.specimens["specimen_id"])
    routes = set(tables.process_steps["process_route_id"])
    _foreign_key(tables.specimens, "process_route_id", routes, "process_steps.csv")
    _foreign_key(tables.images, "specimen_id", specimens, "specimens.csv")
    _foreign_key(tables.mechanical_tests, "specimen_id", specimens, "specimens.csv")

    for index, row in tables.specimens.iterrows():
        where = f"specimens.csv row {index + 2}"
        _require(
            row,
            (
                "specimen_id",
                "group_id",
                "alloy_grade",
                "condition",
                "heat_id",
                "batch_id",
                "composition_wt_pct_json",
                "composition_source",
                "process_route_id",
                "sampling_location",
                "orientation",
            ),
            where,
        )
        _composition(row["composition_wt_pct_json"], where)

    route_steps: dict[str, list[int]] = {}
    for index, row in tables.process_steps.iterrows():
        where = f"process_steps.csv row {index + 2}"
        _require(
            row,
            ("process_route_id", "step_index", "operation", "atmosphere", "equipment_id"),
            where,
        )
        _safe_file(root, row["measured_profile_path"], where, required=False)
        step = _integer(row["step_index"], where, "step_index", minimum=1)
        route_steps.setdefault(row["process_route_id"], []).append(step)
        for column in (
            "start_temperature_c",
            "end_temperature_c",
            "duration_min",
            "cooling_rate_c_per_s",
            "strain",
            "strain_rate_s_inv",
        ):
            _number(row[column], where, column, optional=True)
    for route, steps in route_steps.items():
        expected = list(range(1, len(steps) + 1))
        if sorted(steps) != expected:
            raise ValueError(
                f"process_steps.csv: route {route!r} step_index values must be {expected}"
            )

    seen_image_paths: set[Path] = set()
    for index, row in tables.images.iterrows():
        where = f"images.csv row {index + 2}"
        _require(
            row,
            (
                "image_id",
                "specimen_id",
                "image_path",
                "field_id",
                "modality",
                "scale_um_per_px",
                "magnification",
                "detector",
                "accelerating_voltage_kv",
                "working_distance_mm",
                "preparation",
                "etchant",
                "sampling_location",
                "orientation",
                "acquisition_date",
                "sha256",
            ),
            where,
        )
        for column in (
            "scale_um_per_px",
            "magnification",
            "accelerating_voltage_kv",
            "working_distance_mm",
        ):
            _number(row[column], where, column, positive=True)
        _iso_date(row["acquisition_date"], where, "acquisition_date")
        image_path = _safe_file(root, row["image_path"], where, required=True)
        if image_path in seen_image_paths:
            raise ValueError(f"{where}: duplicate image_path {row['image_path']!r}")
        seen_image_paths.add(image_path)
        expected_hash = row["sha256"].lower()
        if len(expected_hash) != _HASH_LENGTH or any(
            char not in "0123456789abcdef" for char in expected_hash
        ):
            raise ValueError(f"{where}: sha256 must be 64 lowercase hexadecimal characters")
        actual_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"{where}: SHA-256 mismatch for {image_path.name}; "
                f"manifest has {expected_hash}, file is {actual_hash}"
            )

    replicate_keys: set[tuple[str, str, int]] = set()
    units: dict[str, str] = {}
    for index, row in tables.mechanical_tests.iterrows():
        where = f"mechanical_tests.csv row {index + 2}"
        _require(
            row,
            (
                "measurement_id",
                "specimen_id",
                "test_coupon_id",
                "property_name",
                "value",
                "unit",
                "test_method",
                "test_parameters_json",
                "test_temperature_c",
                "replicate_index",
                "sampling_location",
                "orientation",
                "scatter_kind",
                "raw_data_path",
                "test_date",
            ),
            where,
        )
        _number(row["value"], where, "value")
        _json_object(
            row["test_parameters_json"],
            where,
            "test_parameters_json",
            numeric_values=False,
        )
        _number(row["test_temperature_c"], where, "test_temperature_c")
        replicate = _integer(row["replicate_index"], where, "replicate_index", minimum=1)
        key = (row["specimen_id"], row["property_name"], replicate)
        if key in replicate_keys:
            raise ValueError(f"{where}: duplicate specimen/property/replicate {key!r}")
        replicate_keys.add(key)
        if row["scatter_kind"] not in SCATTER_KINDS:
            raise ValueError(
                f"{where}: scatter_kind {row['scatter_kind']!r} not in {list(SCATTER_KINDS)}"
            )
        uncertainty = _number(
            row["uncertainty"], where, "uncertainty", optional=True, positive=True
        )
        if (uncertainty is None) != (row["scatter_kind"] == "unreported"):
            raise ValueError(
                f"{where}: uncertainty must be blank exactly when scatter_kind is 'unreported'"
            )
        _iso_date(row["test_date"], where, "test_date")
        _safe_file(root, row["raw_data_path"], where, required=True)
        previous_unit = units.setdefault(row["property_name"], row["unit"])
        if previous_unit != row["unit"]:
            raise ValueError(
                f"{where}: property {row['property_name']!r} mixes units "
                f"{previous_unit!r} and {row['unit']!r}"
            )


def _unique(frame: pd.DataFrame, column: str, filename: str) -> None:
    nonblank = frame[column].str.strip()
    if (nonblank == "").any():
        row = int((nonblank == "").idxmax()) + 2
        raise ValueError(f"{filename} row {row}: {column} is required")
    duplicate = frame.loc[frame[column].duplicated(), column]
    if not duplicate.empty:
        raise ValueError(f"{filename}: duplicate {column} {duplicate.iloc[0]!r}")


def _foreign_key(
    frame: pd.DataFrame, column: str, allowed: set[str], target: str
) -> None:
    unknown = sorted(set(frame[column]) - allowed)
    if unknown:
        raise ValueError(f"{column} values {unknown} are absent from {target}")


def _require(row: pd.Series, columns: tuple[str, ...], where: str) -> None:
    for column in columns:
        if not str(row[column]).strip():
            raise ValueError(f"{where}: {column} is required")


def _number(
    raw: str,
    where: str,
    column: str,
    *,
    optional: bool = False,
    positive: bool = False,
) -> float | None:
    if not raw.strip():
        if optional:
            return None
        raise ValueError(f"{where}: {column} is required")
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{where}: {column} {raw!r} is not numeric") from None
    if not math.isfinite(value) or (positive and value <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{where}: {column} must be {qualifier}")
    return value


def _integer(raw: str, where: str, column: str, *, minimum: int) -> int:
    value = _number(raw, where, column)
    assert value is not None
    if not value.is_integer() or value < minimum:
        raise ValueError(f"{where}: {column} must be an integer >= {minimum}")
    return int(value)


def _composition(raw: str, where: str) -> None:
    composition = _json_object(
        raw, where, "composition_wt_pct_json", numeric_values=True
    )
    total = 0.0
    for element, raw_value in composition.items():
        value = float(raw_value)
        if value < 0 or value > 100:
            raise ValueError(f"{where}: composition value for {element!r} is outside 0..100")
        total += value
    if total > 100.5:
        raise ValueError(f"{where}: composition weight percentages sum to {total:.3f}")


def _json_object(
    raw: str,
    where: str,
    column: str,
    *,
    numeric_values: bool,
) -> dict:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"{where}: {column} is not valid JSON") from None
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{where}: {column} must be a non-empty object")
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{where}: {column} contains an invalid key")
        if numeric_values and (
            not isinstance(item, (int, float)) or not math.isfinite(float(item))
        ):
            raise ValueError(f"{where}: {column} value for {key!r} is not finite numeric")
    return value


def _iso_date(raw: str, where: str, column: str) -> None:
    try:
        date.fromisoformat(raw)
    except ValueError:
        raise ValueError(f"{where}: {column} must use YYYY-MM-DD") from None


def _safe_file(root: Path, raw: str, where: str, *, required: bool) -> Path | None:
    if not raw.strip():
        if required:
            raise ValueError(f"{where}: file path is required")
        return None
    path = (root / raw).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{where}: path escapes the campaign directory")
    if not path.is_file():
        raise FileNotFoundError(f"{where}: file not found: {path}")
    return path


@register_adapter
class ExperimentalSteelAdapter(BaseAdapter):
    """Direct, specimen-linked steel SEM and mechanical measurements."""

    name = "experimental_steel"
    family = "ferrous"

    def records(self) -> list[CanonicalRecord]:
        root = self.cfg.experimental_campaign_dir
        if not root.exists():
            return []
        tables = load_campaign(root)
        specimens = tables.specimens.set_index("specimen_id")
        properties: dict[str, dict[str, float]] = {}
        for (specimen_id, property_name), rows in tables.mechanical_tests.groupby(
            ["specimen_id", "property_name"]
        ):
            properties.setdefault(specimen_id, {})[property_name] = pd.to_numeric(
                rows["value"]
            ).mean()

        records: list[CanonicalRecord] = []
        for _, image in tables.images.iterrows():
            specimen = specimens.loc[image["specimen_id"]]
            measured = properties.get(image["specimen_id"], {})
            labels = tuple(
                label.strip()
                for label in image["taxonomy_labels"].split("|")
                if label.strip()
            )
            records.append(
                CanonicalRecord(
                    record_id=f"experimental-{image['image_id']}",
                    image_path=root / image["image_path"],
                    scale_um_per_px=float(image["scale_um_per_px"]),
                    modality=image["modality"],
                    group_id=specimen["group_id"],
                    taxonomy_labels=labels or None,
                    properties=dict(measured),
                    property_sources={name: MEASURED for name in measured},
                    alloy_grade=specimen["alloy_grade"],
                    condition=specimen["condition"],
                )
            )
        return records
