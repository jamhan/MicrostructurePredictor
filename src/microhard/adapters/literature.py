"""Cited SEM panels matched to properties reported in the same study.

The manifest is deliberately panel-level. It records where each image came
from, the exact table cell used as its weak label, and why the image and value
are considered a match. Several panels may share one ``specimen_id``; that id
becomes the split group so two magnifications of one material state cannot
leak across train and test.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pandas as pd

from ..records import DISTANT, CanonicalRecord
from . import register_adapter
from .base import BaseAdapter

LITERATURE_COLUMNS = [
    "record_id",
    "image_path",
    "source_id",
    "source_citation",
    "source_doi",
    "source_url",
    "source_pdf_url",
    "license",
    "license_url",
    "source_page",
    "source_figure",
    "source_panel",
    "property_table",
    "property_table_page",
    "source_value_locator",
    "specimen_id",
    "match_relation",
    "match_confidence",
    "exact_physical_specimen_confirmed",
    "modality",
    "source_annotations_present",
    "scale_bar_um",
    "alloy_grade",
    "condition",
    "sampling_location",
    "property_name",
    "property_value",
    "property_unit",
    "hardness_load_kgf",
    "hardness_dwell_s",
    "property_scatter",
    "scatter_kind",
    "n_measurements",
    "property_source",
    "image_sha256",
    "note",
]

MATCH_CONFIDENCE = ("low", "medium", "high")
MATCH_RELATIONS = (
    "same_study_batch_condition",
    "same_study_plate_condition_location",
)
SCATTER_KINDS = (
    "sd",
    "half_range",
    "tolerance_band",
    "reported_plus_minus_unspecified",
    "unreported",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def load_literature_manifest(csv_path: Path) -> pd.DataFrame:
    """Load and validate the panel-level manifest.

    A missing file means this optional adapter has no local records. Once a
    manifest exists, it is strict: every source locator is required, tracked
    image hashes must match, and all panels sharing a specimen id must agree
    on their property match.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return pd.DataFrame(columns=LITERATURE_COLUMNS)
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = set(LITERATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns {sorted(missing)}; "
            f"expected {LITERATURE_COLUMNS}"
        )
    frame = frame[LITERATURE_COLUMNS].copy()
    _validate_manifest(frame, csv_path)
    return frame


def _validate_manifest(frame: pd.DataFrame, csv_path: Path) -> None:
    if frame["record_id"].duplicated().any():
        duplicate = frame.loc[frame["record_id"].duplicated(), "record_id"].iloc[0]
        raise ValueError(f"{csv_path}: duplicate record_id {duplicate!r}")
    if frame["image_path"].duplicated().any():
        duplicate = frame.loc[frame["image_path"].duplicated(), "image_path"].iloc[0]
        raise ValueError(f"{csv_path}: duplicate image_path {duplicate!r}")

    root = csv_path.parent.resolve()
    specimen_matches: dict[tuple[str, str], tuple[str, ...]] = {}
    for index, row in frame.iterrows():
        where = f"{csv_path} row {index + 2}"
        _require_text(row, where)
        _require_choice(row, "match_relation", MATCH_RELATIONS, where)
        _require_choice(row, "match_confidence", MATCH_CONFIDENCE, where)
        _require_choice(row, "scatter_kind", SCATTER_KINDS, where)
        _require_choice(row, "property_source", (DISTANT,), where)
        _require_choice(row, "property_name", ("hardness_hv",), where)
        _require_choice(row, "property_unit", ("HV",), where)
        _require_choice(row, "modality", ("SEM",), where)
        exact = _parse_bool(row["exact_physical_specimen_confirmed"], where)
        _parse_bool(row["source_annotations_present"], where)
        if exact:
            raise ValueError(
                f"{where}: literature records must not claim an exact physical "
                "specimen unless that identity is demonstrated in the source"
            )

        for column in (
            "source_page",
            "property_table_page",
            "scale_bar_um",
            "property_value",
            "hardness_load_kgf",
            "hardness_dwell_s",
        ):
            if _positive_number(row[column], where, column) is None:
                raise ValueError(f"{where}: {column} is required")
        measurements = _positive_number(
            row["n_measurements"], where, "n_measurements"
        )
        if measurements is not None and not measurements.is_integer():
            raise ValueError(f"{where}: n_measurements must be an integer")

        scatter = _positive_number(
            row["property_scatter"], where, "property_scatter", allow_zero=True
        )
        unreported = row["scatter_kind"] == "unreported"
        if (scatter is None) != unreported:
            raise ValueError(
                f"{where}: scatter_kind must be 'unreported' exactly when "
                "property_scatter is blank"
            )

        image_path = (csv_path.parent / row["image_path"]).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError(f"{where}: image_path escapes the literature data directory")
        if not image_path.is_file():
            raise FileNotFoundError(f"{where}: image file not found: {image_path}")
        expected_hash = row["image_sha256"].lower()
        if not _SHA256.fullmatch(expected_hash):
            raise ValueError(f"{where}: image_sha256 must be 64 lowercase hex characters")
        actual_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(
                f"{where}: SHA-256 mismatch for {image_path.name}; "
                f"manifest has {expected_hash}, file is {actual_hash}"
            )

        key = (row["source_id"], row["specimen_id"])
        signature = tuple(
            row[column]
            for column in (
                "alloy_grade",
                "condition",
                "sampling_location",
                "property_name",
                "property_value",
                "property_unit",
                "hardness_load_kgf",
                "hardness_dwell_s",
                "property_scatter",
                "scatter_kind",
                "n_measurements",
                "property_source",
                "match_relation",
            )
        )
        if key in specimen_matches and specimen_matches[key] != signature:
            raise ValueError(
                f"{where}: panels sharing source_id/specimen_id {key!r} "
                "disagree on their property match"
            )
        specimen_matches[key] = signature


def _require_text(row: pd.Series, where: str) -> None:
    optional = {"property_scatter", "n_measurements"}
    for column in LITERATURE_COLUMNS:
        if column not in optional and not str(row[column]).strip():
            raise ValueError(f"{where}: {column} is required")


def _require_choice(
    row: pd.Series, column: str, choices: tuple[str, ...], where: str
) -> None:
    if row[column] not in choices:
        raise ValueError(f"{where}: {column} {row[column]!r} not in {list(choices)}")


def _parse_bool(raw: str, where: str) -> bool:
    if raw not in {"true", "false"}:
        raise ValueError(f"{where}: expected boolean 'true' or 'false', got {raw!r}")
    return raw == "true"


def _positive_number(
    raw: str, where: str, column: str, *, allow_zero: bool = False
) -> float | None:
    if not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{where}: {column} {raw!r} is not numeric") from None
    threshold_ok = value >= 0 if allow_zero else value > 0
    if not math.isfinite(value) or not threshold_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{where}: {column} must be finite and {qualifier}")
    return value


@register_adapter
class LiteratureSteelAdapter(BaseAdapter):
    """Canonical records for redistributed, cited steel SEM panels."""

    name = "literature_steel"
    family = "ferrous"

    def records(self) -> list[CanonicalRecord]:
        manifest = load_literature_manifest(self.cfg.literature_manifest_csv)
        root = self.cfg.literature_manifest_csv.parent
        return [
            CanonicalRecord(
                record_id=row["record_id"],
                image_path=root / row["image_path"],
                scale_um_per_px=None,
                modality=row["modality"],
                group_id=f"literature-{row['source_id']}-{row['specimen_id']}",
                properties={row["property_name"]: float(row["property_value"])},
                property_sources={row["property_name"]: row["property_source"]},
                alloy_grade=row["alloy_grade"],
                condition=row["condition"],
            )
            for _, row in manifest.iterrows()
        ]
