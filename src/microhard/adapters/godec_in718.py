"""Zenodo 14163786: IN718 BSE-SEM fields linked to mechanical workbooks.

The public archive is compact enough to use directly.  Filenames identify the
beam strategy and material state; the supplied workbooks report hardness and
tensile properties for those states.  Hardness is attached as a high-confidence
same-study/same-condition distant label.  Tensile values remain audit
candidates because the micrograph filenames do not identify H/V orientation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..linkage import (
    MaterialState,
    PropertyObservation,
    best_auto_links,
    candidate_links,
)
from ..records import DISTANT, CanonicalRecord
from . import register_adapter
from .base import BaseAdapter

SOURCE_ID = "zenodo-14163786"
SOURCE_URL = "https://zenodo.org/records/14163786"
PROCESS = "laser powder bed fusion"
ALLOY = "IN718"

_IMAGE_NAME = re.compile(
    r"^BEI (?P<state>AB|HT)(?: (?P<temperature>\d+))? "
    r"(?P<strategy>Gauss|Ring) (?P<field>\d+)\.tif$",
    re.IGNORECASE,
)

_CONDITIONS = {
    ("as-built", None): "condition/as_built",
    ("heat-treated", 954): "condition/heat_treat/t954c_1h",
    ("heat-treated", 984): "condition/heat_treat/t984c_1h",
    ("heat-treated", 1034): "condition/heat_treat/t1034c_1h",
    ("heat-treated", 1154): "condition/heat_treat/t1154c_1h",
}

_HARDNESS_COLUMNS = {
    "observation_id",
    "build_strategy",
    "state",
    "temperature_c",
    "hold_minutes",
    "hardness_hv",
    "hardness_sd_hv",
    "n_measurements",
    "source_locator",
}

_TENSILE_COLUMNS = {
    "observation_id",
    "build_strategy",
    "state",
    "temperature_c",
    "hold_minutes",
    "orientation",
    "yield_strength_mpa",
    "ultimate_tensile_strength_mpa",
    "elongation_pct",
    "reduction_area_pct",
    "youngs_modulus_gpa",
    "source_locator",
}

_TENSILE_PROPERTIES = {
    "yield_strength_mpa": "MPa",
    "ultimate_tensile_strength_mpa": "MPa",
    "elongation_pct": "%",
    "reduction_area_pct": "%",
    "youngs_modulus_gpa": "GPa",
}


def _optional_float(raw: object) -> float | None:
    text = str(raw).strip()
    return float(text) if text else None


def _required_columns(frame: pd.DataFrame, expected: set[str], path: Path) -> None:
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def parse_godec_image(path: Path) -> MaterialState:
    """Parse the processing state encoded by one public archive filename."""

    match = _IMAGE_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unrecognised Godec IN718 image filename: {path.name!r}")
    state = "as-built" if match["state"].upper() == "AB" else "heat-treated"
    temperature = (
        float(match["temperature"]) if match["temperature"] is not None else None
    )
    if state == "heat-treated" and temperature is None:
        raise ValueError(f"{path.name}: heat-treated image has no temperature")
    if state == "as-built" and temperature is not None:
        raise ValueError(f"{path.name}: as-built image unexpectedly has a temperature")
    strategy = match["strategy"].title()
    condition_token = "ab" if temperature is None else f"ht-{int(temperature)}c"
    record_id = (
        f"godec-in718-{condition_token}-{strategy.casefold()}-"
        f"{int(match['field']):02d}"
    )
    return MaterialState(
        record_id=record_id,
        source_id=SOURCE_ID,
        alloy=ALLOY,
        process=PROCESS,
        state=state,
        temperature_c=temperature,
        hold_minutes=60.0 if state == "heat-treated" else None,
        build_strategy=strategy,
    )


def load_godec_observations(root: Path) -> list[PropertyObservation]:
    """Load the audited CSV transcriptions of the supplied workbooks."""

    root = Path(root)
    hardness_path = root / "hardness.csv"
    tensile_path = root / "tensile.csv"
    hardness = pd.read_csv(hardness_path, dtype=str, keep_default_na=False)
    tensile = pd.read_csv(tensile_path, dtype=str, keep_default_na=False)
    _required_columns(hardness, _HARDNESS_COLUMNS, hardness_path)
    _required_columns(tensile, _TENSILE_COLUMNS, tensile_path)

    observations: list[PropertyObservation] = []
    for row_number, row in hardness.iterrows():
        value = _optional_float(row["hardness_hv"])
        if value is None:
            raise ValueError(f"{hardness_path} row {row_number + 2}: hardness is required")
        observations.append(
            PropertyObservation(
                observation_id=row["observation_id"],
                source_id=SOURCE_ID,
                property_name="hardness_hv",
                value=value,
                unit="HV1",
                alloy=ALLOY,
                process=PROCESS,
                state=row["state"],
                temperature_c=_optional_float(row["temperature_c"]),
                hold_minutes=_optional_float(row["hold_minutes"]),
                build_strategy=row["build_strategy"],
                scatter=_optional_float(row["hardness_sd_hv"]),
                scatter_kind="sd",
                n_measurements=int(row["n_measurements"]),
                source_locator=row["source_locator"],
            )
        )

    for row_number, row in tensile.iterrows():
        for property_name, unit in _TENSILE_PROPERTIES.items():
            value = _optional_float(row[property_name])
            if value is None:
                continue
            observations.append(
                PropertyObservation(
                    observation_id=f"{row['observation_id']}:{property_name}",
                    source_id=SOURCE_ID,
                    property_name=property_name,
                    value=value,
                    unit=unit,
                    alloy=ALLOY,
                    process=PROCESS,
                    state=row["state"],
                    temperature_c=_optional_float(row["temperature_c"]),
                    hold_minutes=_optional_float(row["hold_minutes"]),
                    build_strategy=row["build_strategy"],
                    orientation=row["orientation"],
                    source_locator=row["source_locator"],
                )
            )

    duplicate_ids = pd.Series([item.observation_id for item in observations]).duplicated()
    if duplicate_ids.any():
        duplicate = [
            item.observation_id for item, is_duplicate in zip(observations, duplicate_ids)
            if is_duplicate
        ][0]
        raise ValueError(f"duplicate observation_id {duplicate!r}")
    return observations


def godec_image_states(root: Path) -> list[tuple[Path, MaterialState]]:
    raw = Path(root) / "raw"
    return [
        (path, parse_godec_image(path))
        for path in sorted(raw.glob("BEI *.tif"))
    ]


def audit_godec_links(root: Path) -> pd.DataFrame:
    """Detailed image/property candidate table for review or export."""

    images = godec_image_states(root)
    observations = load_godec_observations(root)
    observation_by_id = {item.observation_id: item for item in observations}
    state_by_id = {state.record_id: state for _, state in images}
    links = candidate_links((state for _, state in images), observations)
    rows = []
    for link in links:
        state = state_by_id[link.record_id]
        observation = observation_by_id[link.observation_id]
        rows.append(
            {
                "record_id": link.record_id,
                "observation_id": link.observation_id,
                "property_name": link.property_name,
                "value": link.value,
                "unit": link.unit,
                "score": link.score,
                "confidence": link.confidence,
                "training_weight": link.training_weight,
                "auto_attach": link.auto_attach,
                "validation_eligible": link.validation_eligible,
                "image_state": state.state,
                "image_temperature_c": state.temperature_c,
                "image_build_strategy": state.build_strategy,
                "property_orientation": observation.orientation,
                "source_locator": observation.source_locator,
                "reasons": "; ".join(link.reasons),
            }
        )
    return pd.DataFrame(rows)


@register_adapter
class GodecIN718Adapter(BaseAdapter):
    """Canonical records from the downloaded public IN718 subset."""

    name = "godec_in718"
    family = "nickel"

    def records(self) -> list[CanonicalRecord]:
        root = self.cfg.public_in718_dir
        images = godec_image_states(root)
        if not images:
            return []
        observations = load_godec_observations(root)
        links = candidate_links((state for _, state in images), observations)
        auto = best_auto_links(links)

        records: list[CanonicalRecord] = []
        for path, state in images:
            hardness = auto.get((state.record_id, "hardness_hv"))
            properties = {"hardness_hv": hardness.value} if hardness else {}
            property_sources = {"hardness_hv": DISTANT} if hardness else {}
            property_weights = (
                {"hardness_hv": hardness.training_weight} if hardness else {}
            )
            temperature = (
                int(state.temperature_c) if state.temperature_c is not None else None
            )
            condition = _CONDITIONS[(state.state, temperature)]
            condition_token = "ab" if temperature is None else f"ht-{temperature}c"
            records.append(
                CanonicalRecord(
                    record_id=state.record_id,
                    image_path=path,
                    scale_um_per_px=None,
                    modality="BSE-SEM",
                    group_id=(
                        f"godec-in718-{condition_token}-"
                        f"{state.build_strategy.casefold()}"
                    ),
                    properties=properties,
                    property_sources=property_sources,
                    property_weights=property_weights,
                    alloy_grade="grade/nickel/in718",
                    condition=condition,
                )
            )
        return records
