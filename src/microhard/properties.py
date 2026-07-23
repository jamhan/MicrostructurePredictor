"""The (alloy_grade, condition) -> property lookup used for distant supervision.

Bulk properties are reproducible for a given alloy grade and heat treatment:
two labs that quench the same grade from the same temperature measure
hardnesses within a band far narrower than the range across treatments. So a
value published for that pair can be attached to any micrograph of that pair,
which is how a handful of measured samples becomes a usable training set. The
bet is only valid for properties the pair actually fixes; docs/DATASET_PLAN.md
says where it breaks.

The table is data/property_lookup.csv, one row per
(alloy_grade, condition, property_name). Every row carries a citation and a
join confidence, and a row missing either is rejected at load time, because an
uncited property value is indistinguishable from an invented one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .records import DISTANT, MEASURED, CanonicalRecord
from .taxonomy import CONDITION_AXIS, GRADE_AXIS, Taxonomy

PROPERTY_LOOKUP_COLUMNS = [
    "alloy_grade",  # node id, alloy_grade axis
    "condition",  # node id, condition axis
    "property_name",  # key used in CanonicalRecord.properties
    "value",  # number, in the unit PROPERTY_UNITS demands
    "unit",
    "scatter",  # within-condition spread, same unit; blank if unreported
    "scatter_kind",  # how to read `scatter`
    "n_measurements",  # samples behind the value; blank if unreported
    "join_confidence",
    "source_citation",  # free text, but must identify a real document
    "source_url",  # DOI or URL where the value can be checked
    "note",  # assumptions made in matching the source to this key
]

# Adding a property means adding a line here. The unit is part of the
# property_name contract (hardness_hv is Vickers, full stop), so a row whose
# unit disagrees is a transcription error rather than a conversion request.
PROPERTY_UNITS = {
    "hardness_hv": "HV",
    "yield_strength_mpa": "MPa",
    "ultimate_tensile_strength_mpa": "MPa",
    "elongation_pct": "%",
    "reduction_area_pct": "%",
    "youngs_modulus_gpa": "GPa",
}

SCATTER_KINDS = ("sd", "half_range", "tolerance_band", "unreported")

# Ordered weakest to strongest.
CONFIDENCE_LEVELS = ("low", "medium", "high")
CONFIDENCE_WEIGHTS = {"low": 0.25, "medium": 0.55, "high": 0.85}


@dataclass(frozen=True)
class PropertyEntry:
    """One cited property value for one (grade, condition) pair."""

    alloy_grade: str
    condition: str
    property_name: str
    value: float
    unit: str
    scatter: float | None
    scatter_kind: str
    n_measurements: int | None
    join_confidence: str
    source_citation: str
    source_url: str
    note: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.alloy_grade, self.condition)


def load_property_lookup(
    csv_path: Path, taxonomy: Taxonomy | None = None
) -> list[PropertyEntry]:
    """Rows from the lookup CSV, validated.

    Returns an empty list when the file does not exist. The template ships
    with a header and no rows, so an empty table is the normal starting state
    and everything downstream degrades to "no distant labels" rather than
    failing.

    Every key is checked against the bundled taxonomy by default. Pass a
    taxonomy explicitly when loading against a project-specific registry.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = set(PROPERTY_LOOKUP_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing columns {sorted(missing)}; "
            f"expected {PROPERTY_LOOKUP_COLUMNS}"
        )

    entries = [_entry_from_row(row, csv_path, i) for i, row in frame.iterrows()]
    _reject_duplicates(entries, csv_path)
    taxonomy = taxonomy or Taxonomy.load(None)
    taxonomy.require([e.alloy_grade for e in entries], axis=GRADE_AXIS)
    taxonomy.require([e.condition for e in entries], axis=CONDITION_AXIS)
    return entries


def _entry_from_row(row: pd.Series, csv_path: Path, index: int) -> PropertyEntry:
    where = f"{csv_path} row {index + 2}"  # +2: header line, 1-based
    alloy_grade = _required_text(row["alloy_grade"], where, "alloy_grade")
    condition = _required_text(row["condition"], where, "condition")
    property_name = str(row["property_name"]).strip()
    if property_name not in PROPERTY_UNITS:
        raise ValueError(
            f"{where}: unknown property_name {property_name!r}; "
            f"known properties: {sorted(PROPERTY_UNITS)}"
        )
    unit = str(row["unit"]).strip()
    if unit != PROPERTY_UNITS[property_name]:
        raise ValueError(
            f"{where}: {property_name} is recorded in {PROPERTY_UNITS[property_name]}, "
            f"got {unit!r}; convert the value before adding the row"
        )
    confidence = str(row["join_confidence"]).strip().lower()
    if confidence not in CONFIDENCE_LEVELS:
        raise ValueError(
            f"{where}: join_confidence {confidence!r} not in {list(CONFIDENCE_LEVELS)}"
        )
    scatter_kind = str(row["scatter_kind"]).strip().lower() or "unreported"
    if scatter_kind not in SCATTER_KINDS:
        raise ValueError(f"{where}: scatter_kind {scatter_kind!r} not in {list(SCATTER_KINDS)}")
    citation = str(row["source_citation"]).strip()
    if not citation:
        raise ValueError(
            f"{where}: source_citation is empty. Every value needs a document "
            "it can be checked against; see docs/DATASET_PLAN.md."
        )
    value = _required_number(row["value"], where, "value")
    if property_name == "hardness_hv" and value <= 0:
        raise ValueError(f"{where}: hardness_hv must be positive, got {value}")
    scatter = _optional_number(row["scatter"], where, "scatter")
    if scatter is not None and scatter < 0:
        raise ValueError(f"{where}: scatter must be non-negative, got {scatter}")
    if (scatter is None) != (scatter_kind == "unreported"):
        raise ValueError(
            f"{where}: scatter_kind must be 'unreported' exactly when scatter is blank"
        )
    n_measurements = _optional_int(row["n_measurements"], where)
    if confidence == "high" and scatter is None and n_measurements is None:
        raise ValueError(
            f"{where}: high confidence requires reported scatter or n_measurements"
        )
    return PropertyEntry(
        alloy_grade=alloy_grade,
        condition=condition,
        property_name=property_name,
        value=value,
        unit=unit,
        scatter=scatter,
        scatter_kind=scatter_kind,
        n_measurements=n_measurements,
        join_confidence=confidence,
        source_citation=citation,
        source_url=str(row["source_url"]).strip(),
        note=str(row["note"]).strip(),
    )


def _required_text(raw: object, where: str, column: str) -> str:
    text = str(raw).strip()
    if not text:
        raise ValueError(f"{where}: {column} is required")
    return text


def _required_number(raw: object, where: str, column: str) -> float:
    number = _optional_number(raw, where, column)
    if number is None:
        raise ValueError(f"{where}: {column} is required and must be a number")
    return number


def _optional_number(raw: object, where: str, column: str) -> float | None:
    text = str(raw).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"{where}: {column} {text!r} is not a number") from None
    if not math.isfinite(number):
        raise ValueError(f"{where}: {column} {text!r} is not finite")
    return number


def _optional_int(raw: object, where: str) -> int | None:
    number = _optional_number(raw, where, "n_measurements")
    if number is None:
        return None
    if not number.is_integer() or number < 1:
        raise ValueError(f"{where}: n_measurements must be a positive integer, got {number}")
    return int(number)


def _reject_duplicates(entries: Sequence[PropertyEntry], csv_path: Path) -> None:
    seen: dict[tuple[str, str, str], int] = {}
    for i, entry in enumerate(entries):
        triple = (entry.alloy_grade, entry.condition, entry.property_name)
        if triple in seen:
            raise ValueError(
                f"{csv_path}: {triple} appears on rows {seen[triple] + 2} and {i + 2}. "
                "Two published values for one key have to be reconciled by hand, "
                "or the condition split into finer nodes."
            )
        seen[triple] = i


def lookup_index(
    entries: Iterable[PropertyEntry],
) -> dict[tuple[str, str], dict[str, PropertyEntry]]:
    """(alloy_grade, condition) -> {property_name: entry}."""
    index: dict[tuple[str, str], dict[str, PropertyEntry]] = {}
    for entry in entries:
        index.setdefault(entry.key, {})[entry.property_name] = entry
    return index


def join_properties(
    records: Sequence[CanonicalRecord],
    index: dict[tuple[str, str], dict[str, PropertyEntry]],
    min_confidence: str = "low",
) -> list[CanonicalRecord]:
    """Attach looked-up values to records, returning new records.

    The join, exactly:

    1. A record with no ``join_key`` (either half of the pair missing) is
       returned unchanged. There is no fallback to a partial key.
    2. Otherwise every entry for that key whose ``join_confidence`` is at
       least ``min_confidence`` contributes
       ``properties[entry.property_name] = entry.value``, with
       ``property_sources[entry.property_name] = DISTANT``.
    3. A property the record already carries is left alone. A direct
       measurement on this physical sample always beats a value looked up for
       its grade and condition, and a distant label never overwrites another.

    The provenance behind a distant value is not copied onto the record; it
    stays in the table, reachable through the record's join_key, so there is
    one place to correct a citation.
    """
    if min_confidence not in CONFIDENCE_LEVELS:
        raise ValueError(
            f"min_confidence {min_confidence!r} not in {list(CONFIDENCE_LEVELS)}"
        )
    floor = CONFIDENCE_LEVELS.index(min_confidence)

    out: list[CanonicalRecord] = []
    for record in records:
        key = record.join_key
        entries = index.get(key) if key is not None else None
        if not entries:
            out.append(record)
            continue
        additions = {
            name: entry.value
            for name, entry in entries.items()
            if name not in record.properties
            and CONFIDENCE_LEVELS.index(entry.join_confidence) >= floor
        }
        if not additions:
            out.append(record)
            continue
        out.append(
            replace(
                record,
                properties={**record.properties, **additions},
                property_sources={
                    **record.property_sources,
                    **{name: DISTANT for name in additions},
                },
                property_weights={
                    **record.property_weights,
                    **{
                        name: CONFIDENCE_WEIGHTS[entries[name].join_confidence]
                        for name in additions
                    },
                },
            )
        )
    return out


def measured_properties(record: CanonicalRecord) -> dict[str, float]:
    """The record's directly measured properties, dropping distant labels.

    Evaluation uses this: a benchmark scored against weak labels measures
    agreement with the lookup table, not with a durometer.
    """
    return {
        name: value
        for name, value in record.properties.items()
        if record.property_sources.get(name, MEASURED) == MEASURED
    }
