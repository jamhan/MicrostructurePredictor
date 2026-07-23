"""Diversity-first planning for the next direct UHCS measurements.

This is not model-based active learning: seven direct targets are too few for
stable uncertainty estimates. It is a deterministic maximin design over the
recorded process variables, weighted by metadata and image readiness. The
result answers the practical first question: which already-imaged specimens
add the most process-space coverage if measured next?
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .adapters.uhcs import UHCSAdapter, load_hardness_labels, load_metadata
from .config import Config

PLAN_COLUMNS = [
    "rank",
    "group_id",
    "sample_ids",
    "sample_label",
    "temperature_c",
    "hold_min",
    "cooling",
    "image_count",
    "magnification_levels",
    "scale_coverage",
    "exact_condition",
    "novelty_score",
    "metadata_score",
    "imaging_score",
    "priority_score",
    "rationale",
]


def plan_uhcs_measurements(
    cfg: Config,
    limit: int | None = 12,
    *,
    include_unverified: bool = False,
    grade: str | None = "grade/ferrous/uhcs_ac1",
) -> pd.DataFrame:
    """Rank unlabeled, imaged UHCS groups for direct mechanical measurement."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    meta = load_metadata(cfg.sqlite_path)
    if meta.empty:
        return pd.DataFrame(columns=PLAN_COLUMNS)

    group_ids = UHCSAdapter._group_ids(meta)
    working = meta.copy()
    working["_group_id"] = [
        UHCSAdapter._group_id(row, group_ids) for _, row in working.iterrows()
    ]
    hardness = load_hardness_labels(cfg.hardness_csv)
    measured_labels = set(hardness["sample_label"])

    rows: list[dict] = []
    for group_id, group in working.groupby("_group_id", sort=True):
        sample_rows = group.drop_duplicates("sample_key")
        first = sample_rows.iloc[0]
        labels = sorted(
            {str(value) for value in sample_rows["sample_label"] if pd.notna(value)}
        )
        sample_ids = sorted(
            {int(value) for value in sample_rows["sample_key"] if pd.notna(value)}
        )
        label = " | ".join(labels)
        temperature = _common_numeric(sample_rows["anneal_temperature"])
        hold_min = _hold_minutes(first)
        cooling = _common_text(sample_rows["cool_method"])
        condition = UHCSAdapter._exact_condition(first, label if len(labels) == 1 else None)
        grade, _ = UHCSAdapter._join_key(first)
        scales = [UHCSAdapter._scale(row) for _, row in group.iterrows()]
        scale_coverage = sum(value is not None for value in scales) / len(scales)
        magnifications = group["magnification"].dropna().nunique()

        structured = [
            temperature is not None,
            hold_min is not None,
            cooling is not None,
            grade is not None,
        ]
        metadata_score = 0.65 * (sum(structured) / len(structured)) + 0.35 * (
            condition is not None
        )
        imaging_score = (
            0.5 * min(len(group) / 10, 1)
            + 0.3 * scale_coverage
            + 0.2 * min(magnifications / 3, 1)
        )
        rows.append(
            {
                "group_id": group_id,
                "sample_ids": "|".join(map(str, sample_ids)),
                "sample_label": label,
                "temperature_c": temperature,
                "hold_min": hold_min,
                "cooling": cooling,
                "grade": grade,
                "image_count": len(group),
                "magnification_levels": int(magnifications),
                "scale_coverage": scale_coverage,
                "exact_condition": condition or "",
                "direct_hardness": any(value in measured_labels for value in labels),
                "metadata_score": metadata_score,
                "imaging_score": imaging_score,
            }
        )

    frame = pd.DataFrame(rows)
    candidates = frame.loc[
        (~frame["direct_hardness"])
        & (frame["image_count"] > 0)
        & frame["grade"].notna()
        & frame["sample_ids"].ne("")
    ].copy()
    if grade is not None:
        candidates = candidates.loc[candidates["grade"].eq(grade)].copy()
    if not include_unverified:
        candidates = candidates.loc[candidates["exact_condition"].ne("")].copy()
    references = frame.loc[
        frame["direct_hardness"] & frame["grade"].notna() & frame["sample_ids"].ne("")
    ].copy()
    if grade is not None:
        references = references.loc[references["grade"].eq(grade)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=PLAN_COLUMNS)

    _add_normalized_process_coordinates(frame, candidates, references)
    selected: list[dict] = []
    reference_coordinates = [
        _coordinates(row) for _, row in references.iterrows()
    ]
    remaining = candidates.to_dict("records")

    while remaining and (limit is None or len(selected) < limit):
        for row in remaining:
            point = _coordinates(row)
            if not any(value is not None for value in point[:3]):
                novelty = 0.0
            elif reference_coordinates:
                distances = [
                    distance
                    for ref in reference_coordinates
                    if (distance := _process_distance(point, ref)) is not None
                ]
                novelty = min(distances) if distances else 1.0
            else:
                novelty = 1.0
            row["novelty_score"] = novelty
            row["priority_score"] = (
                0.50 * novelty
                + 0.30 * float(row["metadata_score"])
                + 0.20 * float(row["imaging_score"])
            )
        best = max(
            remaining,
            key=lambda row: (
                row["priority_score"],
                row["metadata_score"],
                row["image_count"],
                row["sample_label"],
            ),
        )
        remaining.remove(best)
        reference_coordinates.append(_coordinates(best))
        selected.append(best)

    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
        readiness = (
            "exact process key"
            if row["exact_condition"]
            else "process metadata needs verification"
        )
        row["rationale"] = (
            f"{readiness}; {row['image_count']} images at "
            f"{row['magnification_levels']} magnification levels"
        )

    result = pd.DataFrame(selected)
    for column in ("scale_coverage", "novelty_score", "metadata_score", "imaging_score"):
        result[column] = result[column].astype(float).round(3)
    result["priority_score"] = result["priority_score"].astype(float).round(3)
    return result[PLAN_COLUMNS]


def _hold_minutes(row: pd.Series) -> float | None:
    raw = row.get("anneal_time")
    unit = row.get("anneal_time_unit")
    if not pd.notna(raw) or not pd.notna(unit):
        return None
    normalized = str(unit).strip().upper()
    if normalized == "M":
        return float(raw)
    if normalized == "H":
        return float(raw) * 60
    return None


def _common_numeric(values: pd.Series) -> float | None:
    unique = pd.to_numeric(values, errors="coerce").dropna().unique()
    return float(unique[0]) if len(unique) == 1 else None


def _common_text(values: pd.Series) -> str | None:
    unique = sorted({str(value).strip() for value in values if pd.notna(value)})
    return unique[0] if len(unique) == 1 and unique[0] else None


def _add_normalized_process_coordinates(
    all_rows: pd.DataFrame, candidates: pd.DataFrame, references: pd.DataFrame
) -> None:
    temperatures = pd.to_numeric(all_rows["temperature_c"], errors="coerce")
    holds = pd.to_numeric(all_rows["hold_min"], errors="coerce").map(
        lambda value: math.log1p(value) if pd.notna(value) else np.nan
    )
    temperature_min, temperature_max = temperatures.min(), temperatures.max()
    hold_minimum, hold_maximum = holds.min(), holds.max()

    def normalize(raw: pd.Series, low: float, high: float, *, log: bool = False):
        values = pd.to_numeric(raw, errors="coerce")
        if log:
            values = values.map(lambda value: math.log1p(value) if pd.notna(value) else np.nan)
        if not pd.notna(low) or not pd.notna(high) or math.isclose(low, high):
            return values.map(lambda value: 0.5 if pd.notna(value) else np.nan)
        return (values - low) / (high - low)

    for subset in (candidates, references):
        subset["_temperature_norm"] = normalize(
            subset["temperature_c"], temperature_min, temperature_max
        )
        subset["_hold_norm"] = normalize(
            subset["hold_min"], hold_minimum, hold_maximum, log=True
        )


def _coordinates(
    row: pd.Series | dict,
) -> tuple[float | None, float | None, str | None, str | None]:
    def optional_float(value) -> float | None:
        return float(value) if pd.notna(value) else None

    return (
        optional_float(row["_temperature_norm"]),
        optional_float(row["_hold_norm"]),
        row["cooling"] if pd.notna(row["cooling"]) else None,
        row["grade"] if pd.notna(row["grade"]) else None,
    )


def _process_distance(
    left: tuple[float | None, float | None, str | None, str | None],
    right: tuple[float | None, float | None, str | None, str | None],
) -> float | None:
    components: list[float] = []
    for a, b in zip(left[:2], right[:2]):
        if a is not None and b is not None:
            components.append((a - b) ** 2)
    if left[2] is not None and right[2] is not None:
        components.append(0.0 if left[2] == right[2] else 1.0)
    if not components:
        return None
    if left[3] is not None and right[3] is not None:
        components.append(0.0 if left[3] == right[3] else 1.0)
    return math.sqrt(sum(components) / len(components))
