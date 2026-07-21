"""Audit UHCS metadata before it is used as a distant-supervision join key.

The current adapter can recover a coarse (grade, cooling route) key. Hardness
also depends on the recorded temperature and hold time, so this audit separates
simple, structured conditions from rows that still need interpretation. It does
not add taxonomy nodes or property values.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .normalize import AmbiguousAliasError, normalize_join_key

_SIMPLE_COOLING = frozenset({"WQ", "AR", "FC"})
_KNOWN_COOLING = _SIMPLE_COOLING | {"Q"}
_TEMPERATURE_IN_LABEL = re.compile(r"(?<![-\w])(\d+(?:\.\d+)?)\s*C\b", re.IGNORECASE)


@dataclass(frozen=True)
class SampleKeyAudit:
    sample_id: int
    label: str
    alloy_grade: str | None
    coarse_condition: str | None
    temperature_c: float | None
    hold_value: float | None
    hold_unit: str | None
    cooling_code: str | None
    micrographs: int
    status: str
    flags: tuple[str, ...]

    @property
    def complete_coarse_key(self) -> bool:
        return self.alloy_grade is not None and self.coarse_condition is not None

    @property
    def thermal_signature(self) -> str | None:
        if self.temperature_c is None or self.hold_value is None or self.hold_unit is None:
            return None
        cooling = (self.cooling_code or "unknown").lower()
        return (
            f"t{_number(self.temperature_c)}c_"
            f"{_number(self.hold_value)}{self.hold_unit.lower()}_{cooling}"
        )


@dataclass(frozen=True)
class AuditSummary:
    sample_rows: int
    samples_with_micrographs: int
    micrographs: int
    linked_micrographs: int
    orphan_micrographs: int
    complete_key_samples: int
    complete_key_micrographs: int
    candidate_micrographs: int
    provisional_micrographs: int
    blocked_complete_key_micrographs: int
    incomplete_key_micrographs: int


@dataclass(frozen=True)
class CoarseKeyGroup:
    alloy_grade: str
    condition: str
    samples: int
    micrographs: int
    thermal_signatures: tuple[str, ...]


@dataclass(frozen=True)
class ExactConditionGroup:
    status: str
    alloy_grade: str
    thermal_signature: str
    sample_ids: tuple[int, ...]
    micrographs: int
    flags: tuple[str, ...]


@dataclass(frozen=True)
class JoinKeyAudit:
    samples: tuple[SampleKeyAudit, ...]
    summary: AuditSummary
    coarse_groups: tuple[CoarseKeyGroup, ...]
    exact_groups: tuple[ExactConditionGroup, ...]


def audit_uhcs_join_keys(sqlite_path: Path) -> JoinKeyAudit:
    """Classify every UHCS sample row by join readiness."""
    sqlite_path = Path(sqlite_path)
    with sqlite3.connect(sqlite_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT s.sample_id, s.label, s.anneal_time, s.anneal_time_unit,
                   s.anneal_temperature, s.anneal_temp_unit, s.cool_method,
                   COUNT(m.micrograph_id) AS micrographs
            FROM sample AS s
            LEFT JOIN micrograph AS m ON m.sample_key = s.sample_id
            GROUP BY s.sample_id
            ORDER BY s.sample_id
            """
        ).fetchall()
        micrographs = int(con.execute("SELECT COUNT(*) FROM micrograph").fetchone()[0])

    label_counts = Counter(str(row["label"]) for row in rows if row["label"])
    samples = tuple(_audit_sample(row, label_counts) for row in rows)
    linked = sum(sample.micrographs for sample in samples)
    complete = [sample for sample in samples if sample.complete_coarse_key]
    summary = AuditSummary(
        sample_rows=len(samples),
        samples_with_micrographs=sum(sample.micrographs > 0 for sample in samples),
        micrographs=micrographs,
        linked_micrographs=linked,
        orphan_micrographs=micrographs - linked,
        complete_key_samples=len(complete),
        complete_key_micrographs=sum(sample.micrographs for sample in complete),
        candidate_micrographs=_images_with_status(samples, "candidate"),
        provisional_micrographs=_images_with_status(samples, "provisional"),
        blocked_complete_key_micrographs=sum(
            sample.micrographs
            for sample in complete
            if sample.status not in {"candidate", "provisional"}
        ),
        incomplete_key_micrographs=sum(
            sample.micrographs for sample in samples if not sample.complete_coarse_key
        )
        + (micrographs - linked),
    )
    return JoinKeyAudit(
        samples=samples,
        summary=summary,
        coarse_groups=_coarse_groups(samples),
        exact_groups=_exact_groups(samples),
    )


def _audit_sample(row: sqlite3.Row, label_counts: Counter[str]) -> SampleKeyAudit:
    label = str(row["label"] or "")
    cooling = str(row["cool_method"]) if row["cool_method"] is not None else None
    flags: list[str] = []
    try:
        grade, condition = normalize_join_key(label or None, cooling)
    except AmbiguousAliasError:
        grade = condition = None
        flags.append("ambiguous_alias")

    temperature = _float_or_none(row["anneal_temperature"])
    hold = _float_or_none(row["anneal_time"])
    temp_unit = str(row["anneal_temp_unit"] or "").upper() or None
    hold_unit = str(row["anneal_time_unit"] or "").upper() or None
    if grade is None:
        flags.append("missing_grade")
    if condition is None:
        flags.append("unknown_condition")
    if temperature is None or hold is None or temp_unit != "C" or hold_unit not in {"M", "H"}:
        flags.append("missing_thermal_metadata")
    if cooling == "Q":
        flags.append("unspecified_quench_medium")
    if cooling is not None and cooling not in _KNOWN_COOLING:
        flags.append("special_cooling_code")
    if len(_TEMPERATURE_IN_LABEL.findall(label)) > 1:
        flags.append("multi_step_label")
    if label and label_counts[label] > 1:
        flags.append("duplicate_sample_label")
    if int(row["micrographs"]) == 0:
        flags.append("no_micrographs")

    blocking = {
        "ambiguous_alias",
        "missing_grade",
        "unknown_condition",
        "missing_thermal_metadata",
        "unspecified_quench_medium",
        "special_cooling_code",
        "multi_step_label",
        "no_micrographs",
    }
    if blocking.intersection(flags):
        status = "blocked"
    elif cooling == "WQ":
        status = "candidate"
    else:
        # AR and FC are structurally complete, but their cooling rate and the
        # meaning of AR still need confirmation from the source documentation.
        status = "provisional"

    return SampleKeyAudit(
        sample_id=int(row["sample_id"]),
        label=label,
        alloy_grade=grade,
        coarse_condition=condition,
        temperature_c=temperature if temp_unit == "C" else None,
        hold_value=hold,
        hold_unit=hold_unit,
        cooling_code=cooling,
        micrographs=int(row["micrographs"]),
        status=status,
        flags=tuple(flags),
    )


def _coarse_groups(samples: tuple[SampleKeyAudit, ...]) -> tuple[CoarseKeyGroup, ...]:
    grouped: dict[tuple[str, str], list[SampleKeyAudit]] = defaultdict(list)
    for sample in samples:
        if sample.complete_coarse_key and sample.micrographs:
            grouped[(sample.alloy_grade, sample.coarse_condition)].append(sample)  # type: ignore[arg-type]
    return tuple(
        CoarseKeyGroup(
            alloy_grade=key[0],
            condition=key[1],
            samples=len(group),
            micrographs=sum(sample.micrographs for sample in group),
            thermal_signatures=tuple(
                sorted(
                    signature
                    for signature in {sample.thermal_signature for sample in group}
                    if signature is not None
                )
            ),
        )
        for key, group in sorted(grouped.items())
    )


def _exact_groups(samples: tuple[SampleKeyAudit, ...]) -> tuple[ExactConditionGroup, ...]:
    grouped: dict[tuple[str, str, str], list[SampleKeyAudit]] = defaultdict(list)
    for sample in samples:
        signature = sample.thermal_signature
        if sample.status in {"candidate", "provisional"} and signature is not None:
            grouped[(sample.status, sample.alloy_grade or "", signature)].append(sample)
    return tuple(
        ExactConditionGroup(
            status=key[0],
            alloy_grade=key[1],
            thermal_signature=key[2],
            sample_ids=tuple(sample.sample_id for sample in group),
            micrographs=sum(sample.micrographs for sample in group),
            flags=tuple(sorted({flag for sample in group for flag in sample.flags})),
        )
        for key, group in sorted(grouped.items())
    )


def _images_with_status(samples: tuple[SampleKeyAudit, ...], status: str) -> int:
    return sum(sample.micrographs for sample in samples if sample.status == status)


def _float_or_none(value: object) -> float | None:
    return None if value is None else float(value)


def _number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value).replace(".", "p")
