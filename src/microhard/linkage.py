"""Provenance-aware fuzzy linkage between micrographs and property records.

The matcher is deliberately conservative.  String similarity is useful for
spelling and formatting differences, but it never gets to overrule an explicit
metallurgical conflict such as a different alloy, treatment temperature, build
strategy, or tensile orientation.

Two outputs are kept separate:

``auto_attach``
    The observation is strong enough to be used as a weighted training label.

``validation_eligible``
    The source proves that the image and property came from the same physical
    specimen.  Same-study/same-condition matches are not validation data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

ORIENTATION_SENSITIVE_PROPERTIES = frozenset(
    {
        "yield_strength_mpa",
        "ultimate_tensile_strength_mpa",
        "elongation_pct",
        "reduction_area_pct",
        "youngs_modulus_gpa",
    }
)

CONFIDENCE_LEVELS = ("reject", "review", "medium", "high", "exact")


@dataclass(frozen=True)
class MaterialState:
    """Metadata describing the material state shown by one micrograph."""

    record_id: str
    source_id: str
    alloy: str
    process: str
    state: str
    temperature_c: float | None = None
    hold_minutes: float | None = None
    build_strategy: str | None = None
    orientation: str | None = None
    sampling_location: str | None = None
    physical_specimen_id: str | None = None


@dataclass(frozen=True)
class PropertyObservation:
    """One measured value plus the state metadata reported with it."""

    observation_id: str
    source_id: str
    property_name: str
    value: float
    unit: str
    alloy: str
    process: str
    state: str
    temperature_c: float | None = None
    hold_minutes: float | None = None
    build_strategy: str | None = None
    orientation: str | None = None
    sampling_location: str | None = None
    physical_specimen_id: str | None = None
    scatter: float | None = None
    scatter_kind: str = "unreported"
    n_measurements: int | None = None
    source_locator: str = ""


@dataclass(frozen=True)
class LinkageResult:
    """Auditable score for one image/property candidate pair."""

    record_id: str
    observation_id: str
    property_name: str
    value: float
    unit: str
    score: float
    confidence: str
    training_weight: float
    auto_attach: bool
    validation_eligible: bool
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]


def _normalise(value: str | None) -> str:
    if not value:
        return ""
    value = value.casefold()
    value = value.replace("inconel", "in")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _tokens(value: str | None) -> set[str]:
    return set(_normalise(value).split())


def _string_similarity(left: str | None, right: str | None) -> float:
    a, b = _normalise(left), _normalise(right)
    if not a or not b:
        return 0.0
    sequence = SequenceMatcher(None, a, b).ratio()
    a_tokens, b_tokens = set(a.split()), set(b.split())
    union = a_tokens | b_tokens
    jaccard = len(a_tokens & b_tokens) / len(union) if union else 0.0
    return max(sequence, jaccard)


def _same_text(left: str | None, right: str | None) -> bool:
    return bool(_normalise(left)) and _normalise(left) == _normalise(right)


def _numeric_similarity(
    left: float | None, right: float | None, tolerance: float
) -> float | None:
    if left is None or right is None:
        return None
    return max(0.0, 1.0 - abs(left - right) / tolerance)


def match_state_to_observation(
    image: MaterialState,
    observation: PropertyObservation,
) -> LinkageResult:
    """Score one candidate, rejecting explicit physical conflicts first."""

    blockers: list[str] = []
    reasons: list[str] = []

    alloy_similarity = _string_similarity(image.alloy, observation.alloy)
    if image.alloy and observation.alloy and alloy_similarity < 0.65:
        blockers.append(
            f"explicit alloy conflict: {image.alloy!r} vs {observation.alloy!r}"
        )

    if image.state and observation.state and not _same_text(image.state, observation.state):
        blockers.append(
            f"material-state conflict: {image.state!r} vs {observation.state!r}"
        )

    if (
        image.temperature_c is not None
        and observation.temperature_c is not None
        and abs(image.temperature_c - observation.temperature_c) > 25.0
    ):
        blockers.append(
            "treatment-temperature conflict: "
            f"{image.temperature_c:g} C vs {observation.temperature_c:g} C"
        )

    if (
        image.build_strategy
        and observation.build_strategy
        and not _same_text(image.build_strategy, observation.build_strategy)
    ):
        blockers.append(
            "build-strategy conflict: "
            f"{image.build_strategy!r} vs {observation.build_strategy!r}"
        )

    if (
        image.sampling_location
        and observation.sampling_location
        and not _same_text(image.sampling_location, observation.sampling_location)
    ):
        blockers.append(
            "sampling-location conflict: "
            f"{image.sampling_location!r} vs {observation.sampling_location!r}"
        )

    orientation_sensitive = (
        observation.property_name in ORIENTATION_SENSITIVE_PROPERTIES
    )
    if (
        orientation_sensitive
        and image.orientation
        and observation.orientation
        and not _same_text(image.orientation, observation.orientation)
    ):
        blockers.append(
            f"orientation conflict: {image.orientation!r} vs {observation.orientation!r}"
        )

    if blockers:
        return LinkageResult(
            record_id=image.record_id,
            observation_id=observation.observation_id,
            property_name=observation.property_name,
            value=observation.value,
            unit=observation.unit,
            score=0.0,
            confidence="reject",
            training_weight=0.0,
            auto_attach=False,
            validation_eligible=False,
            reasons=(),
            blockers=tuple(blockers),
        )

    components: list[tuple[str, float, float]] = []

    if image.source_id and observation.source_id:
        same_source = _same_text(image.source_id, observation.source_id)
        components.append(("source", 1.0 if same_source else 0.0, 0.16))
        reasons.append("same source record" if same_source else "different source records")
    else:
        same_source = False

    if image.alloy and observation.alloy:
        components.append(("alloy", alloy_similarity, 0.20))
        reasons.append(f"alloy similarity {alloy_similarity:.2f}")

    process_similarity = _string_similarity(image.process, observation.process)
    if image.process and observation.process:
        components.append(("process", process_similarity, 0.10))
        reasons.append(f"process similarity {process_similarity:.2f}")

    if image.state and observation.state:
        components.append(("state", 1.0, 0.14))
        reasons.append("same material state")

    temperature_similarity = _numeric_similarity(
        image.temperature_c, observation.temperature_c, 25.0
    )
    if temperature_similarity is not None:
        components.append(("temperature", temperature_similarity, 0.15))
        reasons.append(
            f"treatment temperatures differ by "
            f"{abs(image.temperature_c - observation.temperature_c):g} C"
        )

    hold_similarity = _numeric_similarity(
        image.hold_minutes, observation.hold_minutes, 60.0
    )
    if hold_similarity is not None:
        components.append(("hold", hold_similarity, 0.08))
        reasons.append(
            f"hold times differ by "
            f"{abs(image.hold_minutes - observation.hold_minutes):g} min"
        )

    if image.build_strategy and observation.build_strategy:
        components.append(("build strategy", 1.0, 0.12))
        reasons.append("same build strategy")

    if image.sampling_location and observation.sampling_location:
        components.append(("sampling location", 1.0, 0.12))
        reasons.append("same sampling location")

    exact_specimen = bool(
        image.physical_specimen_id
        and observation.physical_specimen_id
        and _same_text(image.physical_specimen_id, observation.physical_specimen_id)
    )
    if image.physical_specimen_id and observation.physical_specimen_id:
        components.append(("physical specimen", 1.0 if exact_specimen else 0.0, 0.25))
        if exact_specimen:
            reasons.append("same explicitly identified physical specimen")

    orientation_missing = bool(
        orientation_sensitive and observation.orientation and not image.orientation
    )
    if orientation_sensitive and image.orientation and observation.orientation:
        components.append(("orientation", 1.0, 0.10))
        reasons.append("same mechanical-test orientation")
    elif orientation_missing:
        reasons.append(
            "image orientation is absent; orientation-specific values remain alternatives"
        )

    denominator = sum(weight for _, _, weight in components)
    score = (
        sum(value * weight for _, value, weight in components) / denominator
        if denominator
        else 0.0
    )
    score = round(max(0.0, min(1.0, score)), 4)

    if exact_specimen and score >= 0.90:
        confidence = "exact"
        training_weight = 1.0
        auto_attach = True
        validation_eligible = True
    elif same_source and score >= 0.85 and not orientation_missing:
        confidence = "high"
        training_weight = 0.85
        auto_attach = True
        validation_eligible = False
    elif score >= 0.72:
        confidence = "medium"
        training_weight = 0.55
        auto_attach = False
        validation_eligible = False
    elif score >= 0.58:
        confidence = "review"
        training_weight = 0.25
        auto_attach = False
        validation_eligible = False
    else:
        confidence = "reject"
        training_weight = 0.0
        auto_attach = False
        validation_eligible = False

    return LinkageResult(
        record_id=image.record_id,
        observation_id=observation.observation_id,
        property_name=observation.property_name,
        value=observation.value,
        unit=observation.unit,
        score=score,
        confidence=confidence,
        training_weight=training_weight,
        auto_attach=auto_attach,
        validation_eligible=validation_eligible,
        reasons=tuple(reasons),
        blockers=(),
    )


def candidate_links(
    images: Iterable[MaterialState],
    observations: Iterable[PropertyObservation],
    *,
    include_rejected: bool = False,
) -> list[LinkageResult]:
    """Score the Cartesian product and retain plausible, auditable candidates."""

    observations = list(observations)
    out = [
        match_state_to_observation(image, observation)
        for image in images
        for observation in observations
    ]
    if not include_rejected:
        out = [link for link in out if link.confidence != "reject"]
    return sorted(
        out,
        key=lambda link: (
            link.record_id,
            link.property_name,
            -link.score,
            link.observation_id,
        ),
    )


def best_auto_links(
    links: Iterable[LinkageResult],
) -> dict[tuple[str, str], LinkageResult]:
    """Best unique auto-attachable link for each (record, property).

    A tie is deliberately omitted: two equally good values are evidence that a
    variable is missing from the image metadata, not permission to average.
    """

    candidates: dict[tuple[str, str], list[LinkageResult]] = {}
    for link in links:
        if link.auto_attach:
            candidates.setdefault((link.record_id, link.property_name), []).append(link)

    out: dict[tuple[str, str], LinkageResult] = {}
    for key, values in candidates.items():
        ranked = sorted(values, key=lambda item: item.score, reverse=True)
        if len(ranked) == 1 or not math.isclose(
            ranked[0].score, ranked[1].score, abs_tol=1e-12
        ):
            out[key] = ranked[0]
    return out
