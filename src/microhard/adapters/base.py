"""BaseAdapter: turn one dataset's native layout into canonical records."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import ClassVar

from ..config import Config
from ..records import PROPERTY_SOURCES, CanonicalRecord
from ..taxonomy import CONDITION_AXIS, GRADE_AXIS, MICROCONSTITUENT_AXIS, Taxonomy


class BaseAdapter(ABC):
    """One concrete subclass per dataset.

    Subclasses declare ``name`` (registry key, also used in feature tables)
    and ``family`` (the taxonomy level-1 node id all records belong to), and
    implement ``records()``. Everything downstream — router, segmenter,
    features, property heads — consumes only the canonical records.
    """

    name: ClassVar[str]
    family: ClassVar[str]

    def __init__(self, cfg: Config, taxonomy: Taxonomy) -> None:
        self.cfg = cfg
        self.taxonomy = taxonomy
        taxonomy.node(self.family)  # an unregistered family is a configuration error

    @abstractmethod
    def records(self) -> list[CanonicalRecord]:
        """All records this dataset provides (image files may or may not be
        on disk yet; consumers filter on existence where it matters)."""

    def validated_records(self) -> list[CanonicalRecord]:
        """records() with every taxonomy reference checked against the registry.

        Each reference is checked against its own axis, so a grade id used as
        a constituent label (or the reverse) fails here rather than producing
        a join key nothing matches.
        """
        out = self.records()
        for record in out:
            if record.taxonomy_labels:
                self.taxonomy.require(record.taxonomy_labels, axis=MICROCONSTITUENT_AXIS)
            if record.mask_class_nodes:
                self.taxonomy.require(record.mask_class_nodes, axis=MICROCONSTITUENT_AXIS)
            if record.alloy_grade:
                self.taxonomy.require([record.alloy_grade], axis=GRADE_AXIS)
            if record.condition:
                self.taxonomy.require([record.condition], axis=CONDITION_AXIS)
            self._check_property_sources(record)
            self._check_property_weights(record)
        return out

    @staticmethod
    def _check_property_sources(record: CanonicalRecord) -> None:
        orphans = sorted(set(record.property_sources) - set(record.properties))
        if orphans:
            raise ValueError(
                f"{record.record_id}: property_sources names {orphans}, which are "
                "not in properties"
            )
        unknown = sorted(set(record.property_sources.values()) - set(PROPERTY_SOURCES))
        if unknown:
            raise ValueError(
                f"{record.record_id}: property source {unknown} not in "
                f"{list(PROPERTY_SOURCES)}"
            )

    @staticmethod
    def _check_property_weights(record: CanonicalRecord) -> None:
        orphans = sorted(set(record.property_weights) - set(record.properties))
        if orphans:
            raise ValueError(
                f"{record.record_id}: property_weights names {orphans}, which are "
                "not in properties"
            )
        invalid = {
            name: value
            for name, value in record.property_weights.items()
            if not math.isfinite(value) or not 0.0 < value <= 1.0
        }
        if invalid:
            raise ValueError(
                f"{record.record_id}: property weights must be finite in (0, 1], "
                f"got {invalid}"
            )
