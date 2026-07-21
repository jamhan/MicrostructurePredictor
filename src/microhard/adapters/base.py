"""BaseAdapter: turn one dataset's native layout into canonical records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ..config import Config
from ..records import CanonicalRecord
from ..taxonomy import Taxonomy


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
        """records() with every taxonomy reference checked against the registry."""
        out = self.records()
        for record in out:
            if record.taxonomy_labels:
                self.taxonomy.require(record.taxonomy_labels)
            if record.mask_class_nodes:
                self.taxonomy.require(record.mask_class_nodes)
        return out
