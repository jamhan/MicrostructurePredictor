"""Hierarchical label registry, one tree per axis.

Loaded from a taxonomy file (YAML by default; TOML and JSON also accepted by
extension). Node ids are path-style: ``ferrous``, ``ferrous/pearlite``,
``ferrous/pearlite/lamellar``. Every label used anywhere in the pipeline must
be a registered node id; free-form label strings are treated as errors, which
is what keeps multiple datasets in one vocabulary.

There are three axes, and a node belongs to exactly one:

``microconstituent``  what is in the image (family / constituent /
                      morphology). This is the original axis and the default,
                      so a taxonomy file that declares no axes is entirely
                      microconstituent.
``alloy_grade``       what the material is, e.g. ``grade/ferrous/aisi_1045``.
``condition``         how it was processed, e.g.
                      ``condition/austenitize/water_quench``.

The last two exist as the join key for distant supervision: bulk properties
are looked up per (alloy_grade, condition) pair. See docs/DATASET_PLAN.md.

A root node picks its axis with an ``axis:`` key and descendants inherit it,
so ids stay stable and the axes never collide (the microconstituent trees are
rooted at family names, the others at ``grade`` and ``condition``).

``Taxonomy.load(None)`` loads the bundled seed file
(src/microhard/taxonomy.yaml).
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

MICROCONSTITUENT_AXIS = "microconstituent"
GRADE_AXIS = "alloy_grade"
CONDITION_AXIS = "condition"

FAMILY_LEVEL = 1
CONSTITUENT_LEVEL = 2
MORPHOLOGY_LEVEL = 3

# Nesting depth allowed per axis. The microconstituent axis is exactly
# family/constituent/morphology. The condition axis gets one more level so a
# processing route can be split further when the property depends on it (see
# the austenitizing-temperature caveat in docs/DATASET_PLAN.md).
AXIS_MAX_LEVEL = {
    MICROCONSTITUENT_AXIS: 3,
    GRADE_AXIS: 3,
    CONDITION_AXIS: 4,
}
MAX_LEVEL = AXIS_MAX_LEVEL[MICROCONSTITUENT_AXIS]


@dataclass(frozen=True)
class TaxonomyNode:
    id: str  # path-style: "ferrous/pearlite/lamellar"
    name: str  # human-readable
    level: int  # depth from the root; 1=family on the microconstituent axis
    parent: str | None
    children: tuple[str, ...]
    axis: str = MICROCONSTITUENT_AXIS


class UnknownNodeError(KeyError):
    pass


class Taxonomy:
    def __init__(self, nodes: dict[str, TaxonomyNode]) -> None:
        self._nodes = nodes

    # --- loading -----------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Taxonomy":
        """Load from a .yaml/.toml/.json file, or the bundled seed if None."""
        if path is None:
            text = resources.files("microhard").joinpath("taxonomy.yaml").read_text()
            raw = _parse_yaml(text)
        else:
            path = Path(path)
            text = path.read_text()
            suffix = path.suffix.lower()
            if suffix in {".yaml", ".yml"}:
                raw = _parse_yaml(text)
            elif suffix == ".toml":
                raw = tomllib.loads(text)
            elif suffix == ".json":
                raw = json.loads(text)
            else:
                raise ValueError(f"Unsupported taxonomy format {suffix!r} (use .yaml/.toml/.json)")
        return cls(_build_nodes(raw))

    # --- lookup ------------------------------------------------------------

    def node(self, node_id: str) -> TaxonomyNode:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise UnknownNodeError(
                f"{node_id!r} is not a taxonomy node. Known ids: {sorted(self._nodes)}"
            ) from None

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes

    def ids(self) -> list[str]:
        return sorted(self._nodes)

    def families(self) -> list[TaxonomyNode]:
        """Level-1 microconstituent nodes — what the router classifies over.

        Grade and condition roots are deliberately excluded: they are
        vocabulary for the property join, not material families.
        """
        return [
            n
            for n in self._nodes.values()
            if n.level == FAMILY_LEVEL and n.axis == MICROCONSTITUENT_AXIS
        ]

    def children(self, node_id: str) -> list[TaxonomyNode]:
        return [self._nodes[c] for c in self.node(node_id).children]

    def axis_of(self, node_id: str) -> str:
        return self.node(node_id).axis

    def in_axis(self, axis: str) -> list[TaxonomyNode]:
        """Every node on one axis, ordered by id."""
        if axis not in AXIS_MAX_LEVEL:
            raise ValueError(f"unknown axis {axis!r}; known axes: {sorted(AXIS_MAX_LEVEL)}")
        return [self._nodes[i] for i in sorted(self._nodes) if self._nodes[i].axis == axis]

    def grades(self) -> list[TaxonomyNode]:
        return self.in_axis(GRADE_AXIS)

    def conditions(self) -> list[TaxonomyNode]:
        return self.in_axis(CONDITION_AXIS)

    def family_of(self, node_id: str) -> str:
        """The level-1 ancestor id of a microconstituent node."""
        node = self.node(node_id)
        if node.axis != MICROCONSTITUENT_AXIS:
            raise ValueError(
                f"{node_id!r} is on the {node.axis!r} axis; family_of applies to "
                f"{MICROCONSTITUENT_AXIS} nodes only"
            )
        return node_id.split("/", 1)[0]

    def require(self, node_ids: list[str] | tuple[str, ...], axis: str | None = None) -> None:
        """Validate a batch of ids; raises UnknownNodeError on the first bad one.

        Passing ``axis`` also rejects ids that are registered but belong to a
        different axis, which is what stops a grade id being used where a
        constituent label is expected.
        """
        for node_id in node_ids:
            node = self.node(node_id)
            if axis is not None and node.axis != axis:
                raise UnknownNodeError(
                    f"{node_id!r} is on the {node.axis!r} axis, expected {axis!r}"
                )


def _parse_yaml(text: str) -> dict:
    import yaml  # local import: the only YAML touchpoint in the codebase

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("taxonomy file must be a mapping of family -> definition")
    return data


def _build_nodes(raw: dict) -> dict[str, TaxonomyNode]:
    nodes: dict[str, TaxonomyNode] = {}

    def walk(key: str, spec: dict, parent: str | None, level: int, axis: str) -> str:
        if not isinstance(spec, dict):
            raise ValueError(f"taxonomy node {key!r} must be a mapping, got {type(spec).__name__}")
        if parent is None:
            axis = str(spec.get("axis", MICROCONSTITUENT_AXIS))
            if axis not in AXIS_MAX_LEVEL:
                raise ValueError(
                    f"taxonomy root {key!r} declares unknown axis {axis!r}; "
                    f"known axes: {sorted(AXIS_MAX_LEVEL)}"
                )
        elif "axis" in spec:
            raise ValueError(
                f"taxonomy node {key!r} sets 'axis', but only root nodes may; "
                "descendants inherit their root's axis"
            )
        if level > AXIS_MAX_LEVEL[axis]:
            raise ValueError(
                f"taxonomy nesting too deep at {key!r}: the {axis!r} axis allows "
                f"{AXIS_MAX_LEVEL[axis]} levels"
            )
        node_id = key if parent is None else f"{parent}/{key}"
        children_spec = spec.get("children", {}) or {}
        child_ids = tuple(walk(k, v, node_id, level + 1, axis) for k, v in children_spec.items())
        nodes[node_id] = TaxonomyNode(
            id=node_id,
            name=str(spec.get("name", key)),
            level=level,
            parent=parent,
            children=child_ids,
            axis=axis,
        )
        return node_id

    for root_key, root_spec in raw.items():
        walk(str(root_key), root_spec, None, FAMILY_LEVEL, MICROCONSTITUENT_AXIS)
    if not nodes:
        raise ValueError("taxonomy file defines no nodes")
    return nodes
