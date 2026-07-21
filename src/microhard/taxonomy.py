"""Hierarchical label registry: family -> constituent -> morphology.

Loaded from a taxonomy file (YAML by default; TOML and JSON also accepted by
extension). Node ids are path-style — ``ferrous``, ``ferrous/pearlite``,
``ferrous/pearlite/lamellar`` — and every label used anywhere in the pipeline
must be a registered node id, never a bare string.

``Taxonomy.load(None)`` loads the bundled seed file
(src/microhard/taxonomy.yaml).
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

FAMILY_LEVEL = 1
CONSTITUENT_LEVEL = 2
MORPHOLOGY_LEVEL = 3
MAX_LEVEL = 3


@dataclass(frozen=True)
class TaxonomyNode:
    id: str  # path-style: "ferrous/pearlite/lamellar"
    name: str  # human-readable
    level: int  # 1=family, 2=constituent, 3=morphology
    parent: str | None
    children: tuple[str, ...]


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
        """Level-1 nodes — what the router classifies over."""
        return [n for n in self._nodes.values() if n.level == FAMILY_LEVEL]

    def children(self, node_id: str) -> list[TaxonomyNode]:
        return [self._nodes[c] for c in self.node(node_id).children]

    def family_of(self, node_id: str) -> str:
        """The level-1 ancestor id of any node."""
        self.node(node_id)  # validates
        return node_id.split("/", 1)[0]

    def require(self, node_ids: list[str] | tuple[str, ...]) -> None:
        """Validate a batch of ids; raises UnknownNodeError on the first bad one."""
        for node_id in node_ids:
            self.node(node_id)


def _parse_yaml(text: str) -> dict:
    import yaml  # local import: the only YAML touchpoint in the codebase

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("taxonomy file must be a mapping of family -> definition")
    return data


def _build_nodes(raw: dict) -> dict[str, TaxonomyNode]:
    nodes: dict[str, TaxonomyNode] = {}

    def walk(key: str, spec: dict, parent: str | None, level: int) -> str:
        if level > MAX_LEVEL:
            raise ValueError(
                f"taxonomy nesting too deep at {key!r}: max is family/constituent/morphology"
            )
        if not isinstance(spec, dict):
            raise ValueError(f"taxonomy node {key!r} must be a mapping, got {type(spec).__name__}")
        node_id = key if parent is None else f"{parent}/{key}"
        children_spec = spec.get("children", {}) or {}
        child_ids = tuple(walk(k, v, node_id, level + 1) for k, v in children_spec.items())
        nodes[node_id] = TaxonomyNode(
            id=node_id,
            name=str(spec.get("name", key)),
            level=level,
            parent=parent,
            children=child_ids,
        )
        return node_id

    for family_key, family_spec in raw.items():
        walk(str(family_key), family_spec, parent=None, level=FAMILY_LEVEL)
    if not nodes:
        raise ValueError("taxonomy file defines no nodes")
    return nodes
