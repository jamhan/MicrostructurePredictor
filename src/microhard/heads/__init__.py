"""Property-head registry.

Register a head class for a (scope, property) pair::

    from microhard.heads import register
    register("ferrous/uhcs", "hardness_hv", HardnessHead)

Scopes are ``family`` or ``family/adapter``. Lookup falls back from the most
specific scope to the bare family, so a family-wide head serves adapters that
lack a dedicated one. ``predict`` abstains when no head resolves.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config
from .base import PropertyHead

_HEADS: dict[tuple[str, str], type[PropertyHead]] = {}


def register(scope: str, property_name: str, head_cls: type[PropertyHead]) -> None:
    if not issubclass(head_cls, PropertyHead):
        raise TypeError(f"{head_cls.__name__} must subclass PropertyHead")
    head_cls.scope = scope
    head_cls.property_name = property_name
    _HEADS[(scope, property_name)] = head_cls


def registered() -> list[tuple[str, str]]:
    return sorted(_HEADS)


def _scope_fallbacks(scope: str) -> list[str]:
    """('ferrous/uhcs') -> ['ferrous/uhcs', 'ferrous']"""
    parts = scope.split("/")
    return ["/".join(parts[: i + 1]) for i in range(len(parts) - 1, -1, -1)]


def head_class(scope: str, property_name: str) -> type[PropertyHead] | None:
    for candidate in _scope_fallbacks(scope):
        if (candidate, property_name) in _HEADS:
            return _HEADS[(candidate, property_name)]
    return None


def heads_for_family(family: str) -> list[tuple[str, str]]:
    """(scope, property) pairs whose scope belongs to the given family."""
    return sorted(key for key in _HEADS if key[0].split("/")[0] == family)


def fitted_path(cfg: Config, scope: str, property_name: str) -> Path:
    return cfg.heads_dir / f"{scope.replace('/', '__')}--{property_name}.pkl"


def load_fitted(cfg: Config, scope: str, property_name: str) -> PropertyHead | None:
    """The fitted head for a scope (walking the fallback chain), or None."""
    for candidate in _scope_fallbacks(scope):
        path = fitted_path(cfg, candidate, property_name)
        if path.exists():
            return PropertyHead.load(path)
    return None


# Import concrete heads so registration happens on package import.
from . import hardness as _hardness  # noqa: E402,F401
