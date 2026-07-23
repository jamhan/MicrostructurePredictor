"""Dataset adapter registry.

Adapters translate a dataset's native layout into canonical records
(records.CanonicalRecord). Register a new one with the ``@register_adapter``
decorator; enable it per-run via ``Config.adapters``.
"""

from __future__ import annotations

from ..config import Config
from ..taxonomy import Taxonomy

_REGISTRY: dict[str, type] = {}


def register_adapter(cls):
    """Class decorator: adds a BaseAdapter subclass to the registry by name."""
    if not getattr(cls, "name", None) or not getattr(cls, "family", None):
        raise ValueError(f"{cls.__name__} must define class attributes 'name' and 'family'")
    _REGISTRY[cls.name] = cls
    return cls


def available_adapters() -> list[str]:
    return sorted(_REGISTRY)


def get_adapter(name: str, cfg: Config, taxonomy: Taxonomy):
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"No adapter named {name!r}; available: {available_adapters()}"
        ) from None
    return cls(cfg, taxonomy)


def enabled_adapters(cfg: Config, taxonomy: Taxonomy) -> list:
    return [get_adapter(name, cfg, taxonomy) for name in cfg.adapters]


# Import concrete adapters so registration happens on package import.
from . import folder as _folder  # noqa: E402,F401
from . import godec_in718 as _godec_in718  # noqa: E402,F401
from . import literature as _literature  # noqa: E402,F401
from . import uhcs as _uhcs  # noqa: E402,F401
from .base import BaseAdapter  # noqa: E402,F401

# Imported after BaseAdapter is defined: experimental_campaign imports the
# registry decorator and subclasses BaseAdapter.
from .. import experimental_campaign as _experimental_campaign  # noqa: E402,F401
