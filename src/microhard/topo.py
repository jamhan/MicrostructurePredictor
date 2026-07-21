"""Optional persistent-homology features from binarized segmentation masks.

Alternative/additional feature set for hardness.py. Backends (auto-detected):

- ``cripser`` (Cubical Ripser) — the ``topo`` extra: ``uv sync --extra topo``
- ``giotto-tda`` — also supported if installed separately (not in the extra:
  it pins scikit-learn==1.3.2, which conflicts with this project's >=1.4)

Everything raises a clear ImportError when neither backend is present; the
rest of the pipeline never imports this module implicitly.

TODO(scaffold): persistence on a raw 0/1 mask is coarse (all finite bars have
persistence <= 1). Running cubical persistence on a signed distance transform
of the mask is the standard upgrade and slots in here.
"""

from __future__ import annotations

import numpy as np

BACKEND: str | None = None
try:
    from gtda.homology import CubicalPersistence  # type: ignore[import-not-found]

    BACKEND = "gtda"
except ImportError:
    try:
        import cripser  # type: ignore[import-not-found]

        BACKEND = "cripser"
    except ImportError:
        pass

HAS_TOPO = BACKEND is not None

_HOMOLOGY_DIMS = (0, 1)
_CRIPSER_INF = 1e300  # cripser marks essential classes with a huge death value


def require_backend() -> None:
    if not HAS_TOPO:
        raise ImportError(
            "No persistent-homology backend installed; topological features are "
            "unavailable. Install with: uv sync --extra topo  (Cubical Ripser)"
        )


def _diagram(binary: np.ndarray) -> np.ndarray:
    """Persistence diagram as (n_points, 3) rows of (birth, death, dim)."""
    if BACKEND == "gtda":
        return CubicalPersistence(homology_dimensions=list(_HOMOLOGY_DIMS)).fit_transform(
            binary[None]
        )[0]
    # cripser.computePH rows: (dim, birth, death, x1, y1, z1, x2, y2, z2)
    ph = cripser.computePH(binary, maxdim=max(_HOMOLOGY_DIMS))
    return np.stack([ph[:, 1], ph[:, 2], ph[:, 0]], axis=1)


def topo_features_for_class(mask: np.ndarray, class_idx: int) -> dict[str, float]:
    """H0/H1 summary stats (count, total & max persistence) for one class."""
    require_backend()
    binary = (mask == class_idx).astype(np.float64)
    diagram = _diagram(binary)
    feats: dict[str, float] = {}
    for dim in _HOMOLOGY_DIMS:
        points = diagram[diagram[:, 2] == dim]
        persistence = points[:, 1] - points[:, 0]
        persistence = persistence[
            np.isfinite(persistence) & (persistence > 0) & (persistence < _CRIPSER_INF)
        ]
        feats[f"h{dim}_count"] = float(len(persistence))
        feats[f"h{dim}_total_persistence"] = float(persistence.sum()) if len(persistence) else 0.0
        feats[f"h{dim}_max_persistence"] = float(persistence.max()) if len(persistence) else 0.0
    return feats


def image_topo_features(mask: np.ndarray, class_nodes: tuple[str, ...]) -> dict[str, float]:
    """Flat topological features for one mask, keyed by taxonomy node id
    (``topo_h0_count:ferrous/network`` …) — merge into a FeatureVector with
    ``FeatureVector.merged_with``."""
    require_backend()
    feats: dict[str, float] = {}
    for i, node in enumerate(class_nodes):
        for key, value in topo_features_for_class(mask, i).items():
            feats[f"topo_{key}:{node}"] = value
    return feats
