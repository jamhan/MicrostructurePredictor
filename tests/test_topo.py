import numpy as np
import pytest

from microhard import topo


def test_optional_dependency_behaviour() -> None:
    """Works when a PH backend is installed; raises a helpful error otherwise."""
    mask = np.zeros((16, 16), dtype=np.int64)
    mask[4:8, 4:8] = 1
    class_nodes = ("ferrous/matrix", "ferrous/spheroidite")
    if topo.HAS_TOPO:
        feats = topo.image_topo_features(mask, class_nodes)
        assert any(key.startswith("topo_h0_count:ferrous/") for key in feats)
        assert all(np.isfinite(v) for v in feats.values())
    else:
        with pytest.raises(ImportError, match="topo"):
            topo.image_topo_features(mask, class_nodes)
