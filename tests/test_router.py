import numpy as np
import pytest

from microhard.router import ConformalAbstainer, train_router


class TestConformalAbstainer:
    def test_confident_singleton(self) -> None:
        abstainer = ConformalAbstainer(alpha=0.1)
        rng = np.random.default_rng(0)
        # calibration: model is right and confident (p_true ~ 0.9)
        n, c = 40, 3
        probs = np.full((n, c), 0.05)
        true = rng.integers(0, c, n)
        probs[np.arange(n), true] = 0.9
        abstainer.calibrate(probs, true)

        confident = np.array([0.92, 0.05, 0.03])
        assert list(abstainer.prediction_set(confident)) == [0]

    def test_uncertain_input_abstains(self) -> None:
        abstainer = ConformalAbstainer(alpha=0.1)
        n, c = 40, 3
        probs = np.full((n, c), 0.05)
        true = np.zeros(n, dtype=int)
        probs[np.arange(n), true] = 0.9
        abstainer.calibrate(probs, true)

        uniform = np.array([1 / 3, 1 / 3, 1 / 3])
        prediction_set = abstainer.prediction_set(uniform)
        assert len(prediction_set) != 1  # empty or multiple -> "unknown family"

    def test_requires_calibration(self) -> None:
        with pytest.raises(RuntimeError, match="calibrate"):
            ConformalAbstainer(alpha=0.1).prediction_set(np.array([0.5, 0.5]))
        with pytest.raises(ValueError):
            ConformalAbstainer(alpha=0.1).calibrate(np.zeros((0, 2)), np.zeros(0, dtype=int))

    def test_alpha_validated(self) -> None:
        with pytest.raises(ValueError):
            ConformalAbstainer(alpha=1.5)


def test_train_router_two_families_smoke(seg_benchmark, aluminum_stub) -> None:
    cfg = seg_benchmark  # same Config object as aluminum_stub (both build on `cfg`)
    cfg.adapters = ["uhcs", "micronet_al"]
    cfg.router_calib_frac = 0.4
    checkpoint = train_router(cfg)
    assert checkpoint.exists()

    from microhard.router import load_router

    model, families, abstainer = load_router(cfg)
    assert families == ["aluminum", "ferrous"]
    assert abstainer.qhat is not None
