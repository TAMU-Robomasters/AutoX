"""Unit tests for ParticleFilterEstimationModule using a fake estimator (no GPU)."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeEstimator:
    """Minimal FullStateEstimator stand-in recording reinit/update calls."""

    def __init__(self):
        self.reinit_prior = None
        self.update_calls = []

    def reinit(self, prior):
        self.reinit_prior = np.asarray(prior, dtype=np.float32)

    def update(self, dt, measurements):
        self.update_calls.append((float(dt), np.asarray(measurements, dtype=np.float32)))
        estimate = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
        confidence = 0.75
        return estimate, confidence

    def update_with_no_observation(self, dt):
        return np.zeros(6, dtype=np.float32), 1.0


def test_run_initializes_filter_and_returns_estimate(monkeypatch):
    """Module reinits the estimator from the first panel and emits its estimate."""
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        from src.subsystems.estimation.full_state import pf as pf_module
        from src.types.autoaim import ArmorPanel, EnemyRobot, FullStateAutoAimContext

        monkeypatch.setattr(pf_module.time, "perf_counter", lambda: 100.0)

        fake = _FakeEstimator()

        panel = ArmorPanel(
            icon=None,
            position=np.array([10.0, 20.0, 0.0], dtype=np.float32),
            orientation=None,
            bbx=None,
            contour=None,
            yaw=0.3,
        )
        target_robot = EnemyRobot(name="sentry", panels=[panel])

        ctx = FullStateAutoAimContext(
            target_robot=target_robot,
            frame_ts=100.4,
            new_observation=True,
        )
        module = pf_module.ParticleFilterEstimationModule(ctx)
        module.set_estimator(fake)

        result_ctx = module.run()

        # Reinit prior: [x, y, vx, vy, theta, omega] seeded from the first panel.
        assert fake.reinit_prior is not None
        np.testing.assert_allclose(
            fake.reinit_prior,
            np.array([10.0, 20.0, 0.0, 0.0, 0.3, 0.0], dtype=np.float32),
        )

        # One observation update with dt = frame_ts - reinit clock.
        assert len(fake.update_calls) == 1
        dt, measurements = fake.update_calls[0]
        assert dt == pytest.approx(0.0)  # reinit resets the clock to frame_ts
        np.testing.assert_allclose(
            measurements, np.array([[10.0, 20.0, 0.3]], dtype=np.float32)
        )

        assert result_ctx.estimate is not None
        np.testing.assert_allclose(
            result_ctx.estimate.value,
            np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        )
        assert result_ctx.estimate.confidence == 0.75
        assert result_ctx.estimate.timestamp == 100.4
    finally:
        sys.argv = original_argv


def test_no_target_returns_none():
    """No target robot -> estimate stays None."""
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        from src.subsystems.estimation.full_state import pf as pf_module
        from src.types.autoaim import FullStateAutoAimContext

        ctx = FullStateAutoAimContext(target_robot=None)
        module = pf_module.ParticleFilterEstimationModule(ctx)
        module.set_estimator(_FakeEstimator())

        result_ctx = module.run()
        assert result_ctx.estimate is None
    finally:
        sys.argv = original_argv
