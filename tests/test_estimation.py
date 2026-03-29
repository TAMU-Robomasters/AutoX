import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Allow importing modules that depend on cupy in environments without GPU/cupy.
if importlib.util.find_spec("cupy") is None:
    fake_cupy = types.SimpleNamespace(
        ndarray=np.ndarray,
        float32=np.float32,
        array=lambda data, dtype=None: np.array(data, dtype=dtype),
        diag=lambda data: np.diag(data),
    )
    sys.modules.setdefault("cupy", fake_cupy)


class _FakeParticleFilter:
    def __init__(self):
        self.reinit_prior = None
        self.update_calls = []

    @staticmethod
    def _to_numpy(array_like):
        if hasattr(array_like, "get"):
            return np.asarray(array_like.get(), dtype=np.float32)
        return np.asarray(array_like, dtype=np.float32)

    def reinit(self, prior):
        self.reinit_prior = self._to_numpy(prior)

    def update(self, current_time, measurement):
        self.update_calls.append((float(current_time), self._to_numpy(measurement)))
        estimate = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
        confidence = 0.75
        return estimate, confidence


def test_run_initializes_filter_and_returns_estimate(monkeypatch):
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        from src.subsystems import estimation as estimation_module
        from src.types.autoaim import ArmorPanel, EnemyRobot, ParticleFilterAutoAimContext

        ParticleFilterEstimationModule = estimation_module.ParticleFilterEstimationModule

        fake_pf = _FakeParticleFilter()

        # Called 3 times on first valid run: start_loop_time, dt calc, timestamp.
        perf_counter_values = iter([100.0, 100.4, 101.0])
        monkeypatch.setattr(
            estimation_module.time, "perf_counter", lambda: next(perf_counter_values)
        )

        panel = ArmorPanel(
            icon=None,
            position=np.array([10.0, 20.0, 0.0], dtype=np.float32),
            orientation=None,
            bbx=None,
            contour=None,
            yaw=0.3,
        )
        target_robot = EnemyRobot(name="sentry", panels=[panel])

        ctx = ParticleFilterAutoAimContext(target_robot=target_robot)
        module = ParticleFilterEstimationModule(ctx)
        module.set_particle_filter(fake_pf)

        result_ctx = module.run()

        assert fake_pf.reinit_prior is not None
        np.testing.assert_allclose(
            fake_pf.reinit_prior,
            np.array([10.0, 20.0, 0.0, 0.0, 0.3, 0.0, 21.0], dtype=np.float32),
        )

        assert len(fake_pf.update_calls) == 1
        dt, measurement = fake_pf.update_calls[0]
        assert dt == pytest.approx(0.4)
        np.testing.assert_allclose(
            measurement, np.array([10.0, 20.0, 0.3], dtype=np.float32)
        )

        assert result_ctx.estimate is not None
        np.testing.assert_allclose(
            result_ctx.estimate.value,
            np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32),
        )
        assert result_ctx.estimate.confidence == 0.75
        assert result_ctx.estimate.timestamp == 101.0
    finally:
        sys.argv = original_argv
