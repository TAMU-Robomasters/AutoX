from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_estimation_runs_with_real_cupy_particle_filter(monkeypatch):
    cp = pytest.importorskip("cupy")

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device available")
    except Exception as exc:
        pytest.skip(f"CUDA runtime unavailable: {exc}")

    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        from src.subsystems import estimation as estimation_module
        from src.subsystems.particle_filter import ParticleFilter
        from src.types.autoaim import ArmorPanel, EnemyRobot, ParticleFilterAutoAimContext

        cp.random.seed(0)

        Q = cp.diag(cp.array([1, 1, 1, 1, 0.1, 0.1, 0.1], dtype=cp.float32))
        R = cp.diag(cp.array([5, 5, 1], dtype=cp.float32))
        prior = cp.array([0, 0, 0, 0, 0, 0, 21], dtype=cp.float32)
        pf = ParticleFilter(num_particles=2048, Q=Q, R=R, prior=prior, num_meas=3)

        perf_counter_values = iter([10.0, 10.05, 10.1])
        monkeypatch.setattr(
            estimation_module.time, "perf_counter", lambda: next(perf_counter_values)
        )

        panel = ArmorPanel(
            icon=None,
            position=np.array([100.0, 120.0, 0.0], dtype=np.float32),
            orientation=None,
            bbx=None,
            contour=None,
            yaw=0.15,
        )
        ctx = ParticleFilterAutoAimContext(
            target_robot=EnemyRobot(name="sentry", panels=[panel])
        )

        module = estimation_module.ParticleFilterEstimationModule(ctx)
        module.set_particle_filter(pf)
        result_ctx = module.run()

        assert result_ctx.estimate is not None
        assert isinstance(result_ctx.estimate.value, cp.ndarray)
        assert result_ctx.estimate.value.shape == (7,)

        estimate_host = cp.asnumpy(result_ctx.estimate.value)
        assert np.all(np.isfinite(estimate_host))
        assert np.isfinite(result_ctx.estimate.timestamp)
        assert 0.0 <= result_ctx.estimate.confidence <= 1.0
    finally:
        sys.argv = original_argv
