"""Integration test for the real pf_cuda_cv CUDA particle filter (skipped without GPU)."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_estimation_runs_with_real_cuda_particle_filter(monkeypatch):
    """End-to-end module run against the real CUDA particle filter."""
    pytest.importorskip("pf_cuda_cv")

    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        from src.subsystems.estimation.full_state import pf as pf_module
        from src.types.autoaim import ArmorPanel, EnemyRobot, FullStateAutoAimContext

        monkeypatch.setattr(pf_module.time, "perf_counter", lambda: 10.0)

        estimator = pf_module._default_particle_filter()

        panel = ArmorPanel(
            icon=None,
            position=np.array([100.0, 120.0, 0.0], dtype=np.float32),
            orientation=None,
            bbx=None,
            contour=None,
            yaw=0.15,
        )
        ctx = FullStateAutoAimContext(
            target_robot=EnemyRobot(name="sentry", panels=[panel]),
            frame_ts=10.05,
            new_observation=True,
        )

        module = pf_module.ParticleFilterEstimationModule(ctx)
        module.set_estimator(estimator)
        result_ctx = module.run()

        assert result_ctx.estimate is not None
        value = np.asarray(result_ctx.estimate.value)
        assert value.shape == (6,)
        assert np.all(np.isfinite(value))
        assert np.isfinite(result_ctx.estimate.timestamp)
        assert 0.0 <= result_ctx.estimate.confidence <= 1.0
    finally:
        sys.argv = original_argv
