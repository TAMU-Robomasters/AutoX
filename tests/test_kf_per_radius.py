"""Tests for per-parity radii: PositionKF per-measurement r + kf.py module stamping."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeKF:
    """FullStateKF stand-in recording update calls (incl. the r kwarg)."""

    def __init__(self):
        self.reinit_prior = None
        self.update_calls = []

    def reinit(self, prior):
        self.reinit_prior = np.asarray(prior, dtype=np.float32)

    def update(self, dt, measurements, r=None, z_meas=None):
        self.update_calls.append((float(dt), np.asarray(measurements), r, z_meas))
        # Echo the measured mid-height at index 6 so aim_z (filtered z) tests see it.
        z = float(z_meas) if z_meas is not None else 0.0
        return np.array([0, 200, 0, 0, 0.5, 1.0, z], dtype=np.float32), 0.9

    def accel(self):
        return None

    def update_with_no_observation(self, dt):
        return np.zeros(7, dtype=np.float32), 1.0


def _panel(x, y, z, yaw, panel_id=None):
    from src.types.autoaim import ArmorPanel

    return ArmorPanel(
        icon=None,
        position=np.array([x, y, z], dtype=np.float32),
        orientation=None,
        bbx=None,
        contour=None,
        yaw=yaw,
        id=panel_id,
    )


def _module(target_robot, estimator, frame_ts=10.0):
    from src.subsystems.estimation.full_state.kf import KalmanFilterEstimationModule
    from src.types.autoaim import FullStateAutoAimContext

    ctx = FullStateAutoAimContext(
        target_robot=target_robot, frame_ts=frame_ts, new_observation=True
    )
    module = KalmanFilterEstimationModule(ctx)
    module.set_estimator(estimator)
    return module


def test_position_kf_per_measurement_radius():
    """Observations of two parities back-project to the same centre with their own radii."""
    sys.argv = [sys.argv[0]]
    from src.subsystems.estimation.filters import PositionKF
    from src.subsystems.estimation.motion_models import ConstantVelocity

    center = np.array([40.0, 250.0])
    theta = 0.6
    r_even, r_odd = 30.0, 20.0

    kf = PositionKF(
        motion_model=ConstantVelocity(q_vx=1.0, q_vy=1.0),
        r_pos=1.0,
        r=23.5,
        init_std=(50, 50, 5, 5),
    )
    kf.reinit(center[0], center[1])

    for k, r in [(0, r_even), (1, r_odd), (2, r_even), (3, r_odd)]:
        angle = theta + k * np.pi / 2
        obs = center + r * np.array([np.cos(angle), np.sin(angle)])
        estimate, _ = kf.update(0.01, obs[0], obs[1], angle, r=r)

    np.testing.assert_allclose(estimate[:2], center, atol=0.5)
    np.testing.assert_allclose(estimate[2:], 0.0, atol=2.0)  # stationary


def test_position_kf_default_radius_preserved():
    """r=None keeps the constructor radius (archived-engine behavior)."""
    sys.argv = [sys.argv[0]]
    from src.subsystems.estimation.filters import PositionKF
    from src.subsystems.estimation.motion_models import ConstantVelocity

    center = np.array([0.0, 100.0])
    kf = PositionKF(
        motion_model=ConstantVelocity(q_vx=1.0, q_vy=1.0),
        r_pos=1.0,
        r=23.5,
        init_std=(50, 50, 5, 5),
    )
    kf.reinit(center[0], center[1])

    obs = center + 23.5 * np.array([np.cos(-np.pi / 2), np.sin(-np.pi / 2)])
    estimate, _ = kf.update(0.01, obs[0], obs[1], -np.pi / 2)  # no r kwarg
    np.testing.assert_allclose(estimate[:2], center, atol=0.5)


def test_module_passes_parity_radius_to_estimator():
    """The estimator receives r_even for even ids and r_odd for odd ids."""
    sys.argv = [sys.argv[0]]
    from src.types.autoaim import EnemyRobot

    for panel_id, expected_r in [(0, 30.0), (2, 30.0), (1, 20.0), (3, 20.0)]:
        fake = _FakeKF()
        robot = EnemyRobot(name="standard", panels=[_panel(0, 100, 5, 0.0, panel_id)])
        module = _module(robot, fake)
        module.set_panel_radii(30.0, 20.0)
        module.run()
        assert fake.update_calls[0][2] == expected_r, f"panel id {panel_id}"


def test_module_unknown_id_uses_mean_radius_and_stamps_output():
    """id=None back-projects with the mean radius; a/b radii stamped from setters."""
    sys.argv = [sys.argv[0]]
    from src.types.autoaim import EnemyRobot

    fake = _FakeKF()
    robot = EnemyRobot(name="standard", panels=[_panel(0, 100, 5, 0.0, None)])
    module = _module(robot, fake)
    module.set_panel_radii(30.0, 20.0)
    ctx = module.run()

    assert fake.update_calls[0][2] == pytest.approx(25.0)
    assert ctx.estimate.a_radius == 30.0
    assert ctx.estimate.b_radius == 20.0


def test_aim_z_requires_geometry_and_uses_parity_sign():
    """aim_z is None before set_aim_geometry; afterwards low pair aims up, high down."""
    sys.argv = [sys.argv[0]]
    from src.types.autoaim import EnemyRobot

    # Before set_aim_geometry: no aim_z.
    fake = _FakeKF()
    robot = EnemyRobot(name="standard", panels=[_panel(0, 100, 10.0, 0.0, 0)])
    module = _module(robot, fake)
    assert module.run().estimate.aim_z is None

    # Low-parity (even) panel at z=10, delta=6 -> aim at 13 (mid-height).
    fake = _FakeKF()
    robot = EnemyRobot(name="standard", panels=[_panel(0, 100, 10.0, 0.0, 0)])
    module = _module(robot, fake)
    module.set_aim_geometry(height_delta=6.0, low_parity=0)
    assert module.run().estimate.aim_z == pytest.approx(13.0)

    # High-parity (odd) panel at z=16 -> aim at 13 as well.
    fake = _FakeKF()
    robot = EnemyRobot(name="standard", panels=[_panel(0, 100, 16.0, 0.0, 1)])
    module = _module(robot, fake)
    module.set_aim_geometry(height_delta=6.0, low_parity=0)
    assert module.run().estimate.aim_z == pytest.approx(13.0)


def test_module_prefers_id_bearing_panel():
    """With a closer id-less panel and a farther id-bearing one, the latter is tracked."""
    sys.argv = [sys.argv[0]]
    from src.types.autoaim import EnemyRobot

    fake = _FakeKF()
    near_no_id = _panel(0, 90, 5, 0.0, None)
    far_with_id = _panel(0, 120, 5, 0.1, 1)
    robot = EnemyRobot(name="standard", panels=[near_no_id, far_with_id])
    module = _module(robot, fake)
    module.set_panel_radii(30.0, 20.0)
    module.run()

    _, measurement, r, _z = fake.update_calls[0]
    assert measurement[0][1] == pytest.approx(120.0)
    assert r == 20.0  # odd parity


def test_reinit_from_panel_suppresses_auto_reinit():
    """reinit_from_panel seeds the prior and the next run() does not reinit again."""
    sys.argv = [sys.argv[0]]
    from src.types.autoaim import EnemyRobot

    fake = _FakeKF()
    panel = _panel(10, 100, 5, 0.3, 0)
    robot = EnemyRobot(name="standard", panels=[panel])
    module = _module(robot, fake)

    module.reinit_from_panel(panel)
    first_prior = fake.reinit_prior.copy()
    # 7-D prior; no aim geometry set -> z0 falls back to the panel z (5).
    np.testing.assert_allclose(first_prior, [10, 100, 0, 0, 0.3, 0, 5])

    module.run()
    np.testing.assert_allclose(fake.reinit_prior, first_prior)  # unchanged
    assert len(fake.update_calls) == 1
