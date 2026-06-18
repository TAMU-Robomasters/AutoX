"""Regression tests for the canonical ballistic coordinate convention.

These lock in the convention documented in src/subsystems/ballistics/solver.py:
internal angles from +x CCW (matching the estimator back-projection), MCU yaw
(0 at +y) only at the output boundary. The panel-normal cases fail against the
old [sin, cos] normals, which flipped handedness (and therefore omega's sign).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# quik-config parses sys.argv at the first config import (the ballistic modules
# read config); strip pytest's CLI args before any src import can trigger that.
sys.argv = [sys.argv[0]]

from src.subsystems.ballistics.solver import (  # noqa: E402
    init_guess,
    make_f_no_spin,
    mcu_yaw_from_xy,
    panel_normals,
    select_panel,
    select_panel_at_time,
    solve_no_spin,
)

GRAVITY = -9.81  # m/s^2
SPEED = 23.0     # m/s
NO_BARREL = np.zeros(3)


def test_mcu_yaw_from_xy():
    """yaw_mcu is 0 at +y, -90deg at +x, +90deg at -x, wrapped to [-pi, pi]."""
    assert mcu_yaw_from_xy(0.0, 1.0) == pytest.approx(0.0)
    assert mcu_yaw_from_xy(1.0, 0.0) == pytest.approx(-np.pi / 2)
    assert mcu_yaw_from_xy(-1.0, 0.0) == pytest.approx(np.pi / 2)
    assert abs(mcu_yaw_from_xy(0.0, -1.0)) == pytest.approx(np.pi)


def test_panel_normals_are_ccw_from_plus_x():
    """Row k is [cos, sin] of theta + k*90deg -- estimator convention, CCW."""
    normals = panel_normals(0.0)
    np.testing.assert_allclose(normals[0], [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(normals[1], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(normals[2], [-1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(normals[3], [0.0, -1.0], atol=1e-12)

    theta = 0.7
    normals = panel_normals(theta)
    for k in range(4):
        angle = theta + k * np.pi / 2
        np.testing.assert_allclose(normals[k], [np.cos(angle), np.sin(angle)], atol=1e-12)


@pytest.mark.parametrize("k", range(4))
@pytest.mark.parametrize("bearing", [0.3, 2.0, -1.4])
def test_select_panel_picks_the_facing_panel(k, bearing):
    """If panel k's outward normal points straight at the turret, select_panel returns k."""
    p_t = 5.0 * np.array([np.cos(bearing), np.sin(bearing), 0.0])
    facing_angle = np.arctan2(-p_t[1], -p_t[0])  # robot-center -> turret direction
    theta = facing_angle - k * np.pi / 2

    assert select_panel(p_t, theta) == k


def test_select_panel_at_time_uses_extrapolated_pose():
    """Panel choice follows theta + omega*t, with omega's true (CCW-positive) sign."""
    p_t = np.array([0.0, 5.0, 0.0])
    facing_angle = -np.pi / 2  # target straight ahead -> facing is -y
    # Panel 1 will face us in t=1s if omega rotates panel 1's normal onto facing.
    omega = 0.5
    theta = facing_angle - np.pi / 2 - omega * 1.0  # panel 1 aligned at t=1
    assert select_panel_at_time(p_t, np.zeros(3), theta, omega, 1.0) == 1


def test_solve_no_spin_stationary_target_yaw_matches_mcu_convention():
    """For a stationary target with no barrel offset, solver yaw == atan2(y,x) - 90deg."""
    for x, y in [(0.0, 5.0), (3.0, 4.0), (-2.0, 6.0)]:
        p_t = np.array([x, y, -0.3])
        result = solve_no_spin(NO_BARREL, SPEED, GRAVITY, p_t, np.zeros(3))
        assert result["success"], result
        assert result["yaw"] == pytest.approx(mcu_yaw_from_xy(x, y), abs=1e-6)
        assert result["time"] > 0


def test_solve_no_spin_moving_target_leads_and_hits():
    """The solution intercepts the extrapolated target point exactly (residual ~ 0)."""
    p_t = np.array([1.0, 6.0, -0.2])
    v_t = np.array([1.5, 0.0, 0.0])  # crossing left-to-right
    result = solve_no_spin(NO_BARREL, SPEED, GRAVITY, p_t, v_t)
    assert result["success"]

    # Lead check: the yaw must point at the future position, not the current one.
    t = result["time"]
    future = p_t + v_t * t
    assert result["yaw"] == pytest.approx(mcu_yaw_from_xy(future[0], future[1]), abs=2e-2)
    assert result["yaw"] < mcu_yaw_from_xy(p_t[0], p_t[1])  # leads toward +x => more negative

    # And the residual function evaluates to ~zero at the solution.
    f = make_f_no_spin(NO_BARREL, SPEED, GRAVITY, p_t, v_t)
    np.testing.assert_allclose(f(np.array([result["yaw"], result["pitch"], t])), 0, atol=1e-4)


def test_solve_no_spin_with_barrel_offset_still_hits():
    """Nonzero barrel tip offset shifts the muzzle, not the impact point."""
    b = np.array([0.0325, 0.150, 0.0])
    p_t = np.array([2.0, 7.0, -0.3])
    result = solve_no_spin(b, SPEED, GRAVITY, p_t, np.zeros(3))
    assert result["success"]
    f = make_f_no_spin(b, SPEED, GRAVITY, p_t, np.zeros(3))
    np.testing.assert_allclose(
        f(np.array([result["yaw"], result["pitch"], result["time"]])), 0, atol=1e-4
    )


def test_init_guess_yaw_is_mcu_convention():
    """The initial guess starts the solver in the same yaw convention it solves in."""
    p_t = np.array([3.0, 4.0, 0.0])
    guess = init_guess(p_t, SPEED, GRAVITY)
    assert guess[0] == pytest.approx(mcu_yaw_from_xy(3.0, 4.0))
    assert guess[2] > 0


# ---------------------------------------------------------------------------
# Module-level tests (full-state shot timing / continuous fire)
# ---------------------------------------------------------------------------


class _FixedPredictor:
    """Holds a fixed state used to stub the static ``predict_ahead``.

    The ballistic modules extrapolate via ``FullStateKF.predict_ahead(state, dt)``
    (no estimator instance). The helpers below monkeypatch that static method to
    return this fixed state regardless of ``dt``, so the alignment/aim math is
    tested in isolation from the constant-velocity extrapolation.
    """

    def __init__(self, state):
        self.state = np.asarray(state, dtype=np.float32)


def _estimate(x, y, theta, omega, a_radius, b_radius, aim_z, now):
    from src.types.autoaim import RobotStateEstimate

    return RobotStateEstimate(
        value=np.array([x, y, 0, 0, theta, omega], dtype=np.float32),
        timestamp=now,
        confidence=1.0,
        a_radius=a_radius,
        b_radius=b_radius,
        aim_z=aim_z,
    )


def _shot_timing_module(estimate, predictor, monkeypatch, now):
    sys.argv = [sys.argv[0]]
    from src.subsystems.ballistics import full_state_shot_timing as st
    from src.types.autoaim import EnemyRobot, FullStateAutoAimContext

    monkeypatch.setattr(st.time, "perf_counter", lambda: now)
    ctx = FullStateAutoAimContext(
        estimate=estimate, target_robot=EnemyRobot(name="standard")
    )
    module = st.FullStateShotTimingModule(ctx)
    monkeypatch.setattr(st.FullStateKF, "predict_ahead", lambda state, dt: predictor.state)
    return module, st


def test_shot_timing_alignment_depends_on_omega_sign(monkeypatch):
    """Hand-computed alignment times for +omega vs -omega (fails on old [sin,cos] normals)."""
    now = 50.0
    d, r = 300.0, 23.5
    delta = 0.3
    facing = -np.pi / 2  # target at (0, d): robot->turret direction is -y

    # omega > 0 (CCW): panel 0's normal is delta short of facing -> waits delta/omega.
    omega = 2.0
    theta0 = facing - delta
    est = _estimate(0.0, d, theta0, omega, r, r, aim_z=-30.0, now=now)
    module, _ = _shot_timing_module(est, _FixedPredictor(est.value), monkeypatch, now)
    ctx = module.run()
    assert ctx.solution is not None and ctx.solution.is_confident
    assert ctx.solution.alignment_time_ms == pytest.approx(delta / omega * 1000, abs=2)
    # Centre aim via solve_no_spin; the barrel-tip x-offset gives a tiny nonzero yaw.
    assert ctx.solution.yaw == pytest.approx(0.0, abs=0.02)  # ~dead ahead

    # omega < 0 (CW): panel 0 must back up; panel 1 arrives first after (pi/2 - delta).
    omega = -2.0
    est = _estimate(0.0, d, theta0, omega, r, r, aim_z=-30.0, now=now)
    module, _ = _shot_timing_module(est, _FixedPredictor(est.value), monkeypatch, now)
    ctx = module.run()
    expected = (np.pi / 2 - delta) / abs(omega) * 1000
    assert ctx.solution.alignment_time_ms == pytest.approx(expected, abs=2)


def test_shot_timing_aims_at_center_with_aim_z(monkeypatch):
    """Pitch/yaw aim the barrel THROUGH the robot centre at aim_z (no d - r)."""
    now = 50.0
    a_radius, b_radius = 30.0, 20.0
    aim_z = -25.0
    d = 400.0
    facing = -np.pi / 2
    delta = 0.2
    omega = 2.0
    theta0 = facing - delta - np.pi / 2

    est = _estimate(0.0, d, theta0, omega, a_radius, b_radius, aim_z, now)
    module, _ = _shot_timing_module(est, _FixedPredictor(est.value), monkeypatch, now)
    ctx = module.run()
    assert ctx.solution is not None and ctx.solution.is_confident

    # Expected pitch/yaw: solve_no_spin at the CENTRE (0, d, aim_z) in metres,
    # the same solver single_panel.py uses (no per-panel radius subtraction).
    from src.toolbox.globals import config

    b = np.asarray(config.ballistic.barrel_offset, dtype=float)
    s = config.ballistic.projectile_velocity / 100.0
    g = -(abs(config.ballistic.gravity) / 100.0)
    p_center = np.array([0.0, d, aim_z]) / 100.0
    expected = solve_no_spin(b, s, g, p_center, np.zeros(3))
    assert ctx.solution.pitch == pytest.approx(expected["pitch"], abs=1e-6)
    assert ctx.solution.yaw == pytest.approx(expected["yaw"], abs=1e-6)


def test_shot_timing_requires_aim_z(monkeypatch):
    """aim_z=None (state 1 / unanchored) -> no solution from this module."""
    now = 50.0
    est = _estimate(0.0, 300.0, 0.0, 2.0, 23.5, 23.5, aim_z=None, now=now)
    module, _ = _shot_timing_module(est, _FixedPredictor(est.value), monkeypatch, now)
    assert module.run().solution is None


def test_shot_timing_out_of_range_tracks_without_firing(monkeypatch):
    """No ballistic arc -> straight-line aim with is_confident=False."""
    now = 50.0
    est = _estimate(0.0, 20000.0, 0.0, 2.0, 23.5, 23.5, aim_z=-30.0, now=now)
    module, _ = _shot_timing_module(est, _FixedPredictor(est.value), monkeypatch, now)
    ctx = module.run()
    assert ctx.solution is not None
    assert ctx.solution.is_confident is False
    assert ctx.solution.yaw == pytest.approx(0.0, abs=1e-6)  # still aimed at target


def _continuous_fire_module(estimate, predictor, monkeypatch, now):
    sys.argv = [sys.argv[0]]
    from src.subsystems.ballistics import full_state_continuous_fire as cf
    from src.types.autoaim import EnemyRobot, FullStateAutoAimContext

    monkeypatch.setattr(cf.time, "perf_counter", lambda: now)
    ctx = FullStateAutoAimContext(
        estimate=estimate, target_robot=EnemyRobot(name="standard")
    )
    module = cf.FullStateContinuousFireModule(ctx)
    monkeypatch.setattr(cf.FullStateKF, "predict_ahead", lambda state, dt: predictor.state)
    return module, cf


def test_cascade_solve_hits_synthetic_panel_with_unequal_radii():
    """The spin solution equals a direct no-spin solve at the constructed panel position."""
    sys.argv = [sys.argv[0]]
    from src.subsystems.ballistics import full_state_continuous_fire as cf

    center = np.array([0.5, 4.0, -0.3])  # metres
    theta = 2.4
    a_radius, b_radius = 0.30, 0.20

    result = cf._cascade_solve(
        b=NO_BARREL, s=SPEED, g=GRAVITY, p_t=center, v_t=np.zeros(3),
        tht_t=theta, omg_t=0.0, a_radius=a_radius, b_radius=b_radius,
    )
    assert result["success"]

    k = result["panel"]
    r = a_radius if k % 2 == 0 else b_radius
    angle = theta + k * np.pi / 2
    panel_pos = center + r * np.array([np.cos(angle), np.sin(angle), 0.0])

    direct = solve_no_spin(NO_BARREL, SPEED, GRAVITY, panel_pos, np.zeros(3))
    assert result["yaw"] == pytest.approx(direct["yaw"], abs=1e-4)
    assert result["pitch"] == pytest.approx(direct["pitch"], abs=1e-4)


def test_cascade_solve_spinning_target_leads_the_panel():
    """With omega != 0 the solution intercepts the panel at theta + omega*t."""
    sys.argv = [sys.argv[0]]
    from src.subsystems.ballistics import full_state_continuous_fire as cf

    center = np.array([0.0, 5.0, -0.3])
    theta, omega = -np.pi / 2 - 0.1, 1.5  # panel 0 rotating toward facing
    a_radius, b_radius = 0.30, 0.20

    result = cf._cascade_solve(
        b=NO_BARREL, s=SPEED, g=GRAVITY, p_t=center, v_t=np.zeros(3),
        tht_t=theta, omg_t=omega, a_radius=a_radius, b_radius=b_radius,
    )
    assert result["success"]

    k, t = result["panel"], result["time"]
    r = a_radius if k % 2 == 0 else b_radius
    angle = theta + omega * t + k * np.pi / 2
    panel_pos = center + r * np.array([np.cos(angle), np.sin(angle), 0.0])

    direct = solve_no_spin(NO_BARREL, SPEED, GRAVITY, panel_pos, np.zeros(3))
    assert result["yaw"] == pytest.approx(direct["yaw"], abs=1e-3)
    assert result["pitch"] == pytest.approx(direct["pitch"], abs=1e-3)


def test_continuous_fire_module_solves_and_requires_aim_z(monkeypatch):
    """End-to-end module run; aim_z=None refuses to solve."""
    now = 50.0
    est = _estimate(0.0, 400.0, 0.7, 0.5, 30.0, 20.0, aim_z=-25.0, now=now)
    module, _ = _continuous_fire_module(est, _FixedPredictor(est.value), monkeypatch, now)
    ctx = module.run()
    assert ctx.solution is not None and ctx.solution.is_confident

    est = _estimate(0.0, 400.0, 0.7, 0.5, 30.0, 20.0, aim_z=None, now=now)
    module, _ = _continuous_fire_module(est, _FixedPredictor(est.value), monkeypatch, now)
    assert module.run().solution is None


def test_continuous_fire_out_of_range_tracks_without_firing(monkeypatch):
    """Unreachable target -> straight-line aim, is_confident=False."""
    now = 50.0
    est = _estimate(0.0, 20000.0, 0.0, 0.0, 23.5, 23.5, aim_z=-30.0, now=now)
    module, _ = _continuous_fire_module(est, _FixedPredictor(est.value), monkeypatch, now)
    ctx = module.run()
    assert ctx.solution is not None
    assert ctx.solution.is_confident is False
    assert ctx.solution.yaw == pytest.approx(0.0, abs=1e-6)
