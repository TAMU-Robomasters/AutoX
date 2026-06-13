"""Unit tests for the single-panel estimation + ballistic modules (state 1 / 2a)."""

import math
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [sys.argv[0]]  # before any config-touching src import

from src.subsystems.ballistics import single_panel as sp_ballistics  # noqa: E402
from src.subsystems.ballistics.solver import mcu_yaw_from_xy  # noqa: E402


def _theta_solver(
    d: float, delta_z: float, g: float, v: float
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Closed-form no-spin ballistic reference (no barrel offset); for tests only.

    Solves the projectile arc for pitch toward a point at horizontal distance
    ``d`` and vertical offset ``delta_z`` (same units as ``g``/``v``). Returns
    two (theta, time_of_flight) solutions, or None if unreachable.
    """
    v2, d2 = v * v, d * d
    A = (g * d2) / (2.0 * v2)
    disc = d2 - 4.0 * A * (delta_z + A)
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(disc)
    inv_denom = 1.0 / (2.0 * A)
    theta1 = math.atan((d + sqrt_disc) * inv_denom)
    theta2 = math.atan((d - sqrt_disc) * inv_denom)
    t1 = d / (v * math.cos(theta1))
    t2 = d / (v * math.cos(theta2))
    return (theta1, t1), (theta2, t2)
from src.subsystems.estimation import single_panel as sp_estimation  # noqa: E402
from src.toolbox.globals import config  # noqa: E402
from src.types.autoaim import (  # noqa: E402
    ArmorPanel,
    EnemyRobot,
    FullStateAutoAimContext,
    PanelEstimate,
)


def _panel(x, y, z=0.0, yaw=0.0, panel_id=None):
    return ArmorPanel(
        icon=None,
        position=np.array([x, y, z], dtype=np.float32),
        orientation=None,
        bbx=None,
        contour=None,
        yaw=yaw,
        id=panel_id,
    )


def _xy_estimate(x, y, vx, vy, z, now, panel_id=None):
    return PanelEstimate(
        value=np.array([x, y, vx, vy], dtype=np.float32),
        z=z,
        panel_id=panel_id,
        timestamp=now,
    )


def _ballistic_module(xy_estimate, monkeypatch, now):
    monkeypatch.setattr(sp_ballistics.time, "perf_counter", lambda: now)
    ctx = FullStateAutoAimContext(xy_estimate=xy_estimate)
    return sp_ballistics.SinglePanelBallisticModule(ctx)


# ---------------------------------------------------------------------------
# SinglePanelBallisticModule
# ---------------------------------------------------------------------------


def test_stationary_panel_matches_closed_form(monkeypatch):
    """Solver pitch agrees with the closed-form _theta_solver; yaw is MCU convention."""
    now = 20.0
    x, y, z = 80.0, 350.0, -25.0  # cm
    module = _ballistic_module(_xy_estimate(x, y, 0, 0, z, now), monkeypatch, now)
    ctx = module.run()

    assert ctx.solution is not None and ctx.solution.is_confident
    # Small deviations allowed: the LM solve includes the barrel-tip offset,
    # which the closed-form reference ignores.
    assert ctx.solution.yaw == pytest.approx(mcu_yaw_from_xy(x, y), abs=0.02)

    solutions = _theta_solver(
        float(np.hypot(x, y)), z, config.ballistic.gravity, config.ballistic.projectile_velocity
    )
    expected_pitch, _ = min(solutions, key=lambda s: s[1])
    # Small deviation allowed: the LM solve includes the barrel-tip offset.
    assert ctx.solution.pitch == pytest.approx(expected_pitch, abs=0.02)


def test_moving_panel_leads_the_intercept(monkeypatch):
    """A crossing target produces a lead: yaw points at the future position."""
    now = 20.0
    x, y, vx = 0.0, 300.0, 150.0  # cm, cm/s, moving toward +x
    module = _ballistic_module(_xy_estimate(x, y, vx, 0, -20.0, now), monkeypatch, now)
    ctx = module.run()

    assert ctx.solution is not None and ctx.solution.is_confident
    assert ctx.solution.yaw < mcu_yaw_from_xy(x, y)  # leads toward +x => more negative


def test_out_of_range_tracks_without_firing(monkeypatch):
    """Beyond max_range: straight-line aim at the panel, is_confident=False."""
    now = 20.0
    x, y, z = 0.0, 2 * float(config.ballistic.max_range), -20.0
    module = _ballistic_module(_xy_estimate(x, y, 0, 0, z, now), monkeypatch, now)
    ctx = module.run()

    assert ctx.solution is not None
    assert ctx.solution.is_confident is False
    assert ctx.solution.yaw == pytest.approx(mcu_yaw_from_xy(x, y), abs=1e-6)
    expected_pitch = np.arctan2(z, np.hypot(x, y))
    assert ctx.solution.pitch == pytest.approx(expected_pitch, abs=1e-6)


def test_no_estimate_returns_none(monkeypatch):
    """No xy_estimate: no solution."""
    module = _ballistic_module(None, monkeypatch, 20.0)
    assert module.run().solution is None


# ---------------------------------------------------------------------------
# SinglePanelEstimationModule
# ---------------------------------------------------------------------------


def _est_module(jump_reinit_cm=15.0):
    ctx = FullStateAutoAimContext(new_observation=True)
    return sp_estimation.SinglePanelEstimationModule(ctx, jump_reinit_cm=jump_reinit_cm), ctx


def _observe(module, ctx, panels, frame_ts):
    ctx.target_robot = EnemyRobot(name="standard", panels=panels)
    ctx.frame_ts = frame_ts
    ctx.new_observation = True
    return module.run().xy_estimate


def test_estimation_tracks_closest_panel_position_and_z():
    """The output tracks the closest panel's xy and carries its raw z."""
    module, ctx = _est_module()
    result = _observe(module, ctx, [_panel(0, 300, -5), _panel(0, 200, -22)], 1.0)

    assert result is not None
    np.testing.assert_allclose(result.value[:2], [0, 200], atol=1e-3)
    assert result.z == pytest.approx(-22.0)


def test_estimation_velocity_converges_on_moving_panel():
    """A panel moving at constant velocity yields that velocity in the state."""
    module, ctx = _est_module()
    result = None
    for i in range(30):
        t = i * 0.01
        result = _observe(module, ctx, [_panel(100.0 * t, 250.0, -20.0)], t)
    assert result.value[2] == pytest.approx(100.0, abs=10.0)  # vx cm/s
    assert abs(result.value[3]) < 10.0


def test_estimation_reinit_on_id_change_and_jump():
    """Tracked-panel id change or a large position jump re-seeds the filter."""
    module, ctx = _est_module(jump_reinit_cm=15.0)
    for i in range(10):
        _observe(module, ctx, [_panel(100.0 * i * 0.01, 250.0, -20, panel_id=1)], i * 0.01)

    # Id change: velocity must reset to ~0 even though position jumped.
    result = _observe(module, ctx, [_panel(50, 250, -20, panel_id=2)], 0.2)
    assert result.panel_id == 2
    np.testing.assert_allclose(result.value[2:], 0, atol=1e-6)

    # No ids: a >15cm jump also re-seeds.
    module, ctx = _est_module(jump_reinit_cm=15.0)
    for i in range(10):
        _observe(module, ctx, [_panel(100.0 * i * 0.01, 250.0, -20)], i * 0.01)
    result = _observe(module, ctx, [_panel(60, 250, -20)], 0.2)
    np.testing.assert_allclose(result.value[2:], 0, atol=1e-6)


def test_estimation_predicts_without_observation():
    """new_observation=False extrapolates instead of updating."""
    module, ctx = _est_module()
    _observe(module, ctx, [_panel(0, 200, -20)], 1.0)

    ctx.new_observation = False
    result = module.run().xy_estimate
    assert result is not None  # predict-only frame still produces an estimate


def test_estimation_resets_on_target_loss():
    """target_robot=None clears the track; the next output is None."""
    module, ctx = _est_module()
    _observe(module, ctx, [_panel(0, 200, -20)], 1.0)

    ctx.target_robot = None
    assert module.run().xy_estimate is None
