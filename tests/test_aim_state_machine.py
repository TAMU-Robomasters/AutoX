"""State-machine tests for FullStateAutoAimEngine (no hardware, no processes).

Drives the transition methods on a real (non-started) engine instance, which
also smoke-tests the module wiring: constructing the engine raises on any
typo'd module input/output.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [sys.argv[0]]  # before any config-touching src import

from src.engines.full_state_autoaim import (  # noqa: E402
    AimState,
    FullStateAutoAimEngine,
)
from src.subsystems.embedded_communicator import CVState  # noqa: E402
from src.subsystems.estimation.robot_constants import (  # noqa: E402
    RobotConstants,
    save_constants,
)
from src.toolbox.globals import config  # noqa: E402
from src.toolbox.storage import JsonStore  # noqa: E402
from src.types.autoaim import (  # noqa: E402
    ArmorPanel,
    BallisticSolution,
    EnemyRobot,
    HeightDeltaEstimate,
    RadiiEstimate,
    RobotStateEstimate,
)

CONSTANTS = RobotConstants(
    r_low=20.0,
    r_high=27.0,
    height_delta=6.0,
    lower_is_minor=True,
    degenerate_height=False,
    degenerate_radii=False,
)


def _engine(tmp_path, saved=None):
    """Real engine instance with a tmp-path store; no drivers, never started."""
    engine = FullStateAutoAimEngine()
    store = JsonStore(tmp_path / "constants.json")
    for name, constants in (saved or {}).items():
        save_constants(store, name, constants)
    engine._init_estimators()
    engine._init_state_machine(JsonStore(store.path))
    # MCU-side fields normally set in initialize(); needed by _publish_solution.
    engine._last_pitch = 0.0
    engine._last_yaw = 0.0
    engine.alignment_time_ms = 255
    engine.cv_state = CVState.NO_TARGET.value
    return engine


def _panel(center, theta, radius, k, z, with_id=True):
    angle = theta + k * np.pi / 2
    return ArmorPanel(
        icon=None,
        position=np.array(
            [center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle), z],
            dtype=np.float64,
        ),
        orientation=None,
        bbx=None,
        contour=None,
        yaw=angle,
        id=k if with_id else None,
    )


# ---------------------------------------------------------------------------
# Initial state selection
# ---------------------------------------------------------------------------


def test_initial_state_depends_on_saved_constants(tmp_path):
    """Unknown robot -> PARAMETER_ESTIMATION; saved robot -> FULL_STATE_INIT."""
    engine = _engine(tmp_path, saved={"hero": CONSTANTS})
    assert engine._state_for("standard") == AimState.PARAMETER_ESTIMATION
    assert engine._state_for("hero") == AimState.FULL_STATE_INIT


def test_force_reestimation_ignores_saved_constants(tmp_path, monkeypatch):
    """force_reestimation discards the loaded constants -> back to state 1."""
    monkeypatch.setitem(config.estimation, "force_reestimation", True)
    engine = _engine(tmp_path, saved={"hero": CONSTANTS})
    assert engine._constants == {}
    assert engine._state_for("hero") == AimState.PARAMETER_ESTIMATION


# ---------------------------------------------------------------------------
# T1: PARAMETER_ESTIMATION -> FULL_STATE_INIT (convergence + save)
# ---------------------------------------------------------------------------


def _converged_estimates(engine, var=0.01, n=100, dz=-6.0):
    engine.ctx.radii_estimate = RadiiEstimate(
        r_even=20.0, r_odd=27.0, var_even=var, var_odd=var, n_updates=n
    )
    engine.ctx.height_delta_estimate = HeightDeltaEstimate(dz=dz, var=var, n_updates=n)


def test_t1_converged_constants_saved_and_state_advances(tmp_path):
    """Converged radii+height write canonical constants to disk and enter INIT."""
    engine = _engine(tmp_path)
    engine._state_for("standard")
    _converged_estimates(engine)  # even pair lower (dz < 0)

    engine._check_constants_converged("standard")

    assert engine._state["standard"] == AimState.FULL_STATE_INIT
    saved = engine._constants["standard"]
    assert saved.r_low == 20.0 and saved.r_high == 27.0
    assert saved.height_delta == 6.0
    # Persisted: a fresh store sees it.
    assert "standard" in JsonStore(engine._store.path)


@pytest.mark.parametrize("var,n", [(1.0, 100), (0.01, 3)])
def test_t1_not_converged_stays_in_state_1(tmp_path, var, n):
    """High variance or too few updates keep PARAMETER_ESTIMATION."""
    engine = _engine(tmp_path)
    engine._state_for("standard")
    _converged_estimates(engine, var=var, n=n)

    engine._check_constants_converged("standard")
    assert engine._state["standard"] == AimState.PARAMETER_ESTIMATION
    assert "standard" not in engine._constants


# ---------------------------------------------------------------------------
# T2: FULL_STATE_INIT -> FULL_STATE_TRACKING (parity anchoring)
# ---------------------------------------------------------------------------


def _two_panel_target(engine, z_a, z_b, r_a=20.0, r_b=27.0, theta=0.4):
    center = (30.0, 250.0)
    a = _panel(center, theta, r_a, 0, z_a, with_id=False)
    b = _panel(center, theta, r_b, 1, z_b, with_id=False)
    engine.ctx.target_robot = EnemyRobot(name="hero", panels=[a, b])
    engine.ctx.new_observation = True
    return a, b


def test_t2_anchors_by_relative_height_lower_first_panel(tmp_path):
    """Anchor panel (id 0) observed LOWER -> even parity gets r_low."""
    engine = _engine(tmp_path, saved={"hero": CONSTANTS})
    engine._state_for("hero")
    _two_panel_target(engine, z_a=100.0, z_b=106.0, r_a=20.0, r_b=27.0)

    engine._try_anchor("hero")

    assert engine._state["hero"] == AimState.FULL_STATE_TRACKING
    assert engine.estimation._r_even == 20.0 and engine.estimation._r_odd == 27.0
    assert engine.estimation._low_parity == 0
    assert engine.estimation._height_delta == 6.0


def test_t2_anchors_by_relative_height_higher_first_panel(tmp_path):
    """Anchor panel observed HIGHER -> even parity gets r_high (works on platforms)."""
    engine = _engine(tmp_path, saved={"hero": CONSTANTS})
    engine._state_for("hero")
    # Same relative geometry but the whole robot 50cm up a platform.
    _two_panel_target(engine, z_a=156.0, z_b=150.0, r_a=27.0, r_b=20.0)

    engine._try_anchor("hero")

    assert engine._state["hero"] == AimState.FULL_STATE_TRACKING
    assert engine.estimation._r_even == 27.0 and engine.estimation._r_odd == 20.0
    assert engine.estimation._low_parity == 1


def test_t2_degenerate_height_anchors_by_measured_radii(tmp_path):
    """height_delta ~ 0: the smaller measured radius identifies the low/minor pair."""
    degenerate = RobotConstants(
        r_low=20.0, r_high=27.0, height_delta=0.0, lower_is_minor=True,
        degenerate_height=True, degenerate_radii=False,
    )
    engine = _engine(tmp_path, saved={"hero": degenerate})
    engine._state_for("hero")
    _two_panel_target(engine, z_a=100.0, z_b=100.0, r_a=26.8, r_b=20.2)

    engine._try_anchor("hero")
    assert engine._state["hero"] == AimState.FULL_STATE_TRACKING
    assert engine.estimation._r_even == 27.0 and engine.estimation._r_odd == 20.0


def test_t2_requires_adjacent_pair_and_fresh_observation(tmp_path):
    """Non-90deg pairs or stale frames never anchor."""
    engine = _engine(tmp_path, saved={"hero": CONSTANTS})
    engine._state_for("hero")

    # Ghost pair: both panels facing the same way.
    center = (30.0, 250.0)
    a = _panel(center, 0.4, 20.0, 0, 100.0, with_id=False)
    ghost = _panel(center, 0.4, 20.0, 0, 100.0, with_id=False)
    engine.ctx.target_robot = EnemyRobot(name="hero", panels=[a, ghost])
    engine.ctx.new_observation = True
    engine._try_anchor("hero")
    assert engine._state["hero"] == AimState.FULL_STATE_INIT

    # Proper pair but stale (no fresh MCU transform this frame).
    _two_panel_target(engine, z_a=100.0, z_b=106.0)
    engine.ctx.new_observation = False
    engine._try_anchor("hero")
    assert engine._state["hero"] == AimState.FULL_STATE_INIT


# ---------------------------------------------------------------------------
# T3 / T5: target loss
# ---------------------------------------------------------------------------


def test_t3_loss_while_tracking_demotes_to_init(tmp_path):
    """TRACKING -> INIT on loss: track-bound filters reset, aim geometry cleared."""
    engine = _engine(tmp_path, saved={"hero": CONSTANTS})
    engine._state["hero"] = AimState.FULL_STATE_TRACKING
    engine._current_target_name = "hero"
    engine.estimation.set_aim_geometry(6.0, 0)
    engine.estimation.is_their_target_prev = True

    engine._on_target_lost()

    assert engine._state["hero"] == AimState.FULL_STATE_INIT
    assert engine.estimation._height_delta is None  # aim geometry cleared
    assert engine.estimation.is_their_target_prev is False  # will reinit
    assert engine.ctx.target_robot is None


def test_t5_loss_in_state_1_keeps_learning_state(tmp_path):
    """PARAM loss keeps the radii/height KFs and flags a parity re-anchor."""
    engine = _engine(tmp_path)
    engine._state_for("standard")
    engine._current_target_name = "standard"
    sentinel = object()
    engine.radii_estimator._filters["standard"] = sentinel  # type: ignore[assignment]

    engine._on_target_lost()

    assert engine._state["standard"] == AimState.PARAMETER_ESTIMATION
    assert engine._needs_parity_anchor["standard"] is True
    assert engine.radii_estimator._filters["standard"] is sentinel  # kept


def test_t5_reanchor_flips_tracker_parity_on_swapped_measurement(tmp_path):
    """A measurement fitting the swapped hypothesis flips the tracker offset."""
    engine = _engine(tmp_path)
    engine._state_for("standard")
    engine._needs_parity_anchor["standard"] = True
    _converged_estimates(engine, dz=-6.0)  # kept means: [20, 27, -6]
    # New track labeled the pair the other way round: measures ~[27, 20, +6].
    center = (30.0, 250.0)
    a = _panel(center, 0.4, 27.0, 0, 106.0)
    b = _panel(center, 0.4, 20.0, 1, 100.0)
    engine.ctx.target_robot = EnemyRobot(name="standard", panels=[a, b])
    engine.ctx.estimate = RobotStateEstimate(
        value=np.array([*center, 0, 0, 0.4, 0], dtype=np.float32),
        timestamp=0.0,
        confidence=1.0,
    )

    engine._maybe_reanchor_parity("standard")

    assert engine.panel_tracking._parity_offset == 1
    assert engine._needs_parity_anchor["standard"] is False


def test_t5_reanchor_keeps_identity_on_matching_measurement(tmp_path):
    """A measurement matching the kept state leaves the tracker offset at 0."""
    engine = _engine(tmp_path)
    engine._state_for("standard")
    engine._needs_parity_anchor["standard"] = True
    _converged_estimates(engine, dz=-6.0)
    center = (30.0, 250.0)
    a = _panel(center, 0.4, 20.0, 0, 100.0)
    b = _panel(center, 0.4, 27.0, 1, 106.0)
    engine.ctx.target_robot = EnemyRobot(name="standard", panels=[a, b])

    engine._maybe_reanchor_parity("standard")

    assert engine.panel_tracking._parity_offset == 0
    assert engine._needs_parity_anchor["standard"] is False


# ---------------------------------------------------------------------------
# T6: target switch
# ---------------------------------------------------------------------------


def test_t6_switch_demotes_previous_and_swaps_state(tmp_path):
    """Switching targets demotes a TRACKING robot and flags a PARAM robot's anchor."""
    engine = _engine(tmp_path, saved={"hero": CONSTANTS})
    engine._state["hero"] = AimState.FULL_STATE_TRACKING
    engine._current_target_name = "hero"

    engine._on_target_switch("standard")

    assert engine._current_target_name == "standard"
    assert engine._state["hero"] == AimState.FULL_STATE_INIT  # demoted
    assert engine._state["standard"] == AimState.PARAMETER_ESTIMATION
    assert engine._needs_parity_anchor["standard"] is True


# ---------------------------------------------------------------------------
# Solution handling
# ---------------------------------------------------------------------------


def test_unconfident_solution_aims_but_holds_fire(tmp_path):
    """is_confident=False forwards pitch/yaw but reports NO_TARGET."""
    engine = _engine(tmp_path)
    engine.ctx.solution = BallisticSolution(
        pitch=0.1, yaw=0.2, alignment_time_ms=255, is_confident=False
    )

    engine._publish_solution(CVState.CONTINUOUS_FIRE.value)

    assert engine.cv_state == CVState.NO_TARGET.value
    assert engine._last_pitch == 0.1
    assert engine._last_yaw == pytest.approx(0.2 + np.deg2rad(config.ballistic.yaw_offset))


def test_confident_solution_uses_active_cv_state(tmp_path):
    """A confident solution reports the active state's cv_state."""
    engine = _engine(tmp_path)
    engine.ctx.solution = BallisticSolution(pitch=0.1, yaw=0.2, alignment_time_ms=120)

    engine._publish_solution(CVState.SHOT_TIMING.value)

    assert engine.cv_state == CVState.SHOT_TIMING.value
    assert engine.alignment_time_ms == 120


def test_no_solution_keeps_fallback(tmp_path):
    """ctx.solution=None leaves the fallback aim and NO_TARGET untouched."""
    engine = _engine(tmp_path)
    engine.ctx.solution = None
    engine._publish_solution(CVState.CONTINUOUS_FIRE.value)
    assert engine.cv_state == CVState.NO_TARGET.value
