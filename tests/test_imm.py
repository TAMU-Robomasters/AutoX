"""Per-axis CV+CA IMM (``src/subsystems/estimation/imm.py``) + 2-D adapter.

Mostly config-free: the IMM internals and ``PositionIMM`` are exercised with
explicit params. The ``make_position_estimator`` factory test monkeypatches
``config.estimation.motion_model`` so it doesn't depend on info.yaml's default.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.subsystems.estimation.filters import (  # noqa: E402
    AngleKF,
    FullStateKF,
    HeightKF,
    PositionKF,
)
from src.subsystems.estimation.imm import (  # noqa: E402
    _CA,
    CAKF,
    CVKF,
    IMM,
    AugBounds,
    PositionIMM,
    expand,
    make_Pi,
    make_position_estimator,
)


def _imm(**kw) -> PositionIMM:
    params = dict(
        r_pos=2.0, r=0.0, init_std=(50, 50, 50, 50), init_std_accel=200,
        sigma_w=25, sigma_j=1000, acc_bound=300, p_cv=0.99, p_ca=0.90,
        mu0_ca=0.1, man_bias=15.0,
    )
    params.update(kw)
    return PositionIMM(**params)


# --- mixing primitives ------------------------------------------------------


def test_expand_truncate_and_uniform_augment():
    """expand() truncates to the leading block and augments with uniform var."""
    mean = np.array([1.0, 2.0, 3.0])
    cov = np.diag([4.0, 5.0, 6.0])
    aug = AugBounds(acc=300.0)
    # truncate 3 -> 2
    m2, c2 = expand(mean, cov, 2, aug)
    np.testing.assert_allclose(m2, [1.0, 2.0])
    np.testing.assert_allclose(c2, np.diag([4.0, 5.0]))
    # expand 2 -> 3: appended accel mean 0, var (2*lim)^2/12
    m3, c3 = expand(np.array([1.0, 2.0]), np.diag([4.0, 5.0]), 3, aug)
    np.testing.assert_allclose(m3, [1.0, 2.0, 0.0])
    assert c3[2, 2] == pytest.approx((2 * 300.0) ** 2 / 12.0)


def test_make_pi_is_row_stochastic_and_diagonally_dominant():
    """make_Pi rows sum to 1 with the self-stay prob on the diagonal."""
    Pi = make_Pi([0.99, 0.90])
    np.testing.assert_allclose(Pi.sum(axis=1), [1.0, 1.0])
    assert Pi[0, 0] == 0.99 and Pi[1, 1] == 0.90


def test_imm_predict_pos_var_includes_q_growth():
    """A single per-axis IMM's predicted variance grows with the horizon (+Q)."""
    cv = CVKF(x0=np.zeros(2), P0=np.diag([4.0, 25.0]), R=4.0, q=25.0)
    ca = CAKF(x0=np.zeros(3), P0=np.diag([4.0, 25.0, 1e4]), R=4.0, q=1000.0)
    imm = IMM([cv, ca], make_Pi([0.99, 0.90]), np.array([0.9, 0.1]), AugBounds(acc=300))
    assert imm.predict_pos_var(0.5, with_Q=True) > imm.predict_pos_var(0.14, with_Q=True)


# --- PositionIMM 2-D adapter ------------------------------------------------


def test_tracks_constant_velocity_and_stays_in_cv():
    """On a clean CV track the IMM converges to the right state, low maneuver prob."""
    imm = _imm()
    imm.reinit(0.0, 100.0)
    t = 0.0
    for _ in range(60):
        t += 1 / 30
        imm.update(1 / 30, 10.0 * t, 100.0, 0.0)  # x at 10 cm/s, y fixed
    x, y, vx, vy = imm.estimate()
    assert x == pytest.approx(10.0 * t, abs=2.0)
    assert vx == pytest.approx(10.0, abs=1.5)
    assert abs(vy) < 1.5
    assert imm.p_man() < 0.3  # mostly the quiescent CV model


def test_maneuver_raises_p_man_and_accel():
    """An accelerating track is tracked by the CA model and lifts the maneuver prob.

    At 30 Hz with cm-level noise a 2 m/s^2 maneuver is only *mildly* detectable
    per step (the harness's own regime, tracking index ~0.014), so the CA
    sub-filter tracks the true accel well while the blended accel stays damped by
    the CV model's zero-mean augmentation and p_man rises modestly -- not to 1.
    """
    imm = _imm()
    imm.reinit(0.0, 0.0)
    t, a = 0.0, 200.0  # cm/s^2
    p_man0 = imm.p_man()
    for _ in range(90):
        t += 1 / 30
        imm.update(1 / 30, 0.5 * a * t * t, 0.0, 0.0)  # x = 0.5 a t^2
    accel = imm.accel()
    assert accel is not None
    # The CA sub-filter itself tracks the true acceleration.
    assert imm._imm_x.models[_CA].x[2] == pytest.approx(a, rel=0.2)
    # The blended accel is positive (maneuver reflected) but conservatively damped.
    assert accel[0] > 30.0
    # The maneuver model gains probability over its 0.1 prior.
    assert imm.p_man() > p_man0


def test_gate_radius_grows_with_horizon_and_includes_man_bias():
    """gate_radius grows over the horizon and the man_bias term inflates it."""
    imm = _imm(man_bias=15.0)
    imm.reinit(0.0, 0.0)
    for _ in range(20):
        imm.update(1 / 30, 0.0, 0.0, 0.0)
    assert imm.gate_radius(0.5) > imm.gate_radius(0.14)
    # Same track, no maneuver bias -> strictly smaller gate (bias adds in quadrature).
    no_bias = _imm(man_bias=0.0)
    no_bias.reinit(0.0, 0.0)
    for _ in range(20):
        no_bias.update(1 / 30, 0.0, 0.0, 0.0)
    assert imm.gate_radius(0.14) > no_bias.gate_radius(0.14)


def test_back_projection_matches_position_kf():
    """PositionIMM.back_project matches PositionKF (centre = panel - r*[cos,sin] yaw)."""
    a = PositionIMM.back_project(10.0, 20.0, 0.5, 23.5)
    b = PositionKF.back_project(10.0, 20.0, 0.5, 23.5)
    np.testing.assert_allclose(a, b)


def test_drop_in_as_full_state_position_half():
    """PositionIMM is a CenterPositionEstimator: FullStateKF accepts it and runs."""
    fs = FullStateKF(_imm(r=23.5), AngleKF(N=4), HeightKF())
    fs.reinit(np.array([0, 0, 0, 0, 0.0, 1.0, 50.0], dtype=np.float32))
    meas = np.array([[10.0, 0.0, 0.0]], dtype=np.float32)  # x,y,yaw
    est, conf = fs.update(1 / 30, meas, r=23.5, z_meas=50.0)
    assert est.shape == (7,)
    assert fs.accel() is not None  # CA-capable position half
    assert fs.gate_radius(0.14) > 0.0


# --- factory ----------------------------------------------------------------


def test_factory_selects_backend_by_config(monkeypatch):
    """make_position_estimator returns PositionKF for cv/ca and PositionIMM for imm."""
    from src.toolbox.globals import config

    monkeypatch.setattr(config.estimation, "motion_model", "constant_velocity")
    assert isinstance(
        make_position_estimator(r_pos=3.0, r=0.0, init_std=(10, 10, 50, 50)), PositionKF
    )
    monkeypatch.setattr(config.estimation, "motion_model", "imm")
    assert isinstance(
        make_position_estimator(r_pos=3.0, r=0.0, init_std=(10, 10, 50, 50)), PositionIMM
    )
