"""Constant-velocity Kalman filters for tracking an N-fold symmetric robot.

Shared estimation primitives used by both the full-state pipeline
(``full_state/kf.py``, ``full_state/pf.py``) and the single-panel pipeline
(``single_panel.py``). Also the home of the two estimator ``Protocol``s
(``FullStateEstimator`` / ``SinglePanelEstimator``) and the static
``predict_ahead`` extrapolators the ballistic modules use to project a state
forward (constant-velocity, or constant-acceleration when handed the filtered
accel) without holding an estimator instance. The CV/CA motion model itself is a
composed strategy -- see ``motion_models.py``.

Companion to ``ParticleFilter`` (`full_state/particle_filter.py`): instead of a
single 6-D particle filter over ``[x, y, vx, vy, theta, omega]``, this module
splits the state into two small linear KFs, each backed by
``filterpy.kalman.KalmanFilter``:

- ``PositionKF`` tracks ``[x, y, vx, vy]`` from a panel back-projected to the
  robot's centre, exactly mirroring the ``cx = x_obs - r*cos(yaw_obs)``
  back-projection used in ``pf_cuda_cv.cu``.
- ``AngleKF`` tracks ``[theta_continuous, omega]`` over the *unwrapped*
  chassis-yaw -- ``theta`` accumulates the full cumulative rotation and is never
  clamped. The ``2*pi/N`` ambiguity (an N-fold symmetric panel layout looks
  identical every ``2*pi/N``) is resolved at correction time: the raw yaw is
  unwrapped onto the *post-predict* state by picking the integer ``k`` that puts
  ``z + k*(2*pi/N)`` nearest the prediction, so the filter's own momentum decides
  which marker is being seen. ``theta_wrapped`` folds it back into ``[0, 2*pi/N)``
  for display/comparison.

``FullStateKF`` combines the two (plus ``HeightKF``) into the 7-D ``[x, y, vx,
vy, theta, omega, z]`` state with the same ``reinit``/``update``/
``update_with_no_observation`` interface as ``ParticleFilter``, so either backend
can be injected into the estimation module via ``set_estimator`` (see
``FullStateEstimator``).
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np
from filterpy.kalman import KalmanFilter

from src.subsystems.estimation.motion_models import (
    CONSTANT_ACCELERATION,
    CONSTANT_VELOCITY,
    ConstantVelocity,
    MotionModel,
)

__all__ = [
    "AngleKF",
    "PositionKF",
    "HeightKF",
    "FullStateKF",
    "FullStateEstimator",
    "SinglePanelEstimator",
    "CenterPositionEstimator",
    "CONSTANT_VELOCITY",
    "CONSTANT_ACCELERATION",
]

#NOTE: get rid of commented code  the old modulo thing 


def _nis(kf: KalmanFilter) -> float:
    """Normalised innovation squared ``y^T S^-1 y`` from the last update."""
    y = np.atleast_1d(kf.y).astype(np.float64).flatten()
    S = np.atleast_2d(kf.S).astype(np.float64)
    return float(y @ np.linalg.solve(S, y))


# --- PREVIOUS IMPLEMENTATION (DISABLED) ------------------------------------
# State wrapped to [0, 2*pi/N) with a wrapped-innovation update + odometer.
# Kept for easy A/B comparison; remove the surrounding ''' to re-enable (and
# comment out the continuous-unwrap AngleKF below).
'''
class AngleKF:
    """Constant-velocity Kalman filter for an N-fold symmetric yaw phase.

    State: ``[theta, omega]``, with ``theta`` wrapped to ``[0, 2*pi/N)``.
    """

    def __init__(
        self,
        N: int = 1,
        r_yaw: float = np.radians(15),
        q_omega: float = 0.6,
        init_std_theta: float = 0.3,
        init_std_omega: float = 15.0,
    ) -> None:
        self.N = N
        self.step_size = 2.0 * np.pi / N
        self.r_yaw = r_yaw
        self.q_omega = q_omega
        self.init_std = np.array([init_std_theta, init_std_omega], dtype=np.float64)

        self._kf = KalmanFilter(dim_x=2, dim_z=1)
        self._kf.H = np.array([[1.0, 0.0]])
        self._kf.R = np.array([[self.r_yaw**2]])
        self._theta_offset = 0.0
        self.reinit(0.0)

    def reinit(self, theta0: float, omega0: float = 0.0) -> None:
        """Re-seed the state, wrapping ``theta0`` into ``[0, 2*pi/N)``."""
        self._kf.x = np.array([theta0 % self.step_size, omega0], dtype=np.float64)
        self._kf.P = np.diag(self.init_std**2)
        self._theta_offset = 0.0

    def _predict(self, dt: float) -> None:
        theta, omega = self._kf.x
        F = np.array([[1.0, dt], [0.0, 1.0]])
        q2 = self.q_omega**2
        Q = q2 * np.array(
            [
                [dt**3 / 3.0, dt**2 / 2.0],
                [dt**2 / 2.0, dt],
            ]
        )

        self._kf.predict(F=F, Q=Q)

        # filterpy leaves theta unbounded; wrap it and track whole-step crossings
        # for the unbounded odometer.
        theta_pred = self._kf.x[0] % self.step_size
        if omega > 0 and theta_pred < theta:
            self._theta_offset += self.step_size
        elif omega < 0 and theta_pred > theta:
            self._theta_offset -= self.step_size
        self._kf.x[0] = theta_pred

    def _correct(self, z: float) -> float:
        """Wrapped-innovation measurement update. Returns the NIS."""
        theta_pred = self._kf.x[0]
        raw_innovation = z - theta_pred
        wrapped_innovation = (
            (raw_innovation + self.step_size / 2.0) % self.step_size
        ) - self.step_size / 2.0

        # Fold the measurement onto the predicted phase so the standard filterpy
        # update sees an innovation equal to the wrapped one.
        self._kf.update(theta_pred + wrapped_innovation)
        # self._kf.x[0] = self._kf.x[0] % self.step_size

        return _nis(self._kf)

    def update(self, dt: float, z: float) -> tuple[np.ndarray, float]:
        """Predict ``dt`` seconds, then correct with a raw yaw observation ``z``."""
        self._predict(dt)
        nis = self._correct(z)
        return self.estimate(), nis

    def update_with_no_observation(self, dt: float) -> tuple[np.ndarray, None]:
        """Predict only (no observation this frame)."""
        self._predict(dt)
        return self.estimate(), None

    def estimate(self) -> np.ndarray:
        """Return the current ``[theta, omega]`` state."""
        return np.array(self._kf.x, dtype=np.float64).flatten()

    @property
    def theta_continuous(self) -> float:
        """Unbounded odometer: accumulated phase rotation since ``reinit``."""
        return self._theta_offset + float(self._kf.x[0])

    def extrapolate(self, dt: float) -> np.ndarray:
        """Constant-velocity extrapolation ``dt`` seconds ahead, no side effects."""
        theta, omega = self._kf.x
        theta_pred = (theta + omega * dt) % self.step_size
        return np.array([theta_pred, omega], dtype=np.float64)
'''
# --- END PREVIOUS IMPLEMENTATION -------------------------------------------


class AngleKF:
    """Constant-velocity KF over an *unwrapped* N-fold-symmetric yaw.

    State: ``[theta_continuous, omega]``. ``theta`` is never clamped -- it holds
    the full cumulative rotation. The ``2*pi/N`` measurement ambiguity is
    resolved at correction time by unwrapping the raw yaw onto the *post-predict*
    state (nearest equivalent to the prediction), so the filter's own momentum
    decides which of the N identical markers is being observed.
    """

    def __init__(
        self,
        N: int = 4,
        r_yaw: float = np.radians(15),
        q_omega: float = 0.6,
        init_std_theta: float = 0.3,
        init_std_omega: float = 15.0,
    ) -> None:
        self.N = N
        self.step_size = 2.0 * np.pi / N
        self.r_yaw = r_yaw
        self.q_omega = q_omega
        self.init_std = np.array([init_std_theta, init_std_omega], dtype=np.float64)

        self._kf = KalmanFilter(dim_x=2, dim_z=1)
        self._kf.H = np.array([[1.0, 0.0]])
        self._kf.R = np.array([[self.r_yaw**2]])
        self.reinit(0.0)

    def reinit(self, theta0: float, omega0: float = 0.0) -> None:
        """Re-seed the state at the (unwrapped) ``theta0``."""
        self._kf.x = np.array([theta0, omega0], dtype=np.float64)
        self._kf.P = np.diag(self.init_std**2)

    def _predict(self, dt: float) -> None:
        F = np.array([[1.0, dt], [0.0, 1.0]])
        q2 = self.q_omega**2
        Q = q2 * np.array(
            [
                [dt**3 / 3.0, dt**2 / 2.0],
                [dt**2 / 2.0, dt],
            ]
        )
        self._kf.predict(F=F, Q=Q)
        # No wrapping: theta stays continuous/unbounded.

    def _correct(self, z: float) -> float:
        """Unwrap ``z`` onto the post-predict state, then update. Returns the NIS.

        The integer ``k`` is chosen so ``z + k*step`` is the symmetry-equivalent
        of the raw yaw nearest the predicted ``theta`` -- this is the critical
        step, and it must run on the post-predict state so the latest motion is
        already folded in before the ambiguity is resolved.
        """
        theta_pred = float(self._kf.x[0])
        k = round((theta_pred - z) / self.step_size)
        z_unwrapped = z + k * self.step_size
        self._kf.update(z_unwrapped)
        return _nis(self._kf)

    def update(self, dt: float, z: float) -> tuple[np.ndarray, float]:
        """Predict ``dt`` seconds, then correct with a raw yaw observation ``z``."""
        self._predict(dt)
        nis = self._correct(z)
        return self.estimate(), nis

    def update_with_no_observation(self, dt: float) -> tuple[np.ndarray, None]:
        """Predict only (no observation this frame)."""
        self._predict(dt)
        return self.estimate(), None

    def estimate(self) -> np.ndarray:
        """Return the current ``[theta_continuous, omega]`` state."""
        return np.array(self._kf.x, dtype=np.float64).flatten()

    @property
    def theta_continuous(self) -> float:
        """Unbounded cumulative orientation (rad)."""
        return float(self._kf.x[0])

    @property
    def theta_wrapped(self) -> float:
        """``theta_continuous`` folded into ``[0, 2*pi/N)`` for display/comparison."""
        return float(self._kf.x[0]) % self.step_size


class PositionKF:
    """Kalman filter for a robot's 2-D centre position (CV or CA motion).

    Measurements are panel observations ``(x_obs, y_obs, yaw_obs)``,
    back-projected to the robot's centre via the marker orbit radius ``r``.

    The motion model (constant-velocity vs constant-acceleration) is supplied as
    a ``MotionModel`` strategy (``motion_models.py``) -- the *only* place the
    CV/CA switch lives (the angle/height filters are always constant-velocity).
    This class owns just the shared *measurement* half (back-projection, ``H``,
    ``R``) and delegates all dynamics (``F``/``Q``/seed/dim/accel) to the model.

    Either way ``estimate()`` returns just ``[x, y, vx, vy]`` -- the acceleration
    stays internal, so every downstream consumer (FullStateKF's 7-D layout,
    ballistics) is unaffected by the model choice. When the model tracks it, the
    acceleration is read back via ``accel()`` and carried *alongside* the state
    (on the estimate dataclasses), never inserted into the published vector.
    """

    def __init__(
        self,
        motion_model: MotionModel,
        r_pos: float = 20.0,
        r: float = 23.5,
        init_std: tuple[float, float, float, float] = (100.0, 100.0, 10.0, 10.0),
    ) -> None:
        self.motion_model = motion_model
        self.r_pos = r_pos
        self.r = r
        self.init_std = np.array(init_std, dtype=np.float64)

        dim_x = motion_model.dim
        self._kf = KalmanFilter(dim_x=dim_x, dim_z=2)
        self._kf.H = np.zeros((2, dim_x))
        self._kf.H[0, 0] = 1.0
        self._kf.H[1, 1] = 1.0
        self._kf.R = np.diag([self.r_pos**2, self.r_pos**2])
        self.reinit(0.0, 0.0)

    def reinit(self, x0: float, y0: float, vx0: float = 0.0, vy0: float = 0.0) -> None:
        """Re-seed the state at ``(x0, y0, vx0, vy0)`` (accel starts at 0 in CA)."""
        self._kf.x = self.motion_model.seed_state(x0, y0, vx0, vy0)
        self._kf.P = self.motion_model.seed_cov(self.init_std)

    @staticmethod
    def back_project(x_obs: float, y_obs: float, yaw_obs: float, r: float) -> tuple[float, float]:
        """Back-project a panel observation to the robot's centre."""
        cx = x_obs - r * np.cos(yaw_obs)
        cy = y_obs - r * np.sin(yaw_obs)
        return cx, cy

    def _predict(self, dt: float) -> None:
        self._kf.predict(
            F=self.motion_model.transition(dt), Q=self.motion_model.process_noise(dt)
        )

    def accel(self) -> Optional[np.ndarray]:
        """Filtered ``[ax, ay]`` acceleration (cm/s^2), or ``None`` for a CV model."""
        return self.motion_model.accel(self._kf.x)

    def predict_pos_var(self, dt: float) -> tuple[float, float]:
        """Predicted ``(var_x, var_y)`` ``dt`` ahead, *including* process noise.

        Mirrors ``IMM.predict_pos_var(dt, with_Q=True)`` from the CV+CA tuning
        harness: ``Pp = F P F^T + Q``, then read the position diagonals. The
        ``+Q`` is essential -- it grows the uncertainty over the horizon; the
        filtered covariance ``P`` alone omits it and reads over-confident.
        """
        F = self.motion_model.transition(dt)
        Q = self.motion_model.process_noise(dt)
        Pp = F @ self._kf.P @ F.T + Q
        return float(Pp[0, 0]), float(Pp[1, 1])

    def gate_radius(self, dt: float) -> float:
        """1-sigma 2-D position gate radius ``dt`` ahead (cm): ``sqrt(var_x + var_y)``.

        The engine compares this against a configurable threshold to decide
        whether the firing solution is trustworthy enough to shoot. Single-filter
        form of the IMM-doc gate (no maneuver-bias term -- that needs an IMM's
        model probability, which a single CV/CA filter does not have).
        """
        var_x, var_y = self.predict_pos_var(dt)
        return float(np.sqrt(var_x + var_y))

    def _correct(self, cx: float, cy: float) -> float:
        """Position-only measurement update. Returns the NIS."""
        self._kf.update(np.array([cx, cy], dtype=np.float64))
        return _nis(self._kf)

    def update(
        self,
        dt: float,
        x_obs: float,
        y_obs: float,
        yaw_obs: float,
        r: Optional[float] = None,
    ) -> tuple[np.ndarray, float]:
        """Predict ``dt`` seconds, then correct with a back-projected panel observation.

        ``r`` overrides the orbit radius for this measurement only (the radius
        depends on which panel pair is being observed); ``None`` keeps the
        constructor default.
        """
        self._predict(dt)
        cx, cy = self.back_project(x_obs, y_obs, yaw_obs, self.r if r is None else r)
        nis = self._correct(cx, cy)
        return self.estimate(), nis

    def update_with_no_observation(self, dt: float) -> tuple[np.ndarray, None]:
        """Predict only (no observation this frame)."""
        self._predict(dt)
        return self.estimate(), None

    def estimate(self) -> np.ndarray:
        """Return the published ``[x, y, vx, vy]`` state (accel stays internal)."""
        return np.array(self._kf.x[:4], dtype=np.float64).flatten()

    @staticmethod
    def predict_ahead(
        state: np.ndarray, dt: float, accel: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extrapolate a ``[x, y, vx, vy]`` state ``dt`` ahead, optionally under constant accel.

        With ``accel=None`` this is constant-velocity (``x += vx*dt``); with a
        ``[ax, ay]`` it is constant-acceleration (``x += vx*dt + ax*dt^2/2``,
        ``vx += ax*dt``). Pure function of the state -- no instance needed -- so
        the single-panel ballistic module can project an estimate forward without
        holding the KF. Satisfies ``SinglePanelEstimator.predict_ahead``.
        """
        x, y, vx, vy = (float(v) for v in state[:4])
        ax, ay = (0.0, 0.0) if accel is None else (float(accel[0]), float(accel[1]))
        return np.array(
            [
                x + vx * dt + 0.5 * ax * dt**2,
                y + vy * dt + 0.5 * ay * dt**2,
                vx + ax * dt,
                vy + ay * dt,
            ],
            dtype=np.float32,
        )


class HeightKF:
    """Position-only Kalman filter for the panel-center height ``z``.

    State: ``[z]`` (cm). Modelled as a random walk with *low* process noise --
    a robot's panel-center height is ~constant, so ``predict`` ~= hold and the
    filter smooths out per-frame z noise / rides through no-observation ticks.
    The measurement is the mid-height between the two panel pairs (see
    ``KalmanFilterEstimationModule._aim_z_for``).
    """

    def __init__(
        self,
        q: float = 0.5,
        r: float = 3.0,
        init_std: float = 50.0,
    ) -> None:
        self.q = q
        self.r = r
        self.init_std = init_std

        self._kf = KalmanFilter(dim_x=1, dim_z=1)
        self._kf.H = np.array([[1.0]])
        self._kf.R = np.array([[self.r**2]])
        self._kf.F = np.array([[1.0]])
        self.reinit(0.0)

    def reinit(self, z0: float) -> None:
        """Re-seed the height state at ``z0``."""
        self._kf.x = np.array([z0], dtype=np.float64)
        self._kf.P = np.array([[self.init_std**2]])

    def _predict(self, dt: float) -> None:
        # Random-walk process noise, scaled by dt so the rate is dt-invariant.
        self._kf.predict(F=np.array([[1.0]]), Q=np.array([[self.q**2 * dt]]))

    def _correct(self, z: float) -> float:
        """Measurement update. Returns the NIS."""
        self._kf.update(np.array([z], dtype=np.float64))
        return _nis(self._kf)

    def update(self, dt: float, z: float) -> tuple[np.ndarray, float]:
        """Predict ``dt`` seconds, then correct with a height observation ``z``."""
        self._predict(dt)
        nis = self._correct(z)
        return self.estimate(), nis

    def update_with_no_observation(self, dt: float) -> tuple[np.ndarray, None]:
        """Predict only (no observation this frame)."""
        self._predict(dt)
        return self.estimate(), None

    def estimate(self) -> np.ndarray:
        """Return the current ``[z]`` state."""
        return np.array(self._kf.x, dtype=np.float64).flatten()


class CenterPositionEstimator(Protocol):
    """The centre/panel 2-D position estimator interface (CV/CA KF or the IMM).

    Both ``PositionKF`` (single CV/CA model) and ``imm.PositionIMM`` (per-axis
    CV+CA IMM) satisfy this structurally, so either can be the position half of
    ``FullStateKF`` or the single-panel tracker -- selected by
    ``config.estimation.motion_model`` via ``imm.make_position_estimator``. State
    layout: ``[x, y, vx, vy]`` (cm, cm/s), turret frame.
    """

    r: float
    """Default back-projection orbit radius (cm); overridden per ``update``."""

    def reinit(self, x0: float, y0: float, vx0: float = 0.0, vy0: float = 0.0) -> None:
        """Re-seed the state at ``(x0, y0, vx0, vy0)`` (accel, if tracked, at 0)."""
        ...

    def update(
        self,
        dt: float,
        x_obs: float,
        y_obs: float,
        yaw_obs: float,
        r: Optional[float] = None,
    ) -> tuple[np.ndarray, float]:
        """Predict ``dt``, correct with a back-projected panel obs; return (estimate, nis)."""
        ...

    def update_with_no_observation(self, dt: float) -> tuple[np.ndarray, None]:
        """Predict only (no observation this frame)."""
        ...

    def estimate(self) -> np.ndarray:
        """Return the published ``[x, y, vx, vy]`` state."""
        ...

    def accel(self) -> Optional[np.ndarray]:
        """Filtered/blended ``[ax, ay]`` accel (cm/s^2), or ``None`` if not modelled."""
        ...

    def predict_pos_var(self, dt: float) -> tuple[float, float]:
        """Predicted ``(var_x, var_y)`` ``dt`` ahead, including process noise."""
        ...

    def gate_radius(self, dt: float) -> float:
        """1-sigma 2-D position gate radius (cm) ``dt`` ahead (maneuver-aware for the IMM)."""
        ...


class FullStateKF:
    """Combine a centre-position estimator + ``AngleKF`` + ``HeightKF`` into a ``ParticleFilter``-compatible estimator.

    State: 7-D ``[x, y, vx, vy, theta, omega, z]`` -- the first six match
    ``ParticleFilter`` (consumers index 0,1,4,5), with the panel-center height
    ``z`` appended at index 6. Drop-in wherever a ``ParticleFilter`` is injected
    via ``set_estimator`` (estimation, shot-timing, continuous-fire modules). The
    position half is any ``CenterPositionEstimator`` (single CV/CA ``PositionKF``
    or the per-axis ``PositionIMM``); the angle/height filters are always CV.
    """

    def __init__(
        self,
        position_kf: CenterPositionEstimator,
        angle_kf: AngleKF,
        height_kf: HeightKF,
    ) -> None:
        self.position_kf = position_kf
        self.angle_kf = angle_kf
        self.height_kf = height_kf
        self.estimate = np.zeros(7, dtype=np.float32)

    def update(
        self,
        dt: float,
        measurements: np.ndarray,
        r: Optional[float] = None,
        z_meas: Optional[float] = None,
    ) -> tuple[np.ndarray, float]:
        """Predict ``dt`` seconds, then correct with the first row of ``measurements``.

        measurements: np.ndarray shape (M, 3) -- [[x, y, yaw], ...]; only the
        first observation is used. ``r`` is the orbit radius of the observed
        panel's pair for this measurement (None = PositionKF default).
        ``z_meas`` is the measured panel-center mid-height (cm); ``None`` (state
        1, height geometry unknown) predicts the height KF only.
        """
        x_obs, y_obs, yaw_obs = (float(v) for v in measurements[0])
        pos_estimate, pos_nis = self.position_kf.update(dt, x_obs, y_obs, yaw_obs, r=r)
        angle_estimate, angle_nis = self.angle_kf.update(dt, yaw_obs)
        if z_meas is None:
            z_estimate, _ = self.height_kf.update_with_no_observation(dt)
        else:
            z_estimate, _ = self.height_kf.update(dt, float(z_meas))
        self.estimate = np.concatenate(
            [pos_estimate, angle_estimate, z_estimate]
        ).astype(np.float32)
        confidence = float(np.exp(-0.5 * (pos_nis + angle_nis)))
        return self.estimate, confidence

    def update_with_no_observation(self, dt: float) -> tuple[np.ndarray, float]:
        """Motion only (no observation this frame)."""
        pos_estimate, _ = self.position_kf.update_with_no_observation(dt)
        angle_estimate, _ = self.angle_kf.update_with_no_observation(dt)
        z_estimate, _ = self.height_kf.update_with_no_observation(dt)
        self.estimate = np.concatenate(
            [pos_estimate, angle_estimate, z_estimate]
        ).astype(np.float32)
        return self.estimate, 1.0

    def reinit(self, prior: np.ndarray) -> None:
        """Re-seed from a ``[x, y, vx, vy, theta, omega(, z)]`` prior.

        A 7-element prior also seeds the height; a 6-element prior leaves the
        height KF untouched (keeps the archived 6-D particle-filter path valid).
        """
        self.position_kf.reinit(float(prior[0]), float(prior[1]), float(prior[2]), float(prior[3]))
        self.angle_kf.reinit(float(prior[4]), float(prior[5]))
        if len(prior) >= 7:
            self.height_kf.reinit(float(prior[6]))
        self.estimate = np.zeros(7, dtype=np.float32)

    def accel(self) -> Optional[np.ndarray]:
        """Filtered centre ``[ax, ay]`` acceleration (cm/s^2), or ``None`` for a CV model."""
        return self.position_kf.accel()

    def gate_radius(self, dt: float) -> float:
        """1-sigma centre-position gate radius ``dt`` ahead (cm); see ``PositionKF.gate_radius``."""
        return self.position_kf.gate_radius(dt)

    @staticmethod
    def predict_ahead(
        state: np.ndarray, dt: float, accel: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extrapolate a 7-D state ``dt`` seconds ahead (constant accel if given).

        ``state``: ``[x, y, vx, vy, theta, omega, z]``. Position advances with its
        velocity (and ``accel`` if provided -- see ``PositionKF.predict_ahead``),
        theta with omega, z is held (the ``HeightKF`` random walk has no velocity).
        The acceleration is assumed constant over the horizon, so it does not
        change. Pure function of the state -- no instance needed -- so the
        full-state ballistic modules project an estimate forward without holding
        an estimator. Satisfies ``FullStateEstimator.predict_ahead``.
        """
        out = np.array(state, dtype=np.float32).copy()
        out[:4] = PositionKF.predict_ahead(state, dt, accel)
        out[4] = float(state[4]) + float(state[5]) * dt  # theta += omega * dt
        return out


class FullStateEstimator(Protocol):
    """The full-state estimator interface injected via ``set_estimator``.

    Implemented structurally by both the CUDA ``ParticleFilter`` (6-D) and
    ``FullStateKF`` (7-D) -- the single place the backend contract is written
    down. State layout: ``[x, y, vx, vy, theta, omega(, z)]`` (cm, cm/s, rad,
    rad/s, cm); consumers index 0,1,4,5, with ``z`` at index 6 when present.

    ``predict_ahead`` is a **static** method (constant-velocity extrapolation of
    a given state): the ballistic modules call it as ``FullStateKF.predict_ahead``
    without holding an estimator instance, which is why shot-timing/continuous-fire
    no longer need ``set_estimator``.
    """

    def reinit(self, prior: np.ndarray) -> None:
        """Re-seed the filter from a 6- or 7-element prior."""
        ...

    def update(
        self,
        dt: float,
        measurements: np.ndarray,
        r: Optional[float] = None,
        z_meas: Optional[float] = None,
    ) -> tuple[np.ndarray, float]:
        """Predict ``dt`` seconds, correct with panel measurements (M, 3); return (estimate, confidence).

        ``r`` overrides the orbit radius for the back-projection; ``z_meas`` is
        the measured panel-center mid-height (FullStateKF only; ignored by the
        6-D particle filter).
        """
        ...

    def update_with_no_observation(self, dt: float) -> tuple[np.ndarray, float]:
        """Predict only; return (estimate, confidence)."""
        ...

    def accel(self) -> Optional[np.ndarray]:
        """Filtered centre ``[ax, ay]`` accel (cm/s^2), or ``None`` if not modelled."""
        ...

    @staticmethod
    def predict_ahead(
        state: np.ndarray, dt: float, accel: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extrapolate ``state`` ``dt`` ahead (constant accel if ``accel`` given), no side effects."""
        ...


class SinglePanelEstimator(Protocol):
    """The single-panel estimator interface (the closest-panel CV tracker).

    Implemented structurally by ``PositionKF`` (used with orbit radius ``r=0`` so
    the back-projection degenerates to identity -- the measurement *is* the panel
    position). State layout: ``[x, y, vx, vy]`` (cm, cm/s), turret frame.

    Like ``FullStateEstimator``, ``predict_ahead`` is **static** so the
    single-panel ballistic module can project the tracked panel forward without
    holding the estimator instance.
    """

    def reinit(self, x0: float, y0: float, vx0: float = 0.0, vy0: float = 0.0) -> None:
        """Re-seed the state at ``(x0, y0, vx0, vy0)``."""
        ...

    def update(
        self,
        dt: float,
        x_obs: float,
        y_obs: float,
        yaw_obs: float,
        r: Optional[float] = None,
    ) -> tuple[np.ndarray, float]:
        """Predict ``dt`` seconds, correct with a panel observation; return (estimate, NIS)."""
        ...

    def update_with_no_observation(self, dt: float) -> tuple[np.ndarray, None]:
        """Predict only (no observation this frame)."""
        ...

    def accel(self) -> Optional[np.ndarray]:
        """Filtered ``[ax, ay]`` accel (cm/s^2), or ``None`` if not modelled."""
        ...

    @staticmethod
    def predict_ahead(
        state: np.ndarray, dt: float, accel: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extrapolate ``[x, y, vx, vy]`` ``dt`` ahead (constant accel if ``accel`` given)."""
        ...
