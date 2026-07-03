"""Linear Kalman filter for a single, fixed-radius, constant-rate spinning target.

State vector (6-D, matching ``ParticleFilter`` so the continuous-fire ballistics
module works unchanged)::

    x = [x_c, y_c, vx, vy, theta, omega]   # cm, cm, cm/s, cm/s, rad, rad/s

Process model is constant-velocity for the centre and constant-rate for the
spin phase (``F`` recomputed each step with the current ``dt``).

Measurement model (per panel observation ``(x_obs, y_obs, yaw_obs)``) splits
into two sequential linear updates:

1. **Position via back-projection.** Given a known orbit radius ``R`` and the
   panel's *measured* yaw, the panel's centre of rotation is at
   ``(x_obs - R*cos(yaw_obs), y_obs - R*sin(yaw_obs))``. That is a direct,
   linear measurement of ``(x_c, y_c)``.
2. **Yaw with (2*pi/N)-wrapped innovation.** The target has ``N``-fold
   rotational symmetry (``N`` identical panels evenly spaced in azimuth), so the
   panel-yaw measurement repeats every ``360/N`` degrees. The innovation is
   wrapped into ``(-pi/N, pi/N]`` before the update so a measurement of the
   "wrong" (but identical) panel doesn't yank theta. We do this update by hand
   because filterpy's ``update`` has no residual-override hook.

This keeps every Jacobian constant -- it is a linear KF with a wrapped yaw
innovation, not a true EKF. Mirrors the ``measurement_backproj`` trick already
used by ``src/subsystems/pf_cuda_cv.cu``.

The aerial rig has ``N = 3`` panels 120deg apart (``config.spinning_target``);
the ground robots the archived pipeline handled had ``N = 4``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter

# 10 RPM in rad/s. Used as the default seed for omega when the spinning_target
# config is unavailable (e.g. in the standalone __main__ sanity check).
_DEFAULT_OMEGA_RAD_S = 10.0 * 2.0 * np.pi / 60.0
# Default rotational symmetry: 3 panels, 120deg apart.
_DEFAULT_N_PANELS = 3

_H_POS = np.array(
    [[1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0]],
    dtype=float,
)
_H_YAW = np.array([[0, 0, 0, 0, 1, 0]], dtype=float)


def _wrap_yaw_innovation(y: float, n_panels: int) -> float:
    """Wrap a yaw residual into ``(-pi/N, pi/N]`` (``N``-fold panel symmetry)."""
    period = 2.0 * np.pi / n_panels
    half = period / 2.0
    return ((y + half) % period) - half


class EkfTracker:
    """Linear Kalman tracker for the spinning-target auto-aim pipeline.

    Attributes:
        x: Current state estimate, shape (6,).
        P: Current state covariance, shape (6, 6).
        radius: Known panel orbit radius (cm).
        omega0: Known nominal spin rate (rad/s); used as the reinit seed.
        n_panels: Rotational symmetry order (evenly-spaced identical panels);
            sets the yaw-innovation wrap period to ``2*pi/n_panels``.
    """

    def __init__(
        self,
        radius: float,
        omega0: float = _DEFAULT_OMEGA_RAD_S,
        n_panels: int = _DEFAULT_N_PANELS,
        r_pos: float = 20.0,
        r_yaw: float = np.radians(15),
        q_vel: Optional[np.ndarray] = None,
        init_std: Optional[np.ndarray] = None,
    ) -> None:
        """Build a tracker tuned for cm / rad units.

        Args:
            radius: Panel orbit radius in cm (same for all arms of the rig).
            omega0: Known constant spin rate in rad/s (e.g. ``10 RPM`` ->
                ``pi/3``). Used as the prior seed; tiny process noise lets the
                filter absorb small deviations.
            n_panels: Number of identical, evenly-spaced panels (3 for the
                aerial rig). Sets the yaw-innovation wrap to ``2*pi/n_panels``.
            r_pos: Position-measurement std (cm). Defaults to the value the
                particle filter is tuned with.
            r_yaw: Yaw-measurement std (rad).
            q_vel: Per-dim process-noise std for ``[vx, vy, omega]`` (cm/s,
                cm/s, rad/s). Mirrors ``_default_particle_filter``'s ``Q_vel``.
            init_std: Per-dim initial spread std for ``[xc, yc, vx, vy, theta,
                omega]``. Sets the diagonal of the initial ``P``.
        """
        self.radius = float(radius)
        self.omega0 = float(omega0)
        self.n_panels = int(n_panels)

        self._r_pos = float(r_pos)
        self._r_yaw = float(r_yaw)

        # Mirror the particle-filter defaults so behaviour is comparable
        # side-by-side. ``Q`` for ``[xc, yc]`` is small because the centre only
        # moves through the velocity terms; ``Q`` for ``omega`` is tiny because
        # the rate is *known*.
        if q_vel is None:
            q_vel = np.array([100.0, 100.0, 0.6], dtype=float)
        if init_std is None:
            init_std = np.array([100.0, 100.0, 10.0, 10.0, 0.3, 15.0], dtype=float)

        self._q_diag = np.array(
            [1.0, 1.0, q_vel[0] ** 2, q_vel[1] ** 2, 0.01, q_vel[2] ** 2],
            dtype=float,
        )
        self._init_std = np.asarray(init_std, dtype=float)

        self._kf = KalmanFilter(dim_x=6, dim_z=2)
        self._kf.H = _H_POS.copy()
        self._kf.R = np.eye(2) * (self._r_pos ** 2)

        # Seed at origin so the filter is well-defined before the first
        # observation arrives. The engine will call ``reinit`` on the first
        # panel seen, so these values don't materially affect tracking.
        self.reinit(np.array([0.0, 0.0, 0.0, 0.0, 0.0, self.omega0]))

    # ------------------------------------------------------------------
    # Public state accessors
    # ------------------------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        return self._kf.x

    @property
    def P(self) -> np.ndarray:
        return self._kf.P

    # ------------------------------------------------------------------
    # Filter operations
    # ------------------------------------------------------------------

    def reinit(self, prior: np.ndarray) -> None:
        """Re-seed the state at ``prior`` with a wide initial covariance.

        Called by the estimation module the first frame a panel becomes
        visible (mirrors ``ParticleFilter.reinit``).
        """
        prior = np.asarray(prior, dtype=float).reshape(6)
        self._kf.x = prior.copy()
        self._kf.P = np.diag(self._init_std ** 2)

    def _set_F(self, dt: float) -> None:
        F = np.eye(6, dtype=float)
        F[0, 2] = dt
        F[1, 3] = dt
        F[4, 5] = dt
        self._kf.F = F
        # Scale process noise with dt so the filter behaves sensibly across
        # variable frame intervals.
        self._kf.Q = np.diag(self._q_diag) * max(dt, 1e-6)

    def predict(self, dt: float) -> None:
        """Advance state + covariance by ``dt`` seconds (no measurement)."""
        self._set_F(dt)
        self._kf.predict()

    def update(self, x_obs: float, y_obs: float, yaw_obs: float) -> None:
        """Fuse a single panel observation in the turret frame.

        Args:
            x_obs: Panel x in the turret frame (cm).
            y_obs: Panel y in the turret frame (cm).
            yaw_obs: Panel yaw in the turret frame (rad).
        """
        # 1. Back-projected position update (linear).
        cx = x_obs - self.radius * np.cos(yaw_obs)
        cy = y_obs - self.radius * np.sin(yaw_obs)
        self._kf.update(np.array([cx, cy], dtype=float))

        # 2. Yaw update with wrapped innovation. Done by hand because
        # filterpy.kalman.KalmanFilter.update has no residual hook.
        H = _H_YAW
        P = self._kf.P
        R_yaw = np.array([[self._r_yaw ** 2]])
        y = _wrap_yaw_innovation(yaw_obs - float(H @ self._kf.x), self.n_panels)
        S = H @ P @ H.T + R_yaw
        K = (P @ H.T) / float(S[0, 0])
        self._kf.x = self._kf.x + (K.flatten() * y)
        I_KH = np.eye(6) - K @ H
        self._kf.P = I_KH @ P

    def update_with_no_observation(self, dt: float) -> None:
        """Predict-only step (matches ``ParticleFilter.update_with_no_observation``)."""
        self.predict(dt)

    def prediction(self, dt: float) -> np.ndarray:
        """Extrapolate state ``dt`` seconds ahead without side effects.

        This is the duck-typed contract the continuous-fire ballistics module
        calls to advance the estimate by the processing + feeder delay before
        solving. Mirrors ``ParticleFilter.prediction`` /
        ``pf_cuda_cv.cu``'s ``pf_prediction``.
        """
        pred = self._kf.x.copy()
        pred[0] += pred[2] * dt
        pred[1] += pred[3] * dt
        pred[4] += pred[5] * dt
        return pred


# ---------------------------------------------------------------------------
# Standalone sanity check -- run with `uv run python -m src.subsystems.ekf`.
# Feeds a circular trajectory with measurement noise and prints the state.
# ---------------------------------------------------------------------------

def _self_test() -> None:
    rng = np.random.default_rng(0)
    R = 23.5
    omega = _DEFAULT_OMEGA_RAD_S          # 10 RPM
    vx, vy = 5.0, -3.0                     # cm/s; centre drift
    dt = 1.0 / 90.0                        # 90 Hz frames

    tracker = EkfTracker(radius=R, omega0=omega)

    true_xc, true_yc, theta = 100.0, 200.0, 0.0
    tracker.reinit(np.array([true_xc + 5, true_yc - 5, 0, 0, theta, omega]))

    for k in range(200):
        true_xc += vx * dt
        true_yc += vy * dt
        theta += omega * dt

        # Panel in the truth frame: centre + R*[cos(theta), sin(theta)]. Which
        # of the N identical panels doesn't matter -- the back-projection trick
        # works for any of them because the yaw measurement is that panel's yaw.
        x_obs = true_xc + R * np.cos(theta) + rng.normal(scale=2.0)
        y_obs = true_yc + R * np.sin(theta) + rng.normal(scale=2.0)
        yaw_obs = theta + rng.normal(scale=np.radians(3.0))

        tracker.predict(dt)
        tracker.update(x_obs, y_obs, yaw_obs)

        if k % 20 == 0:
            est = tracker.x
            print(
                f"k={k:3d} | true xc/yc=({true_xc:6.2f},{true_yc:6.2f}) "
                f"theta={theta:5.2f}  | est xc/yc=({est[0]:6.2f},{est[1]:6.2f}) "
                f"vx/vy=({est[2]:5.2f},{est[3]:5.2f}) "
                f"theta={est[4]:5.2f} omega={est[5]:5.3f}"
            )


if __name__ == "__main__":
    _self_test()
