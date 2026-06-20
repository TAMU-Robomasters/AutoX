"""Per-axis Interacting Multiple Model (IMM) estimator for centre/panel position.

A faithful numpy port of the **validated** tuning harness in
``../estimation_experiments/est_exp/filters/{kf,imm}.py`` (see
``../estimation_experiments/DEPLOY_CVCA.md``), adapted from SI metres to the
estimator's working units (cm, cm/s, cm/s^2, cm/s^3, s).

Topology (the deploy doc's choice -- "the IMM is per-axis 1-D"): one 1-D IMM per
spatial axis, each running a **CV** model ``[pos, vel]`` and a **CA** model
``[pos, vel, acc]`` and blending them by model probability. ``PositionIMM`` wraps
two such per-axis IMMs behind the *same interface* as ``PositionKF`` (back-project
a panel observation to the centre, then feed the per-axis positions), so it is a
drop-in for both the single-panel tracker and the centre sub-filter inside
``FullStateKF``. Only the position is replaced -- the angle/height KFs stay
constant-velocity (the user's "NOT the angular stuff").

Unequal-dimension mixing (Granstrom-Willett-Bar-Shalom, IEEE TAES 2015):
truncate when going to a smaller model (keep the leading block); expand to a
larger one by appending the missing component with a **uniform** augmentation
distribution on ``[-lim, +lim]`` (mean 0, variance ``(2*lim)^2/12``) -- the
magnitude-reflecting choice, not zero-padding. The model-probability update is
done in the log domain with a log-sum-exp normaliser.

Process-noise convention (matches the harness/GDScript exactly):
    CV -> DWNA (discrete white-noise *acceleration*):
            Q = q^2 * G G^T,  G = [dt^2/2, dt]^T,  q = sigma_w  [cm/s^2]
    CA -> continuous white-noise *jerk*:
            Q = q^2 * [[dt^5/20, dt^4/8, dt^3/6],
                       [dt^4/8,  dt^3/3, dt^2/2],
                       [dt^3/6,  dt^2/2, dt    ]],  q = sigma_j  [cm/s^3]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.subsystems.estimation.motion_models import (
    IMM as IMM_MODEL,  # config switch value
)
from src.toolbox.globals import config

# Canonical component order; each model occupies a prefix of this.
POS, VEL, ACC = 0, 1, 2


# ---------------------------------------------------------------------------
# Per-axis 1-D Kalman models
# ---------------------------------------------------------------------------


class LinearKF1D:
    """Single-axis linear KF with a scalar position measurement.

    Subclasses provide ``_F(dt)`` / ``_Q(dt)``. ``predict`` and ``update`` are
    split so the IMM can mix posteriors before the predict step.
    """

    n: int = 0  # state dimension

    def __init__(self, x0: np.ndarray, P0: np.ndarray, R: float, q: float) -> None:
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.R = float(R)
        self.q = float(q)
        # Diagnostics from the last update():
        self.innovation = 0.0
        self.S = self.R
        self.loglik = 0.0  # log N(innovation; 0, S)

    def _F(self, dt: float) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def _Q(self, dt: float) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def predict(self, dt: float) -> None:
        """Advance the state/covariance by ``dt`` (no measurement)."""
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update(self, z: float) -> None:
        """Scalar position update; records innovation, ``S`` and log-likelihood."""
        Px0 = self.P[:, POS]  # P @ H^T (H selects position)
        self.S = float(self.P[POS, POS] + self.R)
        K = Px0 / self.S  # Kalman gain (n,)
        self.innovation = float(z - self.x[POS])
        self.x = self.x + K * self.innovation
        # Joseph-free form mirroring the harness: P -= K (H P)
        self.P = self.P - np.outer(K, self.P[POS, :])
        self.P = 0.5 * (self.P + self.P.T)  # symmetrise
        self.loglik = -0.5 * (
            np.log(2.0 * np.pi) + np.log(self.S) + self.innovation**2 / self.S
        )

    def predict_update(self, z: float, dt: float) -> None:
        """Predict ``dt`` ahead then correct with position ``z``."""
        self.predict(dt)
        self.update(z)

    def predict_only(self, dt: float) -> None:
        """Predict step with no measurement (uniform interface with the IMM)."""
        self.predict(dt)

    def nis(self) -> float:
        """Normalised innovation squared from the last update."""
        return float(self.innovation**2 / self.S)

    def predict_pos_var(self, dt: float, with_Q: bool = True) -> float:
        """Position variance of the ``dt``-ahead prediction (always includes Q)."""
        F = self._F(dt)
        Pp = F @ self.P @ F.T + self._Q(dt)
        return float(Pp[POS, POS])


class CVKF(LinearKF1D):
    """Constant-velocity, DWNA process noise. ``q = sigma_w`` [cm/s^2]."""

    n = 2

    def _F(self, dt: float) -> np.ndarray:
        return np.array([[1.0, dt], [0.0, 1.0]])

    def _Q(self, dt: float) -> np.ndarray:
        G = np.array([0.5 * dt * dt, dt])
        return self.q**2 * np.outer(G, G)


class CAKF(LinearKF1D):
    """Constant-acceleration, continuous white-noise-jerk Q. ``q = sigma_j`` [cm/s^3]."""

    n = 3

    def _F(self, dt: float) -> np.ndarray:
        return np.array(
            [
                [1.0, dt, 0.5 * dt * dt],
                [0.0, 1.0, dt],
                [0.0, 0.0, 1.0],
            ]
        )

    def _Q(self, dt: float) -> np.ndarray:
        d2 = dt * dt
        d3 = d2 * dt
        d4 = d3 * dt
        d5 = d4 * dt
        return self.q**2 * np.array(
            [
                [d5 / 20.0, d4 / 8.0, d3 / 6.0],
                [d4 / 8.0, d3 / 3.0, d2 / 2.0],
                [d3 / 6.0, d2 / 2.0, dt],
            ]
        )


# ---------------------------------------------------------------------------
# Unequal-dimension mixing helpers
# ---------------------------------------------------------------------------


@dataclass
class AugBounds:
    """Uniform augmentation bounds per appended component (symmetric +/-limit).

    ``vel`` is used when expanding into a velocity state (CP->CV); ``acc`` when
    expanding into an acceleration state (CV->CA).
    """

    vel: float = 50.0  # cm/s
    acc: float = 300.0  # cm/s^2

    def limit(self, comp_index: int) -> float:
        """Augmentation half-width for component ``comp_index`` (1=vel, 2=acc)."""
        return self.vel if comp_index == VEL else self.acc


def expand(mean: np.ndarray, cov: np.ndarray, n_to: int, aug: AugBounds):
    """Map a ``(mean, cov)`` of dim ``len(mean)`` into a model of dim ``n_to``.

    Truncates (keep the leading block) if ``n_to`` is smaller; expands with the
    uniform augmentation distribution if larger.
    """
    n_from = mean.shape[0]
    if n_to == n_from:
        return mean.copy(), cov.copy()
    if n_to < n_from:
        return mean[:n_to].copy(), cov[:n_to, :n_to].copy()
    new_mean = np.zeros(n_to)
    new_mean[:n_from] = mean
    new_cov = np.zeros((n_to, n_to))
    new_cov[:n_from, :n_from] = cov
    for c in range(n_from, n_to):
        lim = aug.limit(c)
        new_mean[c] = 0.0  # (a+b)/2 for symmetric bounds
        new_cov[c, c] = (2.0 * lim) ** 2 / 12.0  # (b-a)^2 / 12
    return new_mean, new_cov


def make_Pi(self_probs: List[float]) -> np.ndarray:
    """Diagonally-dominant row-stochastic ``Pi`` from per-model self-stay probs.

    Off-diagonal mass is split evenly among the other models.
    ``Pi[i, j] = P(model j now | model i previously)``.
    """
    M = len(self_probs)
    Pi = np.zeros((M, M))
    for i, p in enumerate(self_probs):
        Pi[i, i] = p
        if M > 1:
            off = (1.0 - p) / (M - 1)
            for j in range(M):
                if j != i:
                    Pi[i, j] = off
    return Pi


# ---------------------------------------------------------------------------
# Per-axis IMM
# ---------------------------------------------------------------------------


class IMM:
    """Interacting Multiple Model estimator over per-axis ``LinearKF1D`` models."""

    def __init__(
        self,
        models: List[LinearKF1D],
        Pi: np.ndarray,
        mu0: np.ndarray,
        aug: Optional[AugBounds] = None,
    ) -> None:
        self.models = models
        self.Pi = np.asarray(Pi, dtype=float)  # Pi[i, j] = P(j_now | i_prev)
        self.mu = np.asarray(mu0, dtype=float).copy()
        self.aug = aug or AugBounds()
        self.max_n = max(m.n for m in models)
        self.M = len(models)
        self.x = np.zeros(self.max_n)  # combined (mixed) estimate, max-dim frame
        self.P = np.eye(self.max_n)
        self._combine()

    def predict_update(self, z: float, dt: float) -> None:
        """One full IMM step: mix, per-model predict+update, reweight, combine."""
        self._mix()
        for m in self.models:
            m.predict_update(z, dt)
        self._update_mu()
        self._combine()

    def predict_only(self, dt: float) -> None:
        """No measurement this step: mix, advance mu via Pi, predict, combine."""
        self._mix()
        c = self.Pi.T @ self.mu
        self.mu = c / c.sum()
        for m in self.models:
            m.predict(dt)
        self._combine()

    def _mix(self) -> None:
        c = self.Pi.T @ self.mu  # predicted model probs c_j = sum_i Pi[i,j] mu_i
        c = np.maximum(c, 1e-300)
        w = (self.Pi * self.mu[:, None]) / c[None, :]  # mixing weights (M, M)

        mixed_x = []
        mixed_P = []
        for j, mj in enumerate(self.models):
            nj = mj.n
            xj = np.zeros(nj)
            contrib = []
            for i, mi in enumerate(self.models):
                xi_j, Pi_j = expand(mi.x, mi.P, nj, self.aug)
                xj += w[i, j] * xi_j
                contrib.append((w[i, j], xi_j, Pi_j))
            Pj = np.zeros((nj, nj))
            for wij, xi_j, Pi_j in contrib:
                d = (xi_j - xj).reshape(-1, 1)
                Pj += wij * (Pi_j + d @ d.T)
            mixed_x.append(xj)
            mixed_P.append(Pj)

        for m, xj, Pj in zip(self.models, mixed_x, mixed_P):
            m.x = xj
            m.P = Pj

    def _update_mu(self) -> None:
        c = self.Pi.T @ self.mu
        c = np.maximum(c, 1e-300)
        log_post = np.log(c) + np.array([m.loglik for m in self.models])
        m_max = np.max(log_post)
        log_norm = m_max + np.log(np.sum(np.exp(log_post - m_max)))  # log-sum-exp
        self.mu = np.exp(log_post - log_norm)
        self.mu = np.maximum(self.mu, 1e-12)
        self.mu /= self.mu.sum()

    def _combine(self) -> None:
        expanded = [expand(m.x, m.P, self.max_n, self.aug) for m in self.models]
        x = np.zeros(self.max_n)
        for mu_j, (xe, _Pe) in zip(self.mu, expanded):
            x += mu_j * xe
        P = np.zeros((self.max_n, self.max_n))
        for mu_j, (xe, Pe) in zip(self.mu, expanded):
            d = (xe - x).reshape(-1, 1)
            P += mu_j * (Pe + d @ d.T)
        self.x = x
        self.P = 0.5 * (P + P.T)

    def mixture_nis(self) -> float:
        """Model-probability-weighted NIS from the last update (mixture innovation)."""
        return float(sum(mu_j * m.nis() for mu_j, m in zip(self.mu, self.models)))

    def predict_pos_var(self, dt: float, with_Q: bool = True) -> float:
        """Position variance of the ``dt``-ahead prediction of the combined state.

        ``with_Q=True`` (the honest gate, DEPLOY_CVCA fix #1): propagate each model
        with its *own* F and Q over ``dt``, expand to the max frame, then
        moment-match with the current model probabilities -- the +Q grows the
        uncertainty over the horizon, which the filtered covariance omits.
        """
        if not with_Q:
            F = self._Fpred(dt)
            return float((F @ self.P @ F.T)[POS, POS])
        xs, Ps = [], []
        for m in self.models:
            Fi = m._F(dt)
            xi = Fi @ m.x
            Pi = Fi @ m.P @ Fi.T + m._Q(dt)
            xe, Pe = expand(xi, Pi, self.max_n, self.aug)
            xs.append(xe)
            Ps.append(Pe)
        x = sum(mu_i * xe for mu_i, xe in zip(self.mu, xs))
        var = 0.0
        for mu_i, xe, Pe in zip(self.mu, xs, Ps):
            d = xe[POS] - x[POS]
            var += mu_i * (Pe[POS, POS] + d * d)
        return float(var)

    def _Fpred(self, dt: float) -> np.ndarray:
        n = self.max_n
        F = np.eye(n)
        if n >= 2:
            F[POS, VEL] = dt
        if n >= 3:
            F[POS, ACC] = 0.5 * dt * dt
            F[VEL, ACC] = dt
        return F


# ---------------------------------------------------------------------------
# 2-D adapter: a drop-in replacement for PositionKF
# ---------------------------------------------------------------------------

# Model order within each per-axis IMM. CA (index 1) is the "maneuver" model
# whose probability drives the maneuver-aware gate inflation (DEPLOY_CVCA section 4).
_CV, _CA = 0, 1


class PositionIMM:
    """Per-axis CV+CA IMM behind the ``PositionKF`` interface (centre/panel position).

    Tracks the robot's 2-D centre (or a panel, with orbit radius ``r=0``) by
    running one 1-D IMM per axis on the back-projected position. ``estimate()``
    returns ``[x, y, vx, vy]`` and ``accel()`` the blended ``[ax, ay]`` -- so every
    downstream consumer (FullStateKF's 7-D layout, the static ``predict_ahead``
    extrapolators, ballistics) is unaffected by the IMM vs single-KF choice. The
    gate radius adds the maneuver-aware inflation the single-filter gate cannot.
    """

    def __init__(
        self,
        *,
        r_pos: float = 50.0,
        r: float = 23.5,
        init_std: tuple[float, float, float, float] = (100.0, 100.0, 10.0, 10.0),
        init_std_accel: float = 200.0,
        sigma_w: float = 25.0,
        sigma_j: float = 1000.0,
        acc_bound: float = 300.0,
        p_cv: float = 0.99,
        p_ca: float = 0.90,
        mu0_ca: float = 0.1,
        man_bias: float = 15.0,
    ) -> None:
        self.r_pos = r_pos
        self.r = r
        self.R = float(r_pos) ** 2  # per-axis measurement-noise variance (cm^2)
        self.init_std = np.asarray(init_std, dtype=np.float64)
        self.init_std_accel = float(init_std_accel)
        self._sigma_w = float(sigma_w)
        self._sigma_j = float(sigma_j)
        self._acc_bound = float(acc_bound)
        self._Pi = make_Pi([float(p_cv), float(p_ca)])
        self._mu0 = np.array([1.0 - float(mu0_ca), float(mu0_ca)], dtype=np.float64)
        self.man_bias = float(man_bias)
        self._imm_x = self._build_axis(0.0, 0.0)
        self._imm_y = self._build_axis(0.0, 0.0)

    def _build_axis(self, pos0: float, vel0: float) -> IMM:
        pos_var = float(self.init_std[0]) ** 2
        vel_var = float(self.init_std[2]) ** 2
        acc_var = self.init_std_accel**2
        cv = CVKF(
            x0=np.array([pos0, vel0]),
            P0=np.diag([pos_var, vel_var]),
            R=self.R,
            q=self._sigma_w,
        )
        ca = CAKF(
            x0=np.array([pos0, vel0, 0.0]),
            P0=np.diag([pos_var, vel_var, acc_var]),
            R=self.R,
            q=self._sigma_j,
        )
        return IMM(
            models=[cv, ca],
            Pi=self._Pi,
            mu0=self._mu0,
            aug=AugBounds(acc=self._acc_bound),
        )

    def reinit(self, x0: float, y0: float, vx0: float = 0.0, vy0: float = 0.0) -> None:
        """Re-seed both per-axis IMMs at ``(x0, y0, vx0, vy0)`` (accel starts at 0)."""
        self._imm_x = self._build_axis(x0, vx0)
        self._imm_y = self._build_axis(y0, vy0)

    @staticmethod
    def back_project(x_obs: float, y_obs: float, yaw_obs: float, r: float) -> tuple[float, float]:
        """Back-project a panel observation to the robot's centre (r=0 -> identity)."""
        return x_obs - r * np.cos(yaw_obs), y_obs - r * np.sin(yaw_obs)

    def update(
        self,
        dt: float,
        x_obs: float,
        y_obs: float,
        yaw_obs: float,
        r: Optional[float] = None,
    ) -> tuple[np.ndarray, float]:
        """Predict ``dt`` then correct with a back-projected panel observation.

        Returns ``([x, y, vx, vy], nis)``; ``nis`` is the model-probability-weighted
        mixture NIS summed across axes (a soft confidence proxy, like PositionKF).
        """
        cx, cy = self.back_project(x_obs, y_obs, yaw_obs, self.r if r is None else r)
        self._imm_x.predict_update(cx, dt)
        self._imm_y.predict_update(cy, dt)
        nis = self._imm_x.mixture_nis() + self._imm_y.mixture_nis()
        return self.estimate(), nis

    def update_with_no_observation(self, dt: float) -> tuple[np.ndarray, None]:
        """Predict only (no observation this frame)."""
        self._imm_x.predict_only(dt)
        self._imm_y.predict_only(dt)
        return self.estimate(), None

    def estimate(self) -> np.ndarray:
        """Published ``[x, y, vx, vy]`` from the two combined per-axis states."""
        return np.array(
            [self._imm_x.x[POS], self._imm_y.x[POS], self._imm_x.x[VEL], self._imm_y.x[VEL]],
            dtype=np.float64,
        )

    def accel(self) -> Optional[np.ndarray]:
        """Blended centre ``[ax, ay]`` acceleration (cm/s^2) from the combined states."""
        return np.array([self._imm_x.x[ACC], self._imm_y.x[ACC]], dtype=np.float64)

    def p_man(self) -> float:
        """Maneuver (CA) model probability: the max over the two axes (most cautious)."""
        return max(float(self._imm_x.mu[_CA]), float(self._imm_y.mu[_CA]))

    def predict_pos_var(self, dt: float) -> tuple[float, float]:
        """Predicted ``(var_x, var_y)`` ``dt`` ahead, moment-matched with +Q."""
        return (
            self._imm_x.predict_pos_var(dt, with_Q=True),
            self._imm_y.predict_pos_var(dt, with_Q=True),
        )

    def gate_radius(self, dt: float) -> float:
        """Maneuver-aware 1-sigma gate radius (cm), DEPLOY_CVCA section 4.

        ``sqrt(var_x + var_y + (man_bias * p_man)^2)`` -- the maneuver-bias term
        (absent from the single-filter gate) inflates the radius while the CA model
        is active, so confidence stays honest mid-maneuver.
        """
        var_x, var_y = self.predict_pos_var(dt)
        return float(np.sqrt(var_x + var_y + (self.man_bias * self.p_man()) ** 2))

    @staticmethod
    def predict_ahead(
        state: np.ndarray, dt: float, accel: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Static ``[x, y, vx, vy]`` extrapolation (delegates to ``PositionKF``).

        Constant-acceleration when ``accel`` is given (the blended IMM accel),
        constant-velocity otherwise -- identical to the single-KF path so the
        ballistic modules need no IMM-specific branch.
        """
        from src.subsystems.estimation.filters import PositionKF

        return PositionKF.predict_ahead(state, dt, accel)


# ---------------------------------------------------------------------------
# Factory: pick PositionKF (CV/CA) or PositionIMM by config
# ---------------------------------------------------------------------------


def make_position_estimator(
    *,
    r_pos: float,
    r: float,
    init_std: tuple[float, float, float, float],
    q_vx: float = 50.0,
    q_vy: float = 50.0,
):
    """Build the centre/panel position estimator named by ``config.estimation.motion_model``.

    ``imm`` -> per-axis CV+CA ``PositionIMM`` (IMM params from ``estimation.imm_*``);
    ``constant_velocity`` / ``constant_acceleration`` -> ``PositionKF`` with the
    matching ``MotionModel``. Both honour the caller's ``r_pos`` (measurement noise
    std), orbit radius ``r``, and ``init_std`` so single-panel vs centre tuning is
    preserved. Returns a ``CenterPositionEstimator`` (see ``filters.py``).
    """
    # Imported here (not at module top) to avoid a filters<->imm import cycle.
    from src.subsystems.estimation.filters import PositionKF
    from src.subsystems.estimation.motion_models import make_motion_model

    est = config.estimation
    model = str(est.motion_model)
    if model == IMM_MODEL:
        return PositionIMM(
            r_pos=r_pos,
            r=r,
            init_std=init_std,
            init_std_accel=float(est.pos_init_std_accel),
            sigma_w=float(est.imm_sigma_w),
            sigma_j=float(est.imm_sigma_j),
            acc_bound=float(est.imm_acc_bound),
            p_cv=float(est.imm_p_cv),
            p_ca=float(est.imm_p_ca),
            mu0_ca=float(est.imm_mu0_ca),
            man_bias=float(est.imm_man_bias),
        )
    motion_model = make_motion_model(
        model,
        q_vx=q_vx,
        q_vy=q_vy,
        q_jerk=float(est.pos_q_jerk),
        init_std_accel=float(est.pos_init_std_accel),
    )
    return PositionKF(motion_model=motion_model, r_pos=r_pos, r=r, init_std=init_std)
