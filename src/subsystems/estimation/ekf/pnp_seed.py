"""EKF seed pose from one frame's panels via IPPE PnP (turret frame).

The EKF engine calls this only when the filter is uninitialised -- the
per-frame detector (``LightPanelDetectorModule``) does no PnP. Port of
Full-Robot-EKF ``pnp_init.py`` (no-decay 7f06e9d), moved from a level camera
to AutoX's turret frame via the camera->turret extrinsic:

  * each panel: ``cv2.solvePnPGeneric(SOLVEPNP_IPPE)`` on the 4 light-bar ends
    -> two candidate poses (best reprojection error first); keep candidates
    whose outward normal faces the camera;
  * two panels: of the candidate combos keep those that ``pair_geometry``
    accepts (yaws ~90 deg apart, centre behind both, sane radii) and take the
    lowest summed error -> ``FullRobotEKF.seed_pair``;
  * otherwise: the lowest-error single candidate -> ``FullRobotEKF.seed``
    (awakening's rule: centre INIT_RADIUS behind the panel).

The panel model is the EKF's own (``model.PANEL_WIDTH`` etc.), so the seed and
h(x) agree on geometry.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2 as cv
import numpy as np

from src.subsystems.estimation.ekf import model
from src.subsystems.estimation.ekf.filter import FullRobotEKF, pair_geometry

# Light-bar ends in the panel plane: x = right (seen from outside), y = up along
# the (tilted) panel, z = outward normal. Left top, left bottom, right bottom, right top.
_LX = model.PANEL_WIDTH / 2.0 - model.LIGHT_WIDTH / 2.0
_HH = model.LIGHT_HEIGHT / 2.0
PANEL_OBJ = np.array(
    [[-_LX, _HH, 0.0], [-_LX, -_HH, 0.0], [_LX, -_HH, 0.0], [_LX, _HH, 0.0]],
    dtype=np.float64,
)


@dataclass
class PanelPose:
    """One IPPE pose of a panel, in the turret frame."""

    pos: np.ndarray   # (3,) panel centre, turret frame (m)
    yaw: float        # outward-normal azimuth from turret +x (rad)
    err: float        # IPPE reprojection error (px RMS)


@dataclass
class Seed:
    """The engine's seed choice: one panel, or an adjacent pair (a -> panel 0)."""

    mode: str                         # "two_panels" | "one_panel"
    a: PanelPose                      # becomes panel 0
    b: Optional[PanelPose] = None     # odd neighbour (two_panels only)

    def apply(self, ekf: FullRobotEKF) -> None:
        """Initialise ``ekf`` from this seed."""
        if self.b is not None:
            ok = ekf.seed_pair(self.a.pos, self.a.yaw, self.b.pos, self.b.yaw)
            assert ok, "choose_seed only returns pairs pair_geometry accepts"
        else:
            ekf.seed(self.a.pos, self.a.yaw)


def _ends(light_corners: np.ndarray) -> np.ndarray:
    """(4, 2) light-bar ends: left top, left bottom, right bottom, right top."""
    lc = np.asarray(light_corners, dtype=np.float64)
    top = (lc[:, 0] + lc[:, 1]) / 2.0
    bottom = (lc[:, 2] + lc[:, 3]) / 2.0
    return np.array([top[0], bottom[0], bottom[1], top[1]])


def panel_candidates(
    light_corners: np.ndarray, K: np.ndarray, dist: np.ndarray, cam_to_turret: np.ndarray,
) -> List[PanelPose]:
    """IPPE poses of one panel that face the camera, best reprojection error first."""
    try:
        _, rvecs, tvecs, errs = cv.solvePnPGeneric(
            PANEL_OBJ, _ends(light_corners), K, dist, flags=cv.SOLVEPNP_IPPE)
    except cv.error:
        return []
    R_ct, t_ct = cam_to_turret[:3, :3], cam_to_turret[:3, 3]
    out = []
    for rv, tv, e in zip(rvecs, tvecs, np.asarray(errs).reshape(-1)):
        t = np.asarray(tv, dtype=np.float64).reshape(3)       # OpenCV: x right, y down, z fwd
        n = cv.Rodrigues(rv)[0][:, 2]                         # outward plane normal
        if n @ -t <= 0 or t[2] <= 0:                          # must face the camera
            continue
        relabel = lambda v: np.array([v[0], v[2], -v[1]])    # -> right / fwd / up
        pos = R_ct @ relabel(t) + t_ct
        n_t = R_ct @ relabel(n)
        if np.hypot(n_t[0], n_t[1]) < 1e-6:
            continue
        out.append(PanelPose(pos=pos, yaw=float(np.arctan2(n_t[1], n_t[0])), err=float(e)))
    return out


def choose_seed(cands: Sequence[List[PanelPose]]) -> Optional[Seed]:
    """Seed from per-panel candidates (index 0 = panel a). None if PnP gave nothing."""
    usable = [c for c in cands if c]
    if len(usable) >= 2:
        best = None
        for ca in usable[0]:
            for cb in usable[1]:
                if pair_geometry(ca.pos, ca.yaw, cb.pos, cb.yaw) is None:
                    continue
                if best is None or ca.err + cb.err < best[0]:
                    best = (ca.err + cb.err, ca, cb)
        if best is not None:
            return Seed("two_panels", best[1], best[2])
    if not usable:
        return None
    return Seed("one_panel", min((c[0] for c in usable), key=lambda p: p.err))
