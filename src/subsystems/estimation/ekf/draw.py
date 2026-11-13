"""EKF debug overlay for the camera window, modelled on Full-Robot-EKF's window.

* blue dots  -- predicted light-bar corners of all 4 panels (hidden ones too)
* red dots   -- predicted corners of the panels the filter matches against
                (``model.visible_panels``: facing the camera and inside the
                horizontal FOV), drawn on top
* green      -- detected light bars (what the EKF update uses)
* yellow     -- HUD text, top left

Colours are BGR (OpenCV). Pure drawing on a given image; the engine decides
when to call it.
"""

from typing import List, Optional, Sequence

import cv2 as cv
import numpy as np

from src.subsystems.estimation.ekf import model

BLUE = (255, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
_MAX_PX = 1e5   # projected corners near the camera plane are finite but huge


def _dots(img: np.ndarray, uv: np.ndarray, color) -> None:
    for u, v in np.asarray(uv, dtype=np.float64).reshape(-1, 2):
        if np.isfinite(u) and np.isfinite(v) and abs(u) < _MAX_PX and abs(v) < _MAX_PX:
            cv.circle(img, (int(round(u)), int(round(v))), 3, color, -1)


def draw_ekf_overlay(
    img: np.ndarray,
    x: Optional[np.ndarray],
    cam: Optional[model.Camera],
    detections: Sequence[np.ndarray],
    hud: List[str],
) -> np.ndarray:
    """Draw predicted lights (if ``x`` and ``cam``), detected lights and the HUD in place."""
    if x is not None and cam is not None:
        vis = model.visible_panels(x, cam)
        for k in range(4):                       # per panel: one behind the camera
            uv = cam.project(model.light_points(x, [k]))   # doesn't hide the rest
            if uv is not None:
                _dots(img, uv, BLUE)
        for k in vis:
            uv = cam.project(model.light_points(x, [k]))
            if uv is not None:
                _dots(img, uv, RED)
    for lc in detections:
        for light in np.asarray(lc, dtype=np.float64).reshape(-1, 4, 2):
            if np.all(np.isfinite(light)):
                cv.polylines(img, [np.round(light).astype(np.int32).reshape(-1, 1, 2)],
                             True, GREEN, 1)
    for i, text in enumerate(hud):
        cv.putText(img, text, (10, 24 + i * 20), cv.FONT_HERSHEY_SIMPLEX, 0.52,
                   YELLOW, 1, cv.LINE_AA)
    return img
