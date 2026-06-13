"""PnP code used to determine the pose of panels."""

import cv2 as cv
import numpy as np

from src.drivers.video_stream import Intrinsics, camera_info

# camera intrinsics (static; loaded once from the cameramodel file, no camera)
intrinsics: Intrinsics = camera_info.intrinsics()
dist = intrinsics.distortion_coefficients
cam_matrix = intrinsics.camera_matrix

small_panel_coordinates = np.array(
    [
        [-13.5 / 2, 12.5 / 2, 0],
        [13.5 / 2, 12.5 / 2, 0],
        [13.5 / 2, -12.5 / 2, 0],
        [-13.5 / 2, -12.5 / 2, 0],
    ],
    dtype=np.float32,
)

small_panel_inner_coordinates = np.array(
    [
        [-12.3 / 2, 5.53 / 2, 0],
        [12.3 / 2, 5.53 / 2, 0],
        [12.3 / 2, -5.53 / 2, 0],
        [-12.3 / 2, -5.53 / 2, 0],
    ],
    dtype=np.float32,
)


hero_coordinates = np.array(
    [
        [-23.0 / 2, 12.7 / 2, 0],
        [23.0 / 2, 12.7 / 2, 0],
        [23.0 / 2, -12.7 / 2, 0],
        [-23.0 / 2, -12.7 / 2, 0],
    ],
    dtype=np.float32,
)

hero_inner_coordinates = np.array(
    [
        [-21.8 / 2, 5.53 / 2, 0],
        [21.8 / 2, 5.53 / 2, 0],
        [21.8 / 2, -5.53 / 2, 0],
        [-21.8 / 2, -5.53 / 2, 0],
    ],
    dtype=np.float32,
)


def get_cord(panel):
    """Use PnP to find tvec and rvec of panels.

    Solves PnP with the inner light-bar corners (the high-confidence pixel
    observations) against the inner object coordinates. On success, also
    reprojects the outer panel object points through the recovered pose so
    ``panel.outer_corners`` holds canonical outer-panel pixel coordinates for
    display and downstream ArmorPanel construction.
    """
    if panel:
        # Use the inner light-bar corners — those are actual pixel observations,
        # not extrapolated from the assumed armor_height_ratio.
        points = np.array(panel.inner_corners, dtype=np.float32).reshape(-1, 2)

        # Ensure we have exactly 4 points
        if points.shape[0] != 4:
            pass
        if panel.id != 0:
            inner_obj = small_panel_inner_coordinates
            outer_obj = small_panel_coordinates
        else:
            inner_obj = hero_inner_coordinates
            outer_obj = hero_coordinates

        success, rvec, tvec = cv.solvePnP(
            inner_obj, points, cam_matrix, dist, flags=cv.SOLVEPNP_ITERATIVE
        )

        if success:
            # Axis relabel (OpenCV x-right/y-down/z-fwd -> right/fwd/up) as a
            # flat (3,) vector. solvePnP's tvec is (3, 1); index the scalars so
            # downstream (ArmorPanel.position, pf.py) sees a 3-vector, not (3, 1)
            # — otherwise position[0] is a 1-element array. This matters when the
            # embedded transform (which used to flatten it) is skipped, e.g. when
            # no serial/embedded board is connected.
            panel.tvec = np.array(
                [tvec[0, 0], tvec[2, 0], -tvec[1, 0]], dtype=np.float64
            )
            panel.rvec = rvec
            # Reproject the full outer-panel object points through the pose we
            # just solved for, to get the canonical outer-panel pixel boundary
            # (shape (4, 1, 2)) for display + ArmorPanel.
            outer_px, _ = cv.projectPoints(outer_obj, rvec, tvec, cam_matrix, dist)
            panel.outer_corners = outer_px.astype(np.int32)
