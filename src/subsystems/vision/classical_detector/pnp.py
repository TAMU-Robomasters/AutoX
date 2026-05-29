"""PnP code used to determine the pose of panels."""

import cv2 as cv
import numpy as np

from src.subsystems.video_streaming.video_stream import Intrinsics, video_stream

# camera intrinsics
intrinsics: Intrinsics = video_stream.get_intrinsics()
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
        [-21.8 / 2, 12.4 / 2, 0],
        [21.8 / 2, 12.4 / 2, 0],
        [21.8 / 2, -12.4 / 2, 0],
        [-21.8 / 2, -12.4 / 2, 0],
    ],
    dtype=np.float32,
)

hero_inner_coordinates = np.array(
    [
        [-21.8 / 2, 12.4 / 2, 0],
        [21.8 / 2, 12.4 / 2, 0],
        [21.8 / 2, -12.4 / 2, 0],
        [-21.8 / 2, -12.4 / 2, 0],
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
            panel.tvec = np.array([tvec[0], tvec[2], -tvec[1]])
            panel.rvec = rvec
            # Reproject the full outer-panel object points through the pose
            # we just solved for. The result is the canonical outer-panel
            # pixel boundary (shape (4, 1, 2)).
            outer_px, _ = cv.projectPoints(outer_obj, rvec, tvec, cam_matrix, dist)
            panel.outer_corners = outer_px.astype(np.int32)
