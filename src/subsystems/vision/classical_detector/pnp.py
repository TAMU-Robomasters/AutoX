"""PnP code used to determine the pose of panels."""

import cv2 as cv
import numpy as np

from src.subsystems.video_streaming.video_stream import Intrinsics, video_stream

# camera intrinsics
intrinsics: Intrinsics = video_stream.get_intrinsics()
dist = intrinsics.distortion_coefficients
cam_matrix = intrinsics.camera_matrix

panel_coordinates = np.array(
    [
        [-12.2 / 2, 12.5 / 2, 0],
        [12.2 / 2, 12.5 / 2, 0],
        [12.2 / 2, -12.5 / 2, 0],
        [-12.2 / 2, -12.5 / 2, 0],
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


def get_cord(panel):
    """Use PnP to find tvec and rvec of panels.

    :param list_of_points: List of points in image coordinates
    :return: [[tvec, rvec], ...]
    """
    if panel:
        points = panel.corners
        # Convert points to the correct format
        points = np.array(points, dtype=np.float32).reshape(-1, 2)

        # Ensure we have exactly 4 points
        if points.shape[0] != 4:
            pass
        if panel.id != 0:
            success, rvec, tvec = cv.solvePnP(
                panel_coordinates, points, cam_matrix, dist, flags=cv.SOLVEPNP_ITERATIVE
            )
        else:
            success, rvec, tvec = cv.solvePnP(
                hero_coordinates, points, cam_matrix, dist, flags=cv.SOLVEPNP_ITERATIVE
            )

        if success:
            panel.tvec = np.array([tvec[0], tvec[2], -tvec[1]])
            panel.rvec = rvec
