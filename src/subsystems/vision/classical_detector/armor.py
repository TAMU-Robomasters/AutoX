"""Computer vision code for finding potential armor panels based on the lights detected."""

import math

import cv2 as cv
import numpy as np

from src.subsystems.display import display
from src.subsystems.vision.classical_detector import tuning
from src.toolbox.globals import config

armor_height_ratio = 12.5 / 5.2
armor_width_ration = 5.5 / 13


class Lights:  # noqa
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle


# TODO: consolidate this type with the other panel type
class Panel:  # noqa
    def __init__(self, corners, inner_corners, center, area):
        # Extrapolated outer corners (approximate, rely on armor_height_ratio).
        # Used by icon_detection.icon_detection to warp the full panel face.
        self.corners = corners
        # Light-bar inner-edge corners — the actual observed pixel locations
        # we feed to solvePnP. These are NOT extrapolated.
        self.inner_corners = inner_corners
        # Reprojected outer panel corners. Filled by pnp.get_cord after a
        # successful solve; this is the canonical outer-panel pixel boundary.
        self.outer_corners = None
        self.center = center
        self.tvec = None
        self.rvec = None
        self.id = None
        self.area = area


def bounding_boxes(contours, frame):
    """Creating bounding boxes around lights.

    :param frame:
    :param contours:
    :return: list of bounding boxes
    """
    b_boxes = []
    for contour in contours:  # have to keep
        rect = cv.minAreaRect(contour)
        h_holder, w_holder = rect[1]

        if rect[1][0] > rect[1][1]:
            h, w = rect[1]
        else:
            w, h = rect[1]

        angle = rect[2]
        if w_holder > h_holder:
            angle += 90

        # filter out bad detections: true if bad
        if abs(angle - 90) > 45:
            continue
        else:
            # TODO: add this to visualizer
            box = cv.boxPoints(rect)  # Get 4 corner points of the rotated box
            box = np.int32(box)  # Convert to integer for drawing
            display.windows["main"].add_contour(box)
            # if config.debug:
            #     box = cv.boxPoints(rect)  # Get 4 corner points of the rotated box
            #     box = np.intp(box)  # Convert to integer for drawing
            #     cv.drawContours(frame, [box], 0, (0, 255, 0), 2)
            b_box = Lights(rect[0][0], rect[0][1], w, h, angle)
            b_boxes.append(b_box)

    return b_boxes


def pairing(b_boxes):
    """Pairs lights together based off of similarity score using vectorized operations.

    :param b_boxes: List of light objects to be paired
    :return: list of pairs of lights
    """
    n = len(b_boxes)
    if n <= 1:
        return []

    # Convert light objects to structured array in one go
    light_data = np.array(
        [(light.cx, light.cy, light.angle, light.h) for light in b_boxes],
        dtype=[("cx", "f8"), ("cy", "f8"), ("angle", "f8"), ("h", "f8")],
    )

    # Extract arrays using structured array fields
    cx = light_data["cx"]
    cy = light_data["cy"]
    angles = light_data["angle"]
    heights = light_data["h"]

    # Create meshgrids for vectorized calculations
    cx1, cx2 = np.meshgrid(cx, cx)
    cy1, cy2 = np.meshgrid(cy, cy)
    angles1, angles2 = np.meshgrid(angles, angles)
    heights1, heights2 = np.meshgrid(heights, heights)

    # Calculate all metrics at once
    dx = cx1 - cx2
    dy = cy1 - cy2
    distances = np.hypot(dx, dy)
    angle_diffs = np.abs(angles1 - angles2)
    misalignment_angles = np.abs(np.degrees(np.arctan2(dy, dx)))
    height_ratios = heights1 / heights2
    avg_heights = (heights1 + heights2) / 2
    expected_distances = np.abs((avg_heights / armor_width_ration) - distances)

    # Score multipliers + valid-mask thresholds. Static from config, or read
    # live from the trackbar window when config.classical.live_tuning is on
    # (see tuning.pairing_params).
    p = tuning.pairing_params()

    # Calculate scores
    scores = (
        angle_diffs * p.angle_diff_multiplier
        + misalignment_angles * p.misalignment_multiplier
        + expected_distances * p.expected_distance_multiplier
        + height_ratios * p.height_ratio_multiplier
    )

    # Create mask for valid pairs
    valid_mask = (
        (angle_diffs < p.angle_diff_thresh)  # Angle difference threshold
        & (misalignment_angles < p.misalignment_thresh)  # Misalignment threshold
        & (height_ratios > p.height_ratio_thresh[0])
        & (height_ratios < p.height_ratio_thresh[1])  # Height ratio threshold
        & (scores < p.score_thresh)  # Score threshold
    )

    # Set invalid pairs (same light) to infinite score
    np.fill_diagonal(scores, np.inf)

    pairs = []
    used = set()

    # Get pairs in order of increasing score
    while True:
        # Find minimum score indices
        min_idx = np.unravel_index(scores.argmin(), scores.shape)
        min_score = scores[min_idx]

        if min_score == np.inf or not valid_mask[min_idx]:
            break

        i, j = min_idx
        if i not in used and j not in used:
            pairs.append([b_boxes[i], b_boxes[j]])
            used.add(i)
            used.add(j)

        # Mark this pair as used by setting its score to infinity
        scores[i, j] = scores[j, i] = np.inf

    return pairs


def armour_corners(pair):
    """An absolute monster of math.

    :param pairs: list of pairs
    :param frame:
    :return: list of 4 points
    """
    # Since pair is a list of two Lights objects
    light1, light2 = pair[0], pair[1]
    # Determine left and right lights based on x-coordinate
    if light1.cx <= light2.cx:
        left = light1
        right = light2
    else:
        left = light2
        right = light1

    top_left = [
        int(
            left.cx
            + left.w * 0.5
            - (left.h * armor_height_ratio) * 0.5 * math.cos(math.radians(left.angle))
        ),
        int(
            (
                left.cy
                - (
                    (left.h * armor_height_ratio)
                    * 0.5
                    * math.sin(math.radians(left.angle))
                )
            )
        ),
    ]
    top_right = [
        int(
            right.cx
            - right.w * 0.5
            - (right.h * armor_height_ratio) * 0.5 * math.cos(math.radians(right.angle))
        ),
        int(
            (
                right.cy
                - (
                    (right.h * armor_height_ratio)
                    * 0.5
                    * math.sin(math.radians(right.angle))
                )
            )
        ),
    ]
    bottom_left = [
        int(
            left.cx
            + left.w * 0.5
            + (left.h * armor_height_ratio) * 0.5 * math.cos(math.radians(left.angle))
        ),
        int(
            (
                left.cy
                + (
                    (left.h * armor_height_ratio)
                    * 0.5
                    * math.sin(math.radians(left.angle))
                )
            )
        ),
    ]
    bottom_right = [
        int(
            right.cx
            - right.w * 0.5
            + (right.h * armor_height_ratio) * 0.5 * math.cos(math.radians(right.angle))
        ),
        int(
            (
                right.cy
                + (
                    (right.h * armor_height_ratio)
                    * 0.5
                    * math.sin(math.radians(right.angle))
                )
            )
        ),
    ]

    # Inner light-bar corners — the actual pixel corners of the light-bar inner
    # edges. These go to PnP. Same shape as the outer extrapolation above but
    # without the armor_height_ratio multiplier on the long-axis extent.
    inner_top_left = [
        int(
            left.cx
            + left.w * 0.5
            - left.h * 0.5 * math.cos(math.radians(left.angle))
        ),
        int(left.cy - left.h * 0.5 * math.sin(math.radians(left.angle))),
    ]
    inner_top_right = [
        int(
            right.cx
            - right.w * 0.5
            - right.h * 0.5 * math.cos(math.radians(right.angle))
        ),
        int(right.cy - right.h * 0.5 * math.sin(math.radians(right.angle))),
    ]
    inner_bottom_left = [
        int(
            left.cx
            + left.w * 0.5
            + left.h * 0.5 * math.cos(math.radians(left.angle))
        ),
        int(left.cy + left.h * 0.5 * math.sin(math.radians(left.angle))),
    ]
    inner_bottom_right = [
        int(
            right.cx
            - right.w * 0.5
            + right.h * 0.5 * math.cos(math.radians(right.angle))
        ),
        int(right.cy + right.h * 0.5 * math.sin(math.radians(right.angle))),
    ]

    points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    points = points.reshape((-1, 1, 2))
    inner_points = np.array(
        [inner_top_left, inner_top_right, inner_bottom_right, inner_bottom_left],
        dtype=np.int32,
    ).reshape((-1, 1, 2))

    panel_center = np.array(
        [
            ((top_left[0] + bottom_left[0] + top_right[0] + bottom_right[0]) / 4),
            (top_left[1] + bottom_left[1] + top_right[1] + bottom_right[1]) / 4,
        ],
        dtype=np.int32,
    )
    area = cv.contourArea(points)  # find area
    panel = Panel(points, inner_points, panel_center, area)

    return panel
