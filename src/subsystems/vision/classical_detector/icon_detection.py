"""Icon detection via XOR template matching.

Icons are loaded in sorted filename order so that indices are deterministic:
  0 -> complete_sentry_icon.jpeg  (sentry)
  1 -> cropped_1.png              (hero)
  2 -> cropped_3.png              (standard)
  3 -> cropped_sentry.png         (sentry)
"""

from pathlib import Path

import cv2 as cv
import numpy as np

from src.toolbox.globals import config, path_to

expected_points = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype=np.float32)

icons_folder = Path(path_to.icons_folder)
icon_list = []
for pic in sorted(icons_folder.iterdir()):
    if not pic.is_file():
        continue
    picture = cv.imread(str(pic))
    if picture is None:
        raise Exception(f"Failed to read icon image: {pic}")
    gray = cv.cvtColor(picture, cv.COLOR_BGR2GRAY)
    icon_list.append(gray)

# Vectorized comparison with all icons at once
# Stack all icons into a 3D array (n_icons x height x width)
icon_stack = np.stack(icon_list)


def icon_detection(panel, frame):
    """Transforms and crops out armour panels to compare to icons.

    :param frame:
    :param list_of_points:
    :return: list of ids, 0 and 3 = sentry, 1 = hero, 2 = standard
    """
    points = np.float32(panel.corners)
    M = cv.getPerspectiveTransform(points, expected_points)
    warped = cv.warpPerspective(frame, M, (300, 300))
    warped = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

    adaptive_thresh = cv.adaptiveThreshold(
        warped,
        255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY,
        config.classical.icon_adaptive_thresh[0],
        config.classical.icon_adaptive_thresh[1],
    )

    icon_contours, _ = cv.findContours(
        adaptive_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    try:
        # Find largest contour after first two
        c = max(icon_contours[2:], key=cv.contourArea)
        x, y, w, h = cv.boundingRect(c)

        # Crop and resize in one step
        cropped = cv.resize(adaptive_thresh[y : y + h, x : x + w], (300, 300))

        # Broadcast cropped image to same shape as icon_stack
        # and compute XOR for all icons simultaneously
        compared = np.bitwise_xor(icon_stack, cropped)

        # Compute mean intensity for all icons at once
        icon_scores = np.mean(compared, axis=(1, 2))

        if config.classical.show_icon_filters:
            for i, comp in enumerate(compared):
                cv.imshow(f"icon + {i}", comp)

        min_score = np.min(icon_scores)
        if min_score > config.classical.icon_tolerance:
            return False

        panel.id = np.argmin(icon_scores)
        return True

    except Exception:
        return False
