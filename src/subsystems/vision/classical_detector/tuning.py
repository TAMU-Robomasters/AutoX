"""Optional live tuning of the light-pairing parameters via OpenCV trackbars.

Toggled by ``config.classical.live_tuning``. When false (the production default)
``pairing_params()`` returns the static ``config.classical.*`` values with zero
overhead — no window, no trackbars. When true, a "pairing tuning" window of
sliders is created on first use and the score multipliers + valid-mask
thresholds in :func:`armor.pairing` are read live from it, so they can be
adjusted while the detector runs.

The window is created/read in the same process that calls ``pairing()`` (the
detection engine child), which is also where ``display.show_windows()`` pumps
``cv.waitKey`` — that key pump is what makes the sliders interactive.

Changes are **ephemeral**: they live only for the current run. Once you find
values you like, copy them back into the ``classical:`` block of ``info.yaml``
by hand.
"""

from types import SimpleNamespace

import cv2 as cv
import numpy as np

from src.toolbox.globals import config

_WINDOW = "pairing tuning"

# Slider spec: (label, config attribute, min, max, scale, stricter).
#
# OpenCV trackbars are integer-only, so float params are exposed scaled up by
# `scale` (e.g. a multiplier shown 0.00-10.00 is stored as 0-1000 ticks with
# scale=100). The two height-ratio bounds map to the two elements of the
# `height_ratio_thresh` list and are handled specially when reading.
#
# `stricter` records which way the slider rejects MORE pairs (shrinks
# valid_mask / pushes scores past score_thresh): "low" = sliding left is
# stricter, "high" = sliding right is stricter. The multipliers only inflate
# `scores`, which must stay under score_thresh, so raising them is stricter.
_SLIDERS = [
    # label               attr                            min  max  scale  stricter
    ("angle_diff_x100", "angle_diff_multiplier", 0, 1000, 100, "high"),
    ("misalign_x100", "misalignment_multiplier", 0, 1000, 100, "high"),
    ("exp_dist_x100", "expected_distance_multiplier", 0, 1000, 100, "high"),
    ("height_ratio_x100", "height_ratio_multiplier", 0, 1000, 100, "high"),
    ("angle_diff_thresh", "angle_diff_thresh", 0, 180, 1, "low"),
    ("misalign_thresh", "misalignment_thresh", 0, 180, 1, "low"),
    ("h_ratio_lo_x100", "height_ratio_thresh_lo", 0, 300, 100, "high"),
    ("h_ratio_hi_x100", "height_ratio_thresh_hi", 0, 300, 100, "low"),
    ("score_thresh", "score_thresh", 0, 2000, 1, "low"),
]

_initialized = False


def _initial_value(attr):
    """Pull the starting slider value for *attr* out of ``config.classical``."""
    if attr == "height_ratio_thresh_lo":
        return config.classical.height_ratio_thresh[0]
    if attr == "height_ratio_thresh_hi":
        return config.classical.height_ratio_thresh[1]
    return getattr(config.classical, attr)


def _legend_image():
    """Render the static "which way is stricter" legend as a BGR image.

    One row per slider showing the trackbar label and an arrow pointing toward
    the direction that rejects MORE pairs (see the `stricter` field above).
    """
    row_h = 26
    img = np.zeros((row_h * (len(_SLIDERS) + 2), 600, 3), dtype=np.uint8)
    font = cv.FONT_HERSHEY_SIMPLEX
    yellow, white, gray = (85, 195, 254), (255, 255, 255), (160, 160, 160)

    cv.putText(img, "arrow points toward STRICTER (rejects more pairs)",
               (10, 18), font, 0.45, gray, 1, cv.LINE_AA)
    for i, (label, _attr, _lo, _hi, _scale, stricter) in enumerate(_SLIDERS):
        y = row_h * (i + 2)
        cv.putText(img, label, (10, y), font, 0.5, white, 1, cv.LINE_AA)
        arrow = "<-- stricter" if stricter == "low" else "stricter -->"
        cv.putText(img, arrow, (300, y), font, 0.5, yellow, 1, cv.LINE_AA)
    return img


def _ensure_window():
    """Create the tuning window + trackbars + legend once, seeded from config."""
    global _initialized
    if _initialized:
        return
    cv.namedWindow(_WINDOW, cv.WINDOW_NORMAL)
    cv.resizeWindow(_WINDOW, 600, 700)
    for label, attr, lo, hi, scale, _stricter in _SLIDERS:
        start = int(round(_initial_value(attr) * scale))
        start = max(lo, min(hi, start))
        # callback is required on some cv builds; reads are pull-based via
        # getTrackbarPos, so the callback does nothing.
        cv.createTrackbar(label, _WINDOW, start, hi, lambda _v: None)
    # Static legend shown in the window body (persists; the global waitKey pump
    # in display.show_windows keeps it drawn).
    cv.imshow(_WINDOW, _legend_image())
    _initialized = True


def pairing_params():
    """Return the current pairing parameters as a namespace.

    Static (straight from config) when ``live_tuning`` is off; read live from
    the trackbars when it is on. Field names match the ``config.classical.*``
    attributes used by :func:`armor.pairing`, with ``height_ratio_thresh``
    reassembled as a ``[lo, hi]`` list.

    When tuning is off this returns ``config.classical`` directly — no OpenCV
    calls and no allocation, so the production hot path is unchanged.
    """
    if not config.classical.live_tuning:
        return config.classical

    _ensure_window()
    values = {}
    for label, attr, _lo, _hi, scale, _stricter in _SLIDERS:
        values[attr] = cv.getTrackbarPos(label, _WINDOW) / scale

    return SimpleNamespace(
        angle_diff_multiplier=values["angle_diff_multiplier"],
        misalignment_multiplier=values["misalignment_multiplier"],
        expected_distance_multiplier=values["expected_distance_multiplier"],
        height_ratio_multiplier=values["height_ratio_multiplier"],
        # thresholds are integer-scaled (scale=1) so these come back as ints/floats
        angle_diff_thresh=values["angle_diff_thresh"],
        misalignment_thresh=values["misalignment_thresh"],
        height_ratio_thresh=[
            values["height_ratio_thresh_lo"],
            values["height_ratio_thresh_hi"],
        ],
        score_thresh=values["score_thresh"],
    )
