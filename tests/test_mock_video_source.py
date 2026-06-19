"""Tests for the looping mock video source (Phase 1 of the circlet work).

These don't touch the camera, iceoryx2, mrcal, or the detector -- just the frame
production of ``MockVideoSource``: synthetic fallback for an un-pulled git-lfs
pointer, resize-to-config, looping, and runtime fps changes.
"""

import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [
    sys.argv[0]
]  # before any config-touching src import (quik_config parses argv)

from src.drivers.video_sources import MockVideoSource, _is_lfs_pointer  # noqa: E402


def test_synthetic_fallback_when_file_missing(tmp_path):
    """A missing video path yields animated synthetic frames at the config size."""
    src = MockVideoSource(160, 120, fps=120, video_path=str(tmp_path / "nope.mp4"))
    src.open()
    try:
        first = src.read(timeout=2.0)
        assert first is not None
        assert first.data.shape == (120, 160, 3)
        assert first.data.dtype == np.uint8
        # Synthetic frames animate, so the next frame differs and seq advances.
        second = src.read(last_seq=first.seq, timeout=2.0)
        assert second is not None
        assert second.seq > first.seq
        assert not np.array_equal(first.data, second.data)
    finally:
        src.close()


def test_unpulled_lfs_pointer_is_detected_and_falls_back():
    """The repo's sentry clip ships as a git-lfs pointer in dev/CI."""
    clip = REPO_ROOT / "assets/videos/1920x1080_30-fps_sentry.mp4"
    if not _is_lfs_pointer(clip):
        pytest.skip("sentry clip has been git-lfs-pulled; pointer path not exercised")
    src = MockVideoSource(160, 120, fps=120, video_path=str(clip))
    src.open()
    try:
        frame = src.read(timeout=2.0)
        assert frame is not None
        assert frame.data.shape == (120, 160, 3)  # synthetic, sized to config
    finally:
        src.close()


def test_read_into_copies_into_destination(tmp_path):
    """read_into writes the newest frame into the caller's buffer and returns (ts, seq)."""
    src = MockVideoSource(64, 48, fps=200, video_path=str(tmp_path / "x.mp4"))
    src.open()
    try:
        dst = np.zeros((48, 64, 3), dtype=np.uint8)
        result = src.read_into(dst, timeout=2.0)
        assert result is not None
        _ts, seq = result
        assert seq >= 1
        assert dst.any()  # something was written into the buffer
    finally:
        src.close()


def test_runtime_fps_change_keeps_producing(tmp_path):
    """Changing target_fps mid-run keeps the stream producing fresh frames."""
    src = MockVideoSource(32, 24, fps=40, video_path=str(tmp_path / "x.mp4"))
    src.open()
    try:
        first = src.read(timeout=2.0)
        assert first is not None
        src.target_fps = 200  # speed the stream up at runtime
        second = src.read(last_seq=first.seq, timeout=2.0)
        assert second is not None
        assert second.seq > first.seq
    finally:
        src.close()


def test_loops_and_resizes_a_real_video(tmp_path):
    """A real clip is resized to config and loops past its own length."""
    path = tmp_path / "tiny.mp4"
    w0, h0, n = 100, 80, 5
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 30, (w0, h0))
    for i in range(n):
        writer.write(np.full((h0, w0, 3), (i + 1) * 30, dtype=np.uint8))
    writer.release()

    cap = cv.VideoCapture(str(path))
    readable = cap.isOpened() and cap.read()[0]
    cap.release()
    if not readable or _is_lfs_pointer(path):
        pytest.skip("no usable mp4 encoder in this environment")

    src = MockVideoSource(50, 40, fps=300, video_path=str(path))
    src.open()
    try:
        last, seqs = -1, []
        for _ in range(n + 3):  # read past the clip length to prove it loops
            frame = src.read(last_seq=last, timeout=2.0)
            assert frame is not None
            assert frame.data.shape == (40, 50, 3)  # resized to config
            last = frame.seq
            seqs.append(frame.seq)
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)
    finally:
        src.close()
