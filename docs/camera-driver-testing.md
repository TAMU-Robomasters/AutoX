# Camera driver: backends, controls, and how to test

The camera is owned by one process, `CameraDriver`
(`src/drivers/video_stream.py`), which decodes frames into an iceoryx2
shared-memory ring buffer; engines read them through a `FrameReader`. The actual
frame *production* is a pluggable **capture backend**
(`src/drivers/video_sources.py`), chosen by config:

| backend  | class                | use                                                            |
| -------- | -------------------- | -------------------------------------------------------------- |
| `pyav`   | `PyAvCameraSource`   | **default.** In-process libav (PyAV) MJPEG decode + V4L2 ctrls |
| `mock`   | `MockVideoSource`    | Laptop testing: loops a video file (no camera)                 |
| `ffmpeg` | `CameraSource`       | Legacy ffmpeg subprocess (kept as a fallback)                  |

`pyav` replaces the old ffmpeg-subprocess capture: PyAV decodes MJPEG in-process
with libav's multi-threaded decoder, reaching the camera's full rate, and camera
controls (exposure/gain/fps) are set with **v4l2-python3** ioctls instead of
shelling out to `v4l2-ctl`.

## Install

```sh
uv sync --extra linux   # adds PyAV (av) + v4l2-python3, the `linux` extra
```

The extra is **optional** and **lazy-imported** — the mock backend and the rest
of the codebase work without it. You only need it for real `pyav` capture.

## 1. Test on a laptop with no camera (mock)

The mock loops a video through the *real* `CameraDriver` → detector → pipeline.
`@MOCK_CAM` sets `video_backend: mock`; pair it with a robot profile for camera
intrinsics (PnP needs them):

```sh
uv run main.py @MOCK_CAM @SENTRY            # full pipeline on the looping video
uv run main.py @MOCK_CAM @SENTRY DISPLAY    # + annotated OpenCV windows
```

- Source clip: `assets/videos/1920x1080_30-fps_sentry.mp4`
  (`hardware.mock_video_path`, repo-root relative), resized to the configured
  `camera_width × camera_height`.
- That clip ships as a **git-lfs pointer**. Until you `git lfs pull` it, the mock
  prints a warning and emits **synthetic frames** (a moving gradient + bar) so
  the plumbing still runs — you just won't see real armor panels.

  ```sh
  git lfs install && git lfs pull   # fetch the real footage
  ```
- Vary the rate at runtime via `hardware.mock_fps` / `hardware.mock_fps_jitter`
  (jitter is a per-frame fps-noise fraction, mimicking a real camera).

Unit test for the source itself (no camera, no mrcal):

```sh
uv run pytest tests/test_mock_video_source.py
```

## 2. Bring up a real camera (PyAV) — the tomorrow checklist

1. Plug the camera in and find its node:
   ```sh
   v4l2-ctl --list-devices            # note the /dev/videoN
   v4l2-ctl -d /dev/videoN --list-formats-ext   # confirm MJPEG + a w×h@fps mode
   ```
2. In the robot profile (e.g. `@SENTRY` in `src/info.yaml`) set
   `hardware.camera_index: N`, `camera_width/height`, `camera_fps`, and
   `camera_exposure` (units 100 µs; `null` leaves auto-exposure on).
   `video_backend` defaults to `pyav`.
3. Run it:
   ```sh
   uv run main.py @SENTRY            # real camera via PyAV
   uv run main.py @SENTRY DISPLAY    # watch the detector overlay
   ```
4. If PyAV/libav can't open the device or you want the old path, set
   `hardware.video_backend: ffmpeg` in the profile (legacy subprocess capture).

## Troubleshooting

- **`No module named 'av'` / `v4l2`** → `uv sync --extra linux`.
- **`No module named 'mrcal'`** → intrinsics (PnP) need the system mrcal package;
  see CLAUDE.md ("Camera calibration"). The mock/detector path can't run without
  it; the `mock` backend's frame production can (it has no PnP).
- **Camera busy / no frames** → the autobooted `cv_dark_boot.service` may hold the
  camera. `sudo systemctl stop cv_dark_boot.service` (see `plans/README.md`).
- **Wrong resolution payload** → `camera_width/height` must match a real camera
  mode; the iceoryx2 payload is sized from them.
