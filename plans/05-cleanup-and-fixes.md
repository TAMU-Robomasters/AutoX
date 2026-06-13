# 05 — Cleanup, archiving, and known bugs

Small, mostly-independent items. Do them whenever convenient.

## A. Archive the old `video_streaming` layer
The new camera path (`src/drivers/video_stream.py` + `camera_info`) replaces
`src/subsystems/video_streaming/`, but a few things still import the old module,
so it can't be deleted yet. A snapshot already exists in `archive/video_streaming/`.

Remaining importers to repoint or remove:
- `src/subsystems/vision/classical_detector/module.py` — still has a **fallback** to
  the `video_stream` singleton when `ctx.frame is None`. Both engines now set
  `ctx.frame`, so the fallback can be deleted (then it stops importing the old
  singleton). Verify no other consumer of `ClassicalDetectorModule` skips setting
  `ctx.frame` first.
- `src/types/ipc.py` — `ImageMessage` is sized from `video_stream.width/height` at
  import (and is now unused; `FramePayload` replaced it). Remove `ImageMessage` (keep
  `LogMessage`, used by `count.py`).
- `src/subsystems/selection.py`, `src/subsystems/vision/classical_detector/depth_module.py`,
  `src/subsystems/vision/__init__.py` — check usage; repoint to `camera_info` or
  remove if dead (depth is out of scope for normal mode).
Once nothing imports it, move `src/subsystems/video_streaming/` fully into `archive/`
(or delete — git history + the snapshot preserve it).

## B. `Engine.stop()` has the same bug `Driver.stop()` had
`Engine.run()` sets `self.active = True` in the child; `Engine.stop()` sets
`self.active = False` in the **parent** — never seen across the process boundary, so
the child loop never exits. Fix exactly like `Driver.stop()`: a
`multiprocessing.Event` created in `__init__`, polled by `run()`'s loop, set by
`stop()`, with a `join(timeout)` + `terminate()` fallback. (Tests currently use
`terminate()` to work around it.)

## C. info.yaml cleanup
- Delete the dead `zed` config blocks — there is no `zed.py` and
  `create_video_stream()` never handled it; the `CAMERA` profile / `resolution`
  string for zed are orphaned.
- Unify the resolution field names. Code already standardizes on
  `config.hardware.camera_width` / `camera_height`; info.yaml still has
  `cam_width/cam_height` (line ~33, unused), `color_stream_width/height`, etc.
  Rename the capture-resolution fields to one name; leave `depth_stream_*` alone.
- Investigate `quik_config` multi-file/include support before splitting the (large)
  `info.yaml` into per-domain files. Verify what the loader (`src/toolbox/globals.py`)
  supports first.

## D. Restore the PF angular-velocity plot (optional)
`orchestrator.start_engines()` dropped the `start_plot_engine` + shared `Queue`
when it switched to `launch_system` (which doesn't thread a plot Queue through). PF
still accepts an optional `queue`. Re-add by extending `launch_system` to pass extra
kwargs, or wire the plot manually alongside `launch_system`.

## E. `CAP_PROP_BUFFERSIZE` — do NOT re-add
On this UVC camera, `cap.set(CAP_PROP_BUFFERSIZE, 1)` dropped capture to 25 fps.
It's intentionally absent. (Moot now that capture is via ffmpeg, but noted.)

## F. Hardware JPEG decode (optional perf/CPU)
Capture is via an ffmpeg subprocess (~90 fps, software decode on CPU). The Jetson
hardware decoder `nvv4l2decoder` (GStreamer) also hits ~86 fps and offloads decode
to NVDEC, freeing CPU for more engines — but this OpenCV build has `GStreamer: NO`,
so it needs either a `gst-launch` subprocess (same pipe pattern as ffmpeg) or an
OpenCV rebuild with GStreamer. Only worth it if CPU becomes contended.
