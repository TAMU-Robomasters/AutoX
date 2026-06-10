# archive/

Reference snapshots of code that was simplified/replaced, kept so it can be
brought back if needed.

## video_streaming/
The original `src/subsystems/video_streaming/` camera layer, including the
mrcal-based lens-correction modes (`warp`, `unproject`, `normal`), RealSense /
webcam / simulation backends, intrinsics loading, and depth helpers.

Replaced by the zero-copy iceoryx2 camera driver in
`src/drivers/video_stream.py`, which currently supports **normal mode only**
(raw passthrough). If you need warp/unproject, depth, or a non-USB backend
again, the implementations are here for reference.
