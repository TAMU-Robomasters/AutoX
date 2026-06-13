"""Standalone demo: CameraDriver publishes frames; FrameReader reads zero-copy.

Run from the repo root:  uv run python scripts/frame_ipc_demo.py

Proves end-to-end on the real camera:
  - frames flow camera-process -> shared memory -> reader-process
  - consumer reads are zero-copy (numpy view shares the shared-memory address)
  - the view is read-only (accidental write raises, not segfault)
  - staleness detection via seq (no double-processing the same frame)
"""

import os
import sys
import time

# Make `import src...` work when run as a plain script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drivers.video_stream import CameraDriver, FrameReader  # noqa: E402


def main() -> None:
    driver = CameraDriver()
    driver.start()  # fork BEFORE creating any iceoryx2 node in this process
    try:
        reader = FrameReader()
        reader.start()

        duration = 5.0
        t0 = time.perf_counter()
        reads = empty_polls = stale_reads = 0
        last_seq = -1
        seqs = set()
        zero_copy = read_only = None

        while time.perf_counter() - t0 < duration:
            frame = reader.latest()
            if frame is None:
                empty_polls += 1
                time.sleep(0.001)
                continue

            reads += 1
            seqs.add(frame.seq)
            if frame.seq == last_seq:
                stale_reads += 1
            last_seq = frame.seq

            if zero_copy is None:
                zero_copy = frame.data.base is not None
                try:
                    frame.data[0, 0, 0] = 0
                    read_only = False
                except ValueError:
                    read_only = True
                mean = float(frame.data.mean())
                print(
                    f"first frame: shape={frame.data.shape} dtype={frame.data.dtype} "
                    f"seq={frame.seq} mean_pixel={mean:.1f}"
                )

            del frame  # release the borrowed sample promptly

        elapsed = time.perf_counter() - t0
        print("\n=== frame IPC demo ===")
        print(f"elapsed:            {elapsed:.1f}s")
        print(f"reads:              {reads}")
        print(f"unique frames:      {len(seqs)}  ({len(seqs)/elapsed:.1f} fps)")
        print(f"stale (repeat) reads:{stale_reads}")
        print(f"empty polls:        {empty_polls}")
        print(f"zero-copy view:     {zero_copy}")
        print(f"read-only enforced: {read_only}")
    finally:
        t = time.perf_counter()
        driver.stop()  # graceful: sets the stop event, child exits its loop
        print(f"driver.stop() returned in {time.perf_counter() - t:.3f}s; "
              f"alive={driver.is_alive()}, exitcode={driver.exitcode}")


if __name__ == "__main__":
    main()
