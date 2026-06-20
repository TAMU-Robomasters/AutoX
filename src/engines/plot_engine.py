"""Visualization engine: live rolling plot of a scalar value from a queue."""

import time
from collections import deque
from multiprocessing import Queue

import matplotlib

# Force the Tk backend. The default Qt backend collides with cv2: cv2 is imported
# in the parent before this plot process forks and points the Qt plugin path at
# its OWN bundled plugins, which are incompatible with matplotlib's Qt, so "xcb"
# fails to load. Tk uses no Qt plugins, sidestepping the conflict entirely (it
# still needs a display -- callers gate on $DISPLAY before spawning this).
matplotlib.use("TkAgg")

import matplotlib.animation as animation  # noqa: E402  (must follow matplotlib.use)
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Config — edit here
# ---------------------------------------------------------------------------
HISTORY_SECONDS: float = 5.0
PLOT_INTERVAL_MS: int = 50   # animation refresh rate


def start_plot_engine(queue: Queue, title: str = "Panel x position (cm)") -> None:
    """Run the live plot in the calling process.

    Reads floats from *queue* and displays a rolling HISTORY_SECONDS window.
    Blocks until the window is closed.  Designed to be the target of a
    multiprocessing.Process so the main process stays free.
    """
    times: deque[float] = deque()
    values: deque[float] = deque()
    t0 = time.perf_counter()
    total = 0  # samples ever received -- distinguishes "no data" from "flat line"

    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], lw=1.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(title)
    ax.set_title("waiting for data...")

    def _update(_frame):
        nonlocal total
        now = time.perf_counter() - t0
        last = None
        # Drain everything currently in the queue
        while not queue.empty():
            try:
                val = queue.get_nowait()
                times.append(now)
                values.append(val)
                last = val
                total += 1
            except Exception:
                break
        ax.set_title(
            f"{title}  |  n={total}"
            + (f"  last={last:.1f}" if last is not None else "  (no samples yet)")
        )

        # Drop data older than HISTORY_SECONDS
        cutoff = now - HISTORY_SECONDS
        while times and times[0] < cutoff:
            times.popleft()
            values.popleft()

        if times:
            line.set_data(list(times), list(values))
            ax.set_xlim(max(0.0, now - HISTORY_SECONDS), now + 0.1)
            ax.relim()
            ax.autoscale(enable=False, axis='y')
            ax.set_ylim(min(values) - 0.1, max(values) + 0.1)

        return (line,)

    ani = animation.FuncAnimation(
        fig, _update, interval=PLOT_INTERVAL_MS, blit=False, cache_frame_data=False
    )
    plt.tight_layout()
    plt.show()
