"""MCU message latency — AutoX side (queue-RPC McuDriver round-trip), plan 09.

This is the auto-aim hot path. The McuDriver runs as a REAL child process (as in
deployment) and we time McuClient round-trips from the parent across the process
boundary -- isolated, then under a concurrent 100 Hz nav odom load (the priority
lane must keep 't' bounded). MCU=MOCK, so there's no serial: this measures the
in-process transport the inverted design keeps auto-aim's 't'/'d' traffic on,
instead of a DDS round-trip (compare bench_ros_latency.py).

    python docker/bench_mcu_latency.py     # in the autox container (or host)
"""

import statistics
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # docker/ -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [sys.argv[0]]  # before any config-touching src import

from src.drivers.mcu import McuDriver  # noqa: E402


def report(samples, label):
    """Print percentile latency stats (microseconds) for a sample list."""
    s = sorted(samples)
    n = len(s) // 1
    pct = lambda p: s[min(len(s) - 1, int(p * len(s)))]  # noqa: E731
    print(
        f"  {label:38s} n={n:5d}  p50={pct(0.5):7.1f}  p90={pct(0.9):7.1f}  "
        f"p99={pct(0.99):8.1f}  max={s[-1]:9.1f}  mean={statistics.mean(s):7.1f}   (us)"
    )


def time_calls(fn, n):
    """Return per-call wall times (us) for n invocations of fn."""
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1e6)
    return out


def main():
    """Bench transform/odom round-trips through a real McuDriver subprocess."""
    conn = McuDriver.provision("mcu", ["AutoAim", "McuBridge"])
    driver = McuDriver.from_conn(conn)
    driver.start()  # real child process: builds MockBackend, services the queues
    try:
        aim = McuDriver.client(conn, "AutoAim")  # auto-aim, high-priority lane
        nav = McuDriver.client(conn, "McuBridge")  # nav bridge, low-priority lane
        for _ in range(300):  # warm up the queues / feeder threads
            aim.get_transformation(5, timeout=1.0)

        n = 5000
        print("AutoX-side MCU round-trip latency (in-process queue-RPC, MCU=MOCK):")
        report(
            time_calls(lambda: aim.get_transformation(5, timeout=1.0), n),
            "transform 't'  (isolated)",
        )
        report(
            time_calls(lambda: aim.get_odometry(timeout=1.0), n),
            "odom 'q'       (isolated)",
        )

        # Hammer the low lane at ~100 Hz (nav odom poll) while re-measuring 't'.
        stop = threading.Event()

        def odom_load():
            while not stop.is_set():
                nav.get_odometry(timeout=1.0)
                time.sleep(0.01)

        loader = threading.Thread(target=odom_load, daemon=True)
        loader.start()
        time.sleep(0.3)
        report(
            time_calls(lambda: aim.get_transformation(5, timeout=1.0), n),
            "transform 't'  (under 100Hz odom load)",
        )
        stop.set()
        loader.join(timeout=1.0)
    finally:
        driver.stop()


if __name__ == "__main__":
    main()
