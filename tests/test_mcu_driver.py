"""Unit tests for the AutoX-owned-UART MCU driver (plan 09, Phase 1).

Cover the pieces the inverted-UART design adds on top of the queue-RPC driver:
the new odom/velocity ops + ``MockBackend`` kinematic odom, the high/low priority
lanes, and the per-consumer response queues (so concurrent consumers -- auto-aim
and the nav bridge -- never dequeue each other's replies).

No serial, no ROS, no camera: ``MCU=MOCK`` (the default profile) builds a
``MockBackend``. The driver loop is run in a background *thread* (not a spawned
process) so the request/response routing + backend dispatch are exercised
deterministically and fast.
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [
    sys.argv[0]
]  # before any config-touching src import (quik_config parses argv)

from src.drivers.mcu import (  # noqa: E402
    _LOW_PRIORITY_OPS,
    McuClient,
    McuDriver,
    MockBackend,
)
from src.types.rpc import McuRequest  # noqa: E402


class _DriverThread:
    """Run an ``McuDriver``'s service loop in a daemon thread (no subprocess)."""

    def __init__(self, conn: dict):
        """Build the driver from ``conn`` and prepare its service thread."""
        self._driver = McuDriver.from_conn(conn)
        self._driver.initialize()  # builds the MockBackend in-process
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        """Service queued requests until stopped."""
        while not self._stop.is_set():
            self._driver.execute()

    def __enter__(self):
        """Start the service thread."""
        self._thread.start()
        return self

    def __exit__(self, *exc):
        """Stop the service thread and join it."""
        self._stop.set()
        self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# MockBackend (no queues)
# ---------------------------------------------------------------------------


def test_mock_backend_kinematic_odom_integrates_velocity():
    """MockBackend odom integrates the last commanded velocity over time."""
    mb = MockBackend()
    assert mb.get_odometry(0.05) == (0.0, 0.0, 0.0)  # first call seeds the clock
    mb.send_velocity(2.0, -1.0)
    time.sleep(0.03)
    x, y, theta = mb.get_odometry(0.05)
    assert x > 0.0 and y < 0.0 and theta == 0.0  # moved per commanded velocity


def test_mock_backend_match_state_active():
    """The mock reports the match as active (so the interlock allows firing)."""
    assert MockBackend().get_match_state() is True


# ---------------------------------------------------------------------------
# Client lane routing (no driver running)
# ---------------------------------------------------------------------------


def test_client_routes_ops_to_priority_lanes():
    """Auto-aim ops go on the high lane; nav ops on the low lane."""
    conn = McuDriver.provision("mcu")  # single "default" consumer
    client = McuDriver.client(conn, "AutoAim")
    req_hi, req_lo = conn["req_hi"], conn["req_lo"]

    client.send_solution(0.1, 0.2, 5, 1)  # auto-aim -> high lane
    client.send_velocity(1.0, 2.0)  # nav -> low lane

    hi: McuRequest = req_hi.get(timeout=1.0)
    lo: McuRequest = req_lo.get(timeout=1.0)
    assert hi.op == "send_solution" and hi.op not in _LOW_PRIORITY_OPS
    assert lo.op == "send_velocity" and lo.op in _LOW_PRIORITY_OPS


# ---------------------------------------------------------------------------
# Full RPC round-trip through the driver loop
# ---------------------------------------------------------------------------


def test_round_trip_all_ops():
    """Every op round-trips through the driver loop against the MockBackend."""
    conn = McuDriver.provision("mcu", ["AutoAim"])
    with _DriverThread(conn):
        client = McuDriver.client(conn, "AutoAim")

        yaw, pitch, matrix = client.get_transformation(5, timeout=1.0)
        assert (yaw, pitch) == (0.0, 0.0)
        assert np.array_equal(matrix, np.eye(4))

        assert client.get_match_state(timeout=1.0) is True

        # Fire-and-forget ops return None and must not raise.
        assert client.send_solution(0.1, 0.2, 3, 1) is None
        assert client.send_velocity(1.0, 0.0) is None

        # Odom integrates the velocity we just sent.
        first = client.get_odometry(timeout=1.0)
        time.sleep(0.03)
        second = client.get_odometry(timeout=1.0)
        assert second is not None and first is not None
        assert second[0] > first[0]  # x advanced under vx=1.0


def test_timeout_returns_none_when_no_driver():
    """A blocking call returns None on timeout instead of hanging forever."""
    conn = McuDriver.provision("mcu")
    client = McuDriver.client(conn, "AutoAim")
    start = time.monotonic()
    assert client.get_transformation(5, timeout=0.05) is None
    assert 0.04 <= time.monotonic() - start < 0.5  # honored the timeout, didn't hang


# ---------------------------------------------------------------------------
# Per-consumer response-queue isolation
# ---------------------------------------------------------------------------


def test_per_consumer_response_queues_are_isolated():
    """Concurrent consumers each get their own replies, no cross-talk."""
    conn = McuDriver.provision("mcu", ["AutoAim", "McuBridge"])
    # Distinct response queues per consumer (the multi-consumer safety fix).
    assert conn["resp"]["AutoAim"] is not conn["resp"]["McuBridge"]

    with _DriverThread(conn):
        aim = McuDriver.client(conn, "AutoAim")
        bridge = McuDriver.client(conn, "McuBridge")

        # Each client gets its own correct reply, no cross-talk, interleaved.
        for _ in range(5):
            assert aim.get_transformation(5, timeout=1.0) is not None
            assert bridge.get_odometry(timeout=1.0) is not None

        # The bridge's reply queue never received an auto-aim transform reply
        # (and vice versa): both queues are drained empty after matched calls.
        assert conn["resp"]["AutoAim"].empty()
        assert conn["resp"]["McuBridge"].empty()


def test_client_falls_back_to_default_queue():
    """A client built with any key binds to the single "default" queue."""
    # Provisioned without a consumer list -> single "default" queue; a client
    # built with any consumer_key still binds to it.
    conn = McuDriver.provision("mcu")
    assert set(conn["resp"]) == {"default"}
    with _DriverThread(conn):
        client = McuDriver.client(conn, "SomeEngine")
        assert client.get_match_state(timeout=1.0) is True
