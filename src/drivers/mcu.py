"""MCU bridge driver: one process owns the UART, engines talk to it via RPC.

See ``plans/01-mcu-driver.md`` and ``plans/09-sentry-nav2-behavior.md``.
``McuDriver`` follows the same factory contract as ``CameraDriver``
(``provision``/``from_conn``/``client``) but the transport is a set of
``multiprocessing.Queue`` objects (control plane, tiny messages) instead of
iceoryx2 shared memory.

Backends are selected by the ``MCU=`` profile (``config.mcu.backend``):

- ``serial`` — :class:`SerialBackend` wraps ``EmbeddedCommunicator`` verbatim;
  AutoX owns ``/dev/ttyTHS1`` directly. This is the **canonical** path: AutoX owns
  the UART even when nav2 runs, and nav2 reaches the MCU through the ROS bridge
  AutoX hosts (``McuBridgeEngine`` publishes ``/odom``, consumes ``/cmd_vel``).
- ``ros`` — :class:`RosBackend` forwards to a nav2-owned serial node. **Deprecated**
  (the nav2-owns direction); kept as a fallback. Lazy-imports ``rclpy``.
- ``mock`` — :class:`MockBackend` returns identity transforms + kinematic odom; no
  serial, no ROS. Laptop default.

The ``McuDriver`` child is the single point of arbitration inside AutoX: every
engine pushes onto a request queue and the driver services them serially, which
also serializes the four UART message types on the wire (the guarantee nav2's
single-threaded executor used to provide).

**Priority lanes.** Two request queues: ``req_hi`` carries the latency-critical
auto-aim traffic (transform query, firing solution, match state); ``req_lo``
carries nav traffic (odometry, velocity). ``execute()`` always services ``req_hi``
first, so a backlog of 100 Hz odom polls can never delay an aim-path query.

**Per-consumer response queues.** With AutoX owning the UART, two engines consume
the driver concurrently (auto-aim *and* the nav bridge). Each gets its own response
queue (allocated parent-side in :meth:`McuDriver.provision`, keyed by consumer) so
replies are never dequeued by the wrong consumer. Plain fork-inherited queues (not
Manager proxies) keep the fast path fast.
"""

import itertools
import os
import queue
import time
from multiprocessing import Queue
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from src.core.driver import Driver
from src.toolbox.globals import config
from src.types.rpc import McuRequest, McuResponse

Transform = Tuple[float, float, npt.NDArray[np.float64]]  # (yaw, pitch, 4x4)
Odometry = Tuple[float, float, float]  # (x, y, theta), embedded frame


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class MockBackend:
    """Identity transform + kinematic odom, no hardware — laptop dev / default.

    ``get_odometry`` integrates the last commanded velocity so nav2's ``/cmd_vel``
    actually moves the reported pose (closes the nav loop on a laptop with no
    Gazebo). ``theta`` stays 0 (yaw is owned by the MCU; the mock keeps it fixed).
    """

    def __init__(self) -> None:
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._vx = 0.0  # last commanded velocity, embedded frame (x=right, y=fwd)
        self._vy = 0.0
        self._last_odom_t: Optional[float] = None

    def get_transformation(
        self, frame_delay_ms: int, timeout: float
    ) -> Optional[Transform]:
        """Return an identity camera->ballistic transform."""
        return 0.0, 0.0, np.eye(4, dtype=np.float64)

    def send_solution(
        self, pitch: float, yaw: float, time_until_fire: int, cv_state: int
    ) -> bool:
        """Pretend to send a firing solution (no-op)."""
        return True

    def get_odometry(self, timeout: float) -> Optional[Odometry]:
        """Integrate the last commanded velocity into a fake pose."""
        now = time.monotonic()
        if self._last_odom_t is not None:
            dt = now - self._last_odom_t
            self._x += self._vx * dt
            self._y += self._vy * dt
        self._last_odom_t = now
        return self._x, self._y, self._theta

    def send_velocity(self, vx: float, vy: float) -> bool:
        """Record the commanded velocity for the kinematic odom integrator."""
        self._vx = float(vx)
        self._vy = float(vy)
        return True

    def get_match_state(self) -> Any:
        """Match-state query: the mock reports the match as active."""
        return True

    def close(self) -> None:
        """Release resources (none)."""
        pass


class SerialBackend:
    """Wraps ``EmbeddedCommunicator`` verbatim; owns the UART in this process."""

    def __init__(self) -> None:
        # Imported lazily so the module doesn't open serial at import time; this
        # runs in the McuDriver child (initialize()), never the parent.
        from src.subsystems.embedded_communicator import EmbeddedCommunicator

        self._comm = EmbeddedCommunicator()

    def get_transformation(
        self, frame_delay_ms: int, timeout: float
    ) -> Optional[Transform]:
        """Query the MCU over UART for the camera->ballistic transform."""
        return self._comm.get_camera_to_ballistic_transformation(frame_delay_ms)

    def send_solution(
        self, pitch: float, yaw: float, time_until_fire: int, cv_state: int
    ) -> bool:
        """Write the firing solution to the MCU over UART."""
        return self._comm.send_angles_to_embedded(pitch, yaw, time_until_fire, cv_state)

    def get_odometry(self, timeout: float) -> Optional[Odometry]:
        """Query the MCU over UART for chassis odometry."""
        return self._comm.get_odometry()

    def send_velocity(self, vx: float, vy: float) -> bool:
        """Write a world-frame velocity command to the MCU over UART."""
        return self._comm.send_velocity(vx, vy)

    def get_match_state(self) -> Any:
        """Match-state query (plan-09 stub: no wire message yet — see firmware)."""
        raise NotImplementedError("get_match_state has no MCU wire message yet")

    def close(self) -> None:
        """Close the serial port."""
        if self._comm.port is not None:
            self._comm.port.close()


class RosBackend:
    """Forward MCU traffic to a nav2 ``mcu_bridge_node`` over ROS (deprecated).

    This is the legacy *nav2-owns-the-UART* direction (``MCU=ROS``); it puts a DDS
    round-trip on the per-frame aim path. Plan 09 inverts ownership (AutoX owns the
    UART; use ``MCU=SERIAL`` with ``ROS=NAV2``). Kept only as a fallback. Odom /
    velocity are nav2's own job in this mode, so they are no-ops here.
    """

    def __init__(self) -> None:
        import rclpy  # type: ignore[import-untyped]
        from mcu_msgs.msg import FiringSolution  # type: ignore[import-not-found]
        from mcu_msgs.srv import QueryTransform  # type: ignore[import-not-found]

        self._rclpy = rclpy
        self._FiringSolution = FiringSolution
        self._QueryTransform = QueryTransform

        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("autox_mcu_backend")
        self._xform_client = self._node.create_client(
            QueryTransform, "/mcu/query_transform"
        )
        self._solution_pub = self._node.create_publisher(
            FiringSolution, "/mcu/firing_solution", 1
        )
        # Don't hard-block forever if the bridge isn't up yet.
        self._xform_client.wait_for_service(timeout_sec=5.0)

    def get_transformation(
        self, frame_delay_ms: int, timeout: float
    ) -> Optional[Transform]:
        """Call the nav2 bridge's QueryTransform service for the transform."""
        if not self._xform_client.service_is_ready():
            return None
        req = self._QueryTransform.Request()
        req.frame_delay_ms = int(frame_delay_ms) & 0xFF
        future = self._xform_client.call_async(req)
        self._rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout)
        if not future.done():
            return None
        resp = future.result()
        if resp is None or not resp.success:
            return None
        matrix = np.array(resp.matrix, dtype=np.float64).reshape((4, 4))
        return float(resp.yaw), float(resp.pitch), matrix

    def send_solution(
        self, pitch: float, yaw: float, time_until_fire: int, cv_state: int
    ) -> bool:
        """Publish the firing solution to the nav2 bridge's topic."""
        msg = self._FiringSolution()
        msg.pitch = float(pitch)
        msg.yaw = float(yaw)
        msg.time_until_fire = int(time_until_fire) & 0xFFFF
        msg.cv_state = int(cv_state) & 0xFF
        self._solution_pub.publish(msg)
        return True

    def get_odometry(self, timeout: float) -> Optional[Odometry]:
        """No-op: in MCU=ROS, nav2 owns odometry directly."""
        return None

    def send_velocity(self, vx: float, vy: float) -> bool:
        """No-op: in MCU=ROS, nav2 drives the chassis directly."""
        return False

    def get_match_state(self) -> Any:
        """Match-state query (deprecated path — not implemented)."""
        raise NotImplementedError("get_match_state is not implemented for RosBackend")

    def close(self) -> None:
        """Tear down the rclpy node."""
        self._node.destroy_node()


def _make_backend():
    """Build the backend selected by ``config.mcu.backend`` (runs in the child)."""
    backend = config.mcu.backend
    if backend == "serial":
        return SerialBackend()
    if backend == "ros":
        return RosBackend()
    if backend == "mock":
        return MockBackend()
    raise ValueError(f"unknown mcu.backend {backend!r} (expected serial|ros|mock)")


# Ops routed to the low-priority lane (nav traffic). Everything else (transform,
# solution, match state) is auto-aim and goes high-priority.
_LOW_PRIORITY_OPS = frozenset({"get_odometry", "send_velocity"})


# ---------------------------------------------------------------------------
# Driver + client
# ---------------------------------------------------------------------------


class McuDriver(Driver):
    """Owns the MCU backend in its child process; services queued RPC requests."""

    def __init__(
        self, req_hi: Queue, req_lo: Queue, resp_map: Dict[str, Queue]
    ) -> None:
        super().__init__(requires="mcu")
        self._req_hi = req_hi
        self._req_lo = req_lo
        self._resp_map = resp_map
        self._backend = None  # built in initialize() (child)

    # --- factory contract (matches CameraDriver) -------------------------
    @staticmethod
    def provision(name: str, consumers: Optional[List[str]] = None) -> dict:
        """Parent-side: create the request + per-consumer response queues.

        ``consumers`` is the list of engine class names that will hold a client
        (passed by ``orchestrator.launch_system``); each gets its own response
        queue. With no list, a single ``"default"`` response queue is used (the
        single-consumer / unit-test case). All queues are created in the parent so
        they survive fork into the driver child and every consumer child.
        """
        keys = list(consumers) if consumers else ["default"]
        return {
            "req_hi": Queue(),
            "req_lo": Queue(),
            "resp": {key: Queue() for key in keys},
        }

    @classmethod
    def from_conn(cls, conn: dict) -> "McuDriver":
        """Parent-side: build the driver process from its provisioned queues."""
        return cls(conn["req_hi"], conn["req_lo"], conn["resp"])

    @staticmethod
    def client(conn: dict, consumer_key: Optional[str] = None) -> "McuClient":
        """Child-side: build a consumer handle bound to its own response queue."""
        resp_map: Dict[str, Queue] = conn["resp"]
        if consumer_key is not None and consumer_key in resp_map:
            key = consumer_key
        elif "default" in resp_map:
            key = "default"  # single-consumer provisioning
        else:
            # Shouldn't happen if launch_system passed the right consumer list;
            # fall back to any queue so we never hand back a dangling client.
            key = next(iter(resp_map))
        return McuClient(conn["req_hi"], conn["req_lo"], resp_map[key], key)

    # --- lifecycle (child) ----------------------------------------------
    def initialize(self) -> None:
        """Build the configured backend (opens serial / rclpy in the child)."""
        self._backend = _make_backend()

    def _next_request(self) -> Optional[McuRequest]:
        """Pop the next request, high-priority lane first.

        Drains ``req_hi`` then ``req_lo`` non-blocking; if both are empty, blocks
        briefly on ``req_hi`` so a freshly-arriving aim-path request wakes the loop
        immediately (the low lane tolerates the few-ms idle wait).
        """
        try:
            return self._req_hi.get_nowait()
        except queue.Empty:
            pass
        try:
            return self._req_lo.get_nowait()
        except queue.Empty:
            pass
        try:
            return self._req_hi.get(timeout=0.005)
        except queue.Empty:
            return None

    def execute(self) -> None:
        """Service one queued request against the backend, reply by consumer/req_id."""
        req = self._next_request()
        if req is None:
            return
        ok, value, error = True, None, None
        try:
            if req.op == "get_transformation":
                frame_delay_ms, rpc_timeout = req.args
                value = self._backend.get_transformation(frame_delay_ms, rpc_timeout)
            elif req.op == "send_solution":
                value = self._backend.send_solution(*req.args)
            elif req.op == "get_odometry":
                (rpc_timeout,) = req.args
                value = self._backend.get_odometry(rpc_timeout)
            elif req.op == "send_velocity":
                value = self._backend.send_velocity(*req.args)
            elif req.op == "get_match_state":
                value = self._backend.get_match_state()
            else:
                ok, error = False, f"unknown op {req.op!r}"
        except Exception as e:  # keep the driver alive; report the failure
            ok, error = False, repr(e)
        if req.req_id is not None:
            resp_q = self._resp_map.get(req.consumer_key)
            if resp_q is not None:
                resp_q.put(McuResponse(req.req_id, ok, value, error))


class McuClient:
    """Engine-side handle. Pushes requests on a priority lane, matches replies.

    Each client owns its own response queue (one per consumer, allocated in
    :meth:`McuDriver.provision`), so concurrent consumers never dequeue each
    other's replies. Auto-aim ops go on the high-priority lane; nav ops on the low
    lane. Within one client there is at most one reply-expecting request in flight
    (engines call sequentially), so ``req_id`` matching on its own queue is exact.
    """

    # Unique across clients in the same process so ids never collide.
    _ids = itertools.count()

    def __init__(
        self, req_hi: Queue, req_lo: Queue, resp: Queue, consumer_key: str
    ) -> None:
        self._req_hi = req_hi
        self._req_lo = req_lo
        self._resp = resp
        self._consumer_key = consumer_key
        self._tag = os.getpid() << 20  # cheap per-process disambiguator

    def _next_id(self) -> int:
        return self._tag + next(self._ids)

    def _call(self, lane: Queue, op: str, args: tuple, timeout: float) -> Any:
        """Push a request and block for its matching reply, or None on timeout."""
        req_id = self._next_id()
        lane.put(McuRequest(op, args, req_id, self._consumer_key))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                resp: McuResponse = self._resp.get(timeout=remaining)
            except queue.Empty:
                return None
            if resp.req_id != req_id:
                continue  # stale reply from an earlier timed-out request; drop it
            return resp.value if resp.ok else None

    # --- auto-aim (high-priority lane) -----------------------------------
    def get_transformation(
        self, frame_delay_ms: int, timeout: Optional[float] = None
    ) -> Optional[Transform]:
        """Blocking request for the camera->ballistic transform, or None."""
        if timeout is None:
            timeout = config.mcu.rpc_timeout
        return self._call(
            self._req_hi, "get_transformation", (frame_delay_ms, timeout), timeout
        )

    def send_solution(
        self, pitch: float, yaw: float, time_until_fire: int, cv_state: int
    ) -> None:
        """Fire-and-forget firing solution to the MCU (no reply)."""
        self._req_hi.put(
            McuRequest(
                "send_solution",
                (pitch, yaw, time_until_fire, cv_state),
                None,
                self._consumer_key,
            )
        )

    def get_match_state(self, timeout: Optional[float] = None) -> Any:
        """Blocking request for the devboard match-active state, or None."""
        if timeout is None:
            timeout = config.mcu.rpc_timeout
        return self._call(self._req_hi, "get_match_state", (), timeout)

    # --- nav (low-priority lane) -----------------------------------------
    def get_odometry(self, timeout: Optional[float] = None) -> Optional[Odometry]:
        """Blocking request for chassis odometry ``(x, y, theta)``, or None."""
        if timeout is None:
            timeout = config.mcu.rpc_timeout
        return self._call(self._req_lo, "get_odometry", (timeout,), timeout)

    def send_velocity(self, vx: float, vy: float) -> None:
        """Fire-and-forget world-frame velocity command to the MCU (no reply)."""
        self._req_lo.put(
            McuRequest("send_velocity", (vx, vy), None, self._consumer_key)
        )
