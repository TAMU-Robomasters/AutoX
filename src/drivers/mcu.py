"""MCU bridge driver: one process owns the UART, engines talk to it via RPC.

See ``plans/01-mcu-driver.md``. ``McuDriver`` follows the same factory contract
as ``CameraDriver`` (``provision``/``from_conn``/``client``) but the transport is
a pair of ``multiprocessing.Queue`` objects (control plane, tiny messages) instead
of iceoryx2 shared memory.

Backends are selected by the ``MCU=`` profile (``config.mcu.backend``):

- ``serial`` — :class:`SerialBackend` wraps ``EmbeddedCommunicator`` verbatim;
  AutoX owns ``/dev/ttyTHS1`` directly (nav2 not running).
- ``ros`` — :class:`RosBackend` forwards to the nav2 ``mcu_bridge_node`` that owns
  the serial (used when nav + auto-aim run concurrently). Lazy-imports ``rclpy``.
- ``mock`` — :class:`MockBackend` returns an identity transform; no serial, no ROS.

The ``McuDriver`` child is the single point of arbitration inside AutoX: every
engine pushes onto one request queue and the driver services them serially.
"""

import itertools
import os
import queue
import time
from multiprocessing import Queue
from typing import Any, Optional, Tuple

import numpy as np
import numpy.typing as npt

from src.core.driver import Driver
from src.toolbox.globals import config
from src.types.rpc import McuRequest, McuResponse

Transform = Tuple[float, float, npt.NDArray[np.float64]]  # (yaw, pitch, 4x4)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class MockBackend:
    """Identity transform, no hardware — laptop dev / default."""

    def get_transformation(self, frame_delay_ms: int, timeout: float) -> Optional[Transform]:
        """Return an identity camera->ballistic transform."""
        return 0.0, 0.0, np.eye(4, dtype=np.float64)

    def send_solution(self, pitch: float, yaw: float, time_until_fire: int, cv_state: int) -> bool:
        """Pretend to send a firing solution (no-op)."""
        return True

    def get_match_state(self) -> Any:
        """Match-state query (plan-02 stub)."""
        raise NotImplementedError("get_match_state is not implemented yet")

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

    def get_transformation(self, frame_delay_ms: int, timeout: float) -> Optional[Transform]:
        """Query the MCU over UART for the camera->ballistic transform."""
        return self._comm.get_camera_to_ballistic_transformation(frame_delay_ms)

    def send_solution(self, pitch: float, yaw: float, time_until_fire: int, cv_state: int) -> bool:
        """Write the firing solution to the MCU over UART."""
        return self._comm.send_angles_to_embedded(pitch, yaw, time_until_fire, cv_state)

    def get_match_state(self) -> Any:
        """Match-state query (plan-02 stub)."""
        raise NotImplementedError("get_match_state is not implemented yet")

    def close(self) -> None:
        """Close the serial port."""
        if self._comm.port is not None:
            self._comm.port.close()


class RosBackend:
    """Forwards MCU traffic to the nav2 ``mcu_bridge_node`` over ROS.

    Used when ``MCU=ROS`` (nav2 owns ``/dev/ttyTHS1``). Lazy-imports ``rclpy`` and
    the ``mcu_msgs`` interfaces so laptops on ``MCU=MOCK``/``SERIAL`` never touch
    ROS. RMW/domain are inherited from the environment (source ROS + the nav2
    overlay before launching AutoX with ``MCU=ROS``).
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
        self._xform_client = self._node.create_client(QueryTransform, "/mcu/query_transform")
        self._solution_pub = self._node.create_publisher(FiringSolution, "/mcu/firing_solution", 1)
        # Don't hard-block forever if the bridge isn't up yet.
        self._xform_client.wait_for_service(timeout_sec=5.0)

    def get_transformation(self, frame_delay_ms: int, timeout: float) -> Optional[Transform]:
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

    def send_solution(self, pitch: float, yaw: float, time_until_fire: int, cv_state: int) -> bool:
        """Publish the firing solution to the nav2 bridge's topic."""
        msg = self._FiringSolution()
        msg.pitch = float(pitch)
        msg.yaw = float(yaw)
        msg.time_until_fire = int(time_until_fire) & 0xFFFF
        msg.cv_state = int(cv_state) & 0xFF
        self._solution_pub.publish(msg)
        return True

    def get_match_state(self) -> Any:
        """Match-state query (plan-02 stub)."""
        raise NotImplementedError("get_match_state is not implemented yet")

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


# ---------------------------------------------------------------------------
# Driver + client
# ---------------------------------------------------------------------------


class McuDriver(Driver):
    """Owns the MCU backend in its child process; services queued RPC requests."""

    def __init__(self, req_queue: Queue, resp_queue: Queue) -> None:
        super().__init__(requires="mcu")
        self._req = req_queue
        self._resp = resp_queue
        self._backend = None  # built in initialize() (child)

    # --- factory contract (matches CameraDriver) -------------------------
    @staticmethod
    def provision(name: str) -> dict:
        """Parent-side: create the request/response queues (survive fork)."""
        return {"req": Queue(), "resp": Queue()}

    @classmethod
    def from_conn(cls, conn: dict) -> "McuDriver":
        """Parent-side: build the driver process from its provisioned queues."""
        return cls(conn["req"], conn["resp"])

    @staticmethod
    def client(conn: dict) -> "McuClient":
        """Child-side: build a consumer handle bound to the same queues."""
        return McuClient(conn["req"], conn["resp"])

    # --- lifecycle (child) ----------------------------------------------
    def initialize(self) -> None:
        """Build the configured backend (opens serial / rclpy in the child)."""
        self._backend = _make_backend()

    def execute(self) -> None:
        """Service one queued request against the backend, reply by req_id."""
        try:
            req: McuRequest = self._req.get(timeout=0.05)
        except queue.Empty:
            return
        ok, value, error = True, None, None
        try:
            if req.op == "get_transformation":
                frame_delay_ms, rpc_timeout = req.args
                value = self._backend.get_transformation(frame_delay_ms, rpc_timeout)
            elif req.op == "send_solution":
                value = self._backend.send_solution(*req.args)
            elif req.op == "get_match_state":
                value = self._backend.get_match_state()
            else:
                ok, error = False, f"unknown op {req.op!r}"
        except Exception as e:  # keep the driver alive; report the failure
            ok, error = False, repr(e)
        if req.req_id is not None:
            self._resp.put(McuResponse(req.req_id, ok, value, error))


class McuClient:
    """Engine-side handle. Pushes requests, matches replies by ``req_id``.

    NOTE: today exactly one engine (the PF auto-aim engine) holds a client, so a
    single shared response queue + ``req_id`` matching is correct (stale/late
    replies are discarded). If multiple engines ever consume the *same* driver
    concurrently they would share this response queue and could dequeue each
    other's replies — at that point give each consumer its own response queue
    (e.g. a ``Manager().Queue`` carried in the request). See plan 02 (AutoNav).
    """

    # Unique across clients in the same process so ids never collide.
    _ids = itertools.count()

    def __init__(self, req_queue: Queue, resp_queue: Queue) -> None:
        self._req = req_queue
        self._resp = resp_queue
        self._tag = os.getpid() << 20  # cheap per-process disambiguator

    def _next_id(self) -> int:
        return self._tag + next(self._ids)

    def get_transformation(
        self, frame_delay_ms: int, timeout: Optional[float] = None
    ) -> Optional[Transform]:
        """Blocking request for the camera->ballistic transform, or None."""
        if timeout is None:
            timeout = config.mcu.rpc_timeout
        req_id = self._next_id()
        self._req.put(McuRequest("get_transformation", (frame_delay_ms, timeout), req_id))
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

    def send_solution(
        self, pitch: float, yaw: float, time_until_fire: int, cv_state: int
    ) -> None:
        """Fire-and-forget firing solution to the MCU (no reply)."""
        self._req.put(
            McuRequest("send_solution", (pitch, yaw, time_until_fire, cv_state), None)
        )

    def get_match_state(self) -> Any:
        """Match-state query (plan-02 stub)."""
        raise NotImplementedError("get_match_state is not implemented yet")
