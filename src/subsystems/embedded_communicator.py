"""UART communication to the embedded MCU for the particle-filter auto-aim pipeline.

Ported from Armor-Panel-Classical/subsystems/communicate.py. Provides:
  - Querying the embedded system for the camera-to-ballistic 4x4 transformation matrix
  - Sending computed pitch, yaw, and alignment time back to the embedded system
"""

import subprocess
from ctypes import Structure, c_float, c_uint8, c_uint16, sizeof
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import serial

from src.toolbox.globals import config

# Serial read timeout (upper bound) for the transform query/response round-trip.
# read() returns the instant all bytes land, so the happy-path latency is just
# the message's transmission time -- at 115200 baud the 69-byte reply takes ~6 ms
# to clock in, so this MUST exceed that or read() returns a partial message and
# the leftover bytes desync the next read (garbage matrix). This bound only
# applies when a reply is missing entirely.
TRANSFORM_READ_TIMEOUT_S = 0.02


# ---------------------------------------------------------------------------
# Protocol enums & C-compatible structs
# ---------------------------------------------------------------------------


class CVState(Enum):
    """State of the computer-vision pipeline as reported to embedded."""

    NO_TARGET: int = 0  # no panel detected; embedded holds fire
    SHOT_TIMING: int = 1  # spinning fast; embedded waits for alignment time
    CONTINUOUS_FIRE: int = 2  # slow/no spin; embedded fires freely


class JetsonMessage(Structure):
    """Jetson -> Embedded: firing solution (12 bytes packed)."""

    _pack_ = 1
    _fields_ = [
        ("magic", c_uint8),
        ("messageType", c_uint8),
        ("pitch", c_float),
        ("yaw", c_float),
        ("timeUntilNextFire", c_uint16),
        ("cvState", c_uint8),
    ]


class QueryToEmbedded(Structure):
    """Jetson -> Embedded: request transformation (3 bytes packed)."""

    _pack_ = 1
    _fields_ = [
        ("magic", c_uint8),
        ("messageType", c_uint8),
        ("frameDelay_ms", c_uint8),
    ]


class EmbeddedTransformationMessage(Structure):
    """Embedded -> Jetson: current turret transformation (69 bytes packed)."""

    _pack_ = 1
    _fields_ = [
        ("magic", c_uint8),
        ("yaw", c_float),
        ("pitch", c_float),
        ("matrix", c_float * 16),
    ]


class OdomQueryToEmbedded(Structure):
    """Jetson -> Embedded: request odometry (2 bytes packed, messageType ``'q'``).

    Distinct from :class:`QueryToEmbedded` (the 3-byte transform query): odom
    carries no ``frameDelay_ms``. The MCU dispatches both by ``messageType`` on the
    same wire. Mirrors ``uart_odom_node.py``'s ``QueryToEmbedded``.
    """

    _pack_ = 1
    _fields_ = [
        ("magic", c_uint8),
        ("messageType", c_uint8),
    ]


class EmbeddedOdometryMessage(Structure):
    """Embedded -> Jetson: chassis odometry (13 bytes packed).

    Embedded frame: ``x`` = right, ``y`` = forward, ``theta`` = gyro heading (rad).
    """

    _pack_ = 1
    _fields_ = [
        ("magic", c_uint8),
        ("x", c_float),
        ("y", c_float),
        ("theta", c_float),
    ]


class VelocityToEmbedded(Structure):
    """Jetson -> Embedded: velocity command (10 bytes packed, messageType ``'v'``).

    Velocities are in the embedded WORLD frame (``vx`` = right, ``vy`` = forward);
    the MCU rotates world->chassis with its own live gyro angle. No omega -- yaw is
    owned by the MCU. Fire-and-forget. Mirrors ``uart_odom_node.py``.
    """

    _pack_ = 1
    _fields_ = [
        ("magic", c_uint8),
        ("messageType", c_uint8),
        ("vx", c_float),
        ("vy", c_float),
    ]


# ---------------------------------------------------------------------------
# Communicator
# ---------------------------------------------------------------------------


class EmbeddedCommunicator:
    """Manages UART communication with the embedded MCU."""

    def __init__(self) -> None:
        serial_port = config.communication.serial_port
        print(f"[EmbeddedCommunicator] Initializing on port: {serial_port}")
        baudrate = config.communication.serial_baudrate
        print(f"[EmbeddedCommunicator] Using baudrate: {baudrate}")
        self._serial_port_path = serial_port
        self._baudrate = baudrate
        self.port: Optional[serial.Serial] = self._setup_serial_port()
        if self.port is not None:
            self.port.reset_input_buffer()
            self.port.reset_output_buffer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_camera_to_ballistic_transformation(
        self, milliseconds_in_the_past: int
    ) -> Optional[Tuple[float, float, npt.NDArray[np.float64]]]:
        """Query embedded for the 4x4 camera-to-ballistic transformation.

        Returns:
            ``(yaw, pitch, 4x4_matrix)`` or ``None`` on failure.
        """
        if self.port is None:
            print("warning: serial port not available for embedded communication")
            return None
        # Clear any stale/partial bytes so the reply we read is frame-aligned to
        # this query (a prior timed-out read could otherwise leave leftover bytes
        # that desync this one into a garbage matrix).
        self.port.reset_input_buffer()
        if self._send_query_to_embedded(milliseconds_in_the_past):
            msg = self._read_transformation_message()
            if msg is not None:
                matrix = np.array([*msg.matrix], dtype=np.float64).reshape((4, 4))
                return msg.yaw, msg.pitch, matrix
        return None

    def send_angles_to_embedded(
        self,
        pitch: float,
        yaw: float,
        time_until_next_fire: int,
        cv_state: int,
        magic: str = "a",
        message_type: str = "d",
    ) -> bool:
        """Send the ballistic solution to the embedded system."""
        if self.port is None:
            return False
        assert time_until_next_fire >= 0, "time_until_next_fire must be non-negative"
        if time_until_next_fire > 255:
            print(f"Warning: time_until_next_fire {time_until_next_fire}")
        message = JetsonMessage(
            magic=ord(magic),
            messageType=ord(message_type),
            pitch=float(pitch),
            yaw=float(yaw),
            timeUntilNextFire=c_uint16(time_until_next_fire),
            cvState=c_uint8(cv_state).value,
        )
        try:
            self.port.write(bytes(message))
            return True
        except Exception as error:
            print(f"[EmbeddedCommunicator] Write error: {error}")
            return False

    def get_odometry(self) -> Optional[Tuple[float, float, float]]:
        """Query embedded for chassis odometry.

        Returns:
            ``(x, y, theta)`` in the embedded frame (x=right, y=forward, theta=rad)
            or ``None`` on failure.
        """
        if self.port is None:
            print("warning: serial port not available for odometry query")
            return None
        # Frame-align the reply to this query (a prior partial read could desync).
        self.port.reset_input_buffer()
        query = OdomQueryToEmbedded(magic=ord("a"), messageType=ord("q"))
        try:
            self.port.write(bytes(query))
        except Exception as error:
            print(f"[EmbeddedCommunicator] Odom query error: {error}")
            return None
        expected_size = sizeof(EmbeddedOdometryMessage())
        data = self.port.read(expected_size)
        if len(data) != expected_size:
            return None
        msg = EmbeddedOdometryMessage.from_buffer_copy(data)
        return float(msg.x), float(msg.y), float(msg.theta)

    def send_velocity(
        self,
        vx: float,
        vy: float,
        magic: str = "a",
        message_type: str = "v",
    ) -> bool:
        """Send a world-frame velocity command (vx=right, vy=forward) to embedded."""
        if self.port is None:
            return False
        command = VelocityToEmbedded(
            magic=ord(magic),
            messageType=ord(message_type),
            vx=float(vx),
            vy=float(vy),
        )
        try:
            self.port.write(bytes(command))
            return True
        except Exception as error:
            print(f"[EmbeddedCommunicator] Velocity write error: {error}")
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _setup_serial_port(self) -> Optional[serial.Serial]:
        if not self._serial_port_path:
            return None
        try:
            return serial.Serial(
                self._serial_port_path,
                baudrate=self._baudrate,
                timeout=TRANSFORM_READ_TIMEOUT_S,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
        except Exception:
            try:
                subprocess.run(
                    [
                        "bash",
                        "-c",
                        f'sudo -S chmod 777 \'{self._serial_port_path}\' <<< "$(cat "$HOME/.pass")"',
                    ],
                    check=False,
                )
                return serial.Serial(
                    self._serial_port_path,
                    baudrate=self._baudrate,
                    timeout=TRANSFORM_READ_TIMEOUT_S,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                )
            except Exception:
                return None

    def _send_query_to_embedded(
        self, frame_delay_ms: int, magic: str = "a", message_type: str = "t"
    ) -> bool:
        query = QueryToEmbedded(
            magic=ord(magic),
            messageType=ord(message_type),
            frameDelay_ms=c_uint8(frame_delay_ms).value,
        )
        try:
            self.port.write(bytes(query))  # type: ignore[union-attr]
            return True
        except Exception as error:
            print(f"[EmbeddedCommunicator] Query error: {error}")
            return False

    def _read_transformation_message(self) -> Optional[EmbeddedTransformationMessage]:
        expected_size = sizeof(EmbeddedTransformationMessage())
        # read() returns as soon as expected_size bytes arrive (typically
        # ~1-2 ms) and otherwise blocks at most the port's timeout
        # (TRANSFORM_READ_TIMEOUT_S) -- no 10 ms busy-poll quantization.
        data = self.port.read(expected_size)  # type: ignore[union-attr]
        if len(data) != expected_size:
            return None
        return EmbeddedTransformationMessage.from_buffer_copy(data)
