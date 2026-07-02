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
from src.toolbox.logger import get_logger

log = get_logger("embedded_communicator")

# Serial read timeout (upper bound) for the transform query/response round-trip.
# read() returns the instant all bytes land, so the happy-path latency is just
# the message's transmission time -- at 460800 baud the 69-byte reply takes ~1.5 ms
# to clock in, so this MUST exceed that or read() returns a partial message and
# the leftover bytes desync the next read (garbage matrix). This bound only
# applies when a reply is missing entirely.
TRANSFORM_READ_TIMEOUT_S = 0.02


# ---------------------------------------------------------------------------
# Protocol enums & C-compatible structs
# ---------------------------------------------------------------------------


class CVState(Enum):
    """State of the computer-vision pipeline as reported to embedded."""

    NO_TARGET: int = 0        # no target at all; embedded holds fire
    SHOT_TIMING: int = 1      # spinning fast; embedded waits for alignment time
    CONTINUOUS_FIRE: int = 2  # slow/no spin; embedded fires freely
    AIMING: int = 3           # target acquired + aiming, but HOLD FIRE (solution not
                              # trustworthy: out of range, no arc, or the predicted
                              # confidence gate failed). FIRMWARE MUST TREAT 3 AS
                              # HOLD-FIRE (same as NO_TARGET); it only forwards pitch/yaw.


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
            log.warning("serial port not available for embedded communication")
            return None
        # Clear any stale/partial bytes so the reply we read is frame-aligned to
        # this query (a prior timed-out read could otherwise leave leftover bytes
        # that desync this one into a garbage matrix).
        self.port.reset_input_buffer()
        # The caller (engine) has already applied the per-robot frame-delay offset
        # and clamped to the u8 wire range; this is a dumb transport.
        frame_delay_ms = milliseconds_in_the_past
        log.debug("frame_delay_ms=%d", frame_delay_ms)
        if self._send_query_to_embedded(frame_delay_ms):
            msg = self._read_transformation_message()
            if msg is not None:
                matrix = np.array([*msg.matrix], dtype=np.float64).reshape((4, 4))
                log.debug("Received transformation: yaw=%.2f, pitch=%.2f", msg.yaw, msg.pitch)
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
            print(
                f"Warning: time_until_next_fire {time_until_next_fire}"
            ) 
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
