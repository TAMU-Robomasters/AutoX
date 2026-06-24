"""Tiny RPC envelopes for the MCU queue-bridge driver.

The ``McuDriver`` (see ``src/drivers/mcu.py``) owns the UART in one process and
services requests from many engines over a ``multiprocessing.Queue``. Engines
push an :class:`McuRequest` and read back the matching :class:`McuResponse`.
These are deliberately small, picklable dataclasses (the control plane is tiny
messages — no shared memory needed).
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass
class McuRequest:
    """An engine -> driver request.

    Attributes:
        op: operation name (``"get_transformation"`` | ``"send_solution"`` |
            ``"get_odometry"`` | ``"send_velocity"`` | ``"get_match_state"``).
        args: positional arguments for the op.
        req_id: monotonic id the client uses to match the response. ``None`` for
            fire-and-forget ops (e.g. ``send_solution``/``send_velocity``) that
            expect no reply.
        consumer_key: which consumer's response queue the driver must reply on.
            One queue per consumer keeps concurrent consumers (auto-aim +
            nav-bridge) from dequeuing each other's replies. ``"default"`` for the
            single-consumer case. Carried on fire-and-forget ops too (harmless).
    """

    op: str
    args: Tuple[Any, ...] = ()
    req_id: Optional[int] = None
    consumer_key: str = "default"


@dataclass
class McuResponse:
    """A driver -> engine reply, matched to a request by ``req_id``."""

    req_id: int
    ok: bool
    value: Any = None
    error: Optional[str] = None
