"""Types used for inter-process communication."""

import ctypes

from src.subsystems.video_streaming.video_stream import video_stream


class LogMessage(ctypes.Structure):
    """Log message sent over IPC."""

    _fields_ = [
        ("log_level", ctypes.c_char),
        ("process_name", ctypes.c_char * 32),
        ("message", ctypes.c_char * 256),
    ]

    def __str__(self):
        return (
            f"[{self.log_level.decode('utf-8')}] "
            f"[{self.process_name.decode('utf-8')}] "
            f"{self.message.decode('utf-8')}"
        )


class ImageMessage(ctypes.Structure):
    """Image message sent over IPC."""

    _fields_ = [
        ("timestamp_ns", ctypes.c_uint64),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        # Flat, inline array allocated in shared memory.
        # Example: 1080p RGB image
        ("pixels", ctypes.c_uint8 * (video_stream.width * video_stream.height * 3)),
    ]
