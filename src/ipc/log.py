import ctypes

class LogMessage(ctypes.Structure):
    _fields_ = [
        ("log_level", ctypes.c_char),
        ("process_name", ctypes.c_char * 32),
        ("message", ctypes.c_char * 256)
    ]

    __str__ = lambda self: f"[{self.log_level.decode('utf-8')}] [{self.process_name.decode('utf-8')}] {self.message.decode('utf-8')}"
