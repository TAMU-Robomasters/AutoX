import time


class Timeout:
    def __init__(self, duration: float):
        """
        Args:
            duration: seconds until expiry
        """
        self.duration = duration
        self._start = time.perf_counter()

    def reset(self):
        """Restart the timer."""
        self._start = time.perf_counter()

    @property
    def is_expired(self) -> bool:
        return time.perf_counter() - self._start >= self.duration

    @property
    def remaining(self) -> float:
        """Seconds left; 0.0 if already expired."""
        return max(0.0, self.duration - (time.perf_counter() - self._start))
