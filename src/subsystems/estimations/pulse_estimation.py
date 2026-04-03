"""Estimates armor panel visibility duration and rotation interval for spinning robots."""

from typing import Optional
from src.core.module import Module, real
from src.types.autoaim import AutoAimContext


class LowPassFilter():
    """Simple low-pass filter to smooth out estimates."""

    def __init__(self, alpha: float):
        self._alpha = alpha
        self._value: Optional[float] = None

    def update(self, sample: float) -> None:
        if self._value is None:
            self._value = sample
        else:
            self._value = self._alpha * sample + (1 - self._alpha) * self._value

    def get_value(self) -> Optional[float]:
        return self._value



class PulseEstimation(Module[AutoAimContext]):
    """Module to estimate the duration of armor panel visibility and rotation interval for spinning robots."""

    def __init__(self, context: AutoAimContext):
        super().__init__(
            name="PulseEstimation",
            context=context,
            inputs=["panels", "timestamp"],
            outputs=["panel_duration", "rotation_interval"]
        )

 
        #two global variables to store the timestamps of the current and previous rising edges of panel visibility
        self.rising_edge = None
        self.falling_edge = None
        self.prev_panels = False  # Previous panel value to detect rising/falling edges

        self.intervalFilter = LowPassFilter(alpha=0.5)  # Adjust alpha as needed for smoothing

        self.durationFilter = LowPassFilter(alpha=0.5)  # Adjust alpha as needed for smoothing

    @real()
    def _run_estimation(self, panels, timestamp) -> tuple[float, float]:
        """Estimate the duration of armor panel visibility and rotation interval."""
        if not panels or timestamp is None:
            return None, None

        if not self.prev_panels and panels:  # Rising edge: Panel just became fully visible
            if self.rising_edge is not None:
                self.intervalFilter.update(timestamp - self.rising_edge)
            self.rising_edge = timestamp

        elif self.prev_panels and not panels:  # Falling edge: Panel becoming invisible
            self.durationFilter.update( timestamp - self.rising_edge)
            if self.falling_edge is not None:
                self.intervalFilter.update(timestamp - self.falling_edge)
            self.falling_edge = timestamp   
            self.rising_edge = None

        self.prev_panels = panels  # Update previous panel state for next iteration     
    
            
        
        
        

        
        return self.durationFilter.get_value(), self.intervalFilter.get_value()
