"""Entry point for the AutoX application."""

from src.engines.autoaim import TestPulseEstimationEngine
from src.types.autoaim import AutoAimContext

if __name__ == "__main__":
    engine = TestPulseEstimationEngine()
    engine.start()
