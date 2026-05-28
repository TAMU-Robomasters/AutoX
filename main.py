"""Entry point for the AutoX application."""
from src.engines.detection_test_engine import DetectionTestEngine

if __name__ == "__main__":
    engine = DetectionTestEngine()
    engine.start()
    try:
        engine.join()
    except KeyboardInterrupt:
        engine.stop()
