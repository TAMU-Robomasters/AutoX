"""Entry point for the AutoX application."""
from matplotlib.pylab import ma

from src.main import main

if __name__ == "__main__":
    from src.engines.autoaim import AdvancedAutoAimEngine
    engine = AdvancedAutoAimEngine()
    engine.start()
    # main()