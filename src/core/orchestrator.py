"""Orchestrator: creates engines, wires shared queues, and starts everything."""

import multiprocessing
from multiprocessing import Queue, Process

from src.engines.particle_filter_autoaim import ParticleFilterAutoAimEngine
from src.engines.plot_engine import start_plot_engine


def start_engines() -> None:
    """Start all engines and block until they finish or the user interrupts."""
    panel_x_queue: Queue = Queue(maxsize=4000)

    autoaim = ParticleFilterAutoAimEngine(queue=panel_x_queue)
    plot = Process(
        target=start_plot_engine,
        args=(panel_x_queue,),
        kwargs={"title": "Panel x position (cm)"},
        daemon=True,
    )

    autoaim.start()
    plot.start()

    try:
        autoaim.join()
    except KeyboardInterrupt:
        autoaim.stop()
    finally:
        plot.terminate()
        plot.join()
