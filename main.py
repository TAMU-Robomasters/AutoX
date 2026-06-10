"""Entry point for the AutoX application."""
# Production entry point (particle-filter auto-aim engine):
# from src.engines.detection_test_engine_cpp import DetectionTestEngineCpp


# def main():
#     """Run the C++ detection test engine."""
#     engine = DetectionTestEngineCpp()
#     engine.start()
#     try:
#         engine.join()
#     except KeyboardInterrupt:
#         engine.stop()


# from src.core.orchestrator import launch_system
# from src.engines.particle_filter_autoaim import ParticleFilterAutoAimEngine

# processes = launch_system([ParticleFilterAutoAimEngine])
# try:
#     for p in processes:
#         p.join()
# except KeyboardInterrupt:
#     for p in processes:
#         p.terminate()
#         p.join()

from src.main import main
if __name__ == "__main__":
    main()
