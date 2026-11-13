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
# from src.engines.full_state_autoaim import FullStateAutoAimEngine

# processes = launch_system([FullStateAutoAimEngine])
# try:
#     for p in processes:
#         p.join()
# except KeyboardInterrupt:
#     for p in processes:
#         p.terminate()
#         p.join()

import os

# One BLAS thread per process, set before numpy loads. Every AutoX process
# (camera, MCU, engine) otherwise starts a full OpenBLAS thread pool, and on the
# Jetson they fight over the cores: the EKF update measured 46 ms in the engine
# vs 6 ms with one thread (fred, 2026-11-13). Our matrices are tiny, so threads
# never help. Override from the shell if ever needed.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from src.main import main  # noqa: E402

if __name__ == "__main__":
    main()
