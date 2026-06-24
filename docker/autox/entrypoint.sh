#!/usr/bin/env bash
# autox service entrypoint: source ROS (for rclpy), sync deps into a
# system-site-packages venv, exec. The AutoX repo is mounted at /autox.
set -e

source "/opt/ros/${ROS_DISTRO}/setup.bash"   # puts rclpy on the system Python path

# Create the venv with access to system site-packages so `import rclpy` resolves.
if [ ! -x "${UV_PROJECT_ENVIRONMENT}/bin/python" ]; then
  echo "[autox-entrypoint] creating venv (${UV_PROJECT_ENVIRONMENT}) with system site-packages…"
  uv venv --python 3.10 --system-site-packages "${UV_PROJECT_ENVIRONMENT}"
fi

# Install DEPENDENCIES only, not the AutoX project itself: building the project
# runs the root CMakeLists (pf_cuda_cv, CUDA sm_87) which isn't present in this
# CPU container and isn't needed for the nav/MCU engines. AutoX is then run from
# the repo root (src/ layout) with the repo as CWD.
#   --inexact: don't prune system-site packages (keeps rclpy).
#   --extra linux: PyAV + v4l2 capture backends used by @MOCK_CAM.
echo "[autox-entrypoint] uv sync (deps only, no project build)…"
uv sync --no-install-project --inexact --extra linux \
  || echo "[autox-entrypoint] WARN: uv sync had errors; check iceoryx2/cargo build"

exec "$@"
