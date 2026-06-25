# ROS 2 Humble (Ubuntu 22.04 -> system Python 3.10) + uv + AutoX.
#
# Why a ROS base image for AutoX: Humble ships rclpy built for Python 3.10, which
# is exactly AutoX's pin (and the Jetson's runtime). So the uv venv -- created
# with --system-site-packages -- can `import rclpy` with no PyPI rclpy (there is
# none). MCU=MOCK + @MOCK_CAM mean no serial, no camera, no mrcal.
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# build-essential/cmake/cargo: iceoryx2==0.9.0 builds from source where no wheel
# exists. libgl1/glib: opencv-python runtime. ffmpeg/v4l-utils: capture backends.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake pkg-config git curl ca-certificates \
      libgl1 libglib2.0-0 \
      ffmpeg v4l-utils \
      cargo rustc \
    && rm -rf /var/lib/apt/lists/*

# nav2_simple_commander (BasicNavigator) + nav2_msgs: the AutoNav engine drives
# nav2 through these. Separate layer so the build-tools layer above stays cached.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-nav2-simple-commander \
      ros-${ROS_DISTRO}-nav2-msgs \
    && rm -rf /var/lib/apt/lists/*

# uv (Astral) on PATH.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Keep the venv OUT of the mounted repo so the host's .venv (built for the host's
# Python) never leaks in, and so system-site rclpy is visible.
ENV UV_PROJECT_ENVIRONMENT=/opt/autox-venv

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /autox
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
