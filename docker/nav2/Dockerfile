# ROS 2 Humble + Nav2 + Gazebo Classic for the sentry nav sim rig.
#
# The nav2-sim-testing workspace is volume-mounted (compose) and colcon-built at
# container start (entrypoint.sh) so edits are live, submodules are handled at
# runtime, and the (private, SSH-only, hardware) dual-ldlidar submodule never has
# to be present to build the simulation packages.
ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-colcon-common-extensions \
      python3-rosdep \
      ros-${ROS_DISTRO}-navigation2 \
      ros-${ROS_DISTRO}-nav2-bringup \
      ros-${ROS_DISTRO}-slam-toolbox \
      ros-${ROS_DISTRO}-gazebo-ros-pkgs \
      ros-${ROS_DISTRO}-gazebo-ros2-control \
      ros-${ROS_DISTRO}-robot-state-publisher \
      ros-${ROS_DISTRO}-joint-state-publisher \
      ros-${ROS_DISTRO}-xacro \
      ros-${ROS_DISTRO}-rviz2 \
      ros-${ROS_DISTRO}-teleop-twist-keyboard \
      ros-${ROS_DISTRO}-tf2-tools \
      git \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
