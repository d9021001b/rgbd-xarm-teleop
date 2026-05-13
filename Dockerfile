FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=jazzy
ENV DISPLAY=:1
ENV QT_X11_NO_MITSHM=1
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MESA_GL_VERSION_OVERRIDE=3.3

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    locales \
    software-properties-common \
  && locale-gen en_US en_US.UTF-8 \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
  && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
      > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    dbus-x11 \
    ffmpeg \
    git \
    libgl1-mesa-dri \
    libglx-mesa0 \
    mesa-utils \
    novnc \
    openbox \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    ros-jazzy-desktop \
    ros-jazzy-gz-ros2-control \
    ros-jazzy-moveit \
    ros-jazzy-ros-gz \
    ros-jazzy-ros2-control \
    ros-jazzy-ros2-controllers \
    ros-jazzy-xacro \
    websockify \
    x11vnc \
    xterm \
    xvfb \
  && rosdep init || true \
  && rosdep update \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /root/ws
COPY xarm_ros2 /root/ws/src/xarm_ros2

RUN touch /root/ws/src/xarm_ros2/thirdparty/realsense_gazebo_plugin/COLCON_IGNORE \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-moveit-servo \
  && source /opt/ros/${ROS_DISTRO}/setup.bash \
  && rosdep install --from-paths src --ignore-src --rosdistro ${ROS_DISTRO} --skip-keys "sdformat14 gz-sim8" -r -y \
  && colcon build --symlink-install --packages-up-to xarm_gazebo xarm_moveit_config --cmake-args -DCMAKE_BUILD_TYPE=Release \
  && rm -rf /var/lib/apt/lists/*

COPY docker/entrypoint.sh /usr/local/bin/xarm-entrypoint
COPY docker/launch-xarm7-gazebo.sh /usr/local/bin/launch-xarm7-gazebo
RUN chmod +x /usr/local/bin/xarm-entrypoint /usr/local/bin/launch-xarm7-gazebo

EXPOSE 6080
ENTRYPOINT ["/usr/local/bin/xarm-entrypoint"]
CMD ["bash"]
