FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# 1. Herramientas base y paquetes ROS 2 necesarios para el EKF
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    libopencv-dev \
    python3-opencv \
    ros-jazzy-robot-localization \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-geometry-msgs \
    ros-jazzy-geographic-msgs \
    ros-jazzy-cv-bridge \
    ros-jazzy-nav-msgs \
    ros-jazzy-sensor-msgs \
    ros-jazzy-geometry-msgs \
    ros-jazzy-std-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/ros2_ws

# 2. Dependencias de Python para el bridge de telemetría y cálculo geomagnético
COPY requirements-ros.txt ./
RUN pip3 install --break-system-packages --no-cache-dir -r requirements-ros.txt

# 3. Copiar paquetes de ROS 2 de La Rovernetta
COPY earth_rovers_sdk ./src/earth_rovers_sdk
COPY er_bringup ./src/er_bringup
COPY mini_plus_localization ./src/mini_plus_localization

# 4. Compilar el workspace
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && colcon build --symlink-install"

# 5. Script de entrada
COPY entrypoint_ekf.sh /root/ros2_ws/entrypoint_ekf.sh
RUN chmod +x /root/ros2_ws/entrypoint_ekf.sh

ENTRYPOINT ["/root/ros2_ws/entrypoint_ekf.sh"]
