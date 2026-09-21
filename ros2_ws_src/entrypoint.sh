#!/bin/bash
set -e

ROVER_MODE="${ROVER_MODE:-manual}"
SDK_URL="${SDK_URL:-http://localhost:8000}"
TARGET_IP="${TARGET_IP:-127.0.0.1}"
TARGET_PORT="${TARGET_PORT:-9876}"

# Configurar entorno de ROS 2
source /opt/ros/jazzy/setup.bash 2>/dev/null || source /opt/ros/lyrical/setup.bash 2>/dev/null || source /opt/ros/humble/setup.bash 2>/dev/null || true

WS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "/root/ros2_ws/install/setup.bash" ]; then
    source "/root/ros2_ws/install/setup.bash"
elif [ -f "$WS_DIR/install/setup.bash" ]; then
    source "$WS_DIR/install/setup.bash"
fi

case "$ROVER_MODE" in
  manual)
    echo "================================================================================"
    echo "[entrypoint] ROVER_MODE=manual -> Contenedor en espera para control interactivo."
    echo "================================================================================"
    exec sleep infinity
    ;;
  ekf)
    echo "================================================================================"
    echo "[entrypoint] ROVER_MODE=ekf -> Iniciando EKF (localization_bypass)..."
    echo "================================================================================"
    exec ros2 launch er_bringup localization_bypass.launch.py \
        sdk_url:="$SDK_URL" \
        target_ip:="$TARGET_IP" \
        target_port:="$TARGET_PORT" \
        "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
