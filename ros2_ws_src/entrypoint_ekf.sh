#!/bin/bash
set -e

source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash

SDK_URL="${SDK_URL:-http://localhost:8000}"
TARGET_IP="${TARGET_IP:-127.0.0.1}"
TARGET_PORT="${TARGET_PORT:-9876}"

echo "===================================================================="
echo "  [La Rovernetta Docker] Filtro de Kalman EKF Activo"
echo "===================================================================="
echo "  • SDK URL:     $SDK_URL"
echo "  • Export UDP:  $TARGET_IP:$TARGET_PORT"
echo "===================================================================="

exec ros2 launch er_bringup localization_bypass.launch.py \
    sdk_url:="$SDK_URL" \
    target_ip:="$TARGET_IP" \
    target_port:="$TARGET_PORT" \
    "$@"
