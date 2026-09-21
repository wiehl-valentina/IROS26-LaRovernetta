#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 1. Detección de la Distribución de ROS 2
DETECTED_ROS=""
if [ -n "$ROS_DISTRO" ] && [ -d "/opt/ros/$ROS_DISTRO" ]; then
    DETECTED_ROS="$ROS_DISTRO"
elif [ -d "/opt/ros/lyrical" ]; then
    DETECTED_ROS="lyrical"
elif [ -d "/opt/ros/jazzy" ]; then
    DETECTED_ROS="jazzy"
elif [ -d "/opt/ros/humble" ]; then
    DETECTED_ROS="humble"
fi

if [ -z "$DETECTED_ROS" ]; then
    echo "[error] No se encontró ninguna distribución de ROS 2 instalada en /opt/ros/"
    echo "Ejecuta primero: ./install_dependencies.sh"
    exit 1
fi

echo "[build] Configurando entorno ROS 2: /opt/ros/$DETECTED_ROS/setup.bash"
source "/opt/ros/$DETECTED_ROS/setup.bash"

echo "[build] Compilando paquetes de ROS 2 en $DIR..."
colcon build --symlink-install "$@"

echo ""
echo "[build] Verificando instalación..."
source install/setup.bash
PKGS=$(ros2 pkg list | grep -E 'earth_rovers_sdk|er_bringup|mini_plus_localization' | tr '\n' ' ')
echo "[build] Paquetes compilados e indexados: $PKGS"
echo "[build] ¡Compilación finalizada con éxito!"
