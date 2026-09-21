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
    echo "[error] No se detectó ninguna instalación de ROS 2 en /opt/ros/"
    echo "Ejecuta primero: ./install_dependencies.sh (o usa Docker con ./run_ekf_docker.sh)"
    exit 1
fi

source "/opt/ros/$DETECTED_ROS/setup.bash"

# 2. Verificación de compilación local
if [ ! -f "$DIR/install/setup.bash" ]; then
    echo "[info] No se detectó la carpeta 'install/'. Compilando automáticamente..."
    "$DIR/build.sh"
fi

source "$DIR/install/setup.bash"

# 3. Verificación de dependencia crítica de runtime (robot_localization)
if ! ros2 pkg prefix robot_localization >/dev/null 2>&1; then
    echo "================================================================================"
    echo "[ERROR] El paquete 'robot_localization' no está instalado en ROS 2 ($DETECTED_ROS)."
    echo "================================================================================"
    echo "Para instalarlo en tu sistema, ejecuta:"
    echo "  sudo apt install ros-${DETECTED_ROS}-robot-localization"
    echo ""
    echo "O ejecuta el script instalador de todas las dependencias:"
    echo "  ./install_dependencies.sh"
    echo ""
    echo "Si prefieres correrlo en Docker aislado sin instalar paquetes en el host:"
    echo "  ./run_ekf_docker.sh"
    echo "================================================================================"
    exit 1
fi

# 4. Parámetros de conexión
SDK_URL="${SDK_URL:-http://localhost:8000}"
TARGET_IP="${TARGET_IP:-127.0.0.1}"
TARGET_PORT="${TARGET_PORT:-9876}"

# Permitir sobreescritura de parámetros posicionales simples
if [ -n "$1" ] && [[ "$1" =~ ^https?:// ]]; then
    SDK_URL="$1"
    shift
fi

if [ -n "$1" ] && [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    TARGET_IP="$1"
    shift
fi

echo "===================================================================="
echo "  [La Rovernetta] Iniciando Filtro de Kalman (ROS 2 EKF Stack)"
echo "===================================================================="
echo "  • SDK URL:       $SDK_URL"
echo "  • Destino UDP:   $TARGET_IP:$TARGET_PORT"
echo "  • ROS Distro:    $DETECTED_ROS"
echo "===================================================================="

# 5. Ejecución del Launch File
exec ros2 launch er_bringup localization_bypass.launch.py \
    sdk_url:="$SDK_URL" \
    target_ip:="$TARGET_IP" \
    target_port:="$TARGET_PORT" \
    "$@"
