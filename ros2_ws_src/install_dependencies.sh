#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "===================================================================="
echo "  Instalador de Dependencias de ROS 2 - La Rovernetta"
echo "===================================================================="

# 1. Detección del Sistema Operativo
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_NAME=$NAME
    OS_VER=$VERSION_ID
    OS_CODENAME=$VERSION_CODENAME
    echo "[info] Sistema Operativo detectado: $OS_NAME $OS_VER ($OS_CODENAME)"
else
    echo "[error] No se pudo determinar la distribución de Linux."
    exit 1
fi

# 2. Detección de la Distribución de ROS 2
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
    echo "[advertencia] No se detectó ninguna instalación de ROS 2 en /opt/ros/"
    case "$OS_VER" in
        "22.04") DETECTED_ROS="humble" ;;
        "24.04") DETECTED_ROS="jazzy" ;;
        "26.04") DETECTED_ROS="lyrical" ;;
        *)       DETECTED_ROS="jazzy" ;;
    esac
    echo "[info] Se configurará e instalará automáticamente ROS 2 $DETECTED_ROS..."

    sudo apt-get update && sudo apt-get install -y software-properties-common curl gnupg
    if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
        echo "[info] Descargando clave de repositorio de ROS 2..."
        sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
    fi

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $OS_CODENAME main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends "ros-${DETECTED_ROS}-ros-base"
fi

echo "[info] Distribución de ROS 2: $DETECTED_ROS (/opt/ros/$DETECTED_ROS)"

# 3. Instalación de Paquetes APT del Sistema y ROS 2
echo "[info] Verificando e instalando paquetes APT necesarios..."

APT_PACKAGES=(
    "build-essential"
    "cmake"
    "git"
    "pkg-config"
    "python3-pip"
    "python3-colcon-common-extensions"
    "python3-rosdep"
    "libopencv-dev"
    "python3-opencv"
    "ros-${DETECTED_ROS}-robot-localization"
    "ros-${DETECTED_ROS}-tf2-ros"
    "ros-${DETECTED_ROS}-tf2-geometry-msgs"
    "ros-${DETECTED_ROS}-geographic-msgs"
    "ros-${DETECTED_ROS}-cv-bridge"
    "ros-${DETECTED_ROS}-nav-msgs"
    "ros-${DETECTED_ROS}-sensor-msgs"
    "ros-${DETECTED_ROS}-geometry-msgs"
    "ros-${DETECTED_ROS}-std-msgs"
)

# Detectar paquetes faltantes
MISSING_PKGS=()
for pkg in "${APT_PACKAGES[@]}"; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        MISSING_PKGS+=("$pkg")
    fi
done

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo "[info] Instalando dependencias faltantes: ${MISSING_PKGS[*]}"
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends "${MISSING_PKGS[@]}"
else
    echo "[ok] Todos los paquetes APT de ROS 2 y del sistema ya están instalados."
fi

# 4. Instalación de Dependencias Python
echo "[info] Verificando dependencias Python..."
PIP_OPTS=""
if python3 -c "import sys; exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)" 2>/dev/null; then
    PIP_OPTS=""
else
    PIP_OPTS="--break-system-packages"
fi

if [ -f "$DIR/requirements-ros.txt" ]; then
    pip3 install $PIP_OPTS -r "$DIR/requirements-ros.txt" || pip install $PIP_OPTS -r "$DIR/requirements-ros.txt" || true
fi

# 5. Inicialización de rosdep
if command -v rosdep >/dev/null 2>&1; then
    if [ ! -d /etc/ros/rosdep/sources.list.d ]; then
        echo "[info] Inicializando rosdep..."
        sudo rosdep init 2>/dev/null || true
    fi
    rosdep update 2>/dev/null || true
    echo "[info] Resolviendo dependencias adicionales del workspace con rosdep..."
    rosdep install --from-paths . --ignore-src -r -y 2>/dev/null || true
fi

# 6. Compilación automática de verificación
echo "[info] Compilando el workspace para verificar la instalación..."
"$DIR/build.sh"

echo ""
echo "===================================================================="
echo "  [ÉXITO TOTAL] Todas las dependencias de ROS 2 están instaladas."
echo "  El workspace quedó compilado y listo para correr."
echo "  Para lanzar el Filtro de Kalman ejecuta: ./run_ekf.sh"
echo "===================================================================="
