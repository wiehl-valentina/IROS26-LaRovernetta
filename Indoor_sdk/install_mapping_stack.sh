#!/usr/bin/env bash
# install_mapping_stack.sh
#
# Instala TODO lo necesario, en una maquina Linux nativa (no WSL2), para
# correr el pipeline de mapeo: bridge ROS2 + RTAB-Map + genie_rover
# (map_session.py). Pensado para Ubuntu 22.04 (ROS2 Humble) o 24.04 (ROS2
# Jazzy) -- detecta la version sola.
#
# Uso:
#   chmod +x install_mapping_stack.sh
#   ./install_mapping_stack.sh                 # bridge + RTAB-Map + genie_rover CPU
#   ./install_mapping_stack.sh --gpu            # + torch con CUDA (para SAM-TP)
#   ./install_mapping_stack.sh --repo ~/mi_repo # si el repo no esta en ~/IROS26-LaRovernetta
#
# Es re-corrible: si algo ya esta instalado, apt/pip lo saltean solos.
# No necesita que el SDK ni el rover esten conectados para instalar.

set -euo pipefail

REPO_PATH="${HOME}/IROS26-LaRovernetta"
WITH_GPU=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) WITH_GPU=1; shift ;;
        --repo) REPO_PATH="$2"; shift 2 ;;
        *) echo "Argumento desconocido: $1"; exit 1 ;;
    esac
done

log()  { echo -e "\n\033[1;32m==> $*\033[0m"; }
warn() { echo -e "\033[1;33m[AVISO] $*\033[0m"; }
die()  { echo -e "\033[1;31m[ERROR] $*\033[0m"; exit 1; }

if [[ "$(uname -s)" != "Linux" ]]; then
    die "Este script es para Linux nativo (Ubuntu). En Windows usa WSL2 y corre esto DENTRO de la Ubuntu de WSL2, no en PowerShell."
fi

. /etc/os-release
UBUNTU_CODENAME_LOCAL="${UBUNTU_CODENAME:-}"
case "$VERSION_ID" in
    22.04) ROS_DISTRO="humble" ;;
    24.04) ROS_DISTRO="jazzy" ;;
    *) die "Ubuntu $VERSION_ID no probado por este script (esperaba 22.04 o 24.04). Ajusta ROS_DISTRO a mano si sabes lo que haces." ;;
esac
log "Ubuntu $VERSION_ID detectado -> ROS2 $ROS_DISTRO"

# ---------------------------------------------------------------- 1. locale
log "1/7 Configurando locale UTF-8"
sudo apt update -y
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# ------------------------------------------------------- 2. repos universe
log "2/7 Habilitando repositorio universe y actualizando apt"
sudo apt install -y software-properties-common curl gnupg
sudo add-apt-repository universe -y
sudo apt update -y

# --------------------------------------------------------------- 3. ROS2
log "3/7 Instalando ROS2 $ROS_DISTRO (esto tarda varios minutos)"
if ! dpkg -l | grep -q "ros-${ROS_DISTRO}-desktop"; then
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
    sudo apt update -y
    sudo apt install -y "ros-${ROS_DISTRO}-desktop"
else
    warn "ros-${ROS_DISTRO}-desktop ya estaba instalado, salteo"
fi

if ! grep -q "source /opt/ros/${ROS_DISTRO}/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
fi
source "/opt/ros/${ROS_DISTRO}/setup.bash"

# ------------------------------------------------------- 4. RTAB-Map + tf2
log "4/7 Instalando RTAB-Map y utilidades ROS2 necesarias"
sudo apt install -y \
    "ros-${ROS_DISTRO}-rtabmap-ros" \
    "ros-${ROS_DISTRO}-tf2-ros" \
    "ros-${ROS_DISTRO}-cv-bridge" \
    "ros-${ROS_DISTRO}-image-transport" \
    python3-pip python3-venv python3-full \
    python3-colcon-common-extensions

# ---------------------------------------------- 5. deps python del bridge
log "5/7 Instalando dependencias Python del bridge ROS2 (en el Python del sistema, con rclpy)"
# Importante: estas van con pip3 al Python DEL SISTEMA (el mismo que trae
# rclpy de ROS2) -- earth_rover_bridge.py y map_session.py con
# rtabmap_correction:true necesitan poder importar rclpy Y estas libs a la
# vez, asi que NO conviene un venv separado para esta parte.
pip3 install --user \
    requests \
    websocket-client \
    opencv-python \
    pyyaml \
    pillow \
    numpy

# --------------------------------------------------- 6. genie_rover (venv)
log "6/7 Preparando entorno de genie_rover"
if [[ -d "$REPO_PATH/genie" ]]; then
    cd "$REPO_PATH/genie"
    if [[ ! -d ".venv" ]]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate

    # rclpy/tf2_ros del sistema tienen que ser visibles DENTRO de este venv
    # para que rtabmap_pose_bridge.py funcione (si vas a usar
    # rtabmap_correction: true). Un venv normal no ve los paquetes del
    # sistema salvo que se cree con --system-site-packages.
    warn "Si mapping.rtabmap_correction.enabled=true en tu config, este venv"
    warn "necesita ver rclpy del sistema. Si no fue creado con"
    warn "--system-site-packages, recrealo asi:"
    warn "  rm -rf .venv && python3 -m venv --system-site-packages .venv"

    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    else
        warn "No encontre genie/requirements.txt -- instalando lo minimo conocido"
        pip install requests pyyaml pillow numpy opencv-python
    fi

    if [[ $WITH_GPU -eq 1 ]]; then
        log "Instalando torch con soporte CUDA (para SAM-TP)"
        nvidia-smi >/dev/null 2>&1 || warn "nvidia-smi no encontro GPU -- segui igual, pero revisa el driver antes de usar SAM-TP"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        warn "Corriste sin --gpu: si genie_rover necesita SAM-TP con GPU,"
        warn "volve a correr este script con --gpu, o instala torch a mano."
    fi
    deactivate
else
    warn "No encontre $REPO_PATH/genie -- salteo la parte de genie_rover."
    warn "Corre de nuevo con --repo /ruta/a/tu/repo si esta en otro lado."
fi

# --------------------------------------------------------------- 7. check
log "7/7 Verificacion"
source "/opt/ros/${ROS_DISTRO}/setup.bash"
echo "ROS2: $(ros2 --version 2>&1 || echo 'FALLO')"
ros2 pkg list 2>/dev/null | grep -q rtabmap_slam && echo "rtabmap_slam: OK" || warn "rtabmap_slam: NO ENCONTRADO"
python3 -c "import cv2, requests, websocket, yaml, PIL, numpy; print('deps python bridge: OK')" \
    || warn "faltan deps python del bridge"

echo
log "Listo. Pasos siguientes:"
echo "  1) Abri una terminal nueva (o: source ~/.bashrc) para que quede el PATH de ROS2."
echo "  2) Copia/confirma que los archivos nuevos esten en:"
echo "       $REPO_PATH/earth-rovers-sdk/examples/ros2/{earth_rover_bridge.py,differential_odometry.py,mapping/}"
echo "       $REPO_PATH/genie/genie_rover/Indoor/{map_session.py,rtabmap_pose_bridge.py}"
echo "       $REPO_PATH/genie/configs/indoor_mapping.yaml"
echo "  3) Aplica el parche de persistent_map_export_addon.py a persistent_map.py (si no lo hiciste)."
echo "  4) Segui la guia de $REPO_PATH/earth-rovers-sdk/examples/ros2/mapping/README.md"
