#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "===================================================================="
echo "  [La Rovernetta] Iniciando Filtro de Kalman (ROS 2 EKF en Docker)"
echo "===================================================================="

# 1. Construir la imagen automáticamente si no existe en esta máquina
if ! docker image inspect larovernetta/ekf:jazzy >/dev/null 2>&1; then
    echo "[info] Imagen Docker 'larovernetta/ekf:jazzy' no encontrada en esta máquina."
    echo "[info] Construyendo imagen automáticamente desde $DIR/Dockerfile..."
    docker build -t larovernetta/ekf:jazzy "$DIR"
    echo "[ok] Imagen construida con éxito."
fi

# 2. Auto-detectar la IP del host accesible desde Docker
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$HOST_IP" ]; then
    HOST_IP=$(ip route get 1 2>/dev/null | awk '{print $7}')
fi
if [ -z "$HOST_IP" ]; then
    HOST_IP="127.0.0.1"
fi

SDK_URL="${SDK_URL:-http://${HOST_IP}:8000}"
TARGET_IP="${TARGET_IP:-${HOST_IP}}"
TARGET_PORT="${TARGET_PORT:-9876}"

# Permitir argumentos posicionales: ./run_ekf_docker.sh <SDK_URL> <TARGET_IP>
if [ -n "$1" ] && [[ "$1" =~ ^https?:// ]]; then
    SDK_URL="$1"
    shift
fi

if [ -n "$1" ] && [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    TARGET_IP="$1"
    shift
fi

echo "  • SDK URL:       $SDK_URL"
echo "  • Destino UDP:   $TARGET_IP:$TARGET_PORT"
echo "===================================================================="

# 3. Limpiar contenedor previo si quedó huérfano (por nombre y por ancestro de imagen)
docker rm -f larovernetta_ekf >/dev/null 2>&1 || true
for cid in $(docker ps -aq --filter "ancestor=larovernetta/ekf:jazzy" 2>/dev/null); do
    echo "[info] Limpiando contenedor EKF huérfano o duplicado ($cid)..."
    docker rm -f "$cid" >/dev/null 2>&1 || true
done

# 4. Ejecución del contenedor
exec docker run --rm --network host --name larovernetta_ekf \
    -e SDK_URL="$SDK_URL" \
    -e TARGET_IP="$TARGET_IP" \
    -e TARGET_PORT="$TARGET_PORT" \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
    larovernetta/ekf:jazzy "$@"
