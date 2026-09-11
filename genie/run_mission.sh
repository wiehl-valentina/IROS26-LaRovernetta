#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PYTHON="/home/marian/miniconda3/envs/sam_tp/bin/python3"

if [ "$1" == "--go" ]; then
    echo "[run_mission] Iniciando La Rovernetta con MOVIMIENTO AUTÓNOMO ACTIVO (--go)..."
    exec "$PYTHON" -m genie_rover.bridge --config configs/frodobot_rover.yaml --go "${@:2}"
else
    echo "[run_mission] Iniciando La Rovernetta en modo SEGURO / DRY-RUN (sin movimiento de motores)..."
    echo "[run_mission] TIP: Para activar movimiento de motores, usá: ./run_mission.sh --go"
    exec "$PYTHON" -m genie_rover.bridge --config configs/frodobot_rover.yaml "$@"
fi
