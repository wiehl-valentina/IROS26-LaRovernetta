#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/genie"

# Busca el entorno conda "sam_tp" sin rutas fijas: respeta CONDA_BASE si está
# definido; si no, usa `conda info --base` o las instalaciones habituales.
ENV_NAME="sam_tp"
for base in "${CONDA_BASE:-}" "$(conda info --base 2>/dev/null || true)" \
            "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge"; do
    if [ -n "$base" ] && [ -x "$base/envs/$ENV_NAME/bin/python3" ]; then
        ENV_BIN="$base/envs/$ENV_NAME/bin"
        break
    fi
done
if [ -z "${ENV_BIN:-}" ]; then
    echo "[run_mission] ERROR: no se encontró el entorno conda '$ENV_NAME'. Definí CONDA_BASE con la raíz de tu conda." >&2
    exit 1
fi
PYTHON="$ENV_BIN/python3"

if [ "$1" == "--go" ]; then
    echo "[run_mission] Iniciando La Rovernetta con MOVIMIENTO AUTÓNOMO ACTIVO (--go)..."
    exec "$PYTHON" -m genie_rover.bridge --config configs/frodobot_rover.yaml --go "${@:2}"
else
    echo "[run_mission] Iniciando La Rovernetta en modo SEGURO / DRY-RUN (sin movimiento de motores)..."
    echo "[run_mission] TIP: Para activar movimiento de motores, usá: ./run_mission.sh --go"
    exec "$PYTHON" -m genie_rover.bridge --config configs/frodobot_rover.yaml "$@"
fi
