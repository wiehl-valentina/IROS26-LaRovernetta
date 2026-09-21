#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR/earth-rovers-sdk"

# Busca el entorno conda "erc_sdk" sin rutas fijas: respeta CONDA_BASE si está
# definido; si no, usa `conda info --base` o las instalaciones habituales.
ENV_NAME="erc_sdk"
for base in "${CONDA_BASE:-}" "$(conda info --base 2>/dev/null || true)" \
            "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge"; do
    if [ -n "$base" ] && [ -x "$base/envs/$ENV_NAME/bin/python3" ]; then
        ENV_BIN="$base/envs/$ENV_NAME/bin"
        break
    fi
done
if [ -z "${ENV_BIN:-}" ]; then
    echo "[start_sdk] ERROR: no se encontró el entorno conda '$ENV_NAME'. Definí CONDA_BASE con la raíz de tu conda." >&2
    exit 1
fi
PYTHON="$ENV_BIN/python3"
HYPERCORN="$ENV_BIN/hypercorn"

echo "[start_sdk] Iniciando el servidor del SDK localmente en puerto 8000..."
exec "$HYPERCORN" main:app --reload --bind 0.0.0.0:8000
