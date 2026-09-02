#!/usr/bin/env bash

# ==============================================================================
# SCRIPT DE REINICIO Y LIMPIEZA COMPLETA DE ENTORNO - ROVERNETTA
# ==============================================================================



# --- 1. CONFIGURACIÓN DE VARIABLES ---
DIR_SDK_NAME="earth-rovers-sdk"
VENV_SDK_NAME=".venv"
CONDA_ENV_NAME="sam_tp"

# --- 2. RESOLUCIÓN DE RUTAS ABSOLUTAS ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PATH_SDK="$ROOT_DIR/$DIR_SDK_NAME"
PATH_VENV_SDK="$PATH_SDK/$VENV_SDK_NAME"

# --- FUNCIONES DE IMPRESIÓN ---
info()    { echo -e "\n\033[1;34m[INFO]\033[0m $1"; }
success() { echo -e "\033[1;32m[OK]\033[0m $1"; }
warn()    { echo -e "\033[1;33m[ADVERTENCIA]\033[0m $1"; }

echo "=========================================================="
echo "      ROVERNETTA: LIMPIEZA Y REINICIO DE ENTORNO          "
echo "=========================================================="
echo "Directorio raíz detectado: $ROOT_DIR"

# --- 3. CONFIRMACIÓN DEL USUARIO ---
read -p "⚠️  ¿Estás seguro de que deseas eliminar los entornos (.venv, Conda) y archivos generados? (s/N): " CONFIRMATION
if [[ ! "$CONFIRMATION" =~ ^[sS]$ ]]; then
    echo "Operación cancelada por el usuario."
    exit 0
fi

# --- 4. DESACTIVAR ENTORNOS ACTIVOS ---
info "Desactivando entornos virtuales en la sesión actual..."
deactivate 2>/dev/null || true
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda deactivate 2>/dev/null || true
fi

# --- 5. ELIMINAR ENTORNO VIRTUAL DEL SDK (.venv) ---
info "Eliminando entorno virtual Python ($VENV_SDK_NAME) del SDK..."
if [ -d "$PATH_VENV_SDK" ]; then
    rm -rf "$PATH_VENV_SDK"
    success "Entorno $VENV_SDK_NAME eliminado correctamente."
else
    warn "No se encontró el entorno $VENV_SDK_NAME en $PATH_SDK."
fi

# --- 6. ELIMINAR ENTORNO DE CONDA (sam_tp) ---
info "Eliminando entorno de Conda '$CONDA_ENV_NAME'..."
if command -v conda &> /dev/null; then
    if conda env list | grep -qE "^${CONDA_ENV_NAME}\s"; then
        conda env remove -n "$CONDA_ENV_NAME" -y
        success "Entorno Conda '$CONDA_ENV_NAME' eliminado."
    else
        warn "El entorno Conda '$CONDA_ENV_NAME' no existe."
    fi
else
    warn "Conda no está instalado o no se encuentra en el PATH."
fi

# --- 7. LIMPIEZA DE CONFIGURACIONES Y ARCHIVOS TEMPORALES ---
info "Eliminando archivos de configuración (.env) y cachés de Python/Playwright..."

# Remover .env generado en SDK
if [ -f "$PATH_SDK/.env" ]; then
    rm -f "$PATH_SDK/.env"
    success "Archivo .env borrado en $DIR_SDK_NAME."
fi

# Limpiar caché global de navegadores Playwright del usuario
if [ -d "$HOME/.cache/ms-playwright" ]; then
    rm -rf "$HOME/.cache/ms-playwright"
    success "Caché de Playwright eliminada."
fi

# Limpiar carpetas __pycache__, .egg-info y .pytest_cache
cd "$ROOT_DIR"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
success "Archivos de caché compilados (__pycache__) eliminados."

echo "=========================================================="
success "¡El entorno ha sido completamente restablecido!"
echo "Para volver a instalar y verificar todo desde cero, ejecuta:"
echo "  ./scripts/setup.sh"
echo "=========================================================="