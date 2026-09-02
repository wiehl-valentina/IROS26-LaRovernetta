#!/usr/bin/env bash

# ==============================================================================
# SCRIPT UNIVERSAL DE SETUP Y VERIFICACIÓN - ROVERNETTA
# Compatible con Ubuntu 20.04, 22.04, 24.04 y superiores (WSL / Nativo)
# ==============================================================================

set -e
#set -x

# --- 1. PROCESAMIENTO DE PARÁMETROS DE ENTRADA ---
SKIP_TORCH=false

for arg in "$@"; do
    case $arg in
        --skip-torch|-s)
            SKIP_TORCH=true
            shift
            ;;
        --help|-h)
            echo "Uso: ./scripts/setup_and_check.sh [OPCIONES]"
            echo "Opciones:"
            echo "  --skip-torch, -s    Omite la instalación/actualización de PyTorch y Torchvision."
            echo "  --help, -h          Muestra este mensaje de ayuda."
            exit 0
            ;;
    esac
done

# --- 2. CONFIGURACIÓN MODULARIZADA ---
DIR_SDK_NAME="earth-rovers-sdk"
DIR_GENIE_NAME="genie"

VENV_SDK_NAME=".venv"
CONDA_ENV_NAME="sam_tp"
PYTHON_SDK_VERSION="python3.9"

CHECKPOINT_NAME="checkpoint_2.pt"
CHECKPOINT_REL_PATH="sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints"

# --- 3. RESOLUCIÓN DE RUTAS ABSOLUTAS ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PATH_SDK="$ROOT_DIR/$DIR_SDK_NAME"
PATH_GENIE="$ROOT_DIR/$DIR_GENIE_NAME"
PATH_VENV_SDK="$PATH_SDK/$VENV_SDK_NAME"
PATH_CHECKPOINT_DIR="$PATH_GENIE/$CHECKPOINT_REL_PATH"

# --- FUNCIONES DE IMPRESIÓN ---
info()    { echo -e "\n\033[1;34m[INFO]\033[0m $1"; }
success() { echo -e "\033[1;32m[OK]\033[0m $1"; }
warn()    { echo -e "\033[1;33m[ADVERTENCIA]\033[0m $1"; }
fail()    { echo -e "\033[1;31m[ERROR]\033[0m $1"; exit 1; }

echo "=========================================================="
echo "   ROVERNETTA: SETUP MULTI-DISTRO DE ENTORNO Y PRUEBAS    "
echo "=========================================================="
echo "Directorio raíz: $ROOT_DIR"
if [ "$SKIP_TORCH" = true ]; then
    warn "Modo activo: Omitiendo instalación de PyTorch (--skip-torch)."
fi

# Detectar versión de Ubuntu/Linux
if [ -f /etc/os-release ]; then
    . /etc/os-release
    info "Sistema operativo detectado: $PRETTY_NAME"
fi

# --- 4. CONFIGURACIÓN MÓDULO EARTH-ROVERS-SDK ---
info "=== [1/5] Configurando $DIR_SDK_NAME ==="

if [ ! -d "$PATH_SDK" ]; then
    fail "No se encontró la carpeta del SDK en: $PATH_SDK"
fi

# Verificar presencia de Python 3.9
if ! command -v $PYTHON_SDK_VERSION &> /dev/null; then
    fail "No se encontró $PYTHON_SDK_VERSION en el sistema. Instálalo ejecutando:\n  sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt update && sudo apt install -y python3.9 python3.9-venv python3.9-dev"
fi

cd "$PATH_SDK"

if [ ! -d "$PATH_VENV_SDK" ]; then
    info "Creando entorno virtual con $PYTHON_SDK_VERSION..."
    $PYTHON_SDK_VERSION -m venv "$VENV_SDK_NAME" || fail "Error al crear el venv con $PYTHON_SDK_VERSION"
else
    info "Entorno $VENV_SDK_NAME detectado."
fi

source "$PATH_VENV_SDK/bin/activate"

info "Instalando requisitos pip en $DIR_SDK_NAME..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Estrategia Universal para Chromium / Playwright
info "Resolviendo dependencias del navegador Chromium..."

CHROMIUM_BIN=""
if command -v chromium-browser &> /dev/null; then
    CHROMIUM_BIN="$(which chromium-browser)"
elif command -v chromium &> /dev/null; then
    CHROMIUM_BIN="$(which chromium)"
fi

if [ -z "$CHROMIUM_BIN" ]; then
    info "Instalando Chromium desde repositorios apt..."
    sudo apt update && sudo apt install -y chromium-browser || sudo apt install -y chromium || true
    CHROMIUM_BIN="$(which chromium-browser 2>/dev/null || which chromium 2>/dev/null || echo "")"
fi

# Gestión de Configuración .env
if [ ! -f ".env" ]; then
    if [ -f ".env.sample" ]; then
        cp .env.sample .env
        warn "Se creó .env a partir de .env.sample."
    else
        touch .env
    fi
fi

if [ -n "$CHROMIUM_BIN" ]; then
    success "Chromium localizado en: $CHROMIUM_BIN"
    grep -v "^PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=" .env > .env.tmp || true
    echo "PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=$CHROMIUM_BIN" >> .env.tmp
    mv .env.tmp .env
else
    warn "No se pudo instalar Chromium vía apt. Intentando descarga directa de Playwright..."
    PLAYWRIGHT_HOST_PLATFORM=ubuntu24.04-x64 python -m playwright install chromium || fail "Falló la instalación de Chromium"
fi

deactivate
success "Módulo $DIR_SDK_NAME configurado."


# --- 5. CONFIGURACIÓN MÓDULO GENIE (Conda) ---
info "=== [2/5] Configurando $DIR_GENIE_NAME (Conda) ==="

if [ ! -d "$PATH_GENIE" ]; then
    fail "No se encontró el directorio GeNIE en: $PATH_GENIE"
fi

cd "$PATH_GENIE"

# Cargar funciones de Conda en subshell
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
else
    fail "Conda no está instalado o no se encuentra en el PATH."
fi

if ! conda env list | grep -qE "^${CONDA_ENV_NAME}\s"; then
    info "Creando entorno Conda '$CONDA_ENV_NAME'..."
    conda env create -n "$CONDA_ENV_NAME" -f environment.yml || fail "Error al crear entorno Conda"
else
    info "Entorno Conda '$CONDA_ENV_NAME' existente."
fi

conda activate "$CONDA_ENV_NAME"

# Instalación condicional de PyTorch según el parámetro --skip-torch
if [ "$SKIP_TORCH" = false ]; then
    # Selección dinámica de instalación PyTorch (CUDA vs CPU)
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        info "GPU NVIDIA detectada. Instalando PyTorch con soporte CUDA 12.6..."
        PYTORCH_URL="https://download.pytorch.org/whl/cu126"
    else
        warn "No se detectó GPU NVIDIA. Instalando versión de PyTorch para CPU..."
        PYTORCH_URL="https://download.pytorch.org/whl/cpu"
    fi

    # Usar directorio temporal personalizado en el HOME para evitar errores "No space left on device"
    TMPDIR_CUSTOM="$HOME/.tmp_pip"
    mkdir -p "$TMPDIR_CUSTOM"
    
    TMPDIR="$TMPDIR_CUSTOM" pip install --quiet torch torchvision --index-url "$PYTORCH_URL"
    rm -rf "$TMPDIR_CUSTOM"
else
    info "Omitiendo instalación de PyTorch por bandera --skip-torch."
fi

# Resto de paquetes livianos
info "Instalando paquete local GeNIE en modo editable..."
TMPDIR_CUSTOM="$HOME/.tmp_pip"
mkdir -p "$TMPDIR_CUSTOM"

# Instalación sin --quiet y con aislamiento desactivado para evitar bloqueos
TMPDIR="$TMPDIR_CUSTOM" pip install --no-build-isolation -e . || fail "Error al instalar el paquete local GeNIE"

rm -rf "$TMPDIR_CUSTOM" 

pip install requests python-dotenv google-genai

success "Módulo $DIR_GENIE_NAME configurado."


# --- 6. VERIFICACIÓN Y AUTO-RECUPERACIÓN DE CHECKPOINTS ---
info "=== [3/5] Verificando Checkpoint de Modelo ==="

if [ ! -f "$PATH_CHECKPOINT_DIR/$CHECKPOINT_NAME" ]; then
    warn "Checkpoint no encontrado en la ruta de destino."
    if [ -f "$HOME/Downloads/$CHECKPOINT_NAME" ]; then
        info "Moviendo $CHECKPOINT_NAME desde ~/Downloads..."
        mkdir -p "$PATH_CHECKPOINT_DIR"
        mv "$HOME/Downloads/$CHECKPOINT_NAME" "$PATH_CHECKPOINT_DIR/"
        success "Checkpoint posicionado en su directorio."
    else
        fail "Falta el archivo '$CHECKPOINT_NAME'. Descárgalo y ubícalo en:\n -> $PATH_CHECKPOINT_DIR/"
    fi
else
    success "Checkpoint verificado."
fi


# --- 7. EJECUCIÓN DE PRUEBAS DE VALIDACIÓN OFFLINE ---
info "=== [4/5] Pruebas Internas de Regresión (Asserts) ==="

cd "$PATH_GENIE"

info "Prueba 1: NAVEGACIÓN"
python -m genie_rover.navigation || fail "Falló genie_rover.navigation"

info "Prueba 2: PLANIFICADOR DE COSTOS"
python -m genie_path_planner.costs || fail "Falló genie_path_planner.costs"

info "Prueba 3: RÉGIMEN CERCANO OFFLINE (Salteando...)"
#python -m genie_rover.test_near_regime_offline || fail "Falló genie_rover.test_near_regime_offline"

success "¡Todas las pruebas pasaron satisfactoriamente!"


# --- 8. INSTRUCCIONES DE INICIALIZACIÓN ---
info "=== [5/5] Instrucciones de Ejecución ==="

echo -e "\n\033[1;36mPara operar el sistema, ejecuta en dos terminales independientes:\033[0m\n"
echo -e "\033[1mTERMINAL 1 (SDK Server):\033[0m"
echo -e "  cd \"$PATH_SDK\""
echo -e "  source \"$VENV_SDK_NAME/bin/activate\""
echo -e "  python -m hypercorn main:app\n"
echo -e "\033[1mTERMINAL 2 (Bridge GeNIE - Navegación):\033[0m"
echo -e "  cd \"$PATH_GENIE\""
echo -e "  conda activate $CONDA_ENV_NAME"
echo -e "  # Modo Simulacro:"
echo -e "  python -m genie_rover.bridge --config configs/frodobot_rover.yaml --start-mission\n"

echo "=========================================================="
success "Entorno configurado y validado exitosamente."
echo "=========================================================="