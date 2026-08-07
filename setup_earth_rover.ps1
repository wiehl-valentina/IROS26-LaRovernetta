<#
.SYNOPSIS
    Instala y configura los dos entornos del proyecto Earth Rover Challenge:
    - genie  (planificador GeNIE / SAM-TP, necesita PyTorch + GPU)
    - sdk    (earth-rovers-sdk, controla el robot vía FastAPI + Chrome headless)

.DESCRIPTION
    Crea un .venv separado dentro de cada carpeta (no comparte dependencias 
    entre los dos, evita conflictos de opencv-python vs opencv-python-headless).

.USAGE
    Parado en la carpeta raíz "Earth Rover" que contiene tanto "genie\" como 
    los archivos del SDK en la raíz:

        powershell -ExecutionPolicy Bypass -File .\setup_earth_rover.ps1

    Si tus rutas son distintas, pasalas por parámetro:

        .\setup_earth_rover.ps1 -GeniePath "C:\ruta\a\genie" -SdkPath "C:\ruta\a\sdk"

    Para saltear la instalación de PyTorch:

        .\setup_earth_rover.ps1 -SkipTorch
#>

param(
    [string]$GeniePath = ".\genie",
    [string]$SdkPath   = ".",
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Write-Warn($msg) {
    Write-Host "!!  $msg" -ForegroundColor Yellow
}

function Write-Ok($msg) {
    Write-Host "OK  $msg" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 0. Chequeos previos
# ---------------------------------------------------------------------------
Write-Step "Chequeando rutas"

if (-not (Test-Path $GeniePath)) {
    throw "No encuentro la carpeta genie en '$GeniePath'. Pasala con -GeniePath 'ruta\completa'."
}
if (-not (Test-Path $SdkPath)) {
    throw "No encuentro la carpeta sdk/raíz en '$SdkPath'. Pasala con -SdkPath 'ruta\completa'."
}

$GeniePath = (Resolve-Path $GeniePath).Path
$SdkPath   = (Resolve-Path $SdkPath).Path

Write-Ok "genie -> $GeniePath"
Write-Ok "sdk   -> $SdkPath"

Write-Step "Chequeando GPU NVIDIA"
try {
    nvidia-smi | Out-Null
    Write-Ok "nvidia-smi responde, hay driver instalado"
} catch {
    Write-Warn "nvidia-smi no funciona. Instalá/actualizá el driver de NVIDIA antes de seguir."
}

# ---------------------------------------------------------------------------
# 1. Entorno de genie (sam_tp)
# ---------------------------------------------------------------------------
Write-Step "Configurando entorno de genie"

Push-Location $GeniePath

if (-not (Test-Path ".\.venv")) {
    Write-Host "Creando .venv en genie..."
    python -m venv .venv
} else {
    Write-Ok ".venv de genie ya existe, lo reuso"
}

$genieVenvPython = ".\.venv\Scripts\python.exe"
& $genieVenvPython -m pip install --upgrade pip | Out-Null

if (Test-Path ".\requirements.txt") {
    Write-Host "Instalando requirements.txt de genie..."
    & $genieVenvPython -m pip install -r requirements.txt
} else {
    Write-Warn "No encontré requirements.txt en genie, sigo igual."
}

if (-not $SkipTorch) {
    Write-Host "Instalando PyTorch (cu126)..."
    & $genieVenvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
} else {
    Write-Warn "Salteando instalación de PyTorch (-SkipTorch)."
}

if (Test-Path ".\pyproject.toml") {
    Write-Host "Instalando genie en modo editable (pip install -e .)..."
    & $genieVenvPython -m pip install -e .
}

Write-Host "Instalando dependencias del puente (requests, pyyaml)..."
& $genieVenvPython -m pip install requests pyyaml

Write-Step "Verificando PyTorch + GPU"
& $genieVenvPython -c "import torch; print('torch:', torch.__version__, '| build CUDA:', torch.version.cuda); print('GPU visible:', torch.cuda.is_available())" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warn "No pude verificar torch. Si falló la instalación, revisá manualmente."
}

Pop-Location
Write-Ok "Entorno de genie listo (.venv dentro de $GeniePath)"

# ---------------------------------------------------------------------------
# 2. Entorno de sdk (raíz del proyecto)
# ---------------------------------------------------------------------------
Write-Step "Configurando entorno de sdk (raíz)"

Push-Location $SdkPath

if (-not (Test-Path ".\.venv")) {
    Write-Host "Creando .venv en sdk..."
    python -m venv .venv
} else {
    Write-Ok ".venv de sdk ya existe, lo reuso"
}

$sdkVenvPython = ".\.venv\Scripts\python.exe"
& $sdkVenvPython -m pip install --upgrade pip | Out-Null

if (Test-Path ".\requirements.txt") {
    Write-Host "Instalando requirements.txt de sdk..."
    & $sdkVenvPython -m pip install -r requirements.txt
} else {
    Write-Warn "No encontré requirements.txt en sdk, sigo igual."
}

Write-Host "Instalando google-genai..."
& $sdkVenvPython -m pip install google-genai

# --- .env ---
Write-Step "Configurando .env"
if (Test-Path ".env") {
    Write-Ok ".env ya existe, no lo toco."
} elseif (Test-Path ".env.sample") {
    Copy-Item ".env.sample" ".env"
    Write-Ok "Creé .env a partir de .env.sample. Faltan completar los valores reales:"
    Write-Host "    - SDK_API_TOKEN"
    Write-Host "    - BOT_SLUG"
    Write-Host "    - CHROME_EXECUTABLE_PATH"
} else {
    Write-Warn "No encontré .env.sample, no pude generar el .env."
}

# --- Chrome ---
Write-Step "Chequeando Google Chrome"
$chromePaths = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LocalAppData\Google\Chrome\Application\chrome.exe"
)
$chromeFound = $chromePaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($chromeFound) {
    Write-Ok "Chrome encontrado en: $chromeFound"
} else {
    Write-Warn "No encontré Chrome instalado. Instalalo desde https://www.google.com/chrome/"
}

Pop-Location
Write-Ok "Entorno de sdk listo (.venv dentro de $SdkPath)"

# ---------------------------------------------------------------------------
# 3. Resumen
# ---------------------------------------------------------------------------
Write-Step "Resumen"
Write-Host "genie: $GeniePath\.venv"
Write-Host "sdk:   $SdkPath\.venv"
Write-Host ""
Write-Host "Para arrancar (en dos terminales separadas):"
Write-Host "  Terminal 1:  cd `"$SdkPath`"   ; .venv\Scripts\Activate.ps1 ; hypercorn main:app --reload"
Write-Host "  Terminal 2:  cd `"$GeniePath`" ; .venv\Scripts\Activate.ps1 ; python -m genie_rover.sdk_client"