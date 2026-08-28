# Instalación desde cero — Ubuntu + RTX 2080

Para una máquina recién formateada. Hardware de referencia: Ryzen 7 3700X,
RTX 2080 (Turing, SM 7.5, 8 GB), 24 GB RAM.

Reservá ~15 GB de disco. Todo el proceso lleva entre 40 y 90 minutos, casi todo
esperando descargas.

Vas a terminar con **dos entornos conda separados** (`sam_tp` y `erc_sdk`) que
corren en **dos terminales distintas** y se hablan por HTTP.

---

## Fase 0 — Sistema base

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget unzip ca-certificates
lsb_release -a
```

Anotá tu versión de Ubuntu por si algo falla después.

---

## Fase 1 — Driver de NVIDIA

**No instales el CUDA Toolkit.** Es el error más común de esta fase. Las ruedas
de PyTorch traen su propio runtime de CUDA adentro; lo único que necesitás del
sistema es el driver. Instalar el toolkit completo agrega 3 GB y una fuente
extra de conflictos de versiones, sin ningún beneficio.

```bash
ubuntu-drivers devices
sudo ubuntu-drivers install        # si tu Ubuntu es viejo: sudo ubuntu-drivers autoinstall
sudo reboot
```

Después del reboot:

```bash
nvidia-smi
```

Tenés que ver `NVIDIA GeForce RTX 2080` y una tabla con la memoria. El campo
`CUDA Version` que muestra arriba a la derecha es **la versión máxima que
soporta el driver**, no algo que tengas instalado. Ignoralo.

> **Si `nvidia-smi` no existe después del reboot** y tenés Secure Boot activado
> en la BIOS: Ubuntu te pidió durante la instalación una contraseña MOK para
> firmar el módulo del kernel. Si te la salteaste, el driver está instalado pero
> el kernel no lo carga. Lo más simple es desactivar Secure Boot en la BIOS y
> reiniciar.

---

## Fase 2 — Google Chrome

El SDK del rover no habla con el robot por un protocolo de robótica: levanta un
Chrome headless que se conecta al canal de video Agora. Sin Chrome no hay video.

```bash
cd /tmp
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install -y ./google-chrome-stable_current_amd64.deb
which google-chrome
```

Guardá la ruta que imprime (normalmente `/usr/bin/google-chrome`), la vas a
necesitar en la Fase 8.

---

## Fase 3 — Miniconda

```bash
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Aceptá la licencia, dejá la ruta por defecto (`~/miniconda3`) y respondé **yes**
a la pregunta de inicializar conda. Después cerrá y volvé a abrir la terminal:

```bash
conda --version
conda config --set auto_activate_base false   # opcional, para que no arranque en (base)
```

> Si preferís evitar los términos de licencia de Anaconda, Miniforge sirve
> igual: el `environment.yml` de GENIE ya usa el canal `conda-forge`
> exclusivamente.

---

## Fase 4 — Los dos repos

```bash
mkdir -p ~/erc && cd ~/erc
unzip ~/Downloads/GENIE-SAMTP-master.zip
unzip ~/Downloads/IROS26_ERC_UNLP_Equipo1-main.zip
mv GENIE-SAMTP-master genie
mv IROS26_ERC_UNLP_Equipo1-main sdk
ls
```

Deberías ver `genie` y `sdk`.

---

## Fase 5 — Entorno de GeNIE

```bash
cd ~/erc/genie
conda env create -n sam_tp -f environment.yml
conda activate sam_tp
```

Esto tarda varios minutos. Ahora PyTorch, que **no** está en el
`environment.yml` a propósito:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> Andá a <https://pytorch.org/get-started/locally/> y usá el selector (Linux,
> Pip, Python, la CUDA estable que ofrezcan) por si el número cambió. Turing está
> soportado por todas las builds actuales, así que cualquier `cu12x` estable
> sirve. **No uses el nightly `cu128` que sugiere el README de GENIE**: ese
> comando apunta a hardware Blackwell y acá solo agrega riesgo.

Instalá el paquete y las dependencias del puente:

```bash
pip install -e .
pip install requests pyyaml
```

### Verificación obligatoria

```bash
python - <<'EOF'
import torch
print("torch:", torch.__version__, "| build CUDA:", torch.version.cuda)
print("GPU visible:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name} | SM {p.major}.{p.minor} | {p.total_memory/2**30:.1f} GB")
    print("bfloat16 nativo:", p.major >= 8)
EOF
```

Salida esperada en tu máquina:

```
GPU visible: True
NVIDIA GeForce RTX 2080 | SM 7.5 | 7.8 GB
bfloat16 nativo: False
```

Ese `False` es justamente por lo que `perception.py` elige fp16 en tu GPU. Si
dice `GPU visible: False`, no sigas: revisá `nvidia-smi` y reinstalá torch.

---

## Fase 6 — El checkpoint de SAM-TP

No viene en el repo y sin él no hay percepción.

```bash
conda activate sam_tp
pip install gdown
cd ~/erc/genie

mkdir -p "sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints"

gdown --folder "https://drive.google.com/drive/folders/190yHH-TcfQVoByZeB1809sPIR62CsBD1" -O /tmp/samtp
find /tmp/samtp -name "*.pt" -exec ls -lh {} \;
```

Copiá el `checkpoint_2.pt` que encuentre:

```bash
cp /tmp/samtp/**/checkpoint_2.pt \
   "sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/"

ls -lh "sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/"
```

Dos trampas acá:

1. **El directorio se llama literalmente `sam2_training_custom2_freezeNoneNone_f57.yaml`.**
   No es un error de tipeo ni un archivo: es una carpeta cuyo nombre termina en
   `.yaml`. Las comillas en los comandos de arriba son necesarias.
2. **Si el archivo pesa unos pocos KB, no es el modelo.** Cuando Drive tira
   límite de cuota, `gdown` guarda la página HTML de error con el nombre del
   archivo. Abrí el link en el navegador y bajalo a mano.

---

## Fase 7 — Los archivos del puente

Copiá al repo `genie` (no al `sdk`):

```bash
cd ~/erc/genie
cp -r ~/Downloads/genie_rover .
cp -r ~/Downloads/tools .
mkdir -p configs && cp ~/Downloads/frodobot_rover.yaml configs/

# la parte matemática no necesita ni GPU ni robot: probala ya
python genie_rover/navigation.py
```

Tiene que terminar con `Todos los asserts pasaron.`

---

## Fase 8 — Entorno del SDK

Entorno aparte porque el SDK pinnea `numpy==1.26.4` y versiones exactas de
FastAPI, y pyppeteer no se lleva bien con Python moderno.

```bash
conda create -n erc_sdk python=3.10 -y
conda activate erc_sdk
cd ~/erc/sdk
pip install -r requirements.txt
pip install google-genai
```

El `pip install google-genai` va aparte porque la última línea del
`requirements.txt` quedó pegada a un comentario
(`#pip install -r requirements.txtgoogle-genai>=1.0.0`) y pip la ignora.

### Sacá la clave filtrada

Antes de tocar nada más:

```bash
grep -n "sk-or-v1" programs/control_nube.py
```

Revocá esa clave en el panel de OpenRouter y editá el archivo para que lea
`os.getenv("OPENROUTER_API_KEY")`. Está en el historial de git, así que revocarla
es lo único que realmente la desactiva.

### Configurá el .env

```bash
cp .env.sample .env
nano .env
```

```bash
SDK_API_TOKEN="tu_token_de_my.frodobots.com"
BOT_SLUG="el_slug_de_tu_rover"
CHROME_EXECUTABLE_PATH="/usr/bin/google-chrome"
MAP_ZOOM_LEVEL=18
MISSION_SLUG=""              # dejalo vacío hasta que quieras correr una misión real
IMAGE_QUALITY=0.8
IMAGE_FORMAT=jpeg
TTS_PROVIDER=edge
```

`IMAGE_FORMAT` y `IMAGE_QUALITY` fijan la resolución y el formato de los frames.
Si los cambiás después de calibrar la cámara, la calibración deja de valer.

---

## Fase 9 — Arrancar el SDK

**Terminal 1**, y dejala abierta:

```bash
conda activate erc_sdk
cd ~/erc/sdk
hypercorn main:app --reload
```

Abrí <http://localhost:8000> y confirmá que se ve el video del rover. Si no se
ve, no sigas: todo lo demás depende de eso.

---

## Fase 10 — Cadena de verificación

**Terminal 2**, en paralelo:

```bash
conda activate sam_tp
cd ~/erc/genie
```

Cada paso valida una cosa. No te saltees ninguno.

```bash
# 1. ¿Habla el SDK? Anotá la resolución del frame. No mueve el robot.
python -m genie_rover.sdk_client

# 2. ¿Carga el modelo y segmenta? Todavía con calibración del Stretch,
#    así que el BEV va a salir mal: solo mirá que el suelo salga rojo.
#    Te imprime también los ms por frame.
python -m genie_rover.perception --config configs/frodobot_rover.yaml \
    --image screenshots/imagen.jpg --out debug/

# 3. Calibrá la cámara (tablero de ajedrez impreso, ~20 capturas)
python tools/calibrate_camera.py capture --out calib_frames/
python tools/calibrate_camera.py solve --frames calib_frames/ \
    --square-size 0.025 --out calib.yaml
#    Pegá image_size, intrinsics y dist_coeffs en configs/frodobot_rover.yaml

# 4. Medí con una regla height_m y pitch_down_deg, poné calibrated: true,
#    y repetí el paso 2. Verificación: una caja a 2 m reales tiene que
#    aparecer a 2 m en el BEV.

# 5. ¿Angular positivo es izquierda o derecha? El robot gira 2 segundos.
python tools/check_angular_sign.py

# 6. Simulacro: calcula todo, imprime los comandos, NO los envía.
python -m genie_rover.bridge --config configs/frodobot_rover.yaml

# 7. Primera corrida real. Al aire libre, con espacio, mano en Ctrl-C.
python -m genie_rover.bridge --config configs/frodobot_rover.yaml \
    --go --max-seconds 30 --debug-dir debug/run1
```

Para el paso 7, bajá antes `max_linear` a `0.15` en el config. Subilo cuando
confíes en lo que ves.

---

## Cómo queda todo

```
~/erc/
├── genie/                          conda activate sam_tp
│   ├── sam2/                       SAM2 + SAM-TP
│   ├── genie_path_planner/         proyección BEV + path fusion
│   ├── genie_rover/                ← el puente
│   ├── tools/                      ← calibración y test de signo
│   ├── configs/frodobot_rover.yaml ← tu config
│   └── sam2_logs/.../checkpoint_2.pt
└── sdk/                            conda activate erc_sdk
    ├── main.py                     FastAPI :8000
    ├── .env                        tokens
    └── programs/                   tus prototipos con Gemini
```

---

## Si algo falla en la instalación

| Síntoma | Causa |
|---|---|
| `nvidia-smi: command not found` tras reboot | driver sin instalar, o Secure Boot bloqueando el módulo |
| `torch.cuda.is_available()` da False | falta reiniciar, o instalaste la rueda CPU (fijate `torch.version.cuda`) |
| `conda env create` se cuelga resolviendo | probá `conda config --set solver libmamba` |
| El checkpoint pesa 2 KB | `gdown` guardó el HTML del error de cuota; bajalo del navegador |
| `Could not find browser executable` al arrancar el SDK | `CHROME_EXECUTABLE_PATH` mal en el `.env` |
| El SDK levanta pero no hay video | `SDK_API_TOKEN` o `BOT_SLUG` mal, o el rover lo está usando otro |
| `ModuleNotFoundError: sam2` | te olvidaste el `pip install -e .`, o estás en el entorno equivocado |
| `ModuleNotFoundError: genie_path_planner` | estás corriendo el bridge desde otro directorio: tiene que ser `~/erc/genie` |
| Warnings compilando `connected_components.cu` | ignoralos, esa extensión CUDA es opcional |
