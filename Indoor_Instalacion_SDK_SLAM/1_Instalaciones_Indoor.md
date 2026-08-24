# Instalación — stack Indoor / SLAM (La Rovernetta)

Qué instalar, en qué entorno, y en qué máquina/OS. Este proyecto usa **tres
entornos separados a propósito** (no es un descuido, es para evitar
conflictos de dependencias) más **ROS2 a nivel sistema**. Esta guía te dice
qué va en cada uno.

---

## 0. Panorama — qué entorno hace qué

| Entorno | Tipo | Para qué | Por qué separado |
|---|---|---|---|
| **`sdk`** | venv (`python -m venv`) | Earth Rovers SDK: FastAPI/hypercorn + Chrome headless, habla con el rover real | pinnea `numpy==1.26.4`, necesita Chrome/Playwright |
| **`genie` / `sam_tp`** | conda | GeNIE / SAM-TP: percepción (segmentación de piso transitable) + planner BEV | necesita PyTorch + CUDA + hydra, versiones que chocan con el SDK |
| **ROS2 (Humble)** | instalación de sistema (no venv/conda) | `earth_rover_bridge.py`, RTAB-Map (SLAM), `rover_launch.sh mapping-ros2` / `indoor-bridge` | ROS2 se instala a nivel del sistema operativo, no es un paquete de pip |

Los tres se hablan por **HTTP/ROS2 topics en `localhost`**, no importan
código entre sí — por eso pueden vivir en entornos totalmente distintos sin
problema, siempre que corran en la **misma máquina** (o en la misma red, si
apuntás `sdk_url` a otra IP).

**Dónde correr todo esto:**

- **Linux nativo** (Ubuntu 22.04 recomendado, por Humble): la opción más
  simple, todo instala directo.
- **Windows**: no instalar ROS2 nativo en Windows — usar **WSL2** con Ubuntu
  22.04 adentro (ver sección 1). GeNIE/SAM-TP y el SDK también van dentro de
  esa misma WSL2, no en Windows directo — así todo habla por `localhost` sin
  configurar networking cruzado.
- **macOS**: ROS2 nativo tampoco es práctico. Para el bridge solo (sin
  SLAM/GeNIE) hay una imagen Docker provista (ver sección 4.4). Para el
  stack completo con RTAB-Map, lo más simple es una VM Linux o un Linux
  real aparte.

---

## 1. Windows: WSL2 primero (saltear si ya estás en Linux)

Requiere Windows 11, o Windows 10 build 19041+ (`winver` para chequear).

PowerShell **como administrador**:

```powershell
wsl --install -d Ubuntu-22.04
```

Reiniciar si lo pide. Al abrir Ubuntu por primera vez, creá un usuario y
contraseña de Linux (independiente del de Windows).

**GPU NVIDIA (si tenés):** no instalar drivers de Linux dentro de WSL — solo
actualizar el driver de **Windows** (Game Ready o Studio, el último). WSL2
pasa la GPU automáticamente. Adentro de Ubuntu, verificar:

```bash
nvidia-smi
```

Si no aparece la GPU, el driver de Windows está desactualizado. Sin GPU
NVIDIA el stack igual funciona en CPU (más lento — ver tabla en sección 3).

**Ventanas gráficas** (rviz2, `rtabmap-databaseViewer`, etc.): en Windows 11
WSLg viene integrado y abre ventanas Linux como ventanas normales de
Windows, sin config extra. Si en algún momento dejan de abrir:

```powershell
# desde PowerShell, no desde Ubuntu
wsl --update
wsl --shutdown
```

y volvé a abrir Ubuntu. Probar con `sudo apt install -y x11-apps && xeyes`
para confirmar que las ventanas salen antes de pelear con algo más
complicado.

**Clonar el repo DENTRO del filesystem de Linux**, no en `/mnt/c/...`
(mucho más lento, y conda/pip a veces fallan por eso):

```bash
cd ~
git clone <url-del-repo>
```

De acá en adelante, todos los comandos de esta guía corren **dentro de esa
Ubuntu de WSL2** (o dentro de tu Linux nativo si no usás Windows).

---

## 2. ROS2 Humble (instalación de sistema)

Necesario para: `earth_rover_bridge.py`, RTAB-Map, `rover_launch.sh
mapping-ros2` / `indoor-bridge`, y para ver/editar mapas con
`rtabmap-databaseViewer`.

```bash
# locale (si no lo tenés ya en UTF-8)
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# repositorio de ROS2
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop   # incluye rviz2, rqt, etc.
```

**Paquetes específicos para este proyecto** (bridge + SLAM + teleop):

```bash
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-tf2-ros \
    ros-humble-rtabmap-ros \
    ros-humble-teleop-twist-keyboard \
    python3-colcon-common-extensions
```

**Sourcear ROS2 en cada terminal nueva** (o dejarlo en `~/.bashrc`):

```bash
echo 'export AMENT_TRACE_SETUP_FILES=0' >> ~/.bashrc   # evita un bug conocido con set -u
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

(el `AMENT_TRACE_SETUP_FILES=0` es porque `rover_launch.sh` corre en modo
estricto `set -u`, y sin esa variable definida el `source` del setup de
ROS2 tira `unbound variable`.)

**Verificar:**

```bash
ros2 doctor --report | head -20
rtabmap-databaseViewer --help   # si no existe con ese nombre exacto:
# dpkg -L ros-humble-rtabmap-ros | grep -i viewer
```

---

## 3. Entorno `genie` / `sam_tp` (GeNIE / SAM-TP, percepción + planner)

Necesita **Miniconda** (no lo instales en Windows si estás en WSL2, instalalo
DENTRO de la WSL):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# reabrí la terminal o `source ~/.bashrc` al terminar
```

**¿GPU o CPU?**

| | tiempo por frame | qué implica |
|---|---|---|
| GPU NVIDIA (≥6 GB) | ~30–60 ms | el modelo va a ~10 Hz; el cuello de botella es el video del rover (3–5 Hz) |
| Solo CPU | ~1–4 s | usable pero hay que bajar `max_linear` a ~0.15 y aceptar que el robot avanza a tramos |

**Instalación** (dentro de la carpeta `GENIE-SAMTP`, donde están `sam2/`,
`genie_path_planner/`, `genie_rover/`, `tools/`, `configs/`):

```bash
cd ~/GENIE-SAMTP   # o donde hayas clonado ese repo
conda env create -n sam_tp -f environment.yml
conda activate sam_tp

# torch acorde a tu CUDA — ver https://pytorch.org/get-started/locally/
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# sin GPU:
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -e .
pip install requests pyyaml opencv-python
```

**El checkpoint del modelo** — no viene en el repo, hay que bajarlo del
Google Drive que linkea el README de GENIE y ponerlo en la ruta exacta (el
directorio se llama literalmente `...yaml`, no es un error de tipeo):

```bash
mkdir -p "sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints"
mv ~/Downloads/checkpoint_2.pt \
   "sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/"
```

**Verificar** (con `sam_tp` activado):

```bash
python -m genie_rover.perception --config configs/frodobot_rover.yaml \
    --image alguna_foto.jpg --out debug/
```

Si carga sin error y genera algo en `debug/`, el checkpoint y el entorno
están bien.

---

## 4. Entorno `sdk` (Earth Rovers SDK — controla el robot)

Venv normal, **no conda**:

```bash
cd ~/IROS26-LaRovernetta/earth-rovers-sdk   # o donde esté tu SDK
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install google-genai        # la última línea de requirements.txt suele venir rota
playwright install chromium     # o: python -m playwright install chromium en Windows
cp .env.sample .env
```

**Completar `.env`:**

- `SDK_API_TOKEN`: token de autenticación del SDK.
- `BOT_SLUG`: identificador del rover.
- `CHROME_EXECUTABLE_PATH`: ruta al Chrome/Chromium instalado.
- `MISSION_SLUG`: opcional, para la API de misiones en bots remotos.
- `TTS_PROVIDER`, `TTS_API_KEY`, `TTS_VOICE`: opcional (voz).

**Verificar:**

```bash
hypercorn main:app
```

Abrí `http://localhost:8000` en el navegador y confirmá que ves el video
del rover antes de seguir con lo demás.

### 4.1 Dependencias del bridge ROS2 (`earth_rover_bridge.py`)

El bridge **no** vive en el venv de `sdk` ni en el conda de `genie` — corre
en la terminal donde ya sourceaste ROS2 (paso 2), porque necesita `rclpy` y
`cv_bridge`, que vienen de la instalación de sistema de ROS2, no de pip.

En esa misma terminal (ROS2 sourceado):

```bash
pip install requests websocket-client opencv-python
```

Si usás un venv dedicado para el bridge, asegurate de que herede los
paquetes de ROS2 (`--system-site-packages` al crearlo), o simplemente usá
el Python de sistema.

### 4.2 Sin GPU/GENIE — orquestador alternativo

Si por ahora solo necesitás navegación reactiva sin todo el stack de GeNIE
(sin conda, sin checkpoint de 130 MB), existe `rover_traversability` como
alternativa más liviana:

```bash
pip install -e './traversability[dev]'
pip install -e './traversability[hf]'   # si querés descarga automática del checkpoint desde HuggingFace
```

No sustituye a `genie_rover.bridge`/`indoor_bridge.py` para el flujo SLAM
completo, pero sirve para levantar navegación rápido mientras instalás el
resto.

### 4.3 Instalación automática (Windows, alternativa)

Si tu setup es Windows con `.venv` propios para `genie` y `sdk` (sin WSL2 —
ver más abajo por qué no es lo recomendado para SLAM), el repo trae un
script que automatiza los pasos 3 y 4:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_earth_rover.ps1
# opcional: -GeniePath, -SdkPath, -SkipTorch
```

⚠️ Esto NO instala ROS2 ni RTAB-Map — ROS2 nativo en Windows no es viable,
así que para todo lo de SLAM/mapeo (secciones 2 y las de este README sobre
mapeo) necesitás sí o sí WSL2, aunque uses este script para `genie`/`sdk`
del lado Windows.

### 4.4 macOS / sin ROS nativo — solo el bridge, vía Docker

Si no necesitás GeNIE/SAM2 y solo querés correr `earth_rover_bridge.py`:

```bash
docker build -t er-ros2 examples/ros2
docker run -it --rm -v "$PWD/examples/ros2:/ws" er-ros2 \
    python3 /ws/earth_rover_bridge.py --ros-args \
    -p sdk_url:=http://host.docker.internal:8000
```

(no uses `--network host` en macOS/Windows — Docker corre en una VM ahí, y
el host networking no llega al `localhost` del host).

Para SLAM/RTAB-Map completo esto no alcanza — armá una VM Linux o un Linux
real aparte.

---

## 5. Submódulos / dependencias del repo (si clonaste desde cero)

```bash
git submodule init
git submodule update
```

---

## 6. Correr todo junto (una vez instalado)

Conviene correr SDK + bridge + GeNIE/SLAM todos **dentro del mismo
entorno/máquina** (misma WSL2 si estás en Windows) para que se hablen por
`localhost` sin configurar networking cruzado.

**Orden de arranque, cada uno en su terminal:**

```bash
# Terminal 1 — entorno `sdk`
cd ~/IROS26-LaRovernetta/earth-rovers-sdk
source .venv/bin/activate
hypercorn main:app

# Terminal 2 — ROS2 sourceado, dependencias del bridge instaladas
source /opt/ros/humble/setup.bash
python3 examples/ros2/earth_rover_bridge.py --ros-args -p sdk_url:=http://localhost:8000

# Terminal 3 — ROS2 sourceado, para SLAM/mapeo
source /opt/ros/humble/setup.bash
./rover_launch.sh mapping-ros2 --db ~/maps/sesion1.db
```

Para el detalle de cómo grabar el mapeo, exportarlo y usarlo con
`indoor_bridge.py` (incluyendo qué hacer si no controlás dónde arranca el
rover), ver el otro documento del proyecto:
**`proceso-mapeo-slam-rtabmap-indoor.md`**.

---

## Checklist rápido de verificación

- [ ] `nvidia-smi` (si tenés GPU) muestra la tarjeta.
- [ ] `ros2 doctor --report` no tira errores graves.
- [ ] `rtabmap-databaseViewer --help` corre.
- [ ] `hypercorn main:app` levanta y `http://localhost:8000` muestra video.
- [ ] `python -m genie_rover.perception --config ... --image ... --out debug/`
      corre sin error de checkpoint faltante.
- [ ] `python3 earth_rover_bridge.py --ros-args -p sdk_url:=http://localhost:8000`
      conecta sin `Connection refused` (con el SDK ya arriba).
- [ ] `xeyes` (o cualquier ventana ROS) se abre visible en pantalla.
