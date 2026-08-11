# Earth Rovers SDK & Genie Rover (La Rovernetta)

Este repositorio contiene el sistema de control, percepción y planificación para los rovers de **Earth Rover Challenge (Frodobots)**.

---

## 1. Instalación y Configuración del Entorno

El proyecto opera con **dos entornos virtuales (`.venv`) independientes** para evitar conflictos de dependencias (por ejemplo, entre `opencv-python` y `opencv-python-headless`):
* **`genie`**: Planificador GeNIE / SAM-TP (requiere PyTorch con soporte GPU CUDA).
* **`sdk`**: Earth Rovers SDK (controla el robot vía FastAPI + Chrome/Chromium headless).

---

### A. Instalación Automática (Recomendada)

Se provee un script en PowerShell (`setup_earth_rover.ps1`) para automatizar la comprobación de GPU, creación de entornos, instalación de PyTorch y configuración inicial.

#### Uso:
Estando en la carpeta raíz del proyecto:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_earth_rover.ps1
```

#### Opciones adicionales:
* **Rutas personalizadas**:
  ```powershell
  .\setup_earth_rover.ps1 -GeniePath "C:\ruta\a\genie" -SdkPath "C:\ruta\a\sdk"
  ```
* **Saltear instalación de PyTorch**:
  ```powershell
  .\setup_earth_rover.ps1 -SkipTorch
  ```

#### ¿Qué realiza el script?
1. **Comprobaciones Previas**:
   - Valida las rutas de `genie` y `sdk`.
   - Verifica la presencia del driver NVIDIA ejecutando `nvidia-smi`.
2. **Entorno `genie`**:
   - Crea/reutiliza `.venv` dentro del directorio `genie`.
   - Instala dependencias de `requirements.txt`.
   - Instala PyTorch con soporte CUDA 12.6 (`cu126`) salvo que se especifique `-SkipTorch`.
   - Instala el paquete local en modo editable (`pip install -e .`) si existe `pyproject.toml`.
   - Instala dependencias del puente (`requests`, `pyyaml`) y verifica la detección de la GPU en PyTorch.
3. **Entorno `sdk`**:
   - Crea/reutiliza `.venv` en la raíz / SDK.
   - Instala `requirements.txt` y `google-genai`.
   - Genera el archivo `.env` a partir de `.env.sample` si no existe.
   - Detecta la presencia de Google Chrome en las rutas estándar del sistema.

---

### B. Instalación Manual y Submódulos Git

Si clonas el repositorio desde cero o faltan submódulos, ejecuta sobre la rama `main`:
```bash
git submodule init
git submodule update
```

Instalación manual de dependencias para el SDK:
```bash
pip3 install -r requirements.txt
playwright install chromium
```
*(En Windows: `python -m playwright install chromium`)*

---

## 2. Configuración (`.env`)

Cuando se genera el archivo `.env`, asegúrate de completar las variables requeridas:
* `SDK_API_TOKEN`: Token de autenticación del SDK.
* `BOT_SLUG`: Identificador único del rover.
* `CHROME_EXECUTABLE_PATH`: Ruta al ejecutable de Chrome.
* `MISSION_SLUG`: (Opcional) Slug de la misión para habilitar la API de Misiones en bots remotos.
* `TTS_PROVIDER`, `TTS_API_KEY`, `TTS_VOICE`: (Opcional) Configuración de síntesis de voz (`edge` o `gemini`).

---

## 3. Comandos de Ejecución y Flujo de Trabajo

### A. Arrancar el Servidor y el Planificador

Abrir **dos terminales separadas**:

* **Terminal 1 (SDK / Servidor)**:
  ```powershell
  cd "ruta\a\sdk"
  .\.venv\Scripts\Activate.ps1
  hypercorn main:app --reload
  ```
  *(Dashboard disponible en `http://localhost:8000`)*

* **Terminal 2 (Genie / Cliente / Puente)**:
  ```powershell
  cd "ruta\a\genie"
  .\.venv\Scripts\Activate.ps1
  python -m genie_rover.sdk_client
  ```

---

### B. Pruebas y Calibración

1. **Validación de Conexión**:
   ```bash
   python -m genie_rover.sdk_client
   ```
   Verifica la comunicación básica, frame de video, señal GPS (`gps_signal > 0`) y checkpoints sin mover el robot.

2. **Prueba de Percepción (SAM-TP)**:
   ```bash
   python -m genie_rover.perception --config configs/frodobot_rover.yaml --image <foto.jpg> --out debug/
   ```
   Procesa una imagen estática para verificar la segmentación del área transitable.

3. **Calibración de Cámara**:
   * Captura de fotogramas:
     ```bash
     python tools/calibrate_camera.py capture --out calib_frames/
     ```
   * Cálculo de intrínsecos y distorsión:
     ```bash
     python tools/calibrate_camera.py solve --frames calib_frames/ --square-size 0.025 --out calib.yaml
     ```

4. **Verificación de Signo Angular (Giro)**:
   ```bash
   python tools/check_angular_sign.py
   ```
   Realiza una prueba de giro de 2s para validar el sentido de la rotación.

---

### C. Ejecución de Misiones / Navegación

* **Modo Simulacro (DRY-RUN / Seguro)**:
  ```bash
  python -m genie_rover.bridge --config configs/frodobot_rover.yaml
  ```
  Imprime comandos en consola sin enviar movimiento al rover.

* **Modo Real (Movimiento Activado)**:
  ```bash
  python -m genie_rover.bridge --config configs/frodobot_rover.yaml --go --max-seconds 300 --debug-dir debug/run5min
  ```
  Envía comandos físicos al rover durante el tiempo indicado. Añadir `--start-mission` para activar la búsqueda autónoma de checkpoints.

---


## 3.B Orquestador Alternativo: `rover_traversability`

Además del puente `genie_rover.bridge`, el repo trae un **segundo
orquestador** más liviano: el paquete `rover_traversability` (carpeta
`traversability/`). Reutiliza el mismo `sam2` vendorizado en `genie/`, pero
en vez de proyectar a BEV y correr el planner completo, decide con una
política de corredores en espacio-imagen — más simple de instalar y de leer.
Se instala **dentro del mismo entorno `genie`**, no necesita un tercer venv:

```bash
# parado en el entorno genie (.venv de genie ya activado)
pip install torch torchvision
pip install --no-build-isolation -e ./genie
pip install -e './traversability[hf]'      # [hf] = descarga automática de pesos desde Hugging Face
```

Ver la sección **3.D** para cómo correrlo y **[docs de `rover_traversability`](traversability/README.md)**
para el detalle completo de parámetros.
## 3.B  Comandos de Ejecución y Flujo de Trabajo `rover_traversability`

`genie_rover.bridge` (sección C) proyecta la máscara a BEV y corre el
planner completo del paper. `rover_traversability` es la alternativa
**más simple**: decide directamente sobre la máscara de SAM-TP en espacio de
imagen, dividiéndola en 9 corredores verticales y manejando hacia el más
transitable (con sesgo opcional hacia el rumbo GPS del checkpoint). Sin
calibración de cámara, sin BEV — pensado para levantar algo andando rápido y
después migrar a `genie_rover.bridge` si hace falta el planner completo.

Se prueba en 4 niveles, de menos a más riesgo:

```bash
# 1. Una imagen suelta, sin rover — genera overlay.png verde/rojo
python -m rover_traversability.demo predict screenshots/imagen.jpg --out overlay.png

# 2. Rover prendido pero SIN enviar comandos — solo guarda overlays en trav_out/
python -m rover_traversability.demo live --save-dir trav_out

# 3. Maneja esquivando obstáculos (sin objetivo GPS) — MUEVE EL ROVER
python -m rover_traversability.demo drive --yes-i-want-the-rover-to-move

# 4. Misión completa: checkpoint por checkpoint usando GPS + percepción — MUEVE EL ROVER
python -m rover_traversability.demo mission --start-mission --yes-i-want-the-rover-to-move
```

`drive` y `mission` se niegan a arrancar sin `--yes-i-want-the-rover-to-move`
(igual filosofía dry-run-por-defecto que el modo simulacro de `bridge`).

También se puede usar sin la CLI, cambiando **una línea** en tu propio loop
de control (en vez de tocar `genie_rover.bridge`):

```python
from rover_traversability import TraversabilityStrategy

strategy = TraversabilityStrategy(drive=True)   # antes: Base64ImageStrategy()
loop = RoverLoop(strategy=strategy, sleep_seconds=0.5, max_iterations=None)
```

Todos los umbrales de manejo (velocidad, sensibilidad de giro, cuándo
frenar) viven en un solo lugar, `PolicyConfig`, y se pueden tunear sin tocar
el resto del código:

```python
from rover_traversability import PolicyConfig, TraversabilityStrategy

cfg = PolicyConfig(
    max_linear=0.35,            # más lento mientras se prueba
    stop_center_fraction=0.5,   # frena menos seguido (default 0.4)
    k_angular=0.8,              # dobla más suave (default 1.2)
)
strategy = TraversabilityStrategy(policy=cfg, drive=True)
```

**Resolución de pesos** (`checkpoint_finetuned_v2.pt`, ~130 MB, no está en
git): argumento explícito → variable de entorno `SAMTP_CHECKPOINT` → caché
local `~/.cache/rover_traversability/` → descarga automática desde Hugging
Face (`sanatem/samtp-mini-traversability`, público, sin token — requiere el
extra `[hf]`).

**Tests** (sin torch, sin checkpoint, sin red):

```bash
pip install -e './traversability[dev]'
pytest traversability/tests
```

#### ¿`genie_rover.bridge` o `rover_traversability`?

| | `genie_rover.bridge` | `rover_traversability` |
|---|---|---|
| Decide sobre | BEV (planta, tras proyección de cámara) | máscara en espacio-imagen, sin proyectar |
| Requiere calibración de cámara | Sí (`tools/calibrate_camera.py`) | No |
| Planner | `genie_path_planner` completo (bancos de rutas polinomiales) | política de corredores (9 bandas, elección por score) |
| Instalación | entorno `genie` completo | entorno `genie` + `pip install -e './traversability[hf]'` |
| Buen punto de partida para | reproducir el sistema del paper tal cual | levantar navegación reactiva rápido, iterar y tunear en el propio field |

Referencia completa de parámetros y comandos:
**[traversability/README.md]()** ·
[Guía en español](Documentos\Modulos\Traversability\Traversability_Comandos.md)

---
## 4. Documentación Completa de Endpoints de la API

La API permite controlar el robot, recibir telemetría en tiempo real, procesar transmisiones de video, gestionar misiones e intervenciones.

Formato general de peticiones HTTP:
- **Query params**: `/<endpoint>?<var1>=<val1>&<var2>=<val2>`
- **Cuerpo JSON**: Peticiones `POST` enviadas con encabezado `Content-Type: application/json`.

---

### A. Base API

#### 1. `POST /control`
Envía comandos de movimiento y control de luz al rover.
> **Importante:** El rover sigue ejecutando el último comando recibido hasta que llega uno nuevo. Para detener el rover se debe enviar `{"linear": 0, "angular": 0}`. Incluye dead-man watchdog (`CONTROL_WATCHDOG_S`, por defecto 3s).

- **Body (JSON):**
  ```json
  {
    "command": {
      "linear": 0.5,
      "angular": -0.2,
      "lamp": 1
    }
  }
  ```
- **Parámetros:**
  - `linear`: Velocidad lineal avance (+1) / retroceso (-1). Rango `[-1, 1]`.
  - `angular`: Velocidad angular izquierda (+1) / derecha (-1). Rango `[-1, 1]`.
  - `lamp`: Control de lámpara (`1` = encendida, `0` = apagada).
- **Respuesta:**
  ```json
  { "message": "Command sent successfully" }
  ```

#### 2. `GET /data`
Retorna la última telemetría del robot (batería, señal, GPS, sensores MPU6050, etc.).
- **Respuesta de ejemplo:**
  ```json
  {
    "battery": 100,
    "signal_level": 5,
    "orientation": 128,
    "lamp": 0,
    "speed": 0,
    "gps_signal": 31.25,
    "latitude": 22.753774,
    "longitude": 114.090950,
    "vibration": 0.31,
    "timestamp": 1724189733.208,
    "accels": [[0.998, 0.003, 0.005, 1725434620.858]],
    "gyros": [[0.521, 0.023, 0.716, 1725434620.913]],
    "mags": [[-1002, 967, 12, 1725434621.194]],
    "rpms": [[0, 0, 0, 0, 1725434567.194]]
  }
  ```

#### 3. `GET /screenshot`
Captura y retorna fotogramas en formato Base64.
- **Query Params:** `view_types` (opciones separadas por coma: `front`, `rear`, `map`). Si se omite, retorna los tres.
- **Ejemplo:** `/screenshot?view_types=rear,map,front`
- **Respuesta:**
  ```json
  {
    "front_frame": "<base64>",
    "rear_frame": "<base64>",
    "map_frame": "<base64>",
    "timestamp": 1724189733.208
  }
  ```

#### 4. `GET /v2/screenshot`
Retorna frames de cámaras cacheados en Base64 con timestamps independientes por cámara (más rápido).
- **Respuesta:**
  ```json
  {
    "front_frame": "<base64>",
    "rear_frame": "<base64>",
    "front_timestamp": 1724189733.198,
    "rear_timestamp": 1724189733.208,
    "timestamp": 1724189733.208
  }
  ```

#### 5. `GET /v2/front`
Retorna únicamente el fotograma de la cámara frontal en Base64.

#### 6. `GET /v2/rear`
Retorna únicamente el fotograma de la cámara trasera en Base64 (si la cámara trasera está disponible).

#### 7. `POST /speak`
Emite un texto mediante voz sintética (Text-to-Speech) a través del parlante del rover vía Agora RTC.
- **Body (JSON):**
  ```json
  { "text": "Hola, iniciando navegación" }
  ```
- **Configuración (`.env`):** `TTS_PROVIDER` (`edge` o `gemini`), `TTS_API_KEY`, `TTS_VOICE`.

#### 8. `GET /feed`
Stream MJPEG continuo en tiempo real (`multipart/x-mixed-replace`). Ideal para consumo en OpenCV o ROS2.
- **Query Params:**
  - `view`: `front` (defecto) o `rear`.
  - `fps`: `1` a `30` (defecto: 15).
- **Ejemplo:** `/feed?view=front&fps=15`

#### 9. `GET /status`
Endpoint liviano de monitoreo de salud del pipeline del SDK.
- **Respuesta:**
  ```json
  {
    "browser_ready": true,
    "mission_started": true,
    "ingest_connected": true,
    "telemetry_age_s": 0.42
  }
  ```

#### 10. `WS /ws/data`
Conexión WebSocket para transmisión continua de telemetría en tiempo real.

---

### B. Missions API

> **Nota de activación:** Requiere definir `MISSION_SLUG=<nombre_mision>` en las variables de entorno para activar la API de misiones en bots remotos.

#### 1. `GET /missions`
Lista las misiones disponibles para el robot configurado.
- **Respuesta:**
  ```json
  {
    "missions": [
      {
        "slug": "mission-1",
        "distance_in_m": 120.5,
        "checkpoints_count": 3
      }
    ]
  }
  ```

#### 2. `POST /start-mission`
Inicia la misión configurada en `MISSION_SLUG`.
- **Respuesta (200 OK):**
  ```json
  { "message": "Mission started successfully" }
  ```

#### 3. `GET /checkpoints-list` (o `POST /checkpoints-list`)
Obtiene la lista completa de checkpoints de la misión y el último alcanzado.
- **Respuesta:**
  ```json
  {
    "checkpoints_list": [
      { "id": 4818, "sequence": 1, "latitude": "30.48243713", "longitude": "114.3026428" },
      { "id": 4819, "sequence": 2, "latitude": "30.48268318", "longitude": "114.3026047" }
    ],
    "latest_scanned_checkpoint": 0
  }
  ```

#### 4. `POST /checkpoint-reached`
Notifica que se ha alcanzado el checkpoint actual. Se aprueba si la distancia relativa entre la ubicación del robot y el checkpoint es **menor a 15 metros**.
- **Respuesta exitosa (200 OK):**
  ```json
  {
    "message": "Checkpoint reached successfully",
    "next_checkpoint_sequence": 2,
    "mission_completed": false
  }
  ```
- **Respuesta de rechazo (400 Bad Request):**
  ```json
  {
    "detail": {
      "error": "Bot is not within 15 meters from the checkpoint",
      "proximate_distance_to_checkpoint": 16.87
    }
  }
  ```

#### 5. `POST /end-mission`
Fuerza la finalización inmediata de la misión (uso en emergencias o reinicio de sesión).
- **Respuesta:**
  ```json
  { "message": "Mission ended successfully" }
  ```

#### 6. `GET /missions-history`
Obtiene el historial de misiones realizadas por el rover.

---

### C. Interventions API

Permite gestionar los periodos en los que el rover requiere intervención manual o atención especial.

#### 1. `POST /interventions/start`
Inicia un registro de intervención almacenando la latitud y longitud actual del robot.
- **Respuesta:**
  ```json
  {
    "message": "Intervention started successfully",
    "intervention_id": "123e4567-e89b-12d3-a456-426614174000"
  }
  ```

#### 2. `POST /interventions/end`
Finaliza la intervención activa almacenando la posición de cierre.
- **Respuesta:**
  ```json
  { "message": "Intervention ended successfully" }
  ```

#### 3. `GET /interventions/history`
Obtiene el historial completo de intervenciones realizadas sobre el rover.
