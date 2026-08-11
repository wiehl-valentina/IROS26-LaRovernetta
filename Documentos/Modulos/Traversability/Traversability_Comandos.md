# `rover_traversability` — Comandos y parámetros en detalle

Percepción de transitabilidad aprendida (SAM-TP fine-tuneado) + política de
manejo por corredores + misión GPS, para el Earth Rover Mini+. Este documento
detalla **todos** los comandos y **todos** los parámetros de configuración del
paquete `traversability/rover_traversability`.

---

## Índice

1. [Instalación](#1-instalación)
2. [Resolución de pesos (checkpoint) y config](#2-resolución-de-pesos-checkpoint-y-config)
3. [CLI: `python -m rover_traversability.demo`](#3-cli-python--m-rover_traversabilitydemo)
   - 3.1 [Flags globales](#31-flags-globales-para-todos-los-subcomandos)
   - 3.2 [`predict`](#32-predict--una-imagen-sin-rover)
   - 3.3 [`live`](#33-live--overlays-en-vivo-no-manda-comandos)
   - 3.4 [`drive`](#34-drive--manejo-reactivo-mueve-el-rover)
   - 3.5 [`mission`](#35-mission--misión-gps-completa-mueve-el-rover)
4. [Uso como librería](#4-uso-como-librería-python)
   - 4.1 [`TraversabilityStrategy`](#41-traversabilitystrategy)
   - 4.2 [`MissionRunner`](#42-missionrunner)
   - 4.3 [`PolicyConfig` — todos los umbrales](#43-policyconfig--todos-los-umbrales)
   - 4.4 [`RoverClient`](#44-roverclient)
   - 4.5 [`TraversabilityPredictor`](#45-traversabilitypredictor)
5. [Variables de entorno — resumen](#5-variables-de-entorno--resumen)
6. [Tests](#6-tests)

---

## 1. Instalación

Desde la raíz del repo, en este orden (son 3 pasos, cada uno necesario):

```bash
# 1. torch (específico de plataforma — CPU/MPS están bien, CUDA si tenés)
pip install torch torchvision

# 2. el código GeNIE/SAM-TP vendorizado (provee el paquete `sam2`)
pip install --no-build-isolation -e ./genie

# 3. este paquete
pip install -e './traversability[hf]'
```

| Extra | Qué agrega | Cuándo usarlo |
|---|---|---|
| `[hf]` | `huggingface_hub>=0.23` | Para que el checkpoint se descargue solo desde Hugging Face. Se puede omitir si vas a poner el archivo `.pt` a mano. |
| `[dev]` | `pytest>=7` | Para correr la suite de tests (no necesita torch/checkpoint/red). |

Requiere **Python ≥ 3.10**. Sin torch instalado, el paquete igual se importa
y `policy.py` / `client.py` / `mission.py` / los tests funcionan — solo
`TraversabilityPredictor` necesita torch + sam2.

---

## 2. Resolución de pesos (checkpoint) y config

`TraversabilityPredictor` busca el checkpoint (`checkpoint_finetuned_v2.pt`,
~130 MB) en este orden, y **se detiene en el primer error explícito** (una
ruta mal tipeada nunca cae silenciosamente a otra fuente):

1. Argumento explícito: `TraversabilityPredictor(checkpoint="/ruta/al/archivo.pt")`
2. Variable de entorno `SAMTP_CHECKPOINT=/ruta/al/archivo.pt`
3. Caché local: `~/.cache/rover_traversability/checkpoint_finetuned_v2.pt`
   (o `$XDG_CACHE_HOME/rover_traversability/...` si esa variable está seteada)
4. Descarga automática desde Hugging Face Hub (requiere `huggingface_hub`,
   ver extra `[hf]`)

sha256 del checkpoint bueno conocido:
`44e508da3d36a63431f8197f16784c980abf43ea94fc4e524bcd19d0646692bd`

> ⚠️ El checkpoint público de GeNIE (`checkpoint_2.pt`, del Google Drive del
> paper) es un modelo **base+**, NO carga contra la config tiny que usa este
> paquete. El archivo esperado es `checkpoint_finetuned_v2.pt`.

### Variables de entorno relacionadas

| Variable | Default | Qué hace |
|---|---|---|
| `SAMTP_CHECKPOINT` | *(no seteada)* | Ruta local explícita al `.pt`. Si está seteada y el archivo no existe → error, no hay fallback. |
| `SAMTP_HF_REPO` | `sanatem/samtp-mini-traversability` | Repo de Hugging Face de donde descargar (público, sin token). |
| `SAMTP_HF_FILENAME` | `checkpoint_finetuned_v2.pt` | Nombre de archivo a buscar/descargar dentro del repo HF. |
| `SAMTP_CONFIG` | *(no seteada)* | Ruta explícita al yaml de inferencia, en vez de resolverlo dentro del paquete `sam2` instalado. |
| `XDG_CACHE_HOME` | `~/.cache` | Base del directorio de caché (`$XDG_CACHE_HOME/rover_traversability/`). |

---

## 3. CLI: `python -m rover_traversability.demo`

```
python -m rover_traversability.demo [flags globales] <comando> [flags del comando]
```

### 3.1 Flags globales (para todos los subcomandos)

| Flag | Default | Descripción |
|---|---|---|
| `--checkpoint PATH` | `None` (usa la resolución automática, ver sección 2) | Ruta explícita al `.pt` |
| `--device {cuda,mps,cpu}` | `None` (auto: cuda > mps > cpu) | Fuerza el dispositivo de inferencia |
| `--no-refine` | `False` | Desactiva el refinamiento por contraste (`refine_traversability_by_contrast`) que baja a "bloqueado" objetos oscuros sobre piso claro. Se recomienda dejarlo activado. |

### 3.2 `predict` — una imagen, sin rover

```bash
python -m rover_traversability.demo predict screenshots/imagen.jpg --out overlay.png
```

| Flag | Default | Descripción |
|---|---|---|
| `image` (posicional) | — | Ruta a la imagen de entrada (obligatorio) |
| `--out PATH` | `overlay.png` | Dónde guardar el overlay verde/rojo |

Salida en consola: latencia de inferencia, % de píxeles transitables, y el
comando `{linear, angular}` que sugeriría la política por defecto.

### 3.3 `live` — overlays en vivo, NO manda comandos

```bash
python -m rover_traversability.demo live --save-dir trav_out --interval 0.5
```

| Flag | Default | Descripción |
|---|---|---|
| `--save-dir PATH` | `trav_out` | Carpeta donde se guardan los overlays (`overlay_<timestamp>_<frame>.png`) |
| `--interval SEGUNDOS` | `0.5` | Pausa entre polls al SDK |
| `--max-frames N` | `None` (infinito) | Corta después de N frames; sin este flag, correr hasta Ctrl-C |

No requiere `--yes-i-want-the-rover-to-move`: nunca envía comandos de
movimiento, solo lee frames y telemetría.

### 3.4 `drive` — manejo reactivo, MUEVE el rover

```bash
python -m rover_traversability.demo drive --yes-i-want-the-rover-to-move --interval 0.5
```

| Flag | Default | Descripción |
|---|---|---|
| `--yes-i-want-the-rover-to-move` | `False` | **Obligatorio** para que el comando haga algo — sin este flag, se niega a arrancar (exit code 2) |
| `--save-dir PATH` | `None` (no guarda) | Si se especifica, guarda overlays igual que `live` |
| `--interval SEGUNDOS` | `0.5` | Pausa entre iteraciones del loop de manejo |
| `--max-iterations N` | `None` (infinito) | Corta después de N iteraciones |

Solo evasión reactiva de obstáculos, sin objetivo GPS. Frena el rover
(`client.stop()`) al salir por Ctrl-C o al terminar `--max-iterations`.

### 3.5 `mission` — misión GPS completa, MUEVE el rover

```bash
python -m rover_traversability.demo mission \
  --start-mission \
  --yes-i-want-the-rover-to-move \
  --arrive-attempt-m 8.0 \
  --interval 0.5
```

| Flag | Default | Descripción |
|---|---|---|
| `--yes-i-want-the-rover-to-move` | `False` | **Obligatorio**, igual que en `drive` |
| `--start-mission` | `False` | Si se pasa, llama `POST /start-mission` antes de arrancar el loop |
| `--arrive-attempt-m METROS` | `8.0` | Distancia al checkpoint desde la cual se intenta `POST /checkpoint-reached` (el backend valida el radio real; un rechazo simplemente significa "seguir manejando") |
| `--interval SEGUNDOS` | `0.5` | Intervalo del loop de misión |
| `--max-steps N` | `None` (infinito) | Corta después de N pasos, con `reason="max_steps reached"` |

Imprime por cada paso: distancia al checkpoint, offset angular al objetivo, y
la decisión (`reason`, `linear`, `angular`). Al finalizar, imprime si la
misión se completó, cuántos checkpoints se alcanzaron y en cuántos pasos.

---

## 4. Uso como librería (Python)

### 4.1 `TraversabilityStrategy`

Duck-type de `RoverStrategy` — reemplazo de una línea en tu `RoverLoop`
existente.

```python
from rover_traversability import TraversabilityStrategy

strategy = TraversabilityStrategy(
    client=None,              # RoverClient() por defecto
    predictor=None,           # TraversabilityPredictor() por defecto (carga lazy)
    policy=None,              # PolicyConfig() por defecto
    drive=False,              # False = dry-run (predice e imprime, no manda nada)
    save_overlays_dir=None,   # carpeta opcional para guardar overlays PNG
    on_decision=None,         # callback: (result, decision) -> None
)
```

| Parámetro | Default | Descripción |
|---|---|---|
| `client` | `RoverClient()` | Cliente HTTP al SDK |
| `predictor` | `TraversabilityPredictor()` (lazy) | Predictor SAM-TP; importar `torch` recién acá |
| `policy` | `PolicyConfig()` | Umbrales de la política de corredores |
| `drive` | `False` | `True` = manda comandos reales; `False` = solo imprime `[dry-run] ...` |
| `save_overlays_dir` | `None` | Si se define, guarda un PNG por frame analizado |
| `on_decision` | `None` | Hook para loguear `(result, decision)` en cada frame, p. ej. a CSV |

### 4.2 `MissionRunner`

```python
from rover_traversability import MissionRunner

runner = MissionRunner(
    client=None,               # RoverClient() por defecto
    predictor=None,            # TraversabilityPredictor() por defecto
    policy=None,                # PolicyConfig() por defecto
    arrive_attempt_m=8.0,       # radio de intento de checkpoint
    interval_s=0.5,             # período del loop
    turn_in_place_deg=70.0,     # a partir de qué offset se hace giro de arco
    arc_turn_linear=0.15,       # velocidad lineal durante el arco hacia el objetivo
    max_steps=None,             # límite de pasos (None = sin límite)
    on_step=None,               # callback: (step_info: dict) -> None
)
result = runner.run()
```

| Parámetro | Default | Descripción |
|---|---|---|
| `arrive_attempt_m` | `8.0` | Distancia (m) desde la que se intenta `POST /checkpoint-reached` |
| `interval_s` | `0.5` | Período objetivo entre pasos del loop |
| `turn_in_place_deg` | `70.0` | `|goal_offset_deg|` por encima del cual el objetivo está "fuera de cámara" y se hace un giro de arco en vez de buscar corredores |
| `arc_turn_linear` | `0.15` | Velocidad lineal durante el giro de arco (nunca 0 — así el heading GPS sigue actualizándose) |
| `max_steps` | `None` | Corta el loop tras N pasos |
| `on_step` | `None` | Callback por paso, recibe `{"step", "distance_m", "heading_deg", "goal_offset_deg", "decision"}` |

`runner.run()` devuelve un `MissionResult(completed, checkpoints_reached, steps, reason, history)`
y **siempre** frena el rover en un `finally`, incluso ante excepción.

### 4.3 `PolicyConfig` — todos los umbrales

Dataclass congelado (`frozen=True`) — todo el ajuste fino de la política de
corredores vive acá, en un solo lugar:

```python
from rover_traversability import PolicyConfig, TraversabilityStrategy

cfg = PolicyConfig(
    roi_top=0.55,
    drivable_thresh=0.5,
    num_corridors=9,
    stop_center_fraction=0.40,
    min_corridor_score=0.35,
    bottom_weight=2.0,
    min_valid_pixels=200,
    max_linear=0.5,
    min_linear=0.15,
    max_angular=0.5,
    k_angular=1.2,
    hfov_deg=92.7,
    goal_sigma=0.5,
    goal_bias_floor=0.2,
)
strategy = TraversabilityStrategy(policy=cfg, drive=True)
```

| Parámetro | Default | Descripción |
|---|---|---|
| `roi_top` | `0.55` | Fracción superior de la imagen que se ignora (cielo/horizonte); se analiza solo lo que está debajo |
| `drivable_thresh` | `0.5` | Valor de máscara por encima del cual un píxel cuenta como transitable |
| `num_corridors` | `9` | Cantidad de bandas verticales (se fuerza impar, ≥3, para tener un corredor central exacto) |
| `stop_center_fraction` | `0.40` | Se considera bloqueado el corredor central si esta fracción o más está por debajo de `drivable_thresh` (test por fracción, no "cualquier píxel bloqueado") |
| `min_corridor_score` | `0.35` | Un corredor con score por debajo de esto no se considera viable para girar/avanzar |
| `bottom_weight` | `2.0` | Peso de las filas más cercanas al rover (rampa lineal 1 → este valor, de arriba a abajo del ROI) |
| `min_valid_pixels` | `200` | Si el ROI tiene menos píxeles finitos que esto → `reason="no_data"` (stop) |
| `max_linear` | `0.5` | Velocidad lineal máxima |
| `min_linear` | `0.15` | Velocidad lineal mínima cuando decide avanzar |
| `max_angular` | `0.5` | Velocidad angular máxima |
| `k_angular` | `1.2` | Ganancia de dirección sobre el offset normalizado de imagen |
| `hfov_deg` | `92.7` | FOV horizontal de la cámara frontal del Mini+ (usado para proyectar `goal_offset_deg` a posición de imagen) |
| `goal_sigma` | `0.5` | Ancho (en x normalizado) de la campana gaussiana del sesgo hacia el objetivo GPS |
| `goal_bias_floor` | `0.2` | Piso del sesgo: los corredores lejos del objetivo conservan al menos esta fracción de su score original (la transitabilidad nunca se anula por estar lejos del objetivo) |

### 4.4 `RoverClient`

```python
from rover_traversability import RoverClient

client = RoverClient(
    base_url=None,       # usa $ROVER_BASE_URL o "http://localhost:8000"
    timeout=5.0,         # segundos, timeout general de requests
    session=None,        # requests.Session() por defecto
)
```

| Parámetro / env var | Default | Descripción |
|---|---|---|
| `base_url` / `ROVER_BASE_URL` | `http://localhost:8000` | URL base del SDK |
| `timeout` | `5.0` | Timeout general (excepto `/control`, que usa `2.0` fijo) |

Métodos relevantes y sus parámetros:

- `send_command(linear: float, angular: float, lamp: int = 0)` — valores de
  `linear`/`angular` se recortan (clamp) a `[-1, 1]`.
- `stop(retries: int = 3)` — reintenta hasta que el comando sea aceptado.
- `get_front_frame_b64()` / `get_front_frame()` — sin parámetros.
- `get_data()` — telemetría cruda (`GET /data`).
- `start_mission()`, `get_checkpoints_list()`, `checkpoint_reached()` — sin
  parámetros; nunca lanzan excepción, devuelven `CommandResult(accepted, status, detail, body)`.

### 4.5 `TraversabilityPredictor`

```python
from rover_traversability import TraversabilityPredictor

predictor = TraversabilityPredictor(
    checkpoint=None,        # ver sección 2 (resolución automática)
    config=None,            # ver sección 2
    device=None,            # None = auto (cuda > mps > cpu)
    hf_repo=None,           # override de SAMTP_HF_REPO
    contrast_refine=True,   # refinamiento por contraste (piso vs objeto oscuro)
    overlay_alpha=0.45,     # opacidad del overlay verde/rojo sobre el frame original
)
result = predictor.predict(payload)   # base64 / path / bytes / PIL / np.ndarray
predictor.warmup()                    # una inferencia dummy para compilar kernels
```

| Parámetro | Default | Descripción |
|---|---|---|
| `checkpoint` | `None` | Ver cadena de resolución (sección 2) |
| `config` | `None` | Ídem, yaml de inferencia |
| `device` | `None` (auto) | `"cuda"`, `"mps"` o `"cpu"` |
| `hf_repo` | `None` (usa `SAMTP_HF_REPO`) | Repo de HF alternativo |
| `contrast_refine` | `True` | Corrige objetos oscuros sobre piso claro marcados como transitables |
| `overlay_alpha` | `0.45` | Mezcla frame/color en el overlay de debug |

`TraversabilityResult` devuelto: `mask` (HxW float32 [0,1]), `logits` (HxW
crudos), `overlay` (HxWx3 uint8), `image` (HxWx3 uint8 decodificado),
`device`, `inference_s`.

---

## 5. Variables de entorno — resumen

| Variable | Usada por | Default |
|---|---|---|
| `ROVER_BASE_URL` | `RoverClient` | `http://localhost:8000` |
| `SAMTP_CHECKPOINT` | `weights.resolve_checkpoint` | *(no seteada)* |
| `SAMTP_HF_REPO` | `weights.resolve_checkpoint` | `sanatem/samtp-mini-traversability` |
| `SAMTP_HF_FILENAME` | `weights.resolve_checkpoint` | `checkpoint_finetuned_v2.pt` |
| `SAMTP_CONFIG` | `weights.resolve_config` | *(no seteada)* |
| `XDG_CACHE_HOME` | `weights.default_cache_dir` | `~/.cache` |
| `PYTORCH_ENABLE_MPS_FALLBACK` | seteada automáticamente por el paquete (`"1"`) | — necesaria en Apple Silicon, algunos ops de Hiera no tienen kernel MPS |

---

## 6. Tests

Sin torch, sin checkpoint, sin red — 67 tests:

```bash
pip install -e './traversability[dev]'
pytest traversability/tests
```

Archivos de test relevantes si querés correr un subconjunto:

```bash
pytest traversability/tests/test_policy.py       # política de corredores
pytest traversability/tests/test_mission.py       # loop de misión GPS
pytest traversability/tests/test_geo.py           # bearing/heading GPS
pytest traversability/tests/test_client.py        # cliente HTTP del SDK
pytest traversability/tests/test_images.py        # decodificación de payloads
pytest traversability/tests/test_strategy.py      # TraversabilityStrategy
pytest traversability/tests/test_weights.py       # resolución de checkpoint/config
pytest traversability/tests/test_calibration.py   # intrínsecos/extrínsecos del Mini+
```

---

## Seguridad — antes de usar `drive` o `mission`

- El rover **repite el último comando para siempre** — el silencio no lo
  frena. Toda decisión de "parar" en este paquete manda activamente
  `{linear: 0, angular: 0}`, con reintentos.
- `drive` y `mission` se niegan a correr sin
  `--yes-i-want-the-rover-to-move`.
- Primera prueba: espacio abierto, dedo en Ctrl-C, verificar que el rover
  dobla para el lado correcto (`angular` positivo = izquierda).
