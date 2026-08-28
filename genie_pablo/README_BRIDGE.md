# Correr GeNIE sobre tu Earth Rover

Puente entre `IROS26_ERC_UNLP_Equipo1` (el SDK que maneja el robot) y
`GENIE-SAMTP` (la percepción y el planner del paper).

## Cómo queda armado

```
Proceso A: SDK de ERC              Proceso B: bridge de GeNIE
─────────────────────              ──────────────────────────
hypercorn main:app                 python -m genie_rover.bridge
  Chrome headless                    /v2/front  ──► SAM-TP ──► BEV
  Agora RTC/RTM                                        │
  :8000  ◄──────── HTTP ────────────  /control ◄── controlador ◄── planner
```

Dos entornos virtuales separados a propósito. El SDK pinnea `numpy==1.26.4` y
necesita Chrome; SAM2 necesita torch y hydra. La frontera es HTTP, así que no
tienen que convivir.

## Antes de arrancar: ¿GPU?

| | tiempo por frame | qué significa |
|---|---|---|
| GPU NVIDIA (≥6 GB) | ~30–60 ms | el modelo va a ~10 Hz como en el paper; el cuello de botella es el video del rover a 3–5 Hz |
| CPU sola | ~1–4 s | usable pero hay que bajar `max_linear` a ~0.15 y aceptar que el robot avanza a tramos |

Hiera-**tiny** a 1024 px es lo que corre acá (`sam2.1_inference_tiny`), no el
large — por eso la CPU es viable aunque incómoda.

## Instalación

**1. Dónde va cada cosa.** Copiá estas carpetas dentro del repo `GENIE-SAMTP`,
no dentro del SDK:

```
GENIE-SAMTP/
├── sam2/                    (ya estaba)
├── genie_path_planner/      (ya estaba)
├── genie_rover/             ← nuevo
├── tools/                   ← nuevo
└── configs/frodobot_rover.yaml  ← nuevo
```

**2. Entorno de GeNIE**

```bash
cd GENIE-SAMTP
conda env create -n sam_tp -f environment.yml
conda activate sam_tp

# torch acorde a tu CUDA (mirá https://pytorch.org/get-started/locally/)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# o, sin GPU:
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -e .
pip install requests pyyaml opencv-python
```

**3. El checkpoint.** No viene en el repo. Bajá `checkpoint_2.pt` del Google
Drive que linkea el README de GENIE y ponelo en esa ruta exacta — ojo que hay un
**directorio que se llama literalmente `...yaml`**, no es un error de tipeo:

```bash
mkdir -p "sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints"
mv ~/Downloads/checkpoint_2.pt \
   "sam2_logs/configs/sam2.1_training_tiny/sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/"
```

**4. Entorno del SDK**, aparte:

```bash
cd IROS26_ERC_UNLP_Equipo1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install google-genai     # la última línea del requirements está rota
cp .env.sample .env          # completá SDK_API_TOKEN, BOT_SLUG, CHROME_EXECUTABLE_PATH
hypercorn main:app --reload
```

Abrí `http://localhost:8000` y confirmá que se ve el video antes de seguir.

## Puesta en marcha, en orden

Cada paso valida algo concreto. No te saltees ninguno: los pasos 4 y 6 son los
que evitan que el robot se vaya contra una pared.

### 1. ¿Habla el SDK?

```bash
python -m genie_rover.sdk_client
```

Te tiene que devolver un frame, GPS con `gps_signal > 0` y la lista de
checkpoints. **Anotá la resolución del frame**, la vas a necesitar. No manda
ningún comando de movimiento.

### 2. ¿Carga el modelo?

```bash
python -m genie_rover.perception --config configs/frodobot_rover.yaml \
    --image alguna_foto_del_rover.jpg --out debug/
```

Todavía con la calibración del Stretch, así que el BEV va a salir mal — lo único
que se valida acá es que el checkpoint carga y que SAM-TP segmenta. Mirá el panel
del medio: el suelo transitable tiene que salir rojo.

### 3. Calibrá la cámara

```bash
python tools/calibrate_camera.py capture --out calib_frames/
python tools/calibrate_camera.py solve --frames calib_frames/ --square-size 0.025 --out calib.yaml
```

Pegá `image_size`, `intrinsics` y `dist_coeffs` en el config.

Esto no es opcional. `project_score_to_bev` asume pinhole puro y no corrige nada,
y el paper aclara que la óptica del rover distorsiona bastante — por eso ellos
usaron un modelo de cámara genérico en vez del pinhole estándar.

### 4. Medí la cámara con una regla

- `height_m`: del piso al centro del lente
- `pitch_down_deg`: cuánto mira hacia abajo respecto de la horizontal

Poné `calibrated: true` y volvé a correr el paso 2. Ahora el BEV tiene que dar un
abanico que se abre hacia adelante, con el suelo libre en verde. **Verificación
concreta**: poné una caja a 2 m exactos del robot y comprobá que aparece a 2 m en
el BEV. Si aparece a 1,4 m o a 3 m, el pitch está mal.

### 5. El signo de `angular`

```bash
python tools/check_angular_sign.py
```

Gira 2 segundos, mirás para dónde fue, y escribís el valor en el config.

### 6. Simulacro

```bash
python -m genie_rover.bridge --config configs/frodobot_rover.yaml
```

Corre el bucle completo e imprime los comandos **sin enviarlos**. Poné el robot
mirando un pasillo libre y verificá que dice `linear≈0.30 angular≈0`. Giralo hacia
una pared y verificá que dice `OBSTACULO al frente`.

### 7. Primera corrida real

```bash
python -m genie_rover.bridge --config configs/frodobot_rover.yaml \
    --go --max-seconds 30 --debug-dir debug/run1
```

Al aire libre, con espacio, y con la mano en Ctrl-C. Empezá con
`max_linear: 0.15` y subilo de a poco.

Recién cuando esto ande, agregá `--start-mission` para navegar checkpoints reales.

## Detalles que importan

**El freno es lo más importante del código.** El SDK mantiene el último comando
indefinidamente: si el proceso muere sin frenar, el rover se sigue moviendo solo.
Por eso `bridge.run()` frena en un `finally`, dos veces, y hay un watchdog que
frena si el video se congela más de 2 segundos.

**Convención de ejes.** El planner trabaja en `[x_right_m, y_forward_m]`
centrado en el robot. `camera_pose_from_height_pitch()` arma la `T_world_camera`
en optical frame (x = derecha de la imagen, y = abajo, z = adelante), que es lo
que `projection.py` espera. Si alguna vez montás la cámara en un pan-tilt, hay
que pasar la pose por frame en vez de usar la fija.

**El rumbo sale del GPS, no de la brújula.** El magnetómetro va montado al lado
de los motores; su lectura suele tener un offset variable. `HeadingEstimator` usa
el desplazamiento GPS acumulado como fuente primaria, que es ruidoso pero no
sesgado. Si querés confiar en `orientation`, poné `trust_orientation: true` y
calibrá el offset comparando ambas fuentes mientras el robot avanza derecho.

## Si algo falla

| Síntoma | Causa probable |
|---|---|
| `FileNotFoundError` del checkpoint | el directorio `...yaml` no está creado con ese nombre literal |
| `CUDA out of memory` | poné `device: "cpu"` en el config |
| El BEV sale casi todo vacío | `pitch_down_deg` mal: la cámara "mira" arriba del horizonte y los rayos nunca cortan el piso |
| Obstáculos a la distancia equivocada | `height_m` mal, o la K no corresponde a la resolución actual |
| El robot gira siempre para el lado contrario | `angular_sign` invertido |
| `El frame es 640x480 pero la calibración es para ...` | `IMAGE_QUALITY`/`IMAGE_FORMAT` del `.env` cambiaron la resolución |
| El planner nunca encuentra camino | `threshold_cost` muy estricto para tu terreno, o el BEV está casi todo desconocido |

## Lo que este puente no tiene

Para que quede claro qué te falta respecto del sistema que ganó el ERC:

- **Recuperación con VLM.** `Bridge._recover()` gira a ciegas. El paper usa un
  VLM en dos etapas (clasificar on-road/off-road, después barrido de 360° en
  cuatro headings y elegir). Es lo primero que le agregaría, y tenés la
  infraestructura de Gemini lista en `programs/genai_program.py`.
- **Fusión temporal de observaciones.** `observation_fusion` existe en el planner
  y acá no se usa: cada frame se planifica de cero. Activarlo requiere llevar la
  pose del robot entre frames.
- **Memoria del entorno.** Los propios autores lo listan como limitación: sin
  memoria, el robot no sabe dónde estuvo y cae en mínimos locales.

Y recordá revocar la clave de OpenRouter que quedó en `control_nube.py`.
