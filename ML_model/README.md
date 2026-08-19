# iros26_erc_unlp_ml_modeling

Repo separado para armar datasets de fine-tuning a partir de corridas
**manuales** del rover (vos manejando, no el bucle autonomo). Sin dependencia
del repo de navegación por ahora — cuando lo integres, `rover_client.py` es
un subconjunto de `genie_rover/sdk_client.py` y se puede reemplazar por ese
import sin tocar el resto.

```
iros26_erc_unlp_ml_modeling/
├── README.md
├── requirements.txt
│
├── rover_client.py          # RoverReader: solo front_frame() + telemetry(), sin control()
│
├── mission_recorder.py      # graba frames + metadata mientras manejás a mano.
│                             # por frame guarda: gps (lat/lon/orientation/gps_signal),
│                             # speed, battery, rpms + gyros (telem.raw, misma
│                             # telemetria cruda que usa odometry.py), source
│                             # (tag "manual"/"teleop"/...), note
│
├── dataset_tools.py          # candidatos (dedup por distancia GPS real) ->
│                             # etiquetar a mano -> armar (layout de training)
│
└── data/
    ├── raw_runs/             # una carpeta POR CORRIDA, la crea mission_recorder.py
    │   └── <nombre_corrida>/         # ej. patio_20260819
    │       ├── session.json          # metadata de la sesion entera
    │       ├── manifest.jsonl        # 1 linea de metadata por frame
    │       └── frames/
    │           ├── 000000_rgb.jpg
    │           ├── 000000_meta.json
    │           └── ...
    │
    └── candidatos/            # la crea `dataset_tools.py candidatos`
        ├── <corrida>_000000_rgb.jpg
        ├── <corrida>_000000_meta.json
        └── <corrida>_000000_mask.png   # <- esto se agrega A MANO (etiquetado)
```

Y lo que `dataset_tools.py armar` crea una vez que etiquetaste algo (no existe
todavía en el repo):

```
training/FOLD_custom/          # --out que le pases a `armar`
├── img_folder/
│   ├── train/<nombre>/00000.jpg
│   └── val/<nombre>/00000.jpg
└── gt_folder/
    ├── train/<nombre>/00000.png
    └── val/<nombre>/00000.png
```

## 1. Instalar

```bash
pip install -r requirements.txt
```

## 2. Grabar una corrida

Con el SDK corriendo (`hypercorn main:app` o como lo levantes) y vos
manejando el rover con lo que sea que uses normalmente:

```bash
python -m mission_recorder \
    --out data/raw_runs/patio_20260819 \
    --interval 0.5 \
    --note "vueltas alrededor del patio, tarde, sombra parcial"
```

Ctrl-C corta en cualquier momento y deja todo lo grabado usable. Este script
**nunca manda comandos de control** — `rover_client.RoverReader` ni siquiera
tiene ese método.

Si el rover se queda quieto mucho rato (por ejemplo mientras ajustás algo),
usá `--min-gps-displacement-m 0.3` para no grabar 200 fotos casi idénticas.

Cada frame guarda, además de GPS/velocidad/batería, `rpms` y `gyros` — la
telemetría cruda de `telem.raw` (la misma que usa `odometry.py` para
reconstruir movimiento fino). Sirve para priorizar candidatos: rueda
patinando (rpms asimétricas sin giro proporcional) o giro brusco (gyro alto)
suelen coincidir con frames donde el modelo se equivoca más. También se
guarda `source` (default `"manual"`, cambiable con `--source`) para poder
distinguir después estas corridas de las que salgan de `bridge.py
--debug-dir` si algún día mezclás ambas fuentes.

## 3. Elegir qué frames etiquetar

No conviene etiquetar todo — conviene etiquetar frames variados y, sobre
todo, frames donde el modelo actual se equivoca. Por ahora esto filtra por
distancia recorrida (evita duplicados casi idénticos); más adelante se puede
sumar un filtro por confianza del modelo si guardás esa metadata también.

```bash
python -m dataset_tools candidatos \
    --runs data/raw_runs/patio_20260819 data/raw_runs/otra_corrida \
    --out data/candidatos \
    --every-n-m 0.5 \
    --max-por-corrida 200
```

## 4. Etiquetar

Por cada `data/candidatos/<nombre>_rgb.jpg`, generar una máscara
`data/candidatos/<nombre>_mask.png` (mismo tamaño, 255 = transitable /
0 = no transitable). Dos formas:

- **Desde cero**: `labelme` (local) o CVAT (web).
- **Semi-automático (más rápido)**: correr el modelo actual sobre esos
  frames, guardar su predicción como punto de partida, y corregir solo
  donde se equivocó — más rápido que dibujar todo a mano.

⚠️ Antes de etiquetar un lote grande, probá **un** par contra el loader real
de entrenamiento (ver nota de formato en el docstring de `dataset_tools.py`)
para confirmar que el formato de máscara es el que espera. Es más barato
confirmar esto con 1 imagen que reetiquetar 500.

## 5. Armar el dataset para entrenar

```bash
python -m dataset_tools armar \
    --labeled data/candidatos \
    --out training/FOLD_custom \
    --val-frac 0.15
```

Esto arma `img_folder/` y `gt_folder/` en el layout que esperan los configs
de `genie/sam2/configs/sam2.1_training_tiny/*.yaml` (un "video" de 1 frame
por imagen). Actualizá `dataset.img_folder` / `dataset.gt_folder` en el yaml
para que apunten a las carpetas `train/` generadas acá, y arrancá desde el
checkpoint fine-tuneado existente (`checkpoint_finetuned_v2.pt` en
`sanatem/samtp-mini-traversability`) en vez de desde cero.
