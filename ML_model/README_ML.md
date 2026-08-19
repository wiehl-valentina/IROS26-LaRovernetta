# ML_model

Dos pipelines conviven acá, y los dos terminan armando datasets para
fine-tunear el modelo de traversability (SAM-TP):

1. **`scripts/pipe_videos_online/`** — datos públicos de FrodoBots bajados de
   Hugging Face (dos datasets distintos, ver abajo). Selección/curación
   previa, sin grabar nada propio.
2. **`scripts/pipe_nuestras_rides/`** — datos propios: grabás vos manejando
   el rover, evaluás con el modelo actual, elegís qué etiquetar, armás el
   dataset de entrenamiento.

Ambos desembocan en el mismo lugar: `training/<nombre>/img_folder` +
`gt_folder`, el layout que espera el training config de SAM2
(`genie/sam2/configs/sam2.1_training_tiny/*.yaml`, en el **repo de
navegación** — ver la sección de abajo para reusarlo desde acá).

## Estructura

```
ML_model/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example                 # plantilla para NAV_REPO_PATH
├── config_paths.py              # reusa genie/sam2/traversability sin vendorearlos
│
├── scripts/
│   ├── pipe_videos_online/                    # datasets públicos (HF)
│   │   ├── 0_check_columns.py                 # inspecciona metadata.parquet de Mini-4K
│   │   ├── 1_descarga_filtrado.py             # select/fetch/clean/curate de Mini-4K
│   │   └── Tool_1_metadata_descarga.py        # metadata de lerobot-7k (otro dataset)
│   │
│   └── pipe_nuestras_rides/                   # datasets propios (rover real)
│       ├── rover_client.py                    # RoverReader: solo lectura del SDK
│       ├── 0_mission_recorder.py              # grabar frames+metadata a mano
│       ├── 1_encontrar_candidatos.py          # elegir candidatos por distancia GPS
│       ├── 2_evaluar_con_modelo.py            # NUEVO: priorizar por incertidumbre del modelo
│       └── 3_armar_dataset.py                 # armar layout de entrenamiento
│
├── notebooks/
│   └── Analisis_dataset_pablo.ipynb           # exploración de metadata.parquet (Mini-4K)
│
├── data/                          # gitignored, se genera corriendo los scripts
│   ├── raw_runs/                  # salida de 0_mission_recorder.py
│   └── candidatos/                # salida de 1_encontrar_candidatos.py (+ mascaras a mano)
│
└── training/                      # gitignored, salida de 3_armar_dataset.py
    └── FOLD_custom/
        ├── img_folder/{train,val}/<nombre>/00000.jpg
        └── gt_folder/{train,val}/<nombre>/00000.png
```

> ⚠️ Si tenés `dataset_tools.py` suelto en `scripts/` (versión vieja, con
> subcomandos `candidatos`/`armar`), reemplazalo por los tres archivos
> numerados de `pipe_nuestras_rides/` de arriba — hacen lo mismo pero
> separados en pasos, y `2_evaluar_con_modelo.py` es nuevo. Confirmá también
> que `rover_client.py` esté en `pipe_nuestras_rides/`: `0_mission_recorder.py`
> lo importa directo (`from rover_client import ...`), así que tiene que
> vivir en la misma carpeta.

## Instalar

```bash
pip install -r requirements.txt
```

## Reusar genie y sam2 (repo de navegación)

Este repo **no** vendorea una copia de `genie/`/`traversability/` — los
reusa desde el repo de navegación. Dos formas, elegí una:

**Opción A (recomendada) — paquete editable, una sola vez:**

```bash
pip install --no-build-isolation -e /ruta/a/tu/repo/de/navegacion/genie
pip install -e '/ruta/a/tu/repo/de/navegacion/traversability[hf]'
```

Con esto, `import sam2` y `from rover_traversability import TraversabilityPredictor`
funcionan en cualquier script o notebook de acá, sin nada más — es lo mismo
que ya usás en el repo de navegación, apuntado a esa ruta.

**Opción B — sin instalar, para notebooks sueltos:**

```bash
cp .env.example .env
# editá .env: NAV_REPO_PATH=/ruta/a/tu/repo/de/navegacion
```

Y antes de importar `sam2`/`rover_traversability`:

```python
import config_paths
from rover_traversability import TraversabilityPredictor
```

`2_evaluar_con_modelo.py` ya usa esto — hoy es el único script de este repo
que necesita el modelo cargado.

## Pipeline 1: datasets de Hugging Face

Dos datasets públicos distintos, cada uno con su propio flujo:

### `BitRobot/FrodoBots-Mini-4K` (`pipe_videos_online/`)

```bash
cd scripts/pipe_videos_online
python 0_check_columns.py                 # ver que columnas trae metadata.parquet
python 1_descarga_filtrado.py select   --out selected_rides.csv --max-per-country 30 --require-rear
python 1_descarga_filtrado.py fetch    --rides selected_rides.csv --raw-dir raw --skip-recordings
python 1_descarga_filtrado.py clean    --raw-dir raw --clean-dir clean --index cleaned_index.csv
python 1_descarga_filtrado.py curate   --index cleaned_index.csv --hours 300 --out curated_rides.csv
```

`clean` sincroniza cada frame con GPS + control + IMU (tolerancia 500 ms) y
tira los puntos GPS sin fix. `curate` deduplica por celda GPS (75 m) para
maximizar diversidad geográfica sin repetir el mismo tramo.

### `BitRobot/berkeley-frodobots-lerobot-7k` (`pipe_videos_online/`)

Formato LeRobot (chunks de parquet + video), metadata liviana primero:

```bash
python Tool_1_metadata_descarga.py
```

Esto todavía es exploratorio — no tiene un pipeline `select/fetch/clean/curate`
como Mini-4K. Si avanza, conviene darle la misma estructura (capaz como
`scripts/pipe_videos_online/lerobot7k/` con sus propios pasos numerados).

## Pipeline 2: datasets propios

```bash
cd scripts/pipe_nuestras_rides

# 0. Grabar una corrida manejando a mano (nunca manda comandos de control)
python 0_mission_recorder.py \
    --out ../../data/raw_runs/patio_20260819 \
    --interval 0.5 \
    --note "vueltas alrededor del patio, tarde, sombra parcial"

# 1. Elegir candidatos, espaciados por distancia GPS real
python 1_encontrar_candidatos.py \
    --runs ../../data/raw_runs/patio_20260819 \
    --out ../../data/candidatos \
    --every-n-m 0.5 --max-por-corrida 200

# 2. (NUEVO) Priorizar por incertidumbre del modelo actual
python 2_evaluar_con_modelo.py \
    --frames-dir ../../data/candidatos \
    --out ../../data/candidatos/model_eval.jsonl \
    --overlays-dir ../../data/candidatos/overlays
# ordená model_eval.jsonl por 'incertidumbre' descendente: eso es lo que
# más te conviene etiquetar primero.

# 3. Etiquetar a mano (labelme/CVAT, o corregir el overlay del paso 2)
#    <nombre>_rgb.jpg -> <nombre>_mask.png (255=transitable, 0=no)

# 4. Armar el layout de entrenamiento
python 3_armar_dataset.py \
    --labeled ../../data/candidatos \
    --out ../../training/FOLD_custom \
    --val-frac 0.15
```

`0_mission_recorder.py` guarda por frame, además de GPS/velocidad/batería,
`rpms` y `gyros` (telemetría cruda de `telem.raw`, la misma que usa
`odometry.py`) y un tag `source` (`"manual"` por default) — útil para
priorizar candidatos (rueda patinando o giro brusco) y para distinguir estas
corridas si el día de mañana las mezclás con las de `bridge.py --debug-dir`.

⚠️ Antes de etiquetar un lote grande, probá **un** par contra el loader real
de entrenamiento (nota de formato en el docstring de `3_armar_dataset.py`)
para confirmar el formato de máscara exacto que espera. Es más barato
confirmar esto con 1 imagen que reetiquetar 500.

## Entrenar

Una vez armado `training/FOLD_custom/`, apuntá `dataset.img_folder` /
`dataset.gt_folder` del yaml de entrenamiento (repo de navegación,
`genie/sam2/configs/sam2.1_training_tiny/`) a las carpetas `train/`
generadas, y arrancá desde `checkpoint_finetuned_v2.pt`
(`sanatem/samtp-mini-traversability`) en vez de desde cero. Referencia de
hiperparámetros que produjeron ese checkpoint: resolución 1024, batch 8,
AdamW, `base_lr 5e-6`, `vision_lr 3e-6`, 5 épocas.
