# ML_model — SAM-TP Earth Rover

Pipelines de datos y fine-tuning para **SAM-TP**, el modelo de *traversability*
(qué parte de la imagen es piso transitable vs. obstáculo) que usa el rover.
Este repo **no entrena el rover en sí** ni contiene el código de inferencia:
arma los *datasets* de entrenamiento (a partir de datos públicos o de rides
propias) y documenta el flujo de fine-tuning sobre el checkpoint actual.

> Ver `DOCUMENTACION_TECNICA.md` para el detalle de cada script, comandos
> completos y la lista de pendientes/mejoras.

## Índice

- [Qué hace este repo](#qué-hace-este-repo)
- [Cómo se relaciona con el repo de navegación](#cómo-se-relaciona-con-el-repo-de-navegación)
- [Instalación](#instalación)
- [Estructura de carpetas](#estructura-de-carpetas)
- [Pipeline 1 — datasets públicos (Hugging Face)](#pipeline-1--datasets-públicos-hugging-face)
- [Pipeline 2 — datasets propios (rover real)](#pipeline-2--datasets-propios-rover-real)
- [Fine-tuning](#fine-tuning)
- [Validación antes de subir a producción](#validación-antes-de-subir-a-producción)
- [Shippear el checkpoint](#shippear-el-checkpoint)
- [Licencias](#licencias)

## Qué hace este repo

Dos pipelines conviven acá, y los dos terminan armando el mismo tipo de
dataset (`img_folder` / `gt_folder`, estilo MOSE) para fine-tunear SAM-TP:

1. **`scripts/pipe_videos_online/`** — datos públicos de FrodoBots bajados de
   Hugging Face. Selección y curación de rides ya grabados, sin filmar nada
   propio.
2. **`scripts/pipe_nuestras_rides/`** — datos propios: grabás manejando el
   rover, evaluás con el modelo actual para priorizar qué etiquetar, y armás
   el dataset de entrenamiento con eso.

El entrenamiento en sí (loop de SAM2) **no vive en este repo**: se corre
clonando `facebookresearch/sam2` aparte y usando los configs que sí están acá
(`genie/sam2/configs/...`, dentro del repo de navegación).

## Cómo se relaciona con el repo de navegación

Este repo **no vendorea** una copia de `genie/` (paquete `sam2`) ni de
`traversability/` (paquete `rover_traversability`) — los reusa desde el repo
de navegación, que se mantiene separado a propósito. Elegí una de estas dos
formas:

**Opción A (recomendada) — instalar como paquete editable, una vez:**

```bash
pip install --no-build-isolation -e /ruta/a/tu/repo/de/navegacion/genie
pip install -e '/ruta/a/tu/repo/de/navegacion/traversability[hf]'
```

Con esto `import sam2` y `from rover_traversability import ...` funcionan en
cualquier lugar de este repo sin nada más.

**Opción B — sin instalar, para notebooks o pruebas sueltas:**

```bash
cp .env.example .env
# editá .env: NAV_REPO_PATH=/ruta/a/tu/repo/de/navegacion
```

Y en el script/notebook, **antes** de importar `sam2` o `rover_traversability`:

```python
import config_paths  # agrega genie/ y traversability/ al sys.path
from rover_traversability import TraversabilityPredictor
```

`2_evaluar_con_modelo.py` ya usa esto — hoy es el único script del repo que
necesita el modelo cargado.

## Instalación

```bash
pip install -r requirements.txt
```

Requiere Python con soporte para `datasets`, `huggingface_hub`, `pandas`,
`numpy`, `pyarrow`, `opencv-python`, `Pillow`, `tqdm`, `PyYAML`,
`matplotlib` y, opcionalmente, Jupyter (ver `requirements.txt`).

## Estructura de carpetas

```
ML_model/
├── README.md
├── DOCUMENTACION_TECNICA.md      # detalle de scripts, comandos y pendientes
├── requirements.txt
├── .gitignore
├── .env.example                  # plantilla para NAV_REPO_PATH
├── config_paths.py               # reusa genie/sam2/traversability sin vendorearlos
│
├── scripts/
│   ├── pipe_videos_online/       # datasets públicos (HF)
│   │   ├── 0_check_columns.py
│   │   ├── 1_descarga_filtrado.py
│   │   └── Tool_1_metadata_descarga.py
│   │
│   └── pipe_nuestras_rides/      # datasets propios (rover real)
│       ├── rover_client.py
│       ├── 0_mission_recorder.py
│       ├── 1_encontrar_candidatos.py
│       ├── 2_evaluar_con_modelo.py
│       └── 3_armar_dataset.py
│
├── notebooks/
│   └── Analisis_dataset_pablo.ipynb
│
├── data/                          # gitignored, se genera corriendo los scripts
│   ├── raw_runs/
│   └── candidatos/
│
└── training/                      # gitignored, salida del armado de dataset
    └── FOLD_custom/
        ├── img_folder/{train,val}/<nombre>/00000.jpg
        └── gt_folder/{train,val}/<nombre>/00000.png
```

> ⚠️ Si tenés `dataset_tools.py` suelto en `scripts/` (versión vieja, con
> subcomandos `candidatos`/`armar`), reemplazalo por los tres archivos
> numerados de `pipe_nuestras_rides/` — hacen lo mismo pero separados en
> pasos, y `2_evaluar_con_modelo.py` es nuevo. Confirmá también que
> `rover_client.py` esté en esa misma carpeta: `0_mission_recorder.py` lo
> importa directo (`from rover_client import ...`).

## Pipeline 1 — datasets públicos (Hugging Face)

Dos datasets distintos, cada uno con su propio flujo.

### `BitRobot/FrodoBots-Mini-4K`

```bash
cd scripts/pipe_videos_online
python 0_check_columns.py
python 1_descarga_filtrado.py select   --out selected_rides.csv --max-per-country 30 --require-rear
python 1_descarga_filtrado.py fetch    --rides selected_rides.csv --raw-dir raw --skip-recordings
python 1_descarga_filtrado.py clean    --raw-dir raw --clean-dir clean --index cleaned_index.csv
python 1_descarga_filtrado.py curate   --index cleaned_index.csv --hours 300 --out curated_rides.csv
```

`clean` sincroniza cada frame con GPS + control + IMU (tolerancia 500 ms) y
descarta los puntos GPS sin fix. `curate` deduplica por celda GPS (75 m) para
maximizar diversidad geográfica sin repetir el mismo tramo.

### `BitRobot/berkeley-frodobots-lerobot-7k`

Formato LeRobot (chunks de parquet + video). Por ahora solo hay descarga de
metadata liviana:

```bash
python Tool_1_metadata_descarga.py
```

Todavía exploratorio — no tiene un pipeline `select/fetch/clean/curate` como
Mini-4K (ver pendientes en `DOCUMENTACION_TECNICA.md`).

## Pipeline 2 — datasets propios (rover real)

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

# 2. Priorizar por incertidumbre del modelo actual
python 2_evaluar_con_modelo.py \
    --frames-dir ../../data/candidatos \
    --out ../../data/candidatos/model_eval.jsonl \
    --overlays-dir ../../data/candidatos/overlays
# ordená model_eval.jsonl por 'incertidumbre' descendente: eso es lo que
# más conviene etiquetar primero.

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
corridas de las de `bridge.py --debug-dir` si el día de mañana se mezclan.

⚠️ Antes de etiquetar un lote grande, probá **un** par contra el loader real
de entrenamiento (ver nota de formato en el docstring de `3_armar_dataset.py`)
para confirmar el formato exacto de máscara que espera. Es mucho más barato
confirmar esto con 1 imagen que reetiquetar 500.

## Fine-tuning

El código de entrenamiento no está vendorizado en `genie/` — hay que clonar
el repo de Meta aparte:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e ".[dev]"
```

1. Copiar `genie/sam2/configs/sam2.1_training_tiny/sam2.1_custom2.yaml` (del
   repo de navegación) a los configs de ese clone.
2. Editar:
   - `dataset.img_folder` / `dataset.gt_folder` → apuntar a
     `training/FOLD_custom/train/...`
   - el `checkpoint_path` de inicialización → `checkpoint_finetuned_v2.pt`
     (`ckpt_state_dict_keys: ['model']`), **nunca** arrancar desde cero.
3. Lanzar:

```bash
python training/train.py \
    -c configs/sam2.1_training_tiny/sam2.1_custom2.yaml \
    --use-cluster 0 --num-gpus 1
```

**Hiperparámetros de referencia** (los que produjeron el checkpoint actual):
resolución 1024, batch 8, AdamW, `base_lr 5e-6`, `vision_lr 3e-6`, 5 épocas,
1 frame por muestra, 1 objeto por frame.

Para un fine-tune chico (unos cientos de frames nuevos, no miles): mantené el
LR y bajá a 2-3 épocas — con pocos datos, más épocas es la receta directa
para *catastrophic forgetting*.

**Monitoreo con TensorBoard** (`log_freq: 10`, cada 10 steps):

```bash
tensorboard --logdir sam2_logs/
```

Mirá `loss_mask`, `loss_dice`, `loss_iou`:
- **Normal:** bajan las primeras épocas y se aplanan; con pocos datos y LR
  bajo la caída va a ser chica, no esperes un dropoff dramático.
- **Alerta:** loss en NaN/inf (LR muy alto o máscara corrupta en el batch), o
  la loss sube en vez de bajar desde el arranque (checkpoint de init mal o
  formato de máscara mal).

Los checkpoints quedan en `sam2_logs/<config>/checkpoints/` según
`save_freq` — quedate con el de la última época salvo que veas la loss
divergir antes.

**Requisitos de hardware:** entrenamiento a resolución 1024 necesita una GPU
real (no alcanza con laptop); unas horas en una A100/4090-class card alcanzan
para un fine-tune chico. Opciones: cluster universitario, Colab Pro,
Lightning/HF cloud GPUs.

## Validación antes de subir a producción

La loss de training **no** dice si el modelo mejoró en la práctica:

```bash
SAMTP_CHECKPOINT=/ruta/al/nuevo.pt \
  python -m rover_traversability.demo predict frame_val.jpg --out check.png
```

1. Correr el checkpoint nuevo sobre los frames de `val/` separados en el
   armado del dataset (esos tienen máscara real, nunca los vio en training).
2. Comparar overlay viejo vs. nuevo, frame por frame — prestá especial
   atención a los frames que el modelo **viejo** ya acertaba: el fallo
   clásico de un fine-tune chico es mejorar en los frames nuevos pero
   empezar a fallar en escenas que antes resolvía bien.
3. Si querés un número: como ya hay máscara real por frame de `val/`, es
   trivial calcular IoU (intersección/unión entre `mask > 0.5` predicha y la
   máscara etiquetada) por frame y promediar. Esto **no viene armado en el
   repo** todavía (ver pendientes).
4. Recién ahí, subir de riesgo en orden: `demo live` (guarda overlays, no
   manda comandos) → `demo drive` en espacio abierto con el dedo en Ctrl-C →
   misión completa.

## Shippear el checkpoint

```bash
hf repo create yourteam/samtp-yourteam --repo-type model --private
hf upload yourteam/samtp-yourteam nuevo.pt checkpoint_finetuned_v2.pt
export SAMTP_HF_REPO=yourteam/samtp-yourteam
```

(Mantené el nombre de archivo o seteá `SAMTP_HF_FILENAME`.) Guardá el sha256
del checkpoint nuevo al lado de tus resultados — así "qué modelo era este
run" siempre se puede responder.

## Licencias

Los pesos base descienden de SAM 2.1 (Apache-2.0) y de metraje de FrodoBots
Mini (el dataset público Mini-4K es CC-BY-SA). Si publicás pesos
fine-tuneados, llevá la nota de procedencia y atribución de la model card
base (`sanatem/samtp-mini-traversability`) con ellos.
