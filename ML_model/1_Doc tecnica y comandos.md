# Documentación técnica — ML_model (SAM-TP Earth Rover)

Complemento del `README.md`. Acá está el detalle de cada pieza del repo, el
cheatsheet de comandos, los formatos de datos y — la parte más importante
para vos ahora — **qué queda pendiente o a medio hacer**, recopilado de las
advertencias que ya están dejadas en `README_ML.md`, `FINETUNING_Santi.md` y
`Entrenamiento.md`.

> Nota: esta documentación describe lo que dicen los README/markdown del
> proyecto y los docstrings referenciados. No tengo acceso al código fuente
> de cada script (`.py`) en esta conversación, así que el comportamiento
> exacto línea por línea de cada uno no está verificado acá — donde algo es
> una inferencia razonable a partir de la documentación existente lo marco
> como tal.

## Índice

1. [Inventario de archivos y qué hace cada uno](#1-inventario-de-archivos-y-qué-hace-cada-uno)
2. [Variables de entorno](#2-variables-de-entorno)
3. [Formatos de datos](#3-formatos-de-datos)
4. [Cheatsheet de comandos](#4-cheatsheet-de-comandos)
5. [Pendientes y mejoras](#5-pendientes-y-mejoras)
6. [Riesgos / puntos de atención al operar](#6-riesgos--puntos-de-atención-al-operar)

---

## 1. Inventario de archivos y qué hace cada uno

### Raíz del repo

| Archivo | Función |
| --- | --- |
| `config_paths.py` | Al importarse, busca `NAV_REPO_PATH` (variable de entorno o `.env`) y agrega `genie/` y `traversability/` de ese repo externo al `sys.path`, para poder hacer `import sam2` / `from rover_traversability import ...` sin instalar nada. Si `NAV_REPO_PATH` no está seteado o la ruta no existe, imprime un aviso y devuelve `False` en vez de fallar. Se auto-ejecuta (`configurar()`) al hacer `import config_paths`. |
| `requirements.txt` | Dependencias: `datasets`, `huggingface_hub`, `pandas`, `numpy`, `pyarrow`, `opencv-python`, `Pillow`, `tqdm`, `PyYAML`, `matplotlib`, `jupyter`, `ipykernel`. No incluye `torch`/`sam2` — esos vienen del repo de navegación (opción A) o vía `config_paths.py` (opción B). |
| `.gitignore` | Ignora entornos virtuales (`.venv/`, `venv/`, `ENV/`), datos pesados (`data/raw/`, `data/frames/`, `data/masks/`, `frodobots_metadata/`), checkpoints y outputs, binarios (`*.tar`, `*.mp4`, `*.pt`, `*.pth`, `*.parquet`), y de forma más amplia `data/`, `checkpoints/`, `external/`, `__pycache__/`. |
| `.env.example` | Plantilla para `NAV_REPO_PATH` (mencionado en el README pero no incluido en los documentos vistos — confirmar que exista en el repo real). |

### `scripts/pipe_videos_online/` — datasets públicos de Hugging Face

| Archivo | Función |
| --- | --- |
| `0_check_columns.py` | Inspecciona las columnas de `metadata.parquet` del dataset `BitRobot/FrodoBots-Mini-4K`, primer paso exploratorio antes de filtrar. |
| `1_descarga_filtrado.py` | Script con subcomandos (`select`, `fetch`, `clean`, `curate`) para todo el flujo de Mini-4K: seleccionar rides por criterios (país, cámara trasera), descargarlas, sincronizar frame+GPS+control+IMU con tolerancia de 500 ms descartando GPS sin fix, y deduplicar por celda GPS de 75 m para maximizar diversidad geográfica. |
| `Tool_1_metadata_descarga.py` | Descarga la metadata liviana del dataset `BitRobot/berkeley-frodobots-lerobot-7k` (formato LeRobot: chunks de parquet + video). Es el único paso implementado hoy para ese dataset — no tiene `select/fetch/clean/curate`. |

### `scripts/pipe_nuestras_rides/` — datasets propios (rover real)

| Archivo | Función |
| --- | --- |
| `rover_client.py` | `RoverReader`: cliente de **solo lectura** del SDK del rover (no manda comandos de control). Es importado directo por `0_mission_recorder.py`, por eso tiene que vivir en la misma carpeta. |
| `0_mission_recorder.py` | Graba una corrida manual: frames a intervalo fijo (`--interval`) + GPS, velocidad, batería, `rpms` y `gyros` (telemetría cruda de `telem.raw`, la misma que usa `odometry.py`), y un tag `source` (default `"manual"`) para distinguir estas corridas de las generadas por `bridge.py --debug-dir`. Nunca manda comandos de control al rover. |
| `1_encontrar_candidatos.py` | Selecciona frames candidatos a etiquetar espaciándolos por distancia GPS real (`--every-n-m`), con un tope por corrida (`--max-por-corrida`), para no sobre-representar tramos donde el rover fue lento o se detuvo. |
| `2_evaluar_con_modelo.py` | Corre el modelo actual (`rover_traversability`, cargado vía `config_paths.py`) sobre los candidatos y genera `model_eval.jsonl` con una métrica de incertidumbre por frame, más overlays de la predicción. Ordenando por incertidumbre descendente se prioriza qué etiquetar primero. Es el único script del repo que hoy necesita el modelo cargado. |
| `3_armar_dataset.py` | Toma los pares etiquetados (`<id>_rgb.jpg` + `<id>_mask.png`, blanco=transitable/negro=no) y arma el layout `img_folder/{train,val}/<id>/00000.jpg` + `gt_folder/{train,val}/<id>/00000.png` (estilo MOSE, cada imagen es un "video" de 1 frame) que espera el training config de SAM2. `--val-frac` controla la separación train/val. Su docstring documenta el formato exacto de máscara esperado por el loader real — revisarlo antes de etiquetar en volumen. |

### `notebooks/`

| Archivo | Función |
| --- | --- |
| `Analisis_dataset_pablo.ipynb` | Exploración de `metadata.parquet` de Mini-4K. |

### Fuera del repo (repo de navegación, referenciado)

| Recurso | Dónde vive | Función |
| --- | --- | --- |
| `genie/sam2/configs/sam2.1_training_tiny/sam2.1_custom2.yaml` | repo de navegación | Config de entrenamiento para la arquitectura actual (backbone tiny + `want_custom_prompt_encoder: 2`), pensado para copiarse dentro del clone de `facebookresearch/sam2`. |
| `rover_traversability` (`traversability/`) | repo de navegación | Paquete de inferencia — expone `TraversabilityPredictor` y el `demo` CLI (`predict`, `live`, `drive`). |
| `sam2` (`genie/`) | repo de navegación | Paquete del modelo SAM2 usado en inferencia. |
| `checkpoint_finetuned_v2.pt` | HF Hub (`sanatem/samtp-mini-traversability`) | Checkpoint actual, punto de partida obligatorio para cualquier fine-tune (nunca arrancar desde cero). `ckpt_state_dict_keys: ['model']`. |
| `facebookresearch/sam2` (`training/`) | clon aparte, no vendorizado | Código real del training loop (`training/train.py`). |

---

## 2. Variables de entorno

| Variable | Usada por | Qué hace |
| --- | --- | --- |
| `NAV_REPO_PATH` | `config_paths.py` | Ruta al repo de navegación local; si falta, `import sam2` / `from rover_traversability import ...` no van a funcionar (opción B de instalación). |
| `SAMTP_CHECKPOINT` | `rover_traversability.demo` | Path al checkpoint a usar para `predict`/`live`/`drive`, sobreescribe el default. |
| `SAMTP_HF_REPO` | `rover_traversability` (carga desde HF) | Repo de Hugging Face desde el que se baja el checkpoint en producción. |
| `SAMTP_HF_FILENAME` | `rover_traversability` (carga desde HF) | Nombre de archivo del checkpoint dentro del repo HF, si no se mantiene `checkpoint_finetuned_v2.pt`. |

## 3. Formatos de datos

- **Máscaras de etiquetado:** PNG binario, mismo tamaño que el frame.
  `255` = piso transitable (blanco), `0` = no transitable (negro).
- **Pares etiquetados (entrada de `3_armar_dataset.py`):**
  `<id>_rgb.jpg` + `<id>_mask.png` en la misma carpeta.
- **Layout de entrenamiento (salida de `3_armar_dataset.py`, entrada de SAM2),
  estilo MOSE / PNG-VOS:**
  ```
  training/FOLD_custom/
    img_folder/{train,val}/<id>/00000.jpg
    gt_folder/{train,val}/<id>/00000.png
  ```
  Cada "video" es en realidad una carpeta de 1 solo frame.
- **Checkpoint:** `state_dict` completo, formato `{"model": state_dict}`,
  usable directo como init para fine-tuning (`ckpt_state_dict_keys: ['model']`
  en el config).

## 4. Cheatsheet de comandos

```bash
# --- instalación ---
pip install -r requirements.txt
pip install --no-build-isolation -e /ruta/genie
pip install -e '/ruta/traversability[hf]'

# --- pipeline 1: Mini-4K ---
python 0_check_columns.py
python 1_descarga_filtrado.py select --out selected_rides.csv --max-per-country 30 --require-rear
python 1_descarga_filtrado.py fetch  --rides selected_rides.csv --raw-dir raw --skip-recordings
python 1_descarga_filtrado.py clean  --raw-dir raw --clean-dir clean --index cleaned_index.csv
python 1_descarga_filtrado.py curate --index cleaned_index.csv --hours 300 --out curated_rides.csv

# --- pipeline 1: lerobot-7k ---
python Tool_1_metadata_descarga.py

# --- pipeline 2: rides propias ---
python 0_mission_recorder.py --out data/raw_runs/<nombre> --interval 0.5 --note "..."
python 1_encontrar_candidatos.py --runs data/raw_runs/<nombre> --out data/candidatos --every-n-m 0.5 --max-por-corrida 200
python 2_evaluar_con_modelo.py --frames-dir data/candidatos --out data/candidatos/model_eval.jsonl --overlays-dir data/candidatos/overlays
python 3_armar_dataset.py --labeled data/candidatos --out training/FOLD_custom --val-frac 0.15

# --- entrenamiento (en el clone de facebookresearch/sam2) ---
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e ".[dev]"
python training/train.py -c configs/sam2.1_training_tiny/sam2.1_custom2.yaml --use-cluster 0 --num-gpus 1
tensorboard --logdir sam2_logs/

# --- validación ---
SAMTP_CHECKPOINT=/ruta/al/nuevo.pt python -m rover_traversability.demo predict frame_val.jpg --out check.png
python -m rover_traversability.demo live
python -m rover_traversability.demo drive

# --- shipping ---
hf repo create yourteam/samtp-yourteam --repo-type model --private
hf upload yourteam/samtp-yourteam nuevo.pt checkpoint_finetuned_v2.pt
export SAMTP_HF_REPO=yourteam/samtp-yourteam
```

## 5. Pendientes y mejoras

Cosas explícitamente marcadas como "todavía no está" o "hay que armarlo" en
los documentos del repo:

- [ ] **Script de IoU sobre el val set.** Hoy la validación es "a ojo"
  (comparar overlays viejo vs. nuevo). El propio `Entrenamiento.md` dice que
  es un script de ~15 líneas: calcular intersección/unión entre
  `mask > 0.5` predicha y la máscara real por frame, y promediar. Es la
  mejora de mayor impacto/menor esfuerzo pendiente.
- [ ] **Pipeline `select/fetch/clean/curate` para `berkeley-frodobots-lerobot-7k`.**
  Por ahora solo existe `Tool_1_metadata_descarga.py` (metadata liviana).
  Si el dataset avanza, conviene darle la misma estructura de pasos
  numerados que tiene Mini-4K (p. ej. `scripts/pipe_videos_online/lerobot7k/`).
- [ ] **Limpieza de `dataset_tools.py` viejo.** Si todavía queda una versión
  vieja de este script suelta en `scripts/` (con subcomandos
  `candidatos`/`armar`), hay que borrarla/reemplazarla por los tres archivos
  numerados de `pipe_nuestras_rides/` — puede generar confusión sobre cuál
  es la fuente de verdad.
- [ ] **Confirmar `.env.example` existe y está actualizado** — el README lo
  referencia (`cp .env.example .env`) pero no forma parte de los documentos
  revisados acá; verificar que tenga el placeholder correcto de
  `NAV_REPO_PATH`.
- [ ] **Comparación de overlays lado a lado (viejo vs. nuevo) como
  herramienta**, no solo como paso manual — mencionado como algo que se
  podría armar además del script de IoU.
- [ ] **Trazabilidad de checkpoints:** el README pide guardar el sha256 del
  checkpoint nuevo "al lado de tus resultados", pero no hay un mecanismo
  (script o convención de archivo) para hacerlo automático todavía — hoy es
  un paso manual que depende de que alguien se acuerde de hacerlo.
- [ ] **Documentar `demo drive`** con más detalle en el README (hoy solo se
  menciona como el paso previo a "misión completa", con el dedo en Ctrl-C,
  pero no hay flags ni ejemplo de comando documentado).

## 6. Riesgos / puntos de atención al operar

- **Catastrophic forgetting** es el fallo más nombrado en la documentación:
  un fine-tune chico puede mejorar en los frames nuevos y empeorar en
  escenas que el modelo viejo ya resolvía bien. Por eso el chequeo de
  overlays "viejo vs. nuevo" pone el foco justo en los frames que el viejo
  ya acertaba, no solo en los nuevos.
- **Nunca arrancar el fine-tune desde cero** — siempre desde
  `checkpoint_finetuned_v2.pt` (o el checkpoint en producción vigente).
- **Confirmar el formato de máscara con 1 imagen antes de etiquetar en
  volumen.** Reetiquetar 500 imágenes por un formato mal interpretado es
  mucho más caro que probar un par contra el loader real primero.
- **`0_mission_recorder.py` y `rover_client.py` son de solo lectura** — no
  mandan comandos de control al rover; el primer paso "de riesgo" real es
  recién en la etapa de validación (`demo live` → `demo drive`).
- **GPU real es obligatoria para entrenar** (resolución 1024); no hay
  fallback documentado para CPU o GPUs chicas.
- **Licenciamiento:** cualquier checkpoint fine-tuneado que se publique
  tiene que llevar la nota de procedencia de SAM 2.1 (Apache-2.0) y de
  FrodoBots Mini (CC-BY-SA), heredada de la model card base
  (`sanatem/samtp-mini-traversability`).
