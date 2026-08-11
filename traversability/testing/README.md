# testing/ — captura, testing y optimización offline de `PolicyConfig`

Tres programas independientes, pensados para correr en este orden:

```
capture_test.py  →  dataset/session01/*.jpg + *.json   (en vivo, dry-run)
policy_test.py   →  results/<config>/  (overlays, results.csv, summary.json) (offline)
policy_tuner.py  →  tuning_results/  (búsqueda progresiva + ranking + best_config.json)
```

Todo importa `rover_traversability` tal cual está — no se reimplementa
`suggest_command()`, `PolicyConfig`, `to_rgb()` ni `TraversabilityPredictor`.
Instalar primero el paquete principal (`pip install -e ./traversability`),
`testing/` vive como paquete hermano.

## 1. `capture_test.py` — captura segura

```bash
python -m testing.capture_test --save-dir dataset/session01 --interval 30
```

- Solo llama `RoverClient.get_front_frame_b64()` y `get_data()` (GET). **Nunca**
  llama `send_command()` ni `/control`.
- `--with-policy` (opcional) también corre SAM-TP + `suggest_command()` por
  frame, pero solo para guardarlo como referencia en el JSON — nunca se
  envía. Si no hay torch/checkpoint disponibles, se degrada solo (sigue
  capturando sin esa info).
- Cada frame: `frame_00007_1723399201.jpg` + `.json` con timestamp, nombre,
  resolución, telemetría de `/data`, y (si `--with-policy`) la decisión de
  referencia con `corridor_scores`/`best_corridor`/`linear`/`angular`/`reason`.

## 2. `policy_test.py` — evaluación offline de una o varias configs

```bash
python -m testing.policy_test --images dataset/session01 --out results --configs configs.json
```

`configs.json` (lista de configs a comparar):

```json
[
  {"name": "config_001", "params": {}},
  {"name": "config_002", "params": {"roi_top": 0.6, "k_angular": 0.9}}
]
```

**Por qué es rápido probar muchas configs:** SAM-TP corre una sola vez por
imagen (cacheado en `dataset/session01/.mask_cache/*.npy`, por hash de
contenido). `suggest_command()` es numpy puro — correrlo 50 veces sobre la
misma máscara con distintos `PolicyConfig` es prácticamente gratis. Por eso
las etapas siguientes (el afinador) no vuelven a tocar el modelo.

Salida por config: `config.json`, `results.csv` (una fila por frame),
`summary.json` (métricas agregadas) y un overlay `.jpg` por frame con ROI,
grilla de corredores, scores, corredor elegido y los parámetros clave de la
config, todo dibujado sobre la imagen. `results/summary.csv` junta un resumen
de todas las configs evaluadas en esa corrida.

## 3. `policy_tuner.py` — búsqueda progresiva

```bash
python -m testing.policy_tuner --images dataset/session01 --out tuning_results
python -m testing.policy_tuner --images dataset/session01 --out tuning_results --labels dataset/session01/labels.csv
```

Sigue el orden que pediste: optimiza `roi_top` (barriendo su grilla, el resto
fijo en los defaults), fija el mejor valor, optimiza `drivable_thresh`, fija,
sigue con `stop_center_fraction` → `k_angular` → `max_linear`. Está en
`DEFAULT_STAGES` en `policy_tuner.py` — agregar o reordenar etapas (por
ejemplo para barrer también `bottom_weight` o `min_corridor_score`) es
agregar una tupla `(nombre_param, [valores])` a esa lista, no reescribir el
loop de búsqueda.

Al final: `tuning_results/best_config.json`, `all_trials_ranked.csv` (cada
trial de cada etapa, ordenado por score) y, por trial, la misma estructura de
`policy_test.py` (overlays incluidos) bajo `tuning_results/stage_<param>/`.

## Métricas calculables sin etiquetar nada a mano

Por frame: `reason` (forward/turning_to_corridor/blocked/no_data), `linear`,
`angular`, `stop`, `best_corridor`, `corridor_scores`.

Agregadas por config: % forward/turn/stop, % izquierda/derecha/recto,
velocidad lineal promedio, score promedio del corredor elegido (proxy de
"confianza"), y una tasa de oscilación (cambios de signo de `angular` entre
frames consecutivos — solo tiene sentido si el dataset es una sesión
cronológica, que es justo lo que arma `capture_test.py`).

Con esto se arma un **score heurístico sin ground truth**
(`score_summary()` en `policy_tuner.py`): 50% confianza del corredor elegido,
30% estabilidad (inverso de oscilación), 20% "actividad" que tiene su pico en
~65% forward y cae para ambos lados — así una config que nunca frena no gana
solo por eso, y una que frena todo el tiempo tampoco.

## Qué NO se puede saber sin etiquetar a mano

Nada en la máscara ni en la telemetría dice si **realmente** había un
obstáculo, ni si el lado elegido era el correcto. Eso exige mirar el frame
(o su overlay) y decidir. Para eso, `labels.csv` opcional:

```csv
frame,expected_reason,expected_side
frame_00007_1723399201,forward,straight
frame_00008_1723399231,blocked,
frame_00009_1723399261,turning_to_corridor,left
```

Si se pasa `--labels`, el score deja de ser el heurístico y pasa a ser
`safety_penalized_accuracy`: exactitud contra las etiquetas, pero un
"debía frenar y no frenó" (`unsafe_misses`) resta el doble que un "frenó
pudiendo avanzar" (`overcautious_misses`) — para no premiar nunca una
config más rápida a costa de ser menos segura, como pediste.

## Reproducibilidad

Cada carpeta de resultados (`config_XXX/` o `stage_<param>/config_XXX/`)
tiene su `config.json` completo, `results.csv` por frame y `summary.json`
agregado — alcanza para reproducir exactamente esa corrida sobre el mismo
`dataset/`.
