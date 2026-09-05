"""Paso 2c (extra) del pipeline de datasets propios: evaluar el pipeline de
percepcion COMPLETO -- SAM-TP + proyeccion a BEV + planner -- sobre una
carpeta de frames, en loop, igual que hacen a mano las herramientas de
testing de genie (`genie_rover.perception --image ...` /
`genie_path_planner.run_image_path_planner`) pero frame por frame sobre
--frames-dir, como ya hace `2_evaluar_con_modelo.py` con SAM-TP solo.

Diferencia con 2_evaluar_con_modelo.py: ese usa `SamTpRunner.traversability`
(solo la mascara en espacio de imagen, para dataset/incertidumbre). Este usa
`PerceptionPipeline.process()` (rectificacion + SAM-TP + proyeccion BEV con
calibracion de camara) y ademas corre `plan_on_bev` -- exactamente lo que
`bridge.py` hace en cada iteracion real -- asi podes ver donde se rompe la
cadena aunque la mascara de SAM-TP se vea perfecta (caso tipico: pasto que
en imagen separa bien pero en el BEV proyectado o en el planner arma cero
caminos validos).

Por cada frame guarda:
  - una fila en el jsonl de salida con metricas de percepcion Y de planner
    (candidatos / filtrados / seleccionados / costo min-mean), para poder
    ordenar y encontrar los frames donde el planner se queda sin caminos
    validos aunque la percepcion este bien;
  - un PNG de 4 paneles en --debug-dir: RGB | SAM-TP (heatmap en espacio de
    imagen) | BEV (transitabilidad proyectada, verde=transitable) | plan
    (visualizacion que ya trae plan_on_bev, con el mapa de costo, los
    caminos candidatos/filtrados y el elegido).

Requiere el repo de navegacion accesible (ver ../../config_paths.py y
../../.env.example, igual que 2_evaluar_con_modelo.py) y --bridge-config
apuntando al yaml COMPLETO del bridge (no solo la seccion samtp: tambien
necesita camera:, projection: y planner:, ej. genie/configs/frodobot_rover.yaml).

Uso:
    python 2c_evaluar_perception_bev.py \
        --frames-dir ../../data/candidatos_mexico \
        --out ../../data/candidatos_mexico/perception_bev_eval.jsonl \
        --debug-dir ../../data/candidatos_mexico/debug_bev \
        --bridge-config /ruta/al/repo/genie/configs/frodobot_rover.yaml \
        --checkpoint /ruta/al/checkpoint_2.pt \
        --samtp-config /ruta/al/sam2.1_custom2.yaml

--checkpoint / --samtp-config son opcionales -- si no se pasan, se usan
samtp.checkpoint_path / samtp.config_path del --bridge-config (mismo
esquema que 2_evaluar_con_modelo.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raiz de ML_model
import config_paths  # noqa: E402  (agrega genie/ al sys.path si esta en .env)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm esta en requirements.txt
    def tqdm(it, **kwargs):
        return it


def _resolver_ruta(valor: str, base_dir: Path, nav_repo_root: Path | None) -> Path:
    """Mismo esquema que 2_evaluar_con_modelo.py: primero relativo a la raiz
    del repo de navegacion (carpeta genie/), despues relativo al propio yaml."""
    p = Path(valor).expanduser()
    if p.is_absolute():
        return p
    if nav_repo_root is not None:
        candidata = nav_repo_root / p
        if candidata.exists():
            return candidata.resolve()
    return (base_dir / p).resolve()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", required=True, help="carpeta con *_rgb.jpg")
    ap.add_argument("--out", required=True, help="jsonl de salida con metricas por frame")
    ap.add_argument("--debug-dir", required=True,
                    help="carpeta donde guardar el panel de 4 imagenes por frame")
    ap.add_argument("--bridge-config", required=True,
                    help="yaml COMPLETO del bridge: samtp:, camera:, projection: y "
                         "planner: (ej. genie/configs/frodobot_rover.yaml) -- a "
                         "diferencia de 2_evaluar_con_modelo.py, aca no alcanza con "
                         "la seccion samtp: sola, porque tambien se proyecta a BEV "
                         "y se corre el planner.")
    ap.add_argument("--checkpoint", default=None,
                    help="pisa samtp.checkpoint_path del yaml (igual que en "
                         "2_evaluar_con_modelo.py, por si el yaml no resuelve bien "
                         "la ruta o queres probar un checkpoint puntual)")
    ap.add_argument("--samtp-config", default=None,
                    help="pisa samtp.config_path del yaml")
    ap.add_argument("--goal-x-m", type=float, default=0.0,
                    help="meta lateral en metros (0 = derecho adelante)")
    ap.add_argument("--goal-y-m", type=float, default=None,
                    help="meta hacia adelante en metros; default: "
                         "projection.forward_range_m del config")
    args = ap.parse_args()

    try:
        from genie_rover.perception import PerceptionPipeline  # noqa: F401
        from genie_path_planner.planner import plan_on_bev
        from genie_path_planner.pipeline import planner_config_from_dict
    except ImportError as exc:
        print(f"[eval-bev] no pude importar genie_rover/genie_path_planner ({exc}).")
        print("Revisa que NAV_REPO_PATH apunte al repo de navegacion (el que tiene "
              "genie/ adentro, con genie_rover/ y genie_path_planner/) -- ver "
              "config_paths.py / .env.")
        return 1

    frames_dir = Path(args.frames_dir)
    frames = sorted(frames_dir.glob("*_rgb.jpg"))
    if not frames:
        print(f"[eval-bev] no encontre *_rgb.jpg en {frames_dir}")
        return 1

    cfg_path = Path(args.bridge_config).expanduser().resolve()
    if not cfg_path.is_file():
        print(f"[eval-bev] --bridge-config {cfg_path} no existe.")
        return 1

    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    for seccion in ("samtp", "camera", "projection", "planner"):
        if seccion not in cfg:
            print(f"[eval-bev] al yaml le falta la seccion '{seccion}:' -- "
                  "necesitas el config COMPLETO del bridge, no solo samtp:.")
            return 1

    # Mismo esquema de resolucion de rutas que 2_evaluar_con_modelo.py: el
    # checkpoint/config de SAM-TP dentro de samtp: son relativos a la carpeta
    # genie/ del repo de navegacion, no a este script ni al yaml.
    env_nav_repo = os.environ.get("NAV_REPO_PATH")
    if env_nav_repo:
        nav_repo_root = Path(env_nav_repo).expanduser().resolve() / "genie"
    else:
        nav_repo_root = cfg_path.parents[1] if len(cfg_path.parents) >= 2 else None

    samtp_cfg = dict(cfg.get("samtp") or {})
    checkpoint_valor = args.checkpoint or samtp_cfg.get("checkpoint_path")
    samtp_config_valor = args.samtp_config or samtp_cfg.get("config_path")
    if not checkpoint_valor or not samtp_config_valor:
        print(f"[eval-bev] falta samtp.checkpoint_path o samtp.config_path -- ni en "
              f"{cfg_path} ni por --checkpoint/--samtp-config.")
        return 1
    samtp_cfg["checkpoint_path"] = str(
        _resolver_ruta(checkpoint_valor, cfg_path.parent, nav_repo_root))
    samtp_cfg["config_path"] = str(
        _resolver_ruta(samtp_config_valor, cfg_path.parent, nav_repo_root))
    cfg["samtp"] = samtp_cfg

    for etiqueta, valor in (("checkpoint", samtp_cfg["checkpoint_path"]),
                            ("config de SAM-TP", samtp_cfg["config_path"])):
        if not Path(valor).is_file():
            print(f"[eval-bev] no encuentro el {etiqueta} en {valor} -- revisa "
                  "NAV_REPO_PATH o las rutas declaradas en el yaml.")
            return 1

    origen = "--checkpoint explícito" if args.checkpoint else f"samtp.checkpoint_path de {cfg_path.name}"
    print("[eval-bev] cargando PerceptionPipeline completo "
          "(rectificacion + SAM-TP + proyeccion BEV)...")
    print(f"[eval-bev] checkpoint: {samtp_cfg['checkpoint_path']} ({origen})")
    print(f"[eval-bev] config SAM-TP: {samtp_cfg['config_path']}")
    pipe = PerceptionPipeline(cfg)
    planner_cfg = planner_config_from_dict(cfg)

    goal_x = float(args.goal_x_m)
    goal_y = (float(args.goal_y_m) if args.goal_y_m is not None
              else float(cfg["projection"]["forward_range_m"]))
    print(f"[eval-bev] meta del planner: x={goal_x:.2f} m, y={goal_y:.2f} m "
          "(relativa al robot, y=adelante)")

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con_plan = 0
    sin_plan = 0

    with open(out_path, "w") as f:
        for frame_path in tqdm(frames, desc="[eval-bev] evaluando frames", unit="frame"):
            rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)

            res = pipe.process(rgb)  # SAM-TP + rectificacion + proyeccion BEV
            planned = plan_on_bev(
                bev_traversability=res.traversability,
                observed_mask=res.observed,
                goal_x_m=goal_x,
                goal_y_m=goal_y,
                bev_resolution_m=pipe.resolution,
                config=planner_cfg,
            )

            meta = planned.metadata
            tiene_plan = meta.get("filtered_paths", 0) > 0
            con_plan += int(tiene_plan)
            sin_plan += int(not tiene_plan)

            row = {
                "frame_file": frame_path.name,
                "drivable_frac_imagen": round(float(res.image_traversability.mean()), 4),
                "bev_observed_cells": res.stats.get("bev_observed_cells"),
                "planner_candidatos": meta.get("candidate_paths"),
                "planner_filtrados": meta.get("filtered_paths"),
                "planner_seleccionados": meta.get("selected_paths"),
                "planner_costo_min": meta.get("cost_stats", {}).get("min"),
                "planner_costo_mean": meta.get("cost_stats", {}).get("mean"),
                "planner_status": meta.get("status"),
            }
            f.write(json.dumps(row) + "\n")

            vis_bev = np.where(res.traversability < 0, np.nan, res.traversability)
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(rgb)
            axes[0].set_title(frame_path.name)
            axes[1].imshow(res.image_traversability, cmap="jet", vmin=0, vmax=1)
            axes[1].set_title("SAM-TP (imagen, rojo=transitable)")
            axes[2].imshow(vis_bev, cmap="RdYlGn", vmin=0, vmax=1, origin="upper")
            axes[2].set_title("BEV (arriba = adelante)")
            axes[3].imshow(planned.visualization)
            estado = "OK" if tiene_plan else "SIN CAMINOS VALIDOS"
            axes[3].set_title(f"plan: {meta.get('filtered_paths')}/"
                              f"{meta.get('candidate_paths')} caminos -- {estado}")
            for a in axes:
                a.axis("off")
            fig.tight_layout()
            fig.savefig(debug_dir / frame_path.name.replace("_rgb.jpg", "_bev.png"), dpi=100)
            plt.close(fig)

    print(f"\n[eval-bev] listo -> {out_path}")
    print(f"[eval-bev] paneles -> {debug_dir}")
    print(f"[eval-bev] frames con plan valido: {con_plan} / sin plan valido: {sin_plan}")
    print("Ordena el jsonl por 'planner_filtrados' ascendente (o filtra ==0): esos son "
          "los frames donde el planner se queda sin caminos aunque SAM-TP este bien -- "
          "abri el panel _bev.png de esos frames primero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
