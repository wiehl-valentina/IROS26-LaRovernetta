#!/usr/bin/env python3
"""Convierte un archivo de poses exportado por RTAB-Map en un waypoints.yaml
que `WaypointRoute` (genie_rover/Indoor/mission.py) pueda leer directo con
`mission.search_mode: "waypoints"`.

De donde sale el archivo de entrada
------------------------------------
Abrí la sesión grabada en el viewer:

    rtabmap-databaseViewer ~/maps/sesion1.db

Menú **File -> Export poses...** -> elegí el formato **"RGBD-SLAM / TUM"**
(una línea por pose: `stamp x y z qx qy qz qw`, separado por espacios) y
guardalo, por ejemplo, en `~/maps/sesion1_poses.txt`.

Por qué hace falta downsamplear
--------------------------------
Una sesión de mapeo genera un nodo cada ~1s de manejo -- para un recorrido
de varios minutos son cientos de poses, demasiado pegadas entre sí para
pasárselas crudas a `WaypointRoute` (que además ya tiene su propio
`waypoint_reach_radius_m`). Este script se queda con un punto cada
`--min-dist-m` metros de distancia recorrida -- alcanza para describir el
trazado sin saturar la lista de waypoints.

Uso
---
    python poses_to_waypoints.py \\
        --poses ~/maps/sesion1_poses.txt \\
        --out configs/waypoints_sesion1.yaml \\
        --min-dist-m 0.5

El resultado es directamente lo que espera `mission.waypoints_path` en tu
config (ver configs/waypoints_example.yaml para el formato).

IMPORTANTE sobre el marco de coordenadas
-----------------------------------------
Estos waypoints quedan expresados en el marco **`map`** de la sesión de
RTAB-Map que grabaste (el origen es donde arrancó esa grabación, no donde
arranca indoor_bridge.py la próxima vez). Para que sean directamente
usables SIN medir nada a mano, indoor_bridge.py tiene que estar tomando su
pose del mismo marco -- es decir, corriendo con
`mapping.rtabmap_correction.enabled: true` Y con una sesión de RTAB-Map en
modo LOCALIZACIÓN (`localization:=true`) cargando el mismo `.db` de donde
salieron estas poses. Si NO hacés eso y dejás que indoor_bridge.py use solo
odometría de rueda+giro (arrancando siempre en 0,0,0 propio), estos
waypoints van a estar desalineados con la realidad tan pronto el punto de
arranque real no coincida con el origen de la grabación original.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml


def read_poses(path: Path) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        # formato RGBD-SLAM/TUM: stamp x y z qx qy qz qw
        # (x adelante, y izquierda -- misma convencion que odometry.py)
        try:
            x, y = float(parts[1]), float(parts[2])
        except ValueError:
            continue  # linea de header/comentario con texto, la saltamos
        pts.append((x, y))
    return pts


def downsample(points: list[tuple[float, float]], min_dist_m: float) -> list[tuple[float, float]]:
    if not points:
        return []
    kept = [points[0]]
    for x, y in points[1:]:
        lx, ly = kept[-1]
        if math.hypot(x - lx, y - ly) >= min_dist_m:
            kept.append((x, y))
    # aseguramos el ultimo punto real del recorrido: si el bucle de arriba
    # lo descarto por quedar muy cerca del anterior, lo agregamos igual --
    # si no, la ruta se corta antes de terminar el trayecto real.
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return kept


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--poses", required=True,
                     help="archivo exportado por RTAB-Map (File -> Export poses..., formato RGBD-SLAM/TUM)")
    ap.add_argument("--out", required=True,
                     help="yaml de salida, formato que espera WaypointRoute.from_file / mission.waypoints_path")
    ap.add_argument("--min-dist-m", type=float, default=0.5,
                     help="distancia minima entre waypoints consecutivos, en metros (default 0.5)")
    args = ap.parse_args()

    poses_path = Path(args.poses).expanduser()
    points = read_poses(poses_path)
    if not points:
        raise SystemExit(
            f"no se leyo ninguna pose de {poses_path} -- revisa que el export haya sido en "
            "formato RGBD-SLAM/TUM (stamp x y z qx qy qz qw)"
        )

    waypoints = downsample(points, args.min_dist_m)

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"waypoints": [{"x_m": round(x, 3), "y_m": round(y, 3)} for x, y in waypoints]}
    header = (
        f"# Generado automaticamente por poses_to_waypoints.py a partir de\n"
        f"# {poses_path}\n"
        f"# {len(points)} poses originales -> {len(waypoints)} waypoints "
        f"(min_dist_m={args.min_dist_m})\n"
        f"# Marco: 'map' de la sesion RTAB-Map de origen -- usar junto con\n"
        f"# mapping.rtabmap_correction.enabled: true + RTAB-Map en modo\n"
        f"# localizacion sobre el mismo .db (ver comentario en este script).\n"
    )
    out_path.write_text(header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    print(f"[poses_to_waypoints] {len(points)} poses -> {len(waypoints)} waypoints -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
