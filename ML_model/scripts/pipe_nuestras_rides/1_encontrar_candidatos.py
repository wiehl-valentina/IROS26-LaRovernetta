"""Paso 1 del pipeline de datasets propios: elegir que frames etiquetar.

No conviene etiquetar todo lo grabado por 0_mission_recorder.py -- conviene
elegir frames espaciados por distancia GPS real recorrida (no por indice ni
por tiempo), para no terminar etiquetando 50 fotos casi identicas del mismo
metro de vereda.

Uso:
    python 1_encontrar_candidatos.py \
        --runs ../../data/raw_runs/patio_20260819 ../../data/raw_runs/otra_corrida \
        --out ../../data/candidatos \
        --every-n-m 0.5 \
        --max-por-corrida 200

Siguiente paso: 2_evaluar_con_modelo.py (opcional, para priorizar por
incertidumbre del modelo) y despues etiquetar a mano, y 3_armar_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path


def _leer_manifest(run_dir: Path) -> list[dict]:
    manifest = run_dir / "manifest.jsonl"
    if not manifest.exists():
        print(f"[candidatos] {run_dir} no tiene manifest.jsonl, salteo")
        return []
    filas = []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                filas.append(json.loads(line))
    return filas


def _dist_m(a: dict, b: dict) -> float:
    ga, gb = a.get("gps"), b.get("gps")
    if not ga or not gb:
        return float("inf")  # sin GPS valido, mejor no descartar el frame
    R = 6378137.0
    dlat = math.radians(gb["lat"] - ga["lat"])
    dlon = math.radians(gb["lon"] - ga["lon"])
    x = dlon * R * math.cos(math.radians((gb["lat"] + ga["lat"]) / 2))
    y = dlat * R
    return math.hypot(x, y)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="carpetas creadas por 0_mission_recorder.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every-n-m", type=float, default=0.5,
                    help="distancia minima en metros entre candidatos elegidos")
    ap.add_argument("--max-por-corrida", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    for run_arg in args.runs:
        run_dir = Path(run_arg)
        filas = _leer_manifest(run_dir)
        if not filas:
            continue

        elegidos = []
        ultimo = None
        for fila in filas:
            if ultimo is None or _dist_m(ultimo, fila) >= args.every_n_m:
                elegidos.append(fila)
                ultimo = fila
            if args.max_por_corrida and len(elegidos) >= args.max_por_corrida:
                break

        print(f"[candidatos] {run_dir.name}: {len(filas)} frames -> "
              f"{len(elegidos)} candidatos (>= {args.every_n_m} m entre si)")

        frames_dir = run_dir / "frames"
        for fila in elegidos:
            src_img = frames_dir / fila["frame_file"]
            if not src_img.exists():
                continue
            dst_name = f"{run_dir.name}_{fila['index']:06d}"
            shutil.copy2(src_img, out_dir / f"{dst_name}_rgb.jpg")
            with open(out_dir / f"{dst_name}_meta.json", "w") as f:
                json.dump({**fila, "run": run_dir.name}, f)
            total += 1

    print(f"\n[candidatos] {total} candidatos copiados a {out_dir}")
    print("Siguiente paso (opcional): 2_evaluar_con_modelo.py, para priorizar "
          "por incertidumbre. O directamente etiquetar: por cada "
          "<nombre>_rgb.jpg generá <nombre>_mask.png.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
