"""Paso 1 del pipeline de datasets propios: elegir, de cada corrida grabada
por 0_mission_recorder.py, los frames candidatos a etiquetar -- espaciados
por distancia GPS real recorrida (no por indice ni por tiempo), para no
terminar etiquetando 50 fotos casi identicas del mismo metro de vereda.

Ademas de copiar los *_rgb.jpg + *_meta.json a --out, este script deja un
`manifest_candidatos.jsonl` en esa misma carpeta con un resumen de todo lo
seleccionado (de que corrida viene cada uno, sus coordenadas GPS exactas,
la distancia al candidato anterior de esa misma corrida, velocidad/bateria
en ese instante, etc.) -- pensado para poder auditar rapido la seleccion
sin abrir cada _meta.json suelto, y para que 3_armar_dataset.py (o vos a
mano) puedan cruzar facil que fue elegido y por que.

Uso:
    python 1_encontrar_candidatos.py \
        --runs ../../data/raw_runs/patio_20260819 ../../data/raw_runs/otra_corrida \
        --out ../../data/candidatos \
        --every-n-m 0.5 \
        --max-por-corrida 200

Siguiente paso: 2_evaluar_con_modelo.py (opcional, para priorizar por
incertidumbre del modelo, y para pre-generar mascaras con --guardar-mascaras)
y despues etiquetar/corregir a mano (ver 2b_corregir_mascaras.py), y por
ultimo 3_armar_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm esta en requirements.txt, pero
    # no queremos que el script se rompa si alguien lo corre sin instalar.
    def tqdm(it, **kwargs):
        return it


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


def _leer_manifest_candidatos_existente(path: Path) -> dict[str, dict]:
    """Carga el manifest_candidatos.jsonl si ya existia (de una corrida
    anterior de este script sobre el mismo --out), indexado por id, para
    poder actualizarlo sin perder ni duplicar entradas al re-correrlo."""
    if not path.exists():
        return {}
    filas = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fila = json.loads(line)
            filas[fila["id"]] = fila
    return filas


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
    manifest_candidatos_path = out_dir / "manifest_candidatos.jsonl"
    candidatos_por_id = _leer_manifest_candidatos_existente(manifest_candidatos_path)

    total = 0

    for run_arg in args.runs:
        run_dir = Path(run_arg)
        filas = _leer_manifest(run_dir)
        if not filas:
            continue

        elegidos = []
        distancias = []  # distancia al candidato anterior de la misma corrida
        ultimo = None
        for fila in filas:
            d = _dist_m(ultimo, fila) if ultimo is not None else None
            if ultimo is None or d >= args.every_n_m:
                elegidos.append(fila)
                distancias.append(d)
                ultimo = fila
            if args.max_por_corrida and len(elegidos) >= args.max_por_corrida:
                break

        print(f"[candidatos] {run_dir.name}: {len(filas)} frames -> "
              f"{len(elegidos)} candidatos (>= {args.every_n_m} m entre si)")

        frames_dir = run_dir / "frames"
        for fila, dist_al_anterior in tqdm(list(zip(elegidos, distancias)),
                                            desc=f"[candidatos] copiando {run_dir.name}",
                                            unit="frame"):
            src_img = frames_dir / fila["frame_file"]
            if not src_img.exists():
                continue
            dst_name = f"{run_dir.name}_{fila['index']:06d}"
            shutil.copy2(src_img, out_dir / f"{dst_name}_rgb.jpg")
            with open(out_dir / f"{dst_name}_meta.json", "w") as f:
                json.dump({**fila, "run": run_dir.name}, f)

            gps = fila.get("gps") or {}
            candidatos_por_id[dst_name] = {
                "id": dst_name,
                "rgb_file": f"{dst_name}_rgb.jpg",
                "run": run_dir.name,
                "index_en_corrida": fila.get("index"),
                "frame_timestamp": fila.get("frame_timestamp"),
                "gps_lat": gps.get("lat"),
                "gps_lon": gps.get("lon"),
                "gps_signal": gps.get("gps_signal"),
                "orientation": gps.get("orientation"),
                "dist_m_al_candidato_anterior": (
                    None if dist_al_anterior in (None, float("inf")) else round(dist_al_anterior, 3)
                ),
                "speed": fila.get("speed"),
                "battery": fila.get("battery"),
                "source": fila.get("source"),
                "note": fila.get("note"),
            }
            total += 1

    # Se reescribe entero (no se apendea) para que quede ordenado y sin
    # duplicados aunque se haya corrido el script varias veces sobre el
    # mismo --out con distintas corridas.
    with open(manifest_candidatos_path, "w") as f:
        for cid in sorted(candidatos_por_id):
            f.write(json.dumps(candidatos_por_id[cid]) + "\n")

    print(f"\n[candidatos] {total} candidatos copiados a {out_dir}")
    print(f"[candidatos] resumen -> {manifest_candidatos_path} "
          f"({len(candidatos_por_id)} candidatos en total en --out)")
    print("Siguiente paso (opcional): 2_evaluar_con_modelo.py, para priorizar "
          "por incertidumbre y/o pre-generar mascaras con --guardar-mascaras.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
