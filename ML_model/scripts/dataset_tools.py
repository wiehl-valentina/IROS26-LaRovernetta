"""Herramientas para pasar de carpetas grabadas (mission_recorder.py) a un
dataset listo para etiquetar y despues entrenar.

Dos comandos:

    # 1. elegir candidatos a etiquetar (evita frames casi duplicados: filtra
    #    por distancia GPS real recorrida entre uno y el siguiente, no por
    #    indice ni por tiempo)
    python -m dataset_tools candidatos \
        --runs data/raw_runs/patio_20260819 data/raw_runs/otra_corrida \
        --out data/candidatos --every-n-m 0.5 --max-por-corrida 200

    # 2. una vez etiquetados (una mascara <nombre>_mask.png al lado de cada
    #    <nombre>_rgb.jpg elegido), armar el layout que espera el training
    #    config de SAM2 (img_folder/gt_folder, un "video" de 1 frame por
    #    imagen, ver genie/sam2/configs/sam2.1_training_tiny/*.yaml)
    python -m dataset_tools armar \
        --labeled data/candidatos --out training/FOLD_custom --val-frac 0.15

IMPORTANTE sobre el formato de mascara: este script asume PNG en escala de
grises, 255 = transitable / 0 = no transitable. El loader de SAM2 para video
(MOSE-style) a veces espera PNG con paleta en vez de escala de grises directa
-- antes de etiquetar en serio un dataset grande, corre una prueba con UN
par (imagen, mascara) contra el training config real y confirma que el
loader los lee bien. Mejor perder 5 minutos ahora que reetiquetar todo
despues.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path


# --------------------------------------------------------------- candidatos

def _leer_manifest(run_dir: Path) -> list[dict]:
    manifest = run_dir / "manifest.jsonl"
    if not manifest.exists():
        print(f"[dataset_tools] {run_dir} no tiene manifest.jsonl, salteo")
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


def cmd_candidatos(args) -> int:
    """Recorre cada corrida grabada y se queda con frames espaciados por
    distancia real, para no terminar etiquetando 50 fotos casi identicas
    del mismo metro de vereda."""
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

        print(f"[dataset_tools] {run_dir.name}: {len(filas)} frames -> "
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

    print(f"\n[dataset_tools] {total} candidatos copiados a {out_dir}")
    print("Etiquetalos ahora: por cada <nombre>_rgb.jpg genera una mascara "
          "<nombre>_mask.png (mismo tamaño, ver nota de formato arriba en "
          "el docstring) con labelme/CVAT, o corrigiendo a mano la "
          "prediccion que ya te da tu modelo actual.")
    return 0


# ------------------------------------------------------------------- armar

def cmd_armar(args) -> int:
    """Toma pares (<nombre>_rgb.jpg, <nombre>_mask.png) ya etiquetados y
    arma el layout MOSE-style (img_folder/<split>/<id>/00000.jpg,
    gt_folder/<split>/<id>/00000.png) que esperan los configs de
    entrenamiento en genie/sam2/configs/."""
    labeled_dir = Path(args.labeled)
    out_dir = Path(args.out)
    img_root = out_dir / "img_folder"
    gt_root = out_dir / "gt_folder"

    pares = []
    for rgb_path in sorted(labeled_dir.glob("*_rgb.jpg")):
        nombre = rgb_path.name[: -len("_rgb.jpg")]
        mask_path = labeled_dir / f"{nombre}_mask.png"
        if not mask_path.exists():
            continue  # todavia no etiquetado, se saltea sin error
        pares.append((nombre, rgb_path, mask_path))

    if not pares:
        print(f"[dataset_tools] no encontre pares _rgb.jpg/_mask.png en {labeled_dir}")
        print("(¿ya etiquetaste algo? el nombre de la mascara tiene que ser "
              "igual al de la imagen, cambiando _rgb.jpg por _mask.png)")
        return 1

    random.seed(args.seed)
    random.shuffle(pares)
    n_val = max(1, int(len(pares) * args.val_frac)) if len(pares) > 1 else 0
    splits = {"val": pares[:n_val], "train": pares[n_val:]}

    for split, items in splits.items():
        for nombre, rgb_path, mask_path in items:
            # cada frame es su propio "video" de 1 cuadro (num_frames: 1 en
            # el scratch del config), asi que no hace falta reconstruir
            # secuencias reales de video.
            video_dir_img = img_root / split / nombre
            video_dir_gt = gt_root / split / nombre
            video_dir_img.mkdir(parents=True, exist_ok=True)
            video_dir_gt.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rgb_path, video_dir_img / "00000.jpg")
            shutil.copy2(mask_path, video_dir_gt / "00000.png")

    print(f"[dataset_tools] {len(pares)} pares etiquetados -> "
          f"{len(splits['train'])} train / {len(splits['val'])} val")
    print(f"  img_folder: {img_root}")
    print(f"  gt_folder:  {gt_root}")
    print("\nActualiza dataset.img_folder / dataset.gt_folder en el yaml de "
          "entrenamiento para que apunten a las carpetas de arriba (train/).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("candidatos", help="elegir frames a etiquetar, espaciados por distancia")
    p.add_argument("--runs", nargs="+", required=True, help="carpetas creadas por mission_recorder.py")
    p.add_argument("--out", required=True)
    p.add_argument("--every-n-m", type=float, default=0.5,
                   help="distancia minima en metros entre candidatos elegidos")
    p.add_argument("--max-por-corrida", type=int, default=None)
    p.set_defaults(fn=cmd_candidatos)

    p = sub.add_parser("armar", help="convertir pares etiquetados al layout de entrenamiento")
    p.add_argument("--labeled", required=True, help="carpeta con *_rgb.jpg y *_mask.png")
    p.add_argument("--out", required=True)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(fn=cmd_armar)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
