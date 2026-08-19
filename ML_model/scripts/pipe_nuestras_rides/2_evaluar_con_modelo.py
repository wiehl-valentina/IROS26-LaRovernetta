"""Paso 2 (NUEVO) del pipeline de datasets propios: correr el modelo YA
entrenado sobre los frames candidatos, para priorizar cuales etiquetar
primero -- los que el modelo predice con mas incertidumbre son justo los
que mas van a mejorar el proximo fine-tune.

Esto es lo que reusa genie/sam2: `rover_traversability.TraversabilityPredictor`
es un wrapper sobre el SAM-TP vendoreado en genie/, cargando el checkpoint
fine-tuneado (`checkpoint_finetuned_v2.pt` via HF, o el que le indiques).

Requiere el repo de navegacion accesible -- ver ../../config_paths.py y
../../.env.example ("Reusar genie y sam2" en el README).

Uso:
    python 2_evaluar_con_modelo.py \
        --frames-dir ../../data/candidatos \
        --out ../../data/candidatos/model_eval.jsonl \
        --overlays-dir ../../data/candidatos/overlays

Con --overlays-dir podés mirar las mascaras verde/rojo antes de etiquetar,
para confirmar a ojo que la incertidumbre alta tiene sentido.

Despues de esto: ordená model_eval.jsonl por 'incertidumbre' descendente y
etiquetá primero esos. Siguiente paso del pipeline: 3_armar_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raiz de ML_model
import config_paths  # noqa: E402  (agrega genie/traversability al sys.path si esta en .env)


def _puntaje_incertidumbre(mask) -> float:
    """0 = el modelo esta seguro en todos lados (todo cerca de 0 o 1).
    1 = el modelo esta confundido en todos lados (todo cerca de 0.5).
    Frames con puntaje alto son buenos candidatos a etiquetar: son
    justamente donde el modelo no sabe que decir."""
    import numpy as np
    incertidumbre_por_pixel = 1.0 - 2.0 * np.abs(mask - 0.5)  # pico en mask=0.5
    return float(incertidumbre_por_pixel.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", required=True, help="carpeta con *_rgb.jpg")
    ap.add_argument("--out", required=True, help="jsonl de salida con el puntaje por frame")
    ap.add_argument("--overlays-dir", default=None,
                    help="si se pasa, guarda ahi el overlay verde/rojo de cada frame")
    args = ap.parse_args()

    try:
        from rover_traversability import TraversabilityPredictor
    except ImportError as exc:
        print(f"[evaluar] no pude importar rover_traversability ({exc}).")
        print("Revisá que el repo de navegacion este instalado (pip install -e ...) "
              "o que NAV_REPO_PATH este seteado en .env -- ver config_paths.py.")
        return 1

    frames_dir = Path(args.frames_dir)
    frames = sorted(frames_dir.glob("*_rgb.jpg"))
    if not frames:
        print(f"[evaluar] no encontre *_rgb.jpg en {frames_dir}")
        return 1

    print("[evaluar] cargando el modelo (el primer frame puede tardar)...")
    predictor = TraversabilityPredictor()

    if args.overlays_dir:
        Path(args.overlays_dir).mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, frame_path in enumerate(frames):
            result = predictor.predict(str(frame_path))
            score = _puntaje_incertidumbre(result.mask)
            row = {
                "frame_file": frame_path.name,
                "incertidumbre": round(score, 4),
                "drivable_frac": round(float(result.mask.mean()), 4),
                "inference_s": round(result.inference_s, 3),
            }
            f.write(json.dumps(row) + "\n")

            if args.overlays_dir:
                from PIL import Image
                Image.fromarray(result.overlay).save(
                    Path(args.overlays_dir) / frame_path.name.replace("_rgb.jpg", "_overlay.png"))

            if (i + 1) % 20 == 0:
                print(f"[evaluar] {i + 1}/{len(frames)} frames evaluados")

    print(f"\n[evaluar] listo -> {out_path}")
    print("Ordená por 'incertidumbre' descendente para saber que etiquetar primero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
