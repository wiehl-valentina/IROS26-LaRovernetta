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

--guardar-mascaras (pre-anotacion automatica / human-in-the-loop):
    python 2_evaluar_con_modelo.py \
        --frames-dir ../../data/candidatos \
        --out ../../data/candidatos/model_eval.jsonl \
        --guardar-mascaras

Con este flag, junto a cada <id>_rgb.jpg en --frames-dir se guarda tambien
una <id>_mask.png con la prediccion del modelo ya binarizada (255 =
transitable, 0 = no transitable -- el mismo formato exacto que espera
3_armar_dataset.py). Asi el trabajo humano deja de ser "dibujar la mascara
desde cero" y pasa a ser "corregir donde el modelo se equivoco", con
2b_corregir_mascaras.py (o CVAT/Label Studio, ver el README) -- reduce el
tiempo por fotograma de minutos a segundos.

Por seguridad, si un <id>_mask.png ya existe (por ejemplo porque ya lo
corregiste a mano) el script NO lo pisa salvo que pases --forzar-mascaras.

Despues de esto: ordená model_eval.jsonl por 'incertidumbre' descendente y
etiquetá/corregí primero esos. Siguiente paso del pipeline: 3_armar_dataset.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # raiz de ML_model
import config_paths  # noqa: E402  (agrega genie/traversability al sys.path si esta en .env)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm esta en requirements.txt, pero
    # no queremos que el script se rompa si alguien lo corre sin instalar.
    def tqdm(it, **kwargs):
        return it


def _puntaje_incertidumbre(mask) -> float:
    """0 = el modelo esta seguro en todos lados (todo cerca de 0 o 1).
    1 = el modelo esta confundido en todos lados (todo cerca de 0.5).
    Frames con puntaje alto son buenos candidatos a etiquetar: son
    justamente donde el modelo no sabe que decir."""
    import numpy as np
    incertidumbre_por_pixel = 1.0 - 2.0 * np.abs(mask - 0.5)  # pico en mask=0.5
    return float(incertidumbre_por_pixel.mean())


def _guardar_mascara_binaria(mask, path: Path) -> None:
    """Guarda `mask` (float, 0..1) como PNG en escala de grises ESTRICTAMENTE
    binario: 255 = transitable, 0 = no transitable. Este es el formato que
    3_armar_dataset.py / el loader de SAM2 esperan -- no tocar sin revisar
    ese docstring."""
    import numpy as np
    from PIL import Image

    mask_bin = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask_bin, mode="L").save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", required=True, help="carpeta con *_rgb.jpg")
    ap.add_argument("--out", required=True, help="jsonl de salida con el puntaje por frame")
    ap.add_argument("--overlays-dir", default=None,
                    help="si se pasa, guarda ahi el overlay verde/rojo de cada frame")
    ap.add_argument("--guardar-mascaras", action="store_true",
                    help="pre-anotacion automatica: guarda <id>_mask.png binarizada "
                         "junto a cada _rgb.jpg en --frames-dir")
    ap.add_argument("--forzar-mascaras", action="store_true",
                    help="con --guardar-mascaras, sobrescribe <id>_mask.png aunque ya "
                         "exista (por default no se pisa una mascara ya corregida a mano)")
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

    mascaras_generadas = 0
    mascaras_saltadas = 0

    with open(out_path, "w") as f:
        for frame_path in tqdm(frames, desc="[evaluar] evaluando frames", unit="frame"):
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

            if args.guardar_mascaras:
                mask_path = frame_path.parent / frame_path.name.replace("_rgb.jpg", "_mask.png")
                if mask_path.exists() and not args.forzar_mascaras:
                    mascaras_saltadas += 1
                else:
                    _guardar_mascara_binaria(result.mask, mask_path)
                    mascaras_generadas += 1

    print(f"\n[evaluar] listo -> {out_path}")
    print("Ordená por 'incertidumbre' descendente para saber que etiquetar/corregir primero.")
    if args.guardar_mascaras:
        print(f"[evaluar] {mascaras_generadas} mascaras pre-generadas "
              f"({mascaras_saltadas} ya existian y no se tocaron -- usá --forzar-mascaras "
              "para pisarlas)")
        print("Siguiente paso: corregí los bordes donde el modelo se equivocó con "
              "2b_corregir_mascaras.py (o CVAT/Label Studio) y despues 3_armar_dataset.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
