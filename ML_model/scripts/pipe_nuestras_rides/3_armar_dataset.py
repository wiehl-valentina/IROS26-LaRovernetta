"""Paso 3 del pipeline de datasets propios: convertir los pares ya
etiquetados (<nombre>_rgb.jpg + <nombre>_mask.png) al layout MOSE-style
(img_folder/gt_folder) que esperan los training configs de SAM2, en
genie/sam2/configs/sam2.1_training_tiny/*.yaml del repo de navegacion.

Uso:
    python 3_armar_dataset.py \
        --labeled ../../data/candidatos \
        --out ../../training/FOLD_custom \
        --val-frac 0.15

IMPORTANTE sobre el formato de mascara: este script asume PNG en escala de
grises, 255 = transitable / 0 = no transitable. El loader de SAM2 para video
(MOSE-style) a veces espera PNG con paleta en vez de escala de grises directa
-- antes de correr esto sobre un dataset grande, probá con UN par contra el
training config real y confirmá que el loader los lee bien.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeled", required=True, help="carpeta con *_rgb.jpg y *_mask.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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
        print(f"[armar] no encontre pares _rgb.jpg/_mask.png en {labeled_dir}")
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

    print(f"[armar] {len(pares)} pares etiquetados -> "
          f"{len(splits['train'])} train / {len(splits['val'])} val")
    print(f"  img_folder: {img_root}")
    print(f"  gt_folder:  {gt_root}")
    print("\nActualizá dataset.img_folder / dataset.gt_folder en el yaml de "
          "entrenamiento (repo de navegacion) para que apunten a las "
          "carpetas train/ de arriba, y arrancá desde "
          "checkpoint_finetuned_v2.pt en vez de desde cero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
