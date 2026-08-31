"""Corre SAM-TP sobre una o varias imagenes y guarda la mascara. Nada mas.

No proyecta a BEV, asi que NO necesita calibracion, ni altura de camara, ni
intrinsecos, ni que la resolucion coincida con nada. Sirve para contestar una
sola pregunta: el modelo reconoce el piso transitable en las imagenes de MI
robot?

    python tools/quick_mask.py mini_frame.jpg
    python tools/quick_mask.py foto1.jpg foto2.jpg foto3.jpg --out debug_mini/

Genera un PNG por imagen con tres paneles: la foto, el mapa de calor, y la
mascara superpuesta (que suele ser el mas facil de juzgar).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from genie_rover.perception import SamTpRunner  # noqa: E402

DEFAULT_CFG = "sam2/configs/sam2.1_inference_tiny/sam2.1_custom2.yaml"
DEFAULT_CKPT = ("sam2_logs/configs/sam2.1_training_tiny/"
                "sam2_training_custom2_freezeNoneNone_f57.yaml/checkpoints/checkpoint_2.pt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="una o mas imagenes")
    ap.add_argument("--out", default="debug_mini")
    ap.add_argument("--config", default=None,
                    help="yaml del que leer las rutas del modelo (opcional)")
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="umbral para la superposicion")
    args = ap.parse_args()

    cfg_path, ckpt_path = DEFAULT_CFG, DEFAULT_CKPT
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        cfg_path = cfg["samtp"]["config_path"]
        ckpt_path = cfg["samtp"]["checkpoint_path"]

    runner = SamTpRunner(config_path=cfg_path, checkpoint_path=ckpt_path)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in args.images:
        p = Path(img_path)
        if not p.is_file():
            print(f"  no existe: {p}")
            continue

        rgb = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        trav = runner.traversability(rgb)

        frac = float(np.mean(trav > args.thresh))

        # Superposicion: verde donde el modelo dice "transitable"
        overlay = rgb.astype(np.float32) / 255.0
        mask = trav > args.thresh
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array([0.15, 0.85, 0.35])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(rgb)
        axes[0].set_title(p.name)
        axes[1].imshow(trav, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title("SAM-TP  (rojo = transitable)")
        axes[2].imshow(overlay)
        axes[2].set_title(f"superpuesto  ({frac*100:.0f}% transitable)")
        for a in axes:
            a.axis("off")
        fig.tight_layout()

        dest = out_dir / f"{p.stem}_mask.png"
        fig.savefig(dest, dpi=100)
        plt.close(fig)
        print(f"  {p.name}  {rgb.shape[1]}x{rgb.shape[0]}  ->  {dest}   "
              f"({frac*100:.0f}% transitable)")

    print(f"\nListo. Mira sobre todo el panel de la derecha: el verde tiene que "
          f"cubrir el piso por donde el robot podria pasar, y nada mas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
