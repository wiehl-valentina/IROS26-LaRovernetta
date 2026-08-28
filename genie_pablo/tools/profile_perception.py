"""Mide cuanto tarda cada etapa del pipeline de percepcion.

Antes de optimizar conviene saber donde se va el tiempo. Este script separa:
  - rectificacion (corregir la distorsion del lente)
  - inferencia de SAM-TP
  - proyeccion a BEV

    python tools/profile_perception.py --config configs/frodobot_rover.yaml --image test_1.jpg
    python tools/profile_perception.py --config configs/frodobot_rover.yaml --image test_1.jpg --downscale 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genie_path_planner.projection import project_score_to_bev  # noqa: E402
from genie_rover.perception import PerceptionPipeline  # noqa: E402


def cronometrar(fn, n=10):
    fn()  # descartar la primera: incluye compilacion de kernels
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
    return (time.perf_counter() - t0) / n, r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--downscale", type=int, default=1,
                    help="factor de reduccion antes de proyectar (1 = sin cambios)")
    ap.add_argument("--repeats", type=int, default=10)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    rgb = np.asarray(Image.open(a.image).convert("RGB"), dtype=np.uint8)
    print(f"Imagen: {rgb.shape[1]}x{rgb.shape[0]}\n")

    pipe = PerceptionPipeline(cfg)
    rect = pipe._rectifier_for((rgb.shape[1], rgb.shape[0]))

    t_rect, rectificada = cronometrar(lambda: rect(rgb), a.repeats)
    t_sam, trav = cronometrar(lambda: pipe.runner.traversability(rectificada), a.repeats)

    k = rect.k
    score = trav
    if a.downscale > 1:
        d = a.downscale
        h2, w2 = trav.shape[0] // d, trav.shape[1] // d
        score = cv2.resize(trav, (w2, h2), interpolation=cv2.INTER_AREA)
        k = k.copy()
        k[0, 0] /= d; k[0, 2] /= d
        k[1, 1] /= d; k[1, 2] /= d
        print(f"Submuestreo /{d}: mapa de scores {trav.shape[1]}x{trav.shape[0]}"
              f" -> {w2}x{h2}\n")

    def proyectar():
        return project_score_to_bev(
            score_map=score,
            camera_k=k,
            camera_pose=pipe.camera_pose,
            ground_z=pipe.ground_z,
            bev_resolution_m_per_px=pipe.resolution,
            bev_forward_range_m=pipe.forward_range,
            bev_side_range_m=pipe.side_range,
            max_ray_distance_m=pipe.max_ray,
        )

    t_proj, (bev, obs, stats) = cronometrar(proyectar, a.repeats)

    total = t_rect + t_sam + t_proj
    print(f"{'etapa':<22} {'ms':>8}  {'%':>6}")
    print("-" * 40)
    for nombre, t in [("rectificacion", t_rect), ("SAM-TP", t_sam), ("proyeccion BEV", t_proj)]:
        print(f"{nombre:<22} {t*1000:>8.1f}  {t/total*100:>5.1f}%")
    print("-" * 40)
    print(f"{'TOTAL':<22} {total*1000:>8.1f}   ({1/total:.1f} Hz)")

    print(f"\nCeldas BEV observadas: {stats['bev_observed_cells']:.0f} de {bev.size}")

    if a.downscale == 1:
        print("\nProba --downscale 2 y compara. La proyeccion recorre cada pixel,")
        print("asi que a la mitad de resolucion deberia costar ~4 veces menos.")
        print("El BEV tiene celdas de 3 cm: la resolucion completa se desperdicia.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
