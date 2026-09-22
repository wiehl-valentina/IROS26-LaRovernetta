#!/usr/bin/env python3
"""Reproduce una grabacion de frames como si fuera la mision, con DINO.

Recorre los hitos en el MISMO orden que la mision (config -> mission.segments:
milestone y despues el `on_milestone` si el tramo gira), puntua cada frame
contra el hito que toca buscar y avanza al siguiente cuando pasa
`vlm.confirm_threshold` las `confirm_hits` veces seguidas que pide el config.
Asi se ve, sin rover y sin ir al lugar, EN QUE FRAME dispararia cada hito.

    python tools/replay_hitos.py --frames debug/Gaborone

Ojo con la trampa: si las fotos de referencia salieron de esa misma
grabacion, el frame identico puntua 1.00 y el resultado queda inflado.
`--excluir-cercanas N` ignora, al puntuar el frame i, las fotos de referencia
llamadas NNNNN_rgb.jpg con |NNNNN - i| <= N (las vecinas en el tiempo), que es
lo mas parecido a mostrarle al rover una escena que nunca vio:

    python tools/replay_hitos.py --frames debug/Gaborone --excluir-cercanas 10

Otras opciones: --cada K (puntuar 1 de cada K frames, mas rapido),
--cooldown-frames (frames ignorados tras cada cambio de hito, ~milestone_cooldown_s
a 5 Hz), --umbral (pisa vlm.confirm_threshold), --traza (imprime el score de
todos los frames, no solo los cambios).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genie_rover.Indoor.hito_detector import (  # noqa: E402
    EXTENSIONES,
    DinoEncoder,
    GaleriaHitos,
)

_NUM = re.compile(r"^(\d+)")


def _numero(p: Path) -> int | None:
    m = _NUM.match(p.name)
    return int(m.group(1)) if m else None


def _secuencia(cfg: dict) -> list[tuple[str, str, int]]:
    """[(tramo, hito_id, confirm_hits)] en el orden en que la mision los busca."""
    pasos = []
    for seg in cfg["mission"]["segments"]:
        ms = seg["milestone"]
        pasos.append((seg["id"], ms["id"], int(ms.get("confirm_hits", 3))))
        giro = seg.get("on_milestone") or seg.get("turn")
        if giro:
            pasos.append((seg["id"] + " (giro " + giro.get("side", "?") + ")",
                          giro["id"], int(giro.get("confirm_hits", 2))))
    return pasos


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, required=True,
                    help="carpeta con los frames grabados (se ordenan por nombre)")
    ap.add_argument("--config", default="configs/indoor_hitos.yaml")
    ap.add_argument("--umbral", type=float, default=None,
                    help="pisa vlm.confirm_threshold del config")
    ap.add_argument("--excluir-cercanas", type=int, default=0, metavar="N")
    ap.add_argument("--cada", type=int, default=1, metavar="K")
    ap.add_argument("--cooldown-frames", type=int, default=15)
    ap.add_argument("--traza", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    vcfg = cfg.get("vlm", {})
    umbral = args.umbral if args.umbral is not None else float(
        vcfg.get("confirm_threshold", 0.72))
    galeria = GaleriaHitos(Path(vcfg.get("reference_dir", "reference_images")),
                           vcfg.get("hito_dirs") or {})
    enc = DinoEncoder(vcfg.get("model", "facebook/dinov2-small"),
                      vcfg.get("device", "auto"), int(vcfg.get("max_side_px", 224)))

    # Las carpetas de debug del bridge mezclan NNNNN_rgb.jpg con _mapa/_plan
    # .png: si hay _rgb, se usan solo esos.
    frames = sorted(p for p in args.frames.iterdir() if p.suffix.lower() in EXTENSIONES)
    if any("_rgb" in p.name for p in frames):
        frames = [p for p in frames if "_rgb" in p.name]
    frames = frames[::max(1, args.cada)]
    pasos = _secuencia(cfg)

    print(f"DINO en {enc.device} | umbral {umbral:.2f} | {len(frames)} frames de "
          f"{args.frames} | excluir vecinas +-{args.excluir_cercanas}")
    print("secuencia: " + " -> ".join(h for _, h, _ in pasos) + "\n")

    k = 0                      # paso actual
    hits = 0
    cooldown = 0
    mejor_del_paso = (0.0, None)
    for frame in frames:
        if k >= len(pasos):
            break
        tramo, hito, requeridos = pasos[k]
        fotos = galeria.fotos_de(hito)
        n = _numero(frame)
        if args.excluir_cercanas and n is not None:
            fotos = [p for p in fotos
                     if _numero(p) is None or "_rgb" not in p.name
                     or abs(_numero(p) - n) > args.excluir_cercanas]
        if not fotos:
            print(f"  [{frame.name}] '{hito}' no tiene fotos -> nunca va a "
                  f"disparar. Corto aca.")
            break

        from PIL import Image
        emb = enc.encode(np.array(Image.open(frame).convert("RGB")))
        scores = [float(np.dot(emb, enc.encode_file(p))) for p in fotos]
        i = int(np.argmax(scores))
        s = scores[i]
        if s > mejor_del_paso[0]:
            mejor_del_paso = (s, frame.name)

        if args.traza:
            print(f"  {frame.name}  {hito:24s} {s:.3f}  vs {fotos[i].name}")

        if cooldown > 0:
            cooldown -= 1
            continue
        hits = hits + 1 if s >= umbral else 0
        if hits >= requeridos:
            print(f"  DISPARA  {frame.name:16s} {tramo:26s} '{hito}'  "
                  f"score {s:.3f} vs {fotos[i].parent.name}/{fotos[i].name}")
            k += 1
            hits = 0
            cooldown = args.cooldown_frames
            mejor_del_paso = (0.0, None)

    print()
    if k >= len(pasos):
        print(f"RESULTADO: los {len(pasos)} hitos dispararon en orden.")
    else:
        tramo, hito, _ = pasos[k]
        s, cual = mejor_del_paso
        print(f"RESULTADO: se trabo en '{hito}' ({tramo}): {k}/{len(pasos)} hitos. "
              f"Mejor score despues del ultimo disparo: {s:.3f}"
              + (f" en {cual}" if cual else "")
              + f" (umbral {umbral:.2f}).")
        print("  Si la grabacion no pasa por ese lugar es esperable; si pasa, "
              "hacen falta fotos mas parecidas a lo que ve la camara ahi.")


if __name__ == "__main__":
    main()
