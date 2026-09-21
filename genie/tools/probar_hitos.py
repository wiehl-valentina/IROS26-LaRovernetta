#!/usr/bin/env python3
"""Calibra los umbrales de reconocimiento de hitos sobre frames reales.

Para que sirve: `confirm_threshold` y `reject_threshold` del yaml deciden si
el rover cree que vio un hito. Los valores por defecto (0.72 / 0.58) son una
suposicion razonable, no una medicion: el numero que sirve depende de tu
pasillo, tu camara y las fotos que hayas sacado. Esta herramienta los mide.

TRES USOS:

1. Ver que tan parecidas son entre si las fotos de un mismo hito. Si dos
   fotos del piano puntuan 0.55 entre ellas, el hito esta mal cubierto (o hay
   una foto que no corresponde) y ningun umbral lo va a salvar:

       python tools/probar_hitos.py --coherencia

2. Ver como puntua un frame cualquiera contra TODOS los hitos. El hito
   correcto tiene que ganar por diferencia clara; si el piano y la puerta
   puntuan casi igual, hacen falta fotos mas distintivas:

       python tools/probar_hitos.py --frame debug/run1/00042_rgb.jpg

3. Recomendar umbrales a partir de frames etiquetados. Necesita una carpeta
   con frames del hito (positivos) y otra con frames de cualquier otro lado
   (negativos):

       python tools/probar_hitos.py --hito piano \\
           --positivos debug/frames_piano --negativos debug/frames_otros

Todo corre local: no usa red, no gasta cuota.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genie_rover.Indoor.hito_detector import (  # noqa: E402
    EXTENSIONES,
    DinoEncoder,
    GaleriaHitos,
)


def _frames_de(carpeta: Path) -> list[Path]:
    return sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES)


def _cargar(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def coherencia(enc: DinoEncoder, galeria: GaleriaHitos) -> None:
    """Cuan parecidas son entre si las fotos de cada hito, y cuanto se
    confunde cada hito con los demas."""
    hitos = sorted(p.name for p in galeria.raiz.iterdir() if p.is_dir())
    embs: dict[str, list[np.ndarray]] = {}

    for hito in hitos:
        fotos = galeria.fotos_de(hito)
        if fotos:
            embs[hito] = [enc.encode_file(p) for p in fotos]

    if not embs:
        print(f"No hay fotos en {galeria.raiz}. Pone al menos 3 por hito.")
        return

    print("\n=== COHERENCIA INTERNA (fotos del mismo hito entre si) ===")
    print("Buscas >= 0.70. Por debajo de 0.60 el hito esta mal cubierto.\n")
    for hito, vecs in embs.items():
        if len(vecs) < 2:
            print(f"  {hito:<24} {len(vecs)} foto(s) -- hacen falta al menos 3")
            continue
        pares = [float(np.dot(vecs[i], vecs[j]))
                 for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        aviso = "" if min(pares) >= 0.60 else "   <-- revisar"
        print(f"  {hito:<24} {len(vecs)} fotos  "
              f"min={min(pares):.3f}  media={np.mean(pares):.3f}{aviso}")

    print("\n=== CONFUSION ENTRE HITOS (maximo entre hitos distintos) ===")
    print("Buscas <= 0.55. Valores altos = dos hitos que se parecen demasiado.\n")
    nombres = list(embs)
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            cruz = max(float(np.dot(va, vb)) for va in embs[a] for vb in embs[b])
            aviso = "" if cruz <= 0.55 else "   <-- se confunden"
            print(f"  {a:<22} vs {b:<22} {cruz:.3f}{aviso}")


def puntuar_frame(enc: DinoEncoder, galeria: GaleriaHitos, frame: Path) -> None:
    """Como puntua un frame contra cada hito."""
    emb = enc.encode(_cargar(frame))
    hitos = sorted(p.name for p in galeria.raiz.iterdir() if p.is_dir())

    filas = []
    for hito in hitos:
        fotos = galeria.fotos_de(hito)
        if not fotos:
            continue
        scores = [float(np.dot(emb, enc.encode_file(p))) for p in fotos]
        filas.append((max(scores), hito, fotos[int(np.argmax(scores))].name))

    if not filas:
        print(f"No hay fotos en {galeria.raiz}.")
        return

    filas.sort(reverse=True)
    print(f"\n=== {frame.name} ===\n")
    for score, hito, foto in filas:
        barra = "#" * int(score * 40)
        print(f"  {hito:<24} {score:.3f}  {barra:<40} ({foto})")

    if len(filas) >= 2:
        margen = filas[0][0] - filas[1][0]
        print(f"\n  ganador: {filas[0][1]} por {margen:.3f}")
        if margen < 0.05:
            print("  AVISO: margen muy chico, estos dos hitos se confunden.")


def recomendar(enc: DinoEncoder, galeria: GaleriaHitos, hito: str,
               positivos: Path, negativos: Path) -> None:
    """Umbrales a partir de frames etiquetados."""
    fotos = galeria.fotos_de(hito)
    if not fotos:
        print(f"El hito '{hito}' no tiene fotos en {galeria.raiz / hito}")
        return

    refs = [enc.encode_file(p) for p in fotos]

    def scores_de(carpeta: Path) -> list[float]:
        return [max(float(np.dot(enc.encode(_cargar(f)), r)) for r in refs)
                for f in _frames_de(carpeta)]

    pos = scores_de(positivos)
    neg = scores_de(negativos)

    if not pos or not neg:
        print("Hacen falta frames en las dos carpetas.")
        return

    print(f"\n=== '{hito}': {len(pos)} positivos, {len(neg)} negativos ===\n")
    print(f"  positivos  min={min(pos):.3f}  media={np.mean(pos):.3f}  max={max(pos):.3f}")
    print(f"  negativos  min={min(neg):.3f}  media={np.mean(neg):.3f}  max={max(neg):.3f}")

    separacion = min(pos) - max(neg)
    print(f"\n  separacion: {separacion:+.3f}")

    if separacion > 0:
        confirm = min(pos) - 0.02
        reject = max(neg) + 0.02
        print("  Separan limpio. Umbrales sugeridos para el yaml:\n")
        print(f"    confirm_threshold: {confirm:.2f}")
        print(f"    reject_threshold:  {reject:.2f}")
    else:
        # Se solapan: elegir el corte que menos errores totales comete.
        cortes = np.arange(0.40, 0.95, 0.01)
        errores = [(sum(s < c for s in pos) + sum(s >= c for s in neg), c)
                   for c in cortes]
        peor, mejor_corte = min(errores)
        print(f"  SE SOLAPAN: ningun umbral separa perfecto ({peor} errores "
              f"en el mejor caso).")
        print(f"  El mejor corte es {mejor_corte:.2f}. Sugerido:\n")
        print(f"    confirm_threshold: {mejor_corte + 0.03:.2f}")
        print(f"    reject_threshold:  {mejor_corte - 0.03:.2f}")
        print("\n  Con solapamiento conviene backend 'hybrid': DINO resuelve los "
              "casos claros y Gemini desempata el resto.")
        print("  Y sacar mas fotos del hito, desde los angulos que fallan.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--reference-dir", default="reference_images")
    ap.add_argument("--coherencia", action="store_true")
    ap.add_argument("--frame", type=Path)
    ap.add_argument("--hito")
    ap.add_argument("--positivos", type=Path)
    ap.add_argument("--negativos", type=Path)
    ap.add_argument("--model", default="facebook/dinov2-small")
    args = ap.parse_args()

    raiz = Path(args.reference_dir)
    if not raiz.is_dir():
        ap.error(f"no existe {raiz}")

    galeria = GaleriaHitos(raiz)
    print(f"fotos de referencia: {galeria.resumen()}")

    enc = DinoEncoder(args.model)
    print(f"DINO: {args.model} en {enc.device}")

    if args.coherencia:
        coherencia(enc, galeria)
    elif args.frame:
        puntuar_frame(enc, galeria, args.frame)
    elif args.hito and args.positivos and args.negativos:
        recomendar(enc, galeria, args.hito, args.positivos, args.negativos)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
