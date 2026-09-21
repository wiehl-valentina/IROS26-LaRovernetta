"""Publica una ruta grabada en el mapa del SDK SIN correr el bridge.

Escribe static/genie_waypoints.json con el mismo formato que bridge_mexico.py
("frame": "gps"), asi map.js / dashboard-map.js dibujan los puntos intermedios.
Sirve para revisar la ruta sin mision, sin video y sin GPS.

Uso (parado en genie/):
    PYTHONPATH=. python tools/mostrar_ruta.py /ruta/al/sdk/static/genie_waypoints.json \
        mexico_checkpoints_help/Mexico_mision2/mexico5 \
        mexico_checkpoints_help/Mexico_mision2/mexico5part3.geojson

Para sacar los puntos del mapa:
    PYTHONPATH=. python tools/mostrar_ruta.py /ruta/al/sdk/static/genie_waypoints.json --apagar
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from genie_rover.route import cargar_ruta


def escribir(destino: Path, payload: dict) -> None:
    # Atomico (tmp + rename), igual que el bridge: el navegador lo lee cada
    # 1.5 s y no tiene que encontrarse nunca un JSON a medio escribir.
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, destino)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("salida", help="static/genie_waypoints.json del SDK")
    ap.add_argument("rutas", nargs="*", help="uno o mas archivos de ruta, en orden")
    ap.add_argument("--apagar", action="store_true",
                    help="saca el overlay del mapa (escribe active: false)")
    args = ap.parse_args()

    salida = Path(args.salida)

    if args.apagar:
        escribir(salida, {"active": False, "ts": time.time()})
        print(f"overlay apagado -> {salida}")
        return 0

    if not args.rutas:
        ap.error("falta al menos un archivo de ruta (o usa --apagar)")

    puntos = []
    for r in args.rutas:
        pts = cargar_ruta(r)
        print(f"  {Path(r).name}: {len(pts)} puntos")
        puntos += pts

    if not puntos:
        print("ninguna ruta tenia puntos, no escribo nada")
        return 1

    escribir(salida, {
        "active": True,
        "frame": "gps",
        "mode": "mexico",
        "ts": time.time(),
        "state": "vista previa (sin bridge)",
        "current_index": 0,
        "waypoints": [{"lat": p.lat, "lon": p.lon, "motivo": p.motivo} for p in puntos],
    })
    print(f"{len(puntos)} puntos publicados -> {salida}")
    print("abri el mapa del SDK; en ~1.5 s aparecen y se encuadran solos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
