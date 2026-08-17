"""Mira GET /status mientras el rover se mueve, para ver si el ingest de
telemetria se corta o se atrasa durante el movimiento (no solo en reposo).

Complementa a diag_odometry.py: ese mide rpm/timestamps desde /data, este
mide si el SDK mismo reporta perdida de conexion con el rover mientras
anda. Si telemetry_age_s se dispara o ingest_connected pasa a false
durante el movimiento, el problema esta en el enlace SDK<->rover (RTM/
video), no en el codigo de integracion de odometria.

    python check_status_while_moving.py --seconds 5 --linear 0.30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genie_rover.sdk_client import RoverClient  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--linear", type=float, default=0.30)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--poll", type=float, default=0.1,
                     help="cada cuanto pedir /status")
    a = ap.parse_args()

    c = RoverClient(a.base_url, timeout=5.0)
    session = requests.Session()

    print(f"Va a avanzar {a.seconds:.0f} s con linear={a.linear}, "
          f"pidiendo /status cada {a.poll*1000:.0f} ms.")
    input("ENTER para arrancar...")

    filas: list[dict] = []
    try:
        t0 = time.time()
        while time.time() - t0 < a.seconds:
            c.control(a.linear, 0.0)
            t_rel = time.time() - t0
            try:
                r = session.get(f"{a.base_url}/status", timeout=3.0)
                r.raise_for_status()
                body = r.json()
            except Exception as exc:
                body = {"error": str(exc)}
            filas.append({"t": t_rel, **body})
            time.sleep(a.poll)
    finally:
        c.stop()
        c.stop()
        print("[frenado]\n")

    if not filas:
        print("No se recibio ninguna respuesta de /status.")
        return 1

    print(f"=== /status durante el movimiento: {len(filas)} lecturas en "
          f"{a.seconds:.1f} s ===\n")
    print(f"{'t (s)':>7}  {'browser':>8}  {'mision':>7}  {'ingest':>7}  "
          f"{'edad_telem (s)':>15}  error")
    print("-" * 70)

    caidas_ingest = 0
    edades = []
    for f in filas:
        if "error" in f:
            print(f"{f['t']:7.2f}  {'':8}  {'':7}  {'':7}  {'':15}  {f['error']}")
            continue
        edad = f.get("telemetry_age_s")
        if edad is not None:
            edades.append(edad)
        ingest = f.get("ingest_connected")
        if ingest is False:
            caidas_ingest += 1
        print(f"{f['t']:7.2f}  {str(f.get('browser_ready')):>8}  "
              f"{str(f.get('mission_started')):>7}  {str(ingest):>7}  "
              f"{('%.2f' % edad) if edad is not None else 'null':>15}")

    print("\n=== resumen ===")
    print(f"  lecturas con ingest_connected=false: {caidas_ingest} de {len(filas)}")
    if edades:
        print(f"  telemetry_age_s: min={min(edades):.2f}  max={max(edades):.2f}  "
              f"media={sum(edades)/len(edades):.2f}")
        picos = [e for e in edades if e > 0.5]
        if picos:
            print(f"  ATENCION: {len(picos)} lecturas con telemetry_age_s > 0.5 s "
                  f"(la ultima telemetria real quedo vieja mientras el rover andaba)")
    if caidas_ingest > 0:
        print("\n  El SDK perdio conexion con el rover durante el movimiento.")
        print("  Esto explica huecos en /data sin que sea un bug de odometry.py:")
        print("  no hay nada que integrar si no llega nada nuevo.")
    elif edades and max(edades) > 0.5:
        print("\n  El SDK sigue 'conectado' pero la telemetria se atrasa igual")
        print("  durante el movimiento (ingest_connected=true no garantiza frescura).")
    else:
        print("\n  El ingest se mantuvo sano durante el movimiento. Los huecos que")
        print("  viste en diag_odometry.py probablemente son de publicacion")
        print("  intermitente del lado del rover, no de la conexion SDK<->rover.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
