"""Mide SOLO la latencia de POST /control, sin pedir telemetria ni status.

Si el loop de diag_odometry.py / check_status_while_moving.py tarda mas de lo
esperado por vuelta, hay que saber si la culpa es de /control (bloqueante) o
de otra cosa. Esto aisla esa unica variable.

    python check_control_latency.py --n 30 --linear 0.30
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genie_rover.sdk_client import RoverClient  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="cantidad de comandos a enviar")
    ap.add_argument("--linear", type=float, default=0.30)
    ap.add_argument("--base-url", default="http://localhost:8000")
    a = ap.parse_args()

    c = RoverClient(a.base_url, timeout=10.0)

    print(f"Va a mandar {a.n} comandos de control seguidos (linear={a.linear}), "
          f"midiendo SOLO la latencia de /control.")
    input("ENTER para arrancar...")

    tiempos = []
    try:
        for i in range(a.n):
            t0 = time.perf_counter()
            c.control(a.linear, 0.0)
            dt = time.perf_counter() - t0
            tiempos.append(dt)
            print(f"  [{i+1:3d}/{a.n}] /control -> {dt*1000:7.1f} ms")
    finally:
        c.stop()
        c.stop()
        print("[frenado]\n")

    print("=== resumen latencia /control ===")
    print(f"  min:     {min(tiempos)*1000:7.1f} ms")
    print(f"  mediana: {statistics.median(tiempos)*1000:7.1f} ms")
    print(f"  media:   {statistics.mean(tiempos)*1000:7.1f} ms")
    print(f"  max:     {max(tiempos)*1000:7.1f} ms")
    lentos = [t for t in tiempos if t > 0.3]
    if lentos:
        print(f"\n  ATENCION: {len(lentos)} de {len(tiempos)} llamadas tardaron "
              f"mas de 300 ms.")
        print("  Si esto se repite, /control esta bloqueando el loop mucho mas")
        print("  de lo que loop_period_s (0.2 s) espera, y explica huecos de")
        print("  telemetria que no son culpa de odometry.py ni del rover.")
    else:
        print("\n  /control responde rapido. El cuello de botella observado en")
        print("  las otras pruebas viene de otro lado (posiblemente SAM-TP en")
        print("  el loop real del bridge, o /status).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
