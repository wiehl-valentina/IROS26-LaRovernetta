"""Mira los datos crudos de /data mientras el rover avanza en linea recta.

Sirve para entender por que la odometria no coincide con la cinta metrica, en
vez de suponer. Contesta cuatro preguntas:

  1. Que rpm reporta de verdad? (de ahi sale la velocidad)
  2. Cada cuanto llegan muestras nuevas, y hay huecos?
  3. Se repiten lotes entre llamadas?
  4. Que velocidad implican las rpm vs la que se mide con cinta?

    python tools/diag_odometry.py --seconds 6 --linear 0.30

Despues de correrlo, MEDI con cinta cuanto avanzo y pasale el dato:

    python tools/diag_odometry.py --seconds 6 --linear 0.30 --measured 2.03
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genie_rover.sdk_client import RoverClient  # noqa: E402

RPM_A_RAD_S = 2.0 * math.pi / 60.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--linear", type=float, default=0.30)
    ap.add_argument("--radius", type=float, default=0.045)
    ap.add_argument("--measured", type=float, default=None,
                    help="metros medidos con cinta, para comparar")
    ap.add_argument("--poll", type=float, default=0.05,
                    help="cada cuanto pedir /data")
    a = ap.parse_args()

    c = RoverClient(timeout=30)
    print(f"Va a avanzar {a.seconds:.0f} s con linear={a.linear}.")
    print("Marca el punto de partida.")
    input("ENTER para arrancar...")

    lotes: list[tuple[float, list]] = []   # (wall_clock, filas de rpm)
    try:
        t0 = time.time()
        while time.time() - t0 < a.seconds:
            c.control(a.linear, 0.0)
            try:
                raw = c.telemetry().raw
                lotes.append((time.time() - t0, raw.get("rpms", [])))
            except Exception as exc:
                print(f"  [error de telemetria: {exc}]")
            time.sleep(a.poll)
    finally:
        c.stop(); c.stop()
        print("[frenado]\n")

    if not lotes:
        print("No se recibio telemetria.")
        return 1

    # ---------------------------------------------------------- llegada de datos
    print(f"=== llamadas a /data: {len(lotes)} en {a.seconds:.1f} s ===")
    tamanos = [len(f) for _, f in lotes]
    print(f"  filas por respuesta: min={min(tamanos)} max={max(tamanos)} "
          f"media={np.mean(tamanos):.1f}")

    # -------------------------------------------------- timestamps y duplicados
    todos: list[float] = []
    for _, filas in lotes:
        for f in filas:
            if len(f) >= 5:
                todos.append(float(f[4]))
    unicos = sorted(set(todos))
    print(f"\n=== timestamps ===")
    print(f"  muestras totales: {len(todos)}   unicas: {len(unicos)}")
    if len(todos) > len(unicos):
        print(f"  se repiten {len(todos)-len(unicos)} "
              f"({100*(1-len(unicos)/len(todos)):.0f}% duplicadas entre llamadas)")
    if len(unicos) >= 2:
        span = unicos[-1] - unicos[0]
        huecos = np.diff(unicos)
        print(f"  cubren {span:.2f} s de los {a.seconds:.1f} s de la prueba "
              f"({100*span/a.seconds:.0f}%)")
        print(f"  intervalo entre muestras: mediana={np.median(huecos)*1000:.1f} ms "
              f"max={huecos.max()*1000:.1f} ms")
        grandes = huecos[huecos > 0.15]
        if len(grandes):
            print(f"  ATENCION: {len(grandes)} huecos mayores a 150 ms "
                  f"(suman {grandes.sum():.2f} s sin datos)")

    # ------------------------------------------------------------------- rpm
    por_ts: dict[float, list[float]] = {}
    for _, filas in lotes:
        for f in filas:
            if len(f) >= 5:
                por_ts[float(f[4])] = [float(v) for v in f[:4]]
    serie = [por_ts[t] for t in unicos]
    arr = np.array(serie)

    print(f"\n=== rpm por canal ===")
    print(f"  media   {np.round(arr.mean(axis=0), 1)}")
    print(f"  mediana {np.round(np.median(arr, axis=0), 1)}")
    print(f"  max     {np.round(arr.max(axis=0), 1)}")
    en_cero = float(np.mean(np.all(np.abs(arr) < 0.5, axis=1)))
    print(f"  muestras con las 4 ruedas en ~0: {en_cero*100:.0f}%")

    v_media = float(arr.mean()) * RPM_A_RAD_S * a.radius
    print(f"\n=== velocidad implicada por las rpm ===")
    print(f"  radio {a.radius} m  ->  v = {v_media:.4f} m/s")
    print(f"  en {a.seconds:.1f} s serian {v_media*a.seconds:.3f} m")

    if a.measured is not None:
        v_real = a.measured / a.seconds
        print(f"\n=== contra la cinta metrica ===")
        print(f"  medido:    {a.measured:.3f} m  ->  {v_real:.4f} m/s")
        print(f"  implicado: {v_media*a.seconds:.3f} m  ->  {v_media:.4f} m/s")
        factor = v_real / v_media if v_media > 1e-9 else float("inf")
        print(f"  factor real/implicado: {factor:.2f}x")
        print()
        if factor > 1.5:
            r_efectivo = a.radius * factor
            print(f"  Las rpm implican MENOS de lo que avanzo. Si las rpm son")
            print(f"  correctas, el radio efectivo seria {r_efectivo:.4f} m")
            print(f"  ({r_efectivo*200:.1f} cm de diametro).")
            print(f"  Si el diametro medido es 9 cm, entonces las rpm no son")
            print(f"  de la rueda: puede haber una reduccion, o el sensor")
            print(f"  cuenta pulsos de motor.")
        elif factor < 0.7:
            print("  Las rpm implican MAS de lo que avanzo: patinaje, o el")
            print("  radio configurado es mayor que el real.")
        else:
            print("  Coinciden razonablemente. El problema esta en la")
            print("  integracion, no en la escala.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
