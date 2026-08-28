"""Calibra la escala de las ruedas contra el GPS RTK.

El problema: 'rpms' no reporta vueltas de rueda. Con radio 0.045 m las cuentas
dan 1.28 m donde la cinta midio 2.03 m. No hace falta saber que mide
exactamente el sensor: alcanza con medir el factor y guardarlo.

El GPS del Mini tiene correccion RTK (hdop ~0.013), o sea precision
centimetrica. Es mejor referencia que la cinta: no depende de donde marcaste,
y se puede repetir muchas veces sin esfuerzo.

Ademas es inmune al otro problema que encontramos: la telemetria llega en
rafagas con huecos de hasta 440 ms. El GPS mide el desplazamiento TOTAL entre
principio y fin, asi que los huecos no lo afectan.

    python tools/calibrate_wheel_scale.py --runs 5

Cada corrida: avanza, mide, retrocede al punto de partida. Necesita unos 3 m
libres y cielo despejado (adentro no hay fix RTK).
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genie_rover.navigation import latlon_to_local_ne  # noqa: E402
from genie_rover.sdk_client import RoverClient  # noqa: E402

RPM_A_RAD_S = 2.0 * math.pi / 60.0


def gps_ok(t) -> bool:
    return abs(t.latitude) <= 90 and abs(t.longitude) <= 180 and t.gps_signal > 0


def una_corrida(c: RoverClient, linear: float, segundos: float,
                radio: float, poll: float) -> dict | None:
    """Avanza recto y devuelve lo medido por GPS y por ruedas."""
    try:
        t_ini = c.telemetry()
    except Exception as exc:
        print(f"    no pude leer la pose inicial: {exc}")
        return None
    if not gps_ok(t_ini):
        print("    sin fix de GPS")
        return None

    vistos: dict[float, list[float]] = {}
    t0 = time.time()
    try:
        while time.time() - t0 < segundos:
            c.control(linear, 0.0)
            try:
                raw = c.telemetry().raw
                for f in raw.get("rpms", []):
                    if len(f) >= 5:
                        vistos[float(f[4])] = [float(v) for v in f[:4]]
            except Exception:
                pass
            time.sleep(poll)
    finally:
        c.stop(); c.stop()

    time.sleep(2.0)   # que el GPS se asiente antes de la lectura final
    try:
        t_fin = c.telemetry()
    except Exception as exc:
        print(f"    no pude leer la pose final: {exc}")
        return None
    if not gps_ok(t_fin):
        print("    perdi el fix durante la corrida")
        return None

    norte, este = latlon_to_local_ne(t_ini.latitude, t_ini.longitude,
                                     t_fin.latitude, t_fin.longitude)
    d_gps = math.hypot(norte, este)

    if len(vistos) < 5:
        print(f"    solo {len(vistos)} muestras de rpm, muy pocas")
        return None

    # Integrar las rpm sobre sus propios timestamps. Los huecos se cubren
    # asumiendo velocidad constante en el intervalo, que es lo mejor que se
    # puede hacer sin datos.
    ts = sorted(vistos)
    d_ruedas = 0.0
    for k in range(1, len(ts)):
        dt = ts[k] - ts[k - 1]
        if dt <= 0 or dt > 1.0:
            continue
        v0 = float(np.mean(vistos[ts[k - 1]])) * RPM_A_RAD_S * radio
        v1 = float(np.mean(vistos[ts[k]])) * RPM_A_RAD_S * radio
        d_ruedas += 0.5 * (v0 + v1) * dt

    return {
        "d_gps": d_gps,
        "d_ruedas": d_ruedas,
        "muestras": len(ts),
        "cobertura": (ts[-1] - ts[0]) / segundos,
        "hdop": t_fin.raw.get("hdop"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--linear", type=float, default=0.30)
    ap.add_argument("--radius", type=float, default=0.045,
                    help="radio configurado hoy (el que queremos corregir)")
    ap.add_argument("--poll", type=float, default=0.05)
    ap.add_argument("--no-return", action="store_true",
                    help="no volver al punto de partida entre corridas")
    a = ap.parse_args()

    c = RoverClient(timeout=30)

    print("Calibracion de escala de ruedas contra GPS RTK.")
    print(f"{a.runs} corridas de {a.seconds:.0f} s a linear={a.linear}.")
    print("Necesita ~3 m libres y cielo despejado (adentro no hay fix).\n")

    try:
        t = c.telemetry()
        print(f"GPS: lat={t.latitude:.7f} lon={t.longitude:.7f} "
              f"signal={t.gps_signal} hdop={t.raw.get('hdop')}")
        if not gps_ok(t):
            print("\nSin fix de GPS. Salí al exterior y esperá a que enganche.")
            return 1
    except Exception as exc:
        print(f"No pude leer telemetria: {exc}")
        return 1

    input("\nENTER para arrancar...")

    resultados = []
    try:
        for i in range(1, a.runs + 1):
            print(f"\n--- corrida {i}/{a.runs} ---")
            r = una_corrida(c, a.linear, a.seconds, a.radius, a.poll)
            if r is None:
                continue
            factor = r["d_gps"] / r["d_ruedas"] if r["d_ruedas"] > 1e-6 else float("nan")
            resultados.append(factor)
            print(f"    GPS: {r['d_gps']:.3f} m   ruedas: {r['d_ruedas']:.3f} m"
                  f"   factor: {factor:.3f}")
            print(f"    {r['muestras']} muestras, cobertura {r['cobertura']*100:.0f}%,"
                  f" hdop {r['hdop']}")

            if i < a.runs and not a.no_return:
                print("    volviendo al punto de partida ...")
                t0 = time.time()
                while time.time() - t0 < a.seconds:
                    c.control(-a.linear, 0.0)
                    time.sleep(0.1)
                c.stop(); c.stop()
                time.sleep(2.0)
    finally:
        c.stop(); c.stop()
        print("\n[frenado]")

    validos = [f for f in resultados if f == f and 0.2 < f < 10]
    if len(validos) < 2:
        print(f"\nSolo {len(validos)} corridas validas. Repeti con mejor senal.")
        return 1

    media = statistics.mean(validos)
    print("\n" + "=" * 56)
    print(f"factores: {[round(f, 3) for f in validos]}")
    print(f"media: {media:.3f}")
    if len(validos) >= 3:
        desvio = statistics.stdev(validos)
        print(f"desvio: {desvio:.3f}  ({desvio/media*100:.0f}% de dispersion)")
        if desvio / media > 0.15:
            print("\nADVERTENCIA: mucha dispersion entre corridas. Puede ser")
            print("patinaje, o el GPS no esta tan firme como parece.")

    radio_efectivo = a.radius * media
    print(f"\nRadio efectivo = {a.radius} x {media:.3f} = {radio_efectivo:.4f} m")
    print("\nPone esto en configs/frodobot_rover.yaml, seccion odometry:\n")
    print(f"  wheel_radius_m: {radio_efectivo:.4f}   # EFECTIVO, calibrado")
    print(f"                                # contra GPS RTK en {len(validos)} corridas.")
    print(f"                                # El diametro real es 9 cm: 'rpms'")
    print(f"                                # no reporta vueltas de rueda.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
