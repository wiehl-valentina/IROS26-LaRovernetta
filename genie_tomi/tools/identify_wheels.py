"""Determina que posicion del array 'rpms' corresponde a cada lado del rover.

/data devuelve rpms como 4 valores por muestra, pero el SDK no documenta el
orden. Sin saberlo no se puede calcular odometria: hace falta separar las
ruedas de la izquierda de las de la derecha.

Este script mueve el rover de tres formas y observa el signo y la magnitud de
cada canal:

  1. avance recto  -> las cuatro deberian ir en el mismo sentido
  2. giro derecha  -> izquierda adelante, derecha atras (o mas lento)
  3. giro izquierda-> al reves

Necesita espacio libre alrededor. El rover se mueve unos segundos.

    python tools/identify_wheels.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genie_rover.sdk_client import RoverClient  # noqa: E402


def muestrear(client: RoverClient, linear: float, angular: float,
              duracion: float = 2.0) -> np.ndarray:
    """Aplica un comando y devuelve las RPM medidas mientras dura."""
    filas: list[list[float]] = []
    t0 = time.time()
    while time.time() - t0 < duracion:
        client.control(linear, angular)
        try:
            t = client.telemetry()
            for fila in t.raw.get("rpms", []):
                filas.append([float(v) for v in fila[:4]])
        except Exception as exc:
            print(f"    error leyendo telemetria: {exc}")
        time.sleep(0.2)
    client.stop()
    time.sleep(1.0)
    return np.array(filas) if filas else np.zeros((0, 4))


def main() -> int:
    client = RoverClient(timeout=30)

    print("Identificacion de ruedas.")
    print("El rover va a moverse: avanzar, girar a la derecha y a la izquierda.")
    print("Cada movimiento dura 2 segundos. Necesita espacio libre.\n")
    if input("Escribi 'si' para continuar: ").strip().lower() not in {"si", "sí", "s"}:
        print("Cancelado.")
        return 0

    for i in (3, 2, 1):
        print(f"  {i} ...")
        time.sleep(1)

    resultados = {}
    try:
        for etiqueta, lin, ang in [("avance recto", 0.35, 0.0),
                                   ("giro a la DERECHA", 0.0, -0.45),
                                   ("giro a la IZQUIERDA", 0.0, 0.45)]:
            print(f"\n  {etiqueta} ...")
            datos = muestrear(client, lin, ang)
            if len(datos) == 0:
                print("    sin muestras")
                continue
            medias = datos.mean(axis=0)
            resultados[etiqueta] = medias
            print("    medias por canal: " +
                  "  ".join(f"[{i}]={v:+8.1f}" for i, v in enumerate(medias)))
            print(f"    ({len(datos)} muestras)")
    finally:
        client.stop()
        client.stop()

    print("\n" + "=" * 60)

    der = resultados.get("giro a la DERECHA")
    izq = resultados.get("giro a la IZQUIERDA")
    if der is None or izq is None:
        print("Faltan datos de los giros; no puedo deducir el mapeo.")
        return 1

    # En un giro a la derecha, las ruedas izquierdas empujan hacia adelante y
    # las derechas hacia atras (o mucho mas lento). La diferencia entre ambos
    # giros amplifica esa asimetria y cancela cualquier offset del sensor.
    contraste = der - izq
    print("Contraste (giro derecha menos giro izquierda) por canal:")
    for i, v in enumerate(contraste):
        print(f"  canal [{i}]: {v:+8.1f}")

    orden = np.argsort(contraste)
    lado_a = sorted(int(i) for i in orden[:2])
    lado_b = sorted(int(i) for i in orden[2:])

    # El lado con contraste mayor giro mas hacia adelante al doblar a la
    # derecha: ese es el lado izquierdo.
    print(f"\n  Canales del lado IZQUIERDO: {lado_b}")
    print(f"  Canales del lado DERECHO:   {lado_a}")

    magnitud = float(np.mean(np.abs(contraste)))
    if magnitud < 5.0:
        print("\n  ADVERTENCIA: el contraste es muy chico. Puede que el rover no")
        print("  haya girado de verdad (bateria baja, mucha friccion), o que")
        print("  las rpm no se esten reportando. Revisa y repeti.")

    print("\nPone esto en configs/frodobot_rover.yaml:\n")
    print("odometry:")
    print(f"  wheel_radius_m: 0.045")
    print(f"  track_width_m: 0.15")
    print(f"  left_rpm_indices: {lado_b}")
    print(f"  right_rpm_indices: {lado_a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
