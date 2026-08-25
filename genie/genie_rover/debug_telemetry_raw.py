#!/usr/bin/env python3
"""Diagnostico rapido: que trae telem.raw de verdad, y por que la odometria
puede estar quedando congelada en (0,0,0) durante una mision de bridge.py.

No mueve el rover -- solo pide telemetria un par de veces y muestra:
  1) las claves top-level que trae telem.raw (para confirmar si "rpms" y
     "gyros" existen con ese nombre exacto, que es lo que Odometry.update()
     espera).
  2) forma/contenido de "rpms" y "gyros" si existen (deberian ser listas de
     listas, cada fila con 5 numeros: 4 valores + timestamp).
  3) si los timestamps de esas filas avanzan entre dos llamadas seguidas
     (si no avanzan, o vienen todos en 0, Odometry los descarta igual que si
     no existieran).

Uso:
    python -m genie_rover.debug_telemetry_raw --base-url http://localhost:8000

(o copialo dentro de genie/genie_rover/ y corré
 `python3 debug_telemetry_raw.py --base-url http://localhost:8000`,
 lo que sea mas comodo -- no importa el import package-relative, no
 depende de nada mas de genie_rover)
"""

import argparse
import json
import time

import requests


def _show_batch(name: str, batch) -> None:
    if batch is None:
        print(f"  {name}: AUSENTE (la clave no esta en telem.raw)")
        return
    if not isinstance(batch, list):
        print(f"  {name}: presente pero no es una lista (tipo={type(batch).__name__}): {batch!r}")
        return
    if len(batch) == 0:
        print(f"  {name}: presente pero VACIA (lista de 0 filas) <-- esto congela la odometria")
        return
    print(f"  {name}: {len(batch)} filas. primera={batch[0]!r}  ultima={batch[-1]!r}")
    malas = [f for f in batch if not isinstance(f, list) or len(f) < 5]
    if malas:
        print(f"    AVISO: {len(malas)} filas con menos de 5 elementos "
              f"(Odometry._wheel_series/_gyro_series las descarta con 'if len(fila) < 5: continue')")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--pauses-s", type=float, default=1.0,
                    help="pausa entre las 2 lecturas, para ver si los timestamps avanzan")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")

    print(f"GET {base}/data  (lectura 1)")
    r1 = requests.get(f"{base}/data", timeout=5.0)
    r1.raise_for_status()
    raw1 = r1.json()

    print("\nclaves top-level de /data:")
    print(f"  {sorted(raw1.keys())}")

    print("\nlectura 1:")
    _show_batch("rpms", raw1.get("rpms"))
    _show_batch("gyros", raw1.get("gyros"))
    print(f"  latitude={raw1.get('latitude')!r}  longitude={raw1.get('longitude')!r}  "
          f"orientation={raw1.get('orientation')!r}")

    time.sleep(args.pauses_s)

    print(f"\nGET {base}/data  (lectura 2, {args.pauses_s:.1f}s despues)")
    r2 = requests.get(f"{base}/data", timeout=5.0)
    r2.raise_for_status()
    raw2 = r2.json()
    _show_batch("rpms", raw2.get("rpms"))
    _show_batch("gyros", raw2.get("gyros"))

    # Comparacion de timestamps entre las dos lecturas (columna 5, indice 4).
    def _ultimo_ts(batch):
        if not batch or not isinstance(batch, list):
            return None
        fila = batch[-1]
        if not isinstance(fila, list) or len(fila) < 5:
            return None
        return fila[4]

    ts1_rpm, ts2_rpm = _ultimo_ts(raw1.get("rpms")), _ultimo_ts(raw2.get("rpms"))
    ts1_gy, ts2_gy = _ultimo_ts(raw1.get("gyros")), _ultimo_ts(raw2.get("gyros"))

    print("\n--- diagnostico ---")
    if raw1.get("rpms") is None and raw1.get("gyros") is None:
        print("PROBLEMA CONFIRMADO: /data no trae ni 'rpms' ni 'gyros'. Odometry.update()")
        print("nunca tiene datos para integrar -> pose se queda congelada en (0,0,0) para")
        print("siempre, sin ningun error. Hay que revisar bajo que nombre trae el SDK real")
        print("las revoluciones de rueda / giroscopo (puede ser otro campo, o venir anidado")
        print("en otra clave -- mirar el 'claves top-level' de arriba) y ajustar")
        print("RoverClient.telemetry() (sdk_client.py) para que Telemetry.raw las incluya")
        print("con las claves 'rpms'/'gyros' que Odometry.update() espera.")
    elif ts1_rpm is not None and ts2_rpm is not None and ts1_rpm == ts2_rpm:
        print("PROBLEMA CONFIRMADO: el timestamp de la ultima fila de 'rpms' NO avanzo entre")
        print("las dos lecturas -- el SDK esta devolviendo un lote repetido/cacheado, o el")
        print("timestamp viene mal armado. Odometry descarta todo intervalo con dt<=0.")
    elif ts1_rpm is not None and ts2_rpm is not None:
        print(f"rpms: timestamp avanzo de {ts1_rpm} a {ts2_rpm} (dt={ts2_rpm - ts1_rpm:.3f}s) -- OK")
        if ts2_rpm - ts1_rpm > 0.5:
            print("  AVISO: ese dt es mayor a max_dt_s (0.5s por defecto en OdometryConfig).")
            print("  Con --pauses-s mas chico (o en el loop real de bridge.py, que llama")
            print("  update() cada iteracion) el gap deberia ser mucho menor -- esto solo")
            print("  importa si tambien lo ves asi durante una mision real.")
    else:
        print("No se pudo comparar timestamps (revisar formato de 'rpms' arriba a mano).")

    print("\nraw completo de la lectura 1 (por si el nombre de campo es otro):")
    print(json.dumps(raw1, indent=2, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
