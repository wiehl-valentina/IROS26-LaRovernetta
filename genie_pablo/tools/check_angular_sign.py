"""Determina si 'angular' positivo hace girar el rover a izquierda o derecha.

Hace falta porque tu repo se contradice:
  * programs/genai_program.py  -> "Positive = Left, Negative = Right"
  * programs/control_nube.py   -> "1.0 es derecha maxima"

Uno de los dos esta mal, y si el bridge usa el signo equivocado el robot se
aleja de cada meta en vez de acercarse.

    python tools/check_angular_sign.py

El rover gira 2 segundos con angular=+0.4 y despues frena. Miralo y contesta.
Necesita espacio libre alrededor.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genie_rover.sdk_client import RoverClient  # noqa: E402

TEST_ANGULAR = 0.4
TEST_SECONDS = 2.0


def main() -> int:
    client = RoverClient()

    print("Chequeo del signo de 'angular'.")
    print(f"El rover va a girar en el lugar {TEST_SECONDS} s con angular=+{TEST_ANGULAR}.")
    print("Aseguráte de que tenga espacio libre alrededor.\n")
    if input("Escribí 'si' para continuar: ").strip().lower() not in {"si", "sí", "s"}:
        print("Cancelado.")
        return 0

    for i in (3, 2, 1):
        print(f"  {i} ...")
        time.sleep(1)

    try:
        print(f"  girando con angular=+{TEST_ANGULAR} ...")
        t0 = time.time()
        while time.time() - t0 < TEST_SECONDS:
            client.control(0.0, TEST_ANGULAR)
            time.sleep(0.25)
    finally:
        client.stop()
        client.stop()
        print("  frenado.\n")

    ans = ""
    while ans not in {"i", "d"}:
        ans = input("Para donde giro, visto desde arriba? [i]zquierda / [d]erecha: ").strip().lower()

    if ans == "i":
        sign = -1.0
        print("\nangular positivo = IZQUIERDA (coincide con genai_program.py)")
    else:
        sign = +1.0
        print("\nangular positivo = DERECHA (coincide con el prompt de control_nube.py)")

    print(f"\nPoné esto en configs/frodobot_rover.yaml:\n\n  navigation:\n    angular_sign: {sign}\n")
    print("Aprovechá y arreglá el docstring que quedo mal en tu repo, "
          "antes de que confunda a alguien mas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
