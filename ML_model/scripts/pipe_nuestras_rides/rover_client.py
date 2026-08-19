"""Cliente HTTP minimo para el Earth Rovers SDK, pensado SOLO para lectura.

Repo separado (iros26_erc_unlp_ml_modeling): esto NO depende del paquete
genie_rover del repo de navegacion, a proposito. Este repo es de dataset y
modelado, no de control del robot -- por eso ni siquiera existe un metodo
`control()` aca: no hay forma de que este codigo mueva el rover por
accidente, ni aunque alguien se equivoque llamando algo.

Cuando en el futuro integres esto al repo de navegacion, esto es un
subconjunto de genie_rover/sdk_client.py (mismos endpoints, misma forma de
los datos): alcanza con reemplazar el import por el del otro repo, el resto
del codigo no cambia.

Prueba standalone (con el SDK corriendo):
    python -m rover_client --base-url http://localhost:8000
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from PIL import Image


class RoverError(RuntimeError):
    pass


@dataclass
class Telemetry:
    latitude: float
    longitude: float
    orientation: float
    speed: float
    battery: float
    gps_signal: float
    timestamp: float
    raw: dict[str, Any]


class RoverReader:
    """Solo lectura: front_frame() y telemetry(). Nada que mueva el rover."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.session = requests.Session()

    def front_frame(self) -> tuple[np.ndarray, float]:
        """Devuelve (imagen RGB HxWx3 uint8, timestamp unix)."""
        r = self.session.get(f"{self.base_url}/v2/front", timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        b64 = data.get("front_frame")
        if not b64:
            raise RoverError(
                "El SDK no devolvio 'front_frame' (rover apagado, o video no conectado?)"
            )
        return _decode_b64_image(b64), float(data.get("timestamp", time.time()))

    def telemetry(self) -> Telemetry:
        r = self.session.get(f"{self.base_url}/data", timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        return Telemetry(
            latitude=float(d.get("latitude", 0.0)),
            longitude=float(d.get("longitude", 0.0)),
            orientation=float(d.get("orientation", 0.0)),
            speed=float(d.get("speed", 0.0)),
            battery=float(d.get("battery", 0.0)),
            gps_signal=float(d.get("gps_signal", 0.0)),
            timestamp=float(d.get("timestamp", time.time())),
            raw=d,
        )


def _decode_b64_image(b64: str) -> np.ndarray:
    if b64.startswith("data:"):  # por si alguna version manda data-URI
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


# --------------------------------------------------------------------- prueba

def _self_test(base_url: str) -> None:
    r = RoverReader(base_url)
    print("-> GET /data")
    t = r.telemetry()
    print(f"   lat={t.latitude:.7f} lon={t.longitude:.7f} "
          f"gps_signal={t.gps_signal} bateria={t.battery}%")
    if t.gps_signal <= 0:
        print("   ADVERTENCIA: gps_signal en 0, la metadata GPS va a salir vacia.")

    print("-> GET /v2/front")
    img, ts = r.front_frame()
    print(f"   frame {img.shape} dtype={img.dtype} timestamp={ts}")
    print("\nOK -- este script nunca mando ningun comando de movimiento.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    _self_test(ap.parse_args().base_url)
