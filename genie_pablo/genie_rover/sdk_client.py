"""Cliente HTTP para el Earth Rovers SDK v5.2.

El SDK expone el rover en http://localhost:8000. Este modulo encapsula los
endpoints que necesita el bridge de GeNIE y nada mas.

Prueba standalone (con el SDK corriendo):
    python -m genie_rover.sdk_client --base-url http://localhost:8000
"""

from __future__ import annotations

import atexit
import base64
import io
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from PIL import Image


class RoverError(RuntimeError):
    pass


class _TelemetryPoller:
    """Hilo de fondo que consulta /data a alta frecuencia (hz) y acumula
    todas las muestras de rpms/gyros que van pasando, en vez de dejar que
    cada llamada a telemetry() se quede solo con la ventana angosta (~80ms)
    que devuelve el SDK en cada request.

    El SDK v5.2 responde /data con un buffer circular de tamano fijo (~5
    muestras) sin importar cada cuanto se lo consulte: si el bridge tarda
    ~5s por iteracion (por la inferencia de SAM-TP), cada telemetry() solo
    veia ~80ms de movimiento real de esos 5s — ~1.7% del recorrido real
    llegaba a integrarse en la odometria. Este poller consulta cada 1/hz
    segundos (por defecto 20 Hz => cada 50ms, mas rapido que la ventana del
    SDK) y va acumulando en self._rpms/self._gyros lo que no se haya visto
    todavia (deduplicado por el timestamp que trae cada fila), asi que
    drain() devuelve TODO lo que paso desde el ultimo drain, sin importar
    cuanto haya tardado el que hace la consulta.
    """

    def __init__(self, base_url: str, timeout: float, hz: float = 20.0):
        self._base_url = base_url
        self._timeout = timeout
        self._interval = 1.0 / hz
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._rpms: list = []
        self._gyros: list = []
        self._seen_rpm_ts: set = set()
        self._seen_gyro_ts: set = set()
        self._latest: dict = {}
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                r = self._session.get(f"{self._base_url}/data", timeout=self._timeout)
                r.raise_for_status()
                d = r.json()
                with self._lock:
                    self._latest = d
                    for fila in d.get("rpms", []) or []:
                        if len(fila) >= 5 and fila[4] not in self._seen_rpm_ts:
                            self._seen_rpm_ts.add(fila[4])
                            self._rpms.append(fila)
                    for fila in d.get("gyros", []) or []:
                        if len(fila) >= 4 and fila[-1] not in self._seen_gyro_ts:
                            self._seen_gyro_ts.add(fila[-1])
                            self._gyros.append(fila)
            except Exception:
                # Un fallo puntual de red no debe matar el hilo de fondo;
                # la proxima vuelta reintenta sola.
                pass
            time.sleep(max(0.0, self._interval - (time.monotonic() - t0)))

    def drain(self) -> dict:
        """Devuelve el ultimo /data recibido, pero con rpms/gyros
        reemplazados por TODO lo acumulado desde el drain anterior (y no
        solo lo que traiga la ultima respuesta). Vacia el acumulador."""
        with self._lock:
            raw = dict(self._latest)
            raw["rpms"] = self._rpms
            raw["gyros"] = self._gyros
            self._rpms = []
            self._gyros = []
            self._seen_rpm_ts = set()
            self._seen_gyro_ts = set()
            return raw

    def stop(self) -> None:
        self._running = False


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


@dataclass
class Checkpoint:
    id: int
    sequence: int
    latitude: float
    longitude: float


class RoverClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 5.0,
                 stop_on_exit: bool = True):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.session = requests.Session()
        self._sent_motion = False
        # Hilo de fondo que consulta /data a 20 Hz para no perder movimiento
        # real entre iteraciones lentas del bridge (ver _TelemetryPoller).
        self._telemetry_poller = _TelemetryPoller(self.base_url, self.timeout)
        atexit.register(self._telemetry_poller.stop)
        if stop_on_exit:
            # El SDK mantiene el ultimo comando indefinidamente: si el proceso
            # muere sin frenar, el rover se sigue moviendo solo. Esto cubre los
            # scripts sueltos que no tienen su propio try/finally.
            atexit.register(self._stop_quietly)

    def _stop_quietly(self) -> None:
        if not self._sent_motion:
            return
        for _ in range(2):
            try:
                self.session.post(
                    f"{self.base_url}/control",
                    json={"command": {"linear": 0.0, "angular": 0.0, "lamp": 0}},
                    timeout=2.0)
            except Exception:
                pass

    # ---------------------------------------------------------------- camara

    def front_frame(self) -> tuple[np.ndarray, float]:
        """Devuelve (imagen RGB HxWx3 uint8, timestamp unix)."""
        r = self.session.get(f"{self.base_url}/v2/front", timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        b64 = data.get("front_frame")
        if not b64:
            raise RoverError("El SDK no devolvio 'front_frame' (video no conectado?)")
        return _decode_b64_image(b64), float(data.get("timestamp", time.time()))

    # ------------------------------------------------------------- telemetria

    def telemetry(self) -> Telemetry:
        # Antes: un unico GET /data por llamada, que solo trae la ventana
        # angosta (~80ms) que el SDK guarda en ese momento. Ahora: se drena
        # todo lo que el _TelemetryPoller acumulo en el hilo de fondo desde
        # la ultima llamada, sin importar cuanto haya tardado esta iteracion.
        d = self._telemetry_poller.drain()
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

    # ---------------------------------------------------------------- control

    def control(self, linear: float, angular: float, lamp: int = 0) -> None:
        """Envia velocidades. linear/angular en [-1, 1]."""
        lin = float(np.clip(linear, -1.0, 1.0))
        ang = float(np.clip(angular, -1.0, 1.0))
        if lin != 0.0 or ang != 0.0:
            self._sent_motion = True
        payload = {"command": {"linear": lin, "angular": ang, "lamp": int(lamp)}}
        r = self.session.post(f"{self.base_url}/control", json=payload, timeout=self.timeout)
        r.raise_for_status()

    def stop(self) -> None:
        """Frena. Se llama en todos los caminos de salida, incluidos los de error."""
        try:
            self.control(0.0, 0.0)
        except Exception as exc:  # nunca propagar desde el freno
            print(f"[sdk] ADVERTENCIA: no se pudo enviar el comando de freno: {exc}")

    # --------------------------------------------------------------- misiones

    def start_mission(self) -> dict[str, Any]:
        r = self.session.post(f"{self.base_url}/start-mission", timeout=15.0)
        if r.status_code == 400:
            raise RoverError(f"No se pudo iniciar la mision: {r.text}")
        r.raise_for_status()
        return r.json()

    def checkpoints(self) -> tuple[list[Checkpoint], int]:
        r = self.session.get(f"{self.base_url}/checkpoints-list", timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        cps = [
            Checkpoint(
                id=int(c["id"]),
                sequence=int(c["sequence"]),
                latitude=float(c["latitude"]),
                longitude=float(c["longitude"]),
            )
            for c in d.get("checkpoints_list", [])
        ]
        cps.sort(key=lambda c: c.sequence)
        return cps, int(d.get("latest_scanned_checkpoint", 0))

    def claim_checkpoint(self) -> tuple[bool, str]:
        """Intenta reclamar el checkpoint actual.

        Devuelve (exito, mensaje). Un 400 no es excepcional: significa que
        todavia estamos lejos, y el mensaje trae la distancia.
        """
        r = self.session.post(f"{self.base_url}/checkpoint-reached", json={}, timeout=self.timeout)
        if r.status_code == 200:
            return True, str(r.json())
        return False, r.text

    # ---------------------------------------------------------- intervenciones

    def start_intervention(self) -> None:
        self.session.post(f"{self.base_url}/interventions/start", timeout=self.timeout)

    def end_intervention(self) -> None:
        self.session.post(f"{self.base_url}/interventions/end", timeout=self.timeout)


def _decode_b64_image(b64: str) -> np.ndarray:
    if b64.startswith("data:"):  # por si alguna version manda data-URI
        b64 = b64.split(",", 1)[1]
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


# --------------------------------------------------------------------- self test

def _self_test(base_url: str) -> None:
    c = RoverClient(base_url)
    print("→ GET /data")
    t = c.telemetry()
    print(f"   lat={t.latitude:.7f} lon={t.longitude:.7f} orientation={t.orientation} "
          f"gps_signal={t.gps_signal} bateria={t.battery}%")
    if t.gps_signal <= 0:
        print("   ADVERTENCIA: gps_signal en 0. Sin GPS no hay navegacion a checkpoints.")

    print("→ GET /v2/front")
    img, ts = c.front_frame()
    print(f"   frame {img.shape} dtype={img.dtype} timestamp={ts}")

    print("→ GET /checkpoints-list")
    try:
        cps, latest = c.checkpoints()
        print(f"   {len(cps)} checkpoints, ultimo escaneado: {latest}")
        for cp in cps[:5]:
            print(f"   #{cp.sequence}: {cp.latitude}, {cp.longitude}")
    except Exception as exc:
        print(f"   sin checkpoints ({exc}) — normal si no hay MISSION_SLUG")

    print("\nNO se envio ningun comando de movimiento.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    _self_test(ap.parse_args().base_url)
