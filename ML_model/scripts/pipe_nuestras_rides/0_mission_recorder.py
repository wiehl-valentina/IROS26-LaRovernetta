"""Grabador de frames + metadata para corridas MANUALES del rover.

A diferencia de `bridge.py --debug-dir` (que graba mientras el bucle
autonomo maneja), esto esta pensado para cuando VOS manejas el rover a mano
(joystick, la app del SDK, teleop, lo que sea): este script solo escucha
/v2/front y /data en paralelo, en un thread separado, y graba. Nunca manda
comandos de control -- ni podria, `RoverReader` no tiene ese metodo.

Uso:
    python -m mission_recorder \
        --base-url http://localhost:8000 \
        --out data/raw_runs/patio_20260819 \
        --interval 0.5 \
        --note "vueltas alrededor del patio, tarde, sombra parcial"

    # opcional: saltear frames si el rover no se movio (evita grabar 200
    # fotos identicas mientras esta parado)
    ... --min-gps-displacement-m 0.3

Cada corrida deja:
    <out>/
      session.json            <- metadata de la sesion entera (una vez)
      manifest.jsonl           <- una linea de metadata por frame (para
                                  filtrar rapido con pandas/jq sin abrir
                                  cada .json suelto)
      frames/000000_rgb.jpg
      frames/000000_meta.json  <- mismo contenido que su linea del manifest
      frames/000001_rgb.jpg
      ...

Ctrl-C corta limpio y deja todo lo grabado hasta ese momento usable.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from rover_client import RoverError, RoverReader, Telemetry


@dataclass
class RecorderConfig:
    out_dir: str
    interval_s: float = 0.5
    max_seconds: float | None = None
    max_frames: int | None = None
    note: str = ""
    jpeg_quality: int = 90
    # 0 = desactivado (graba todo). >0 = saltea frames si el rover no se
    # movio al menos esta distancia en metros desde el ultimo frame grabado.
    min_gps_displacement_m: float = 0.0
    # Tag libre: "manual", "teleop", "autonomo", etc. Sirve para cuando mas
    # adelante mezcles esto con corridas de bridge.py --debug-dir y quieras
    # saber de que fuente salio cada frame (similar a 'drive_mode' en el
    # dataset FrodoBots-Mini-4K).
    source: str = "manual"


class MissionRecorder:
    def __init__(self, reader: RoverReader, cfg: RecorderConfig):
        self.reader = reader
        self.cfg = cfg
        self.out_dir = Path(cfg.out_dir)
        self.frames_dir = self.out_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / "manifest.jsonl"
        self._manifest_f = open(self.manifest_path, "a")
        self._n = 0
        self._last_gps: tuple[float, float] | None = None
        self._stop = False

    # -------------------------------------------------------------- helpers

    def _distancia_gps_m(self, lat: float, lon: float) -> float:
        """Aproximacion plana (equirectangular), suficiente para decidir 'se
        movio o no' a esta escala de pocos metros. No usar para navegacion."""
        if self._last_gps is None:
            return float("inf")
        lat0, lon0 = self._last_gps
        R = 6378137.0
        dlat = math.radians(lat - lat0)
        dlon = math.radians(lon - lon0)
        x = dlon * R * math.cos(math.radians((lat + lat0) / 2))
        y = dlat * R
        return math.hypot(x, y)

    def _guardar_frame(self, rgb, frame_ts: float, telem: Telemetry) -> None:
        from PIL import Image

        idx = self._n
        img_name = f"{idx:06d}_rgb.jpg"
        Image.fromarray(rgb).save(self.frames_dir / img_name, quality=self.cfg.jpeg_quality)

        meta = {
            "index": idx,
            "frame_file": img_name,
            "frame_timestamp": frame_ts,
            "recorded_at": time.time(),
            "gps": {
                "lat": telem.latitude,
                "lon": telem.longitude,
                "gps_signal": telem.gps_signal,
                "orientation": telem.orientation,
            },
            "speed": telem.speed,
            "battery": telem.battery,
            # Telemetria cruda de mas alta frecuencia (lo que usa odometry.py
            # para reconstruir movimiento fino entre frames). rpms = velocidad
            # de cada rueda, gyros = giroscopo. Sirve para priorizar candidatos:
            # rpms asimetricas sin giro proporcional = rueda patinando; gyro
            # alto = giro brusco -- ambos suelen coincidir con frames donde el
            # modelo se equivoca mas.
            "rpms": telem.raw.get("rpms"),
            "gyros": telem.raw.get("gyros"),
            "source": self.cfg.source,
            "note": self.cfg.note,
        }
        with open(self.frames_dir / f"{idx:06d}_meta.json", "w") as f:
            json.dump(meta, f)

        self._manifest_f.write(json.dumps(meta) + "\n")
        self._manifest_f.flush()
        self._n += 1
        self._last_gps = (telem.latitude, telem.longitude)

    # ------------------------------------------------------------------ run

    def run(self) -> None:
        def _on_sigint(signum, frame):
            self._stop = True
            print("\n[recorder] Ctrl-C recibido, cierro despues del frame actual...")

        signal.signal(signal.SIGINT, _on_sigint)

        t0 = time.time()
        with open(self.out_dir / "session.json", "w") as f:
            json.dump({
                "started_at": t0,
                "note": self.cfg.note,
                "interval_s": self.cfg.interval_s,
                "min_gps_displacement_m": self.cfg.min_gps_displacement_m,
            }, f, indent=2)

        print(f"[recorder] grabando en {self.out_dir} (Ctrl-C para cortar)")
        errores_seguidos = 0

        while not self._stop:
            if self.cfg.max_seconds is not None and (time.time() - t0) > self.cfg.max_seconds:
                print("[recorder] limite de tiempo alcanzado")
                break
            if self.cfg.max_frames is not None and self._n >= self.cfg.max_frames:
                print("[recorder] limite de frames alcanzado")
                break

            try:
                rgb, frame_ts = self.reader.front_frame()
                telem = self.reader.telemetry()
                errores_seguidos = 0
            except RoverError as exc:
                errores_seguidos += 1
                print(f"[recorder] error leyendo SDK ({errores_seguidos}): {exc}")
                if errores_seguidos >= 10:
                    print("[recorder] demasiados errores seguidos, corto")
                    break
                time.sleep(1.0)
                continue

            if self.cfg.min_gps_displacement_m > 0:
                d = self._distancia_gps_m(telem.latitude, telem.longitude)
                if d < self.cfg.min_gps_displacement_m:
                    time.sleep(self.cfg.interval_s)
                    continue

            self._guardar_frame(rgb, frame_ts, telem)
            if self._n % 20 == 0:
                print(f"[recorder] {self._n} frames grabados")

            time.sleep(self.cfg.interval_s)

        self._manifest_f.close()
        elapsed = time.time() - t0
        print(f"\n[recorder] listo: {self._n} frames en {elapsed:.0f}s -> {self.out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--out", required=True, help="carpeta de salida para esta corrida")
    ap.add_argument("--interval", type=float, default=0.5, help="segundos entre frames")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--note", default="", help="texto libre: lugar, condiciones, etc.")
    ap.add_argument("--min-gps-displacement-m", type=float, default=0.0,
                    help="saltea frames si el rover no se movio al menos esto (0=desactivado)")
    ap.add_argument("--source", default="manual",
                    help="tag de origen de la corrida, ej. manual/teleop (default: manual)")
    args = ap.parse_args()

    reader = RoverReader(args.base_url)
    cfg = RecorderConfig(
        out_dir=args.out,
        interval_s=args.interval,
        max_seconds=args.max_seconds,
        max_frames=args.max_frames,
        note=args.note,
        min_gps_displacement_m=args.min_gps_displacement_m,
        source=args.source,
    )
    MissionRecorder(reader, cfg).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
