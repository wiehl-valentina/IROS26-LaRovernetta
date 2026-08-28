"""Prueba _recover() del bridge REAL sin SDK ni rover encendidos.

No inventa una copia de la logica: importa la clase Bridge de verdad y le
llama Bridge._recover(stub) sobre un objeto minimo que imita los atributos
que _recover() necesita, reemplazando front_frame() por una imagen guardada
en disco en vez de pedirsela al SDK por HTTP.

Esto prueba el circuito completo:
    _recover() -> self.client.front_frame() (falso, imagen de disco)
               -> ask_recovery_heading()      (real, llama a Gemini)
               -> self.send(...)              (falso, solo imprime)

Lo que NO prueba (necesita SDK/rover de verdad): _step(), telemetria GPS,
front_is_blocked sobre un frame en vivo, el planner sobre un BEV real.

Uso:
    python -m genie_rover.test_recover_offline --image genie/debug/obst2/00007_rgb.jpg
    python -m genie_rover.test_recover_offline --image genie/debug/obst2/00007_rgb.jpg --no-vlm
"""

from __future__ import annotations

import argparse
import time
import types

import numpy as np
from PIL import Image

from .bridge import Bridge
from .navigation import DriveCommand, HeadingEstimator, PathFollower


def _build_stub(image_path: str, use_vlm: bool, timeout_s: float, min_confidence: float):
    """Objeto minimo con exactamente los atributos que _recover() toca.

    No es un Bridge real (no carga SAM-TP, no abre conexion HTTP): es mas
    rapido y mas facil de inspeccionar para esta prueba puntual.
    """
    rgb = np.asarray(Image.open(image_path).convert("RGB"))

    stub = types.SimpleNamespace()
    # Nombres exactos que lee bridge.Bridge (antes eran vlm_timeout_s /
    # vlm_min_confidence, que NO existen en el bridge: la prueba pasaba pero
    # el circuito real nunca se ejercitaba).
    stub.use_vlm_recovery = use_vlm
    stub.vlm_recovery_timeout_s = timeout_s
    stub.vlm_recovery_min_confidence = min_confidence
    stub._vlm_recovery_failed = False
    stub.recovery_turn_s = 1.5
    stub._consecutive_empty = 3
    stub._stop_requested = False

    # Sin memoria espacial ni odometria: _pick_recovery_heading solo puede
    # apoyarse en el VLM, que es justo lo que esta prueba quiere ejercitar.
    stub.pmap = None
    stub.odometry = None
    stub.recovery_headings_deg = [0.0, 90.0, -90.0, 180.0]
    stub.recovery_min_cobertura_pct = 25.0
    stub._track_pose_during_maneuver = lambda: None

    # Reporter minimo: _recover ahora reporta por MissionConsoleReporter.
    stub._reporter = types.SimpleNamespace(
        event=lambda titulo, msg, color=None: print(f"  [{titulo}] {msg}")
    )

    stub.client = types.SimpleNamespace(
        front_frame=lambda: (rgb, time.time())
    )
    stub.follower = PathFollower(angular_sign=-1.0, turn_speed=0.35, max_linear=0.35)
    stub.heading_est = HeadingEstimator()

    sent = []

    def _send(cmd: DriveCommand) -> None:
        sent.append(cmd)
        print(f"  [SIMULADO] linear={cmd.linear:+.2f} angular={cmd.angular:+.2f}  {cmd.reason}")

    stub.send = _send
    stub._sent = sent
    return stub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="frame guardado, ej. genie/debug/obst2/00007_rgb.jpg")
    ap.add_argument("--no-vlm", action="store_true", help="forzar el barrido ciego, para comparar")
    ap.add_argument("--timeout", type=float, default=4.0)
    ap.add_argument("--min-confidence", type=float, default=0.35)
    args = ap.parse_args()

    print(f"=== probando Bridge._recover() con {args.image} ===")
    print(f"    use_vlm_recovery = {not args.no_vlm}\n")

    if not args.no_vlm:
        # El stub se saltea Bridge.__init__() a proposito (no carga el modelo
        # SAM-TP ni abre conexion HTTP), pero por eso tambien se saltea la
        # linea que ahi llama a load_credentials(). La replicamos aca.
        from programs.client.genai_client import load_credentials
        load_credentials()

    stub = _build_stub(args.image, use_vlm=not args.no_vlm,
                       timeout_s=args.timeout, min_confidence=args.min_confidence)

    t0 = time.time()
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    Bridge._recover(stub, rgb)  # metodo REAL de bridge.py, sobre el stub
    dt = time.time() - t0

    print(f"\n=== resultado ===")
    print(f"  comandos enviados: {len(stub._sent)}")
    print(f"  tiempo total: {dt:.2f}s")
    if stub._sent:
        print("  el bridge giro (mira arriba el motivo que reporto RECUPERACION)")
    else:
        print("  el bridge decidio NO girar: el rumbo actual ya parecia libre")


if __name__ == "__main__":
    main()
