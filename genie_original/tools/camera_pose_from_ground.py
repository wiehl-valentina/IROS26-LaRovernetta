"""Obtiene la pose de la camara respecto del suelo, sin regla ni transportador.

Por que existe: project_score_to_bev necesita saber donde esta la camara
respecto del plano del piso. Medir la altura con regla se puede, pero medir la
inclinacion a ojo es impreciso, y en un Earth Rover Mini —bajo y con la camara
bastante inclinada— unos pocos grados de error mueven los obstaculos varios
metros en el BEV.

La idea: apoyas el MISMO tablero de ajedrez plano en el piso delante del robot,
sacas una foto, y solvePnP calcula la pose de la camara respecto del plano del
tablero. Como el tablero esta sobre el piso, ese plano ES el piso.

No importa como quede rotado el tablero sobre el suelo. La altura, el pitch y
el roll no dependen de eso, y el "adelante" del mundo se define como el eje
optico de la camara proyectado sobre el piso, que es exactamente lo que hace
camera_planar_axes() en el repo.

Uso:

  1) Apoya el tablero plano en el piso, ~1 m delante del robot, bien visible.
     Que quede realmente plano: si se comba, la pose sale mal.

  2) python tools/camera_pose_from_ground.py capture --out pose_frame.png

  3) python tools/camera_pose_from_ground.py solve --image pose_frame.png \
         --calib calib.yaml --board 9x6 --square-size 0.025

Pega la matriz 'pose' que imprime en configs/frodobot_rover.yaml.

Autoprueba de la matematica (no necesita robot ni fotos):
  python tools/camera_pose_from_ground.py selftest
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


# ------------------------------------------------------------------ nucleo

def pose_from_board_on_ground(rvec: np.ndarray, tvec: np.ndarray) -> dict:
    """Convierte la pose del tablero (salida de solvePnP) en T_world_camera.

    Mundo: x adelante, y izquierda, z arriba, origen en el piso bajo la camara.
    Camara: optical frame (x derecha de imagen, y abajo, z adelante).
    """
    r, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))
    t = np.asarray(tvec, dtype=np.float64).reshape(3)

    # Ejes de la camara expresados en el marco del tablero
    x_cam_b = r.T[:, 0]
    z_cam_b = r.T[:, 2]
    # Centro de la camara en el marco del tablero
    c_b = -r.T @ t

    # El tablero esta en el plano z=0 de su propio marco. Su normal es [0,0,1],
    # pero puede apuntar hacia arriba o hacia abajo segun como lo detecto
    # OpenCV. La camara esta arriba, asi que orientamos la normal hacia ella.
    s = 1.0 if c_b[2] >= 0 else -1.0
    up_b = np.array([0.0, 0.0, s])

    height = abs(float(c_b[2]))
    if height < 1e-6:
        raise ValueError("La camara quedo sobre el plano del tablero. Algo esta mal.")

    # Adelante = eje optico proyectado sobre el piso
    fwd_b = z_cam_b - np.dot(z_cam_b, up_b) * up_b
    n = np.linalg.norm(fwd_b)
    if n < 1e-6:
        raise ValueError("La camara apunta perpendicular al piso; no se puede "
                         "definir una direccion de avance.")
    fwd_b /= n
    left_b = np.cross(up_b, fwd_b)

    # Cambio de base: expresar los ejes de la camara en coordenadas del mundo
    basis = np.stack([fwd_b, left_b, up_b], axis=0)  # filas = ejes del mundo
    r_world_cam = np.stack([basis @ r.T[:, i] for i in range(3)], axis=1)

    pose = np.eye(4)
    pose[:3, :3] = r_world_cam
    pose[:3, 3] = np.array([0.0, 0.0, height])

    pitch_down = math.degrees(math.asin(float(np.clip(-np.dot(z_cam_b, up_b), -1, 1))))
    roll = math.degrees(math.asin(float(np.clip(np.dot(x_cam_b, up_b), -1, 1))))

    return {
        "pose": pose,
        "height_m": height,
        "pitch_down_deg": pitch_down,
        "roll_deg": roll,
        "board_distance_m": float(np.linalg.norm(t)),
    }


# ------------------------------------------------------------------ comandos

def cmd_capture(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from genie_rover.sdk_client import RoverClient

    cols, rows = _parse_board(args.board)
    client = RoverClient(args.base_url)

    print("Apoya el tablero PLANO en el piso, delante del robot.")
    print("ENTER para capturar, 'q'+ENTER para salir.\n")
    while True:
        img, _ = client.front_frame()
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        found, _ = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        state = "TABLERO DETECTADO" if found else "no lo veo"
        key = input(f"  {img.shape[1]}x{img.shape[0]} — {state} > ").strip().lower()
        if key == "q":
            return 0
        if not found:
            print("     acercalo, mejora la luz, o corregi el angulo")
            continue
        cv2.imwrite(args.out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"     guardado {args.out} ({img.shape[1]}x{img.shape[0]})")
        return 0


def cmd_solve(args) -> int:
    cols, rows = _parse_board(args.board)
    square = float(args.square_size)

    calib = yaml.safe_load(Path(args.calib).read_text())
    k = np.asarray(calib["intrinsics"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(calib.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1)
    calib_w, calib_h = [int(v) for v in calib["image_size"]]

    img = cv2.imread(args.image)
    if img is None:
        print(f"No pude abrir {args.image}")
        return 1
    h, w = img.shape[:2]

    if (w, h) != (calib_w, calib_h):
        sx, sy = w / calib_w, h / calib_h
        print(f"La foto es {w}x{h} y la calibracion {calib_w}x{calib_h}: reescalo K")
        k = k.copy()
        k[0, 0] *= sx; k[0, 2] *= sx
        k[1, 1] *= sy; k[1, 2] *= sy

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
    if not found:
        print("No se detecta el tablero en esa imagen.")
        return 1
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

    ok, rvec, tvec = cv2.solvePnP(objp, corners, k, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        print("solvePnP no convergio.")
        return 1

    proj, _ = cv2.projectPoints(objp, rvec, tvec, k, dist)
    err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)))

    res = pose_from_board_on_ground(rvec, tvec)

    print(f"\nError de reproyeccion: {err:.3f} px")
    if err > 2.0:
        print("  ADVERTENCIA: alto. Revisa que los intrinsecos sean de esta camara")
        print("  y que el tablero este realmente plano.")
    print(f"Tablero a {res['board_distance_m']:.2f} m de la camara\n")
    print(f"  height_m       = {res['height_m']:.4f}")
    print(f"  pitch_down_deg = {res['pitch_down_deg']:.2f}")
    print(f"  roll_deg       = {res['roll_deg']:.2f}")
    if abs(res["roll_deg"]) > 5:
        print("  (roll alto: la camara esta torcida respecto del horizonte)")

    print("\nPega esto en configs/frodobot_rover.yaml, bajo camera:\n")
    print("  pose:")
    for row in res["pose"]:
        print("    - [" + ", ".join(f"{v:.9f}" for v in row) + "]")
    print("\nCon 'pose' presente se ignoran height_m y pitch_down_deg.")
    return 0


def cmd_selftest(_args) -> int:
    """Verifica la matematica sintetizando una pose conocida."""
    print("Genero un tablero en el piso visto por una camara de pose conocida,")
    print("proyecto, corro solvePnP y comparo con la verdad.\n")

    cols, rows, square = 9, 6, 0.025
    k = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from genie_rover.perception import camera_pose_from_height_pitch

    worst = 0.0
    for height, pitch in [(0.18, 15.0), (0.30, 25.0), (0.22, 35.0), (1.30, 25.0)]:
        pose_true = camera_pose_from_height_pitch(height, pitch)
        r_wc, t_wc = pose_true[:3, :3], pose_true[:3, 3]

        # Tablero apoyado en el piso, rotado un angulo arbitrario y corrido
        yaw = math.radians(37.0)
        c, s = math.cos(yaw), math.sin(yaw)
        r_wb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        t_wb = np.array([0.9, -0.06, 0.0])

        objp = np.zeros((rows * cols, 3))
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
        pts_world = (r_wb @ objp.T).T + t_wb

        # mundo -> camara
        r_cw = r_wc.T
        pts_cam = (r_cw @ (pts_world - t_wc).T).T
        if np.any(pts_cam[:, 2] <= 0):
            print(f"  h={height} pitch={pitch}: el tablero cae detras de la camara, salteo")
            continue
        uv = (k @ pts_cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]

        ok, rvec, tvec = cv2.solvePnP(objp.astype(np.float32), uv.astype(np.float32),
                                      k, None, flags=cv2.SOLVEPNP_ITERATIVE)
        assert ok
        res = pose_from_board_on_ground(rvec, tvec)

        dh = abs(res["height_m"] - height)
        dp = abs(res["pitch_down_deg"] - pitch)
        dpose = float(np.max(np.abs(res["pose"] - pose_true)))
        worst = max(worst, dpose)
        print(f"  h={height:.2f} pitch={pitch:.0f} -> recuperado "
              f"h={res['height_m']:.4f} (err {dh*1000:.2f} mm), "
              f"pitch={res['pitch_down_deg']:.2f} (err {dp:.3f} grados), "
              f"roll={res['roll_deg']:+.3f}")
        assert dh < 1e-3 and dp < 0.05, "la reconstruccion no coincide"

    print(f"\nMaxima diferencia en la matriz 4x4: {worst:.2e}")
    print("La matematica cierra. El error real va a venir de la calibracion")
    print("y de que el tablero este bien plano, no de esta cuenta.")
    return 0


def _parse_board(s: str) -> tuple[int, int]:
    a, b = s.lower().split("x")
    return int(a), int(b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture")
    c.add_argument("--out", default="pose_frame.png")
    c.add_argument("--base-url", default="http://localhost:8000")
    c.add_argument("--board", default="9x6")
    c.set_defaults(func=cmd_capture)

    s = sub.add_parser("solve")
    s.add_argument("--image", required=True)
    s.add_argument("--calib", default="calib.yaml")
    s.add_argument("--board", default="9x6")
    s.add_argument("--square-size", type=float, default=0.025)
    s.set_defaults(func=cmd_solve)

    t = sub.add_parser("selftest")
    t.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
