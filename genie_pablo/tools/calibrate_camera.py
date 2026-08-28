"""Calibra la camara frontal del rover con un tablero de ajedrez.

Sin esto, project_score_to_bev proyecta con la K del Stretch y el BEV sale
deformado: el robot cree que el suelo esta donde no esta.

Uso, en dos pasos:

  1) Capturar. Imprimi un tablero (9x6 esquinas internas es el estandar),
     pegalo sobre algo rigido, y movelo delante del rover: distintos angulos,
     distancias y zonas de la imagen. Unas 20 capturas buenas alcanzan.

        python tools/calibrate_camera.py capture --out calib_frames/

     Apreta ENTER para capturar, 'q' para terminar. Va avisando si detecta el
     tablero en cada frame.

  2) Calibrar.

        python tools/calibrate_camera.py solve --frames calib_frames/ \
            --board 9x6 --square-size 0.025 --out calib.yaml

     Pega intrinsics, dist_coeffs e image_size en configs/frodobot_rover.yaml.

Un error de reproyeccion por debajo de ~0.5 px esta bien. Por encima de 1.0 px,
recapturá con mejor iluminacion y mas variedad de poses.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


def cmd_capture(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from genie_rover.sdk_client import RoverClient

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = RoverClient(args.base_url)
    cols, rows = _parse_board(args.board)

    print(f"Capturando en {out}/  — ENTER para guardar, 'q'+ENTER para salir")
    n = 0
    while True:
        try:
            img, _ = client.front_frame()
        except Exception as exc:
            print(f"  error leyendo el frame: {exc}")
            time.sleep(1.0)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        found, _ = cv2.findChessboardCorners(
            gray, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
        )
        status = "TABLERO DETECTADO" if found else "no se ve el tablero"
        key = input(f"  [{n} guardados] {img.shape[1]}x{img.shape[0]} — {status} > ").strip().lower()
        if key == "q":
            break
        if not found:
            print("     salteado (movelo o mejora la luz)")
            continue
        path = out / f"calib_{n:03d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        n += 1
        print(f"     guardado {path}")

    print(f"\n{n} imagenes en {out}. Ahora corré el subcomando 'solve'.")
    return 0


def cmd_solve(args) -> int:
    cols, rows = _parse_board(args.board)
    square = float(args.square_size)

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    files = sorted(p for p in Path(args.frames).iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files:
        print(f"No hay imagenes en {args.frames}")
        return 1

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != image_size:
            print(f"  {f.name}: resolucion distinta, salteada")
            continue

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if not found:
            print(f"  {f.name}: sin tablero")
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        print(f"  {f.name}: ok")

    if len(obj_points) < 6:
        print(f"\nSolo {len(obj_points)} vistas validas. Hacen falta al menos 6, "
              "idealmente 15-20.")
        return 1

    rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    total = 0.0
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], k, dist)
        total += cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
    mean_err = total / len(obj_points)

    print(f"\nRMS: {rms:.4f}")
    print(f"Error medio de reproyeccion: {mean_err:.4f} px")
    if mean_err > 1.0:
        print("  ADVERTENCIA: por encima de 1 px. Recapturá con mas variedad de poses.")
    print(f"\nfx={k[0,0]:.2f} fy={k[1,1]:.2f} cx={k[0,2]:.2f} cy={k[1,2]:.2f}")

    result = {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "intrinsics": [[float(v) for v in row] for row in k],
        "dist_coeffs": [float(v) for v in dist.reshape(-1)],
        "reprojection_error_px": float(mean_err),
    }
    Path(args.out).write_text(yaml.safe_dump(result, sort_keys=False))
    print(f"\nEscrito {args.out}. Copiá image_size, intrinsics y dist_coeffs "
          "a configs/frodobot_rover.yaml.")
    return 0


def _parse_board(s: str) -> tuple[int, int]:
    a, b = s.lower().split("x")
    return int(a), int(b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture")
    c.add_argument("--out", default="calib_frames")
    c.add_argument("--base-url", default="http://localhost:8000")
    c.add_argument("--board", default="9x6", help="esquinas internas, ej 9x6")
    c.set_defaults(func=cmd_capture)

    s = sub.add_parser("solve")
    s.add_argument("--frames", default="calib_frames")
    s.add_argument("--board", default="9x6")
    s.add_argument("--square-size", type=float, default=0.025,
                   help="lado del cuadrado en metros")
    s.add_argument("--out", default="calib.yaml")
    s.set_defaults(func=cmd_solve)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
