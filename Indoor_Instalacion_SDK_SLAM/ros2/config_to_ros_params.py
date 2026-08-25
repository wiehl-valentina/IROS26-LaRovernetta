#!/usr/bin/env python3
"""Traduce un config YAML de genie_rover a parametros del stack ROS2.

POR QUE EXISTE
--------------
Habia DOS fuentes de verdad para los mismos numeros fisicos:

  * `genie/configs/*.yaml`      -> lo que usa genie_rover (perception, odometry,
                                   PersistentMap, planner).
  * defaults hardcodeados en    -> lo que usaba el stack ROS2
    `earth_rover_bridge.py` y
    `rtabmap_mapping.launch.py`

y no coincidian. Ejemplos reales encontrados en el repo:

    parametro                genie yaml     ROS2 (default)   consecuencia
    ---------------------------------------------------------------------
    wheel_radius_m           0.0527         0.045            odometria ~17% corta
    rotation_sign            +1.0           -1.0             el robot "gira al reves"
                                                             para RTAB-Map
    camera_height_m          0.150          0.20             TF de camara mal
    camera_pitch_down_deg    1.85           15.0             TF de camara MUY mal
    dist_coeffs (k1)         -0.2626        0 (sin usar)     camera_info sin distorsion

Con la odometria y el TF de camara distintos entre los dos lados, la
correccion `map -> odom` que publica RTAB-Map no es aplicable a la pose de
genie_rover: es justamente el escenario que `map_session.py` intenta usar.

Este script lee el yaml UNA vez y emite los parametros para que el stack ROS2
salga de la misma fuente de verdad.

USO
---
    # argumentos para "ros2 launch rtabmap_mapping.launch.py ..."
    python3 config_to_ros_params.py genie/configs/indoor_mapping.yaml

    # argumentos "-p k:=v" para correr earth_rover_bridge.py suelto
    python3 config_to_ros_params.py --style ros-args genie/configs/indoor_mapping.yaml

    # para leerlo a ojo
    python3 config_to_ros_params.py --style pretty genie/configs/indoor_mapping.yaml

Solo necesita PyYAML. Si el yaml no tiene alguna seccion, ese parametro
simplemente no se emite (el launch/nodo se queda con su default).
"""

from __future__ import annotations

import argparse
import math
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    print("Falta PyYAML: pip3 install pyyaml", file=sys.stderr)
    raise SystemExit(2)


def _pose_to_height_pitch_offset(pose) -> tuple[float, float, float] | None:
    """De la matriz 4x4 T_world_camera saca (altura, pitch_abajo_deg, offset_adelante).

    Convencion (la misma que usa perception.camera_pose_from_height_pitch):
      * columna 3 = traslacion de la camara en el marco del robot,
      * el eje optico de la camara es +Z en el marco de camara, asi que la
        direccion a la que apunta en el mundo es la 3ra COLUMNA de R.

    Devuelve None si la matriz no tiene la forma esperada.
    """
    try:
        rows = [[float(v) for v in row] for row in pose]
    except (TypeError, ValueError):
        return None
    if len(rows) != 4 or any(len(r) != 4 for r in rows):
        return None

    tx, _ty, tz = rows[0][3], rows[1][3], rows[2][3]
    # 3ra columna de R = direccion del eje optico en coordenadas del mundo
    fx_, _fy_, fz_ = rows[0][2], rows[1][2], rows[2][2]
    norm = math.sqrt(fx_ * fx_ + _fy_ * _fy_ + fz_ * fz_)
    if norm < 1e-9:
        return None
    pitch_down_deg = math.degrees(math.asin(max(-1.0, min(1.0, -fz_ / norm))))
    return tz, pitch_down_deg, tx


def derive(cfg: dict) -> dict:
    """dict {nombre_de_parametro_ros: valor}. Solo lo que el yaml realmente define."""
    out: dict[str, object] = {}

    rover = cfg.get("rover") or {}
    if rover.get("base_url"):
        out["sdk_url"] = str(rover["base_url"])

    cam = cfg.get("camera") or {}
    size = cam.get("image_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        out["camera_width"] = int(size[0])
        out["camera_height"] = int(size[1])

    K = cam.get("intrinsics")
    if isinstance(K, (list, tuple)) and len(K) == 3:
        out["camera_fx"] = float(K[0][0])
        out["camera_fy"] = float(K[1][1])
        out["camera_cx"] = float(K[0][2])
        out["camera_cy"] = float(K[1][2])

    dist = cam.get("dist_coeffs")
    if isinstance(dist, (list, tuple)) and len(dist) >= 4:
        # plumb_bob espera 5 (k1,k2,p1,p2,k3); rellenamos con 0 si vienen 4.
        d = [float(v) for v in dist][:5]
        while len(d) < 5:
            d.append(0.0)
        out["camera_d"] = d

    # extrinsecos: 'pose' (4x4) tiene prioridad sobre height_m/pitch_down_deg,
    # igual que en perception.py.
    hpo = _pose_to_height_pitch_offset(cam["pose"]) if isinstance(cam.get("pose"), (list, tuple)) else None
    if hpo is not None:
        height, pitch, forward = hpo
        out["camera_pose_height_m"] = round(height, 6)
        out["camera_pose_pitch_down_deg"] = round(pitch, 4)
        out["camera_pose_forward_offset_m"] = round(forward, 6)
    else:
        if cam.get("height_m") is not None:
            out["camera_pose_height_m"] = float(cam["height_m"])
        if cam.get("pitch_down_deg") is not None:
            out["camera_pose_pitch_down_deg"] = float(cam["pitch_down_deg"])

    odo = cfg.get("odometry") or {}
    for key, cast in (
        ("wheel_radius_m", float),
        ("track_width_m", float),
        ("rotation_sign", float),
    ):
        if odo.get(key) is not None:
            out[key] = cast(odo[key])
    for key in ("left_rpm_indices", "right_rpm_indices"):
        if isinstance(odo.get(key), (list, tuple)):
            out[key] = [int(v) for v in odo[key]]
    for key in ("use_gyro_for_rotation", "gps_correction"):
        if odo.get(key) is not None:
            out[key] = bool(odo[key])

    return out


# Los nombres que entiende cada consumidor. El launch file recibe algunos con
# otro nombre porque ahi son argumentos de launch, no parametros de nodo.
LAUNCH_ONLY = {
    "sdk_url", "camera_width", "camera_height",
    "camera_fx", "camera_fy", "camera_cx", "camera_cy", "camera_d",
}
# parametros que consume earth_rover_bridge.py (nodo)
BRIDGE_PARAM_NAMES = {
    "camera_pose_height_m": "camera_height_m",
    "camera_pose_pitch_down_deg": "camera_pitch_down_deg",
    "camera_pose_forward_offset_m": "camera_forward_offset_m",
}


def _fmt_scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _fmt_value(v) -> str:
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_fmt_scalar(x) for x in v) + "]"
    return _fmt_scalar(v)


def emit_launch(params: dict) -> str:
    """`clave:=valor` para `ros2 launch`. Las listas van como texto entre
    corchetes; el launch file las parsea (ver rtabmap_mapping.launch.py)."""
    parts = []
    for key, val in params.items():
        parts.append(f"{key}:={_fmt_value(val)}")
    return " ".join(parts)


def emit_ros_args(params: dict) -> str:
    """`-p clave:=valor` para correr earth_rover_bridge.py directo."""
    parts = []
    for key, val in params.items():
        if key in LAUNCH_ONLY and key != "sdk_url":
            continue  # esos los consume camera_info_publisher, no el bridge
        name = BRIDGE_PARAM_NAMES.get(key, key)
        parts.append(f"-p {name}:={_fmt_value(val)}")
    return " ".join(parts)


def emit_pretty(params: dict) -> str:
    width = max((len(k) for k in params), default=0)
    return "\n".join(f"{k.ljust(width)} = {_fmt_value(v)}" for k, v in params.items())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="config_to_ros_params.py",
        description="Traduce un config de genie_rover a parametros del stack ROS2.",
    )
    ap.add_argument("config", help="ej. genie/configs/indoor_mapping.yaml")
    ap.add_argument("--style", choices=["launch", "ros-args", "pretty"], default="launch")
    args = ap.parse_args(argv)

    try:
        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except OSError as exc:
        print(f"no pude leer {args.config}: {exc}", file=sys.stderr)
        return 1
    except yaml.YAMLError as exc:
        print(f"yaml invalido en {args.config}: {exc}", file=sys.stderr)
        return 1

    params = derive(cfg)
    if not params:
        print(f"{args.config} no tiene ninguna seccion conocida "
              f"(rover/camera/odometry)", file=sys.stderr)
        return 1

    if args.style == "launch":
        print(emit_launch(params))
    elif args.style == "ros-args":
        print(emit_ros_args(params))
    else:
        print(emit_pretty(params))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
