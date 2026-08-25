#!/usr/bin/env python3
"""Sesion de mapeo: earth_rover_bridge + camera_info + RTAB-Map (mono, odom externa).

Que hace RTAB-Map aca (y que NO hace):

  SI hace: usa /earth_rover/odom (rueda+giro, con escala metrica correcta)
  como odometria base, y le suma cierre de bucles por apariencia visual
  (RGB) para corregir la deriva acumulada cuando el rover vuelve a pasar por
  un lugar ya visto. Publica la correccion como TF `map -> odom`.

  NO hace: una grilla de ocupacion 2D confiable para navegar. Sin
  profundidad ni lidar (esta camara es RGB monocular), `Grid/FromDepth` se
  deja en false a proposito -- no tiene con que reconstruir una grilla de
  obstaculos util. Esa parte la resuelve, por separado, el programa de
  mapeo de genie_rover (`genie/genie_rover/Indoor/map_session.py`), que
  arma la grilla con su propia percepcion (SAM-TP) usando como pose la
  correccion `map -> odom` que este launch publica.

Requiere:
    sudo apt install ros-$ROS_DISTRO-rtabmap-ros
    pip install requests websocket-client opencv-python

Uso RECOMENDADO (deriva todos los parametros del yaml de genie, asi la
odometria/extrinsecos de este stack y los de genie_rover no divergen):

    ./rover_launch.sh mapping-ros2 --db ~/maps/sesion1.db \\
        --config genie/configs/indoor_mapping.yaml

Uso manual equivalente:

    ros2 launch rtabmap_mapping.launch.py \\
        database_path:=~/maps/sesion1.db \\
        $(python3 ../config_to_ros_params.py ../../../genie/configs/indoor_mapping.yaml)

CAMBIOS respecto de la version anterior de este archivo:
  * `database_path` se expande (`~`) y se le crea la carpeta padre ANTES de
    arrancar rtabmap. Antes, si la carpeta no existia o el path traia `~`
    sin expandir, rtabmap caia silenciosamente a `~/.ros/rtabmap_mapping.db`
    y la sesion "se perdia" (esta documentado en 2_Funcionamiento_Mapeo.md).
  * `camera_d` (coeficientes de distorsion) ahora se puede pasar: la camara
    del Mini tiene barril fuerte (k1 ~ -0.26) y publicar `d = [0,0,0,0,0]`
    le daba a RTAB-Map una geometria equivocada.
  * Los parametros de odometria (`wheel_radius_m`, `track_width_m`,
    `rotation_sign`, indices de rpm, giroscopo, correccion GPS) y los
    extrinsecos de camara (`camera_pose_*`) se pueden pasar por launch en
    vez de quedar en los defaults hardcodeados del bridge, que NO coincidian
    con los del yaml de genie.
  * `delete_db_on_start` es un argumento (antes estaba fijo en el codigo, asi
    que apuntar a una base existente la borraba sin avisar).
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)  # .../ros2/


ARGS = [
    ("sdk_url", "http://localhost:8000"),
    ("feed_fps", "15"),
    ("database_path", "~/maps/rtabmap_mapping.db"),
    ("delete_db_on_start", "true"),

    # --- camara: mismos valores que camera.* del yaml de genie -------------
    ("camera_width", "1920"),
    ("camera_height", "1080"),
    ("camera_fx", "925.265722853314"),
    ("camera_fy", "924.6288195383473"),
    ("camera_cx", "962.3052531643399"),
    ("camera_cy", "528.3891677741947"),
    # plumb_bob: k1,k2,p1,p2,k3. "[]" = sin distorsion (lo de antes).
    ("camera_d", "[-0.26257227523222204,0.06503885855790396,"
                 "-0.00031427372195389234,-0.0011895820486559178,"
                 "-0.0073726075647136185]"),
    ("camera_pose_height_m", "0.150"),
    ("camera_pose_pitch_down_deg", "1.85"),
    ("camera_pose_forward_offset_m", "0.0"),

    # --- odometria: mismos valores que odometry.* del yaml de genie --------
    ("wheel_radius_m", "0.0527"),
    ("track_width_m", "0.15"),
    ("rotation_sign", "1.0"),
    ("left_rpm_indices", "[0,2]"),
    ("right_rpm_indices", "[1,3]"),
    ("use_gyro_for_rotation", "true"),
    ("gps_correction", "false"),
]


def _resolve_db_path(raw: str) -> str:
    """expanduser + crear la carpeta padre. Devuelve un path absoluto.

    Sin esto rtabmap se guarda la sesion en ~/.ros/rtabmap_mapping.db cuando
    el path pedido no se puede abrir, y el usuario se entera recien al final.
    """
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(raw)))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def _launch_setup(context, *_args, **_kwargs):
    def cfg(name: str) -> str:
        return LaunchConfiguration(name).perform(context)

    db_path = _resolve_db_path(cfg("database_path"))
    delete_db = cfg("delete_db_on_start").strip().lower() in ("1", "true", "yes", "on")

    rtabmap_params = {
        "frame_id": "base_link",
        "odom_frame_id": "odom",
        "map_frame_id": "map",
        "subscribe_depth": False,
        "subscribe_rgb": True,
        "subscribe_odom_info": False,
        "approx_sync": True,
        "database_path": db_path,
        # Robot plano (el mundo de genie_rover es 2D: x,y,theta) -> restringe
        # la optimizacion del grafo a 3DoF, mucho mas estable sin depth/lidar.
        "Reg/Force3DoF": "true",
        # Sin depth/lidar no hay con que armar una grilla de obstaculos
        # confiable -- lo dejamos apagado a proposito, ver docstring arriba.
        "Grid/FromDepth": "false",
        "RGBD/NeighborLinkRefining": "false",
        "RGBD/ProximityBySpace": "false",
        # Cierre de bucles por apariencia (Bayes) es lo que SI aporta valor
        # aca: corrige la deriva de la odometria de rueda cuando reconoce un
        # lugar ya visitado.
        "RGBD/OptimizeFromGraphEnd": "false",
        "Rtabmap/DetectionRate": "1.0",
    }

    bridge_params = [
        "sdk_url", "feed_fps",
        "wheel_radius_m", "track_width_m", "rotation_sign",
        "left_rpm_indices", "right_rpm_indices",
        "use_gyro_for_rotation", "gps_correction",
    ]
    bridge_cmd = ["python3", os.path.join(PARENT_DIR, "earth_rover_bridge.py"), "--ros-args"]
    for name in bridge_params:
        bridge_cmd += ["-p", f"{name}:={cfg(name)}"]
    # los extrinsecos usan otro nombre del lado del nodo
    for launch_name, node_name in (
        ("camera_pose_height_m", "camera_height_m"),
        ("camera_pose_pitch_down_deg", "camera_pitch_down_deg"),
        ("camera_pose_forward_offset_m", "camera_forward_offset_m"),
    ):
        bridge_cmd += ["-p", f"{node_name}:={cfg(launch_name)}"]

    caminfo_cmd = ["python3", os.path.join(THIS_DIR, "camera_info_publisher.py"), "--ros-args"]
    for launch_name, node_name in (
        ("camera_width", "width"), ("camera_height", "height"),
        ("camera_fx", "fx"), ("camera_fy", "fy"),
        ("camera_cx", "cx"), ("camera_cy", "cy"),
        ("camera_d", "d"),
    ):
        caminfo_cmd += ["-p", f"{node_name}:={cfg(launch_name)}"]

    rtabmap_args = ["--delete_db_on_start"] if delete_db else []

    print(f"[rtabmap_mapping.launch.py] database_path -> {db_path}"
          f"{' (se borra al arrancar)' if delete_db else ' (se continua la base existente)'}")

    return [
        # --- bridge del SDK: imagen, imu, gps, /earth_rover/odom y TF -------
        ExecuteProcess(cmd=bridge_cmd, output="screen"),
        # --- camera_info: misma calibracion que usa genie_rover ------------
        ExecuteProcess(cmd=caminfo_cmd, output="screen"),
        # --- RTAB-Map: odometria externa (nuestra) + loop closure visual ---
        Node(
            package="rtabmap_slam",
            executable="rtabmap",
            name="rtabmap",
            output="screen",
            parameters=[rtabmap_params],
            remappings=[
                ("rgb/image", "/earth_rover/front/image_raw"),
                ("rgb/camera_info", "/earth_rover/front/camera_info"),
                ("odom", "/earth_rover/odom"),
            ],
            arguments=rtabmap_args,
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(name, default_value=default) for name, default in ARGS]
        + [OpaqueFunction(function=_launch_setup)]
    )
