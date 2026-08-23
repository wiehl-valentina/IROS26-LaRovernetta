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

Uso:
    ros2 launch rtabmap_mapping.launch.py \\
        sdk_url:=http://localhost:8000 \\
        camera_fx:=900.0 camera_fy:=900.0 camera_cx:=640.0 camera_cy:=360.0 \\
        camera_width:=1280 camera_height:=720 \\
        database_path:=/home/user/maps/sesion1.db

Al terminar (Ctrl+C), la base .db queda en database_path. Para pasarla a
formato ROS map_server (.yaml+.pgm) usa el Database Viewer de RTAB-Map
("File -> Export 2D occupancy grid") -- aunque, por la limitacion de arriba,
la grilla que exporte va a ser pobre; la fuente de verdad para la grilla de
obstaculos es el export de PersistentMap (ver genie/genie_rover/Indoor/map_session.py).
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(this_dir)  # examples/ros2/

    sdk_url = LaunchConfiguration("sdk_url")
    feed_fps = LaunchConfiguration("feed_fps")
    database_path = LaunchConfiguration("database_path")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")
    camera_fx = LaunchConfiguration("camera_fx")
    camera_fy = LaunchConfiguration("camera_fy")
    camera_cx = LaunchConfiguration("camera_cx")
    camera_cy = LaunchConfiguration("camera_cy")
    wheel_radius_m = LaunchConfiguration("wheel_radius_m")
    track_width_m = LaunchConfiguration("track_width_m")

    rtabmap_params = {
        "frame_id": "base_link",
        "odom_frame_id": "odom",
        "map_frame_id": "map",
        "subscribe_depth": False,
        "subscribe_rgb": True,
        "subscribe_odom_info": False,
        "approx_sync": True,
        "database_path": database_path,
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

    return LaunchDescription([
        DeclareLaunchArgument("sdk_url", default_value="http://localhost:8000"),
        DeclareLaunchArgument("feed_fps", default_value="15"),
        DeclareLaunchArgument("database_path", default_value="~/.ros/rtabmap_mapping.db"),
        DeclareLaunchArgument("camera_width", default_value="1920"),
        DeclareLaunchArgument("camera_height", default_value="1080"),
        DeclareLaunchArgument("camera_fx", default_value="925.265722853314"),
        DeclareLaunchArgument("camera_fy", default_value="924.6288195383473"),
        DeclareLaunchArgument("camera_cx", default_value="962.3052531643399"),
        DeclareLaunchArgument("camera_cy", default_value="528.3891677741947"),
        DeclareLaunchArgument("wheel_radius_m", default_value="0.0527"),
        DeclareLaunchArgument("track_width_m", default_value="0.15"),


        # --- bridge del SDK: imagen, imu, gps, y /earth_rover/odom + TF -----
        ExecuteProcess(
            cmd=[
                "python3", PathJoinSubstitution([parent_dir, "earth_rover_bridge.py"]),
                "--ros-args",
                "-p", ["sdk_url:=", sdk_url],
                "-p", ["feed_fps:=", feed_fps],
                "-p", ["wheel_radius_m:=", wheel_radius_m],
                "-p", ["track_width_m:=", track_width_m],
            ],
            output="screen",
        ),

        # --- camera_info: misma calibracion que uses en genie_rover ---------
        ExecuteProcess(
            cmd=[
                "python3", PathJoinSubstitution([this_dir, "camera_info_publisher.py"]),
                "--ros-args",
                "-p", ["width:=", camera_width],
                "-p", ["height:=", camera_height],
                "-p", ["fx:=", camera_fx],
                "-p", ["fy:=", camera_fy],
                "-p", ["cx:=", camera_cx],
                "-p", ["cy:=", camera_cy],
            ],
            output="screen",
        ),

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
            arguments=["--delete_db_on_start"],
        ),
    ])
