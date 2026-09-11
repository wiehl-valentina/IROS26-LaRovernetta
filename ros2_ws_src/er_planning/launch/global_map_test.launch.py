import os
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node


def conda_python_env():
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        return {}

    conda_site_packages = os.path.join(
        conda_prefix,
        'lib',
        f'python{sys.version_info.major}.{sys.version_info.minor}',
        'site-packages',
    )
    if not os.path.isdir(conda_site_packages):
        return {}

    pythonpath = os.environ.get('PYTHONPATH', '')
    paths = [conda_site_packages]
    if pythonpath:
        paths.append(pythonpath)
    return {'PYTHONPATH': os.pathsep.join(paths)}


def generate_launch_description():
    er_planning_dir = get_package_share_directory('er_planning')
    node_env = conda_python_env()

    planner_yaml = os.path.join(er_planning_dir, 'config', 'planner_params.yaml')
    map_yaml = os.path.join(er_planning_dir, 'config', 'persistent_map_params.yaml')
    global_planner_yaml = os.path.join(er_planning_dir, 'config', 'global_planner_params.yaml')

    # 1. Nodo de Planificación Local BEV y Percepción SAM-TP
    bev_planner_node = Node(
        package='er_planning',
        executable='bev_planner_node',
        name='bev_planner_node',
        output='screen',
        parameters=[planner_yaml],
        additional_env=node_env,
    )

    # 2. Nodo de Mapa Persistente Global (Acumulador con Confianza y Decaimiento)
    persistent_map_node = Node(
        package='er_planning',
        executable='persistent_map_node',
        name='persistent_map_node',
        output='screen',
        parameters=[map_yaml],
        additional_env=node_env,
    )

    # 3. Nodo de Planificación Global Incremental D* Lite
    global_planner_node = Node(
        package='er_planning',
        executable='global_planner_node',
        name='global_planner_node',
        output='screen',
        parameters=[global_planner_yaml],
        additional_env=node_env,
    )

    return LaunchDescription([
        LogInfo(msg="[TEST LAUNCH] Iniciando Pipeline de Mapa Persistente y Planificador Global D* Lite (Modo Aislado)..."),
        bev_planner_node,
        persistent_map_node,
        global_planner_node,
    ])
