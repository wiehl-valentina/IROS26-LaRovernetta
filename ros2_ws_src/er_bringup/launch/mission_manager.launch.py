import os
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
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
    # --------------------------------------------------------------------------
    # 1. Directorios de Paquetes ROS 2
    # --------------------------------------------------------------------------
    er_navigation_dir = get_package_share_directory('er_navigation')
    er_planning_dir = get_package_share_directory('er_planning')
    er_mission_dir = get_package_share_directory('er_mission')
    mpl_dir = get_package_share_directory('mini_plus_localization')

    # Entorno Python adicional
    node_env = conda_python_env()

    # --------------------------------------------------------------------------
    # 2. Argumentos de Lanzamiento
    # --------------------------------------------------------------------------
    enable_global_planning_arg = DeclareLaunchArgument(
        'enable_global_planning',
        default_value='false',
        description='Habilita el mapa persistente Bayesiano y el planificador global D* Lite (true/false)',
    )

    mission_slug_arg = DeclareLaunchArgument(
        'mission_slug',
        default_value='mission-1',
        description='Identificador de la misión en FrodoBots SDK (default: mission-1)',
    )

    bot_slug_arg = DeclareLaunchArgument(
        'bot_slug',
        default_value='',
        description='Slug del rover en FrodoBots (si está vacío, utiliza el configurado en .env)',
    )

    sdk_url_arg = DeclareLaunchArgument(
        'sdk_url',
        default_value='http://127.0.0.1:8000',
        description='URL base del servidor SDK FastAPI / Hypercorn',
    )

    enable_global_planning = LaunchConfiguration('enable_global_planning')
    sdk_url = LaunchConfiguration('sdk_url')

    # --------------------------------------------------------------------------
    # 3. Inclusión del Pipeline de Fusión Sensorial EKF (robot_localization)
    # --------------------------------------------------------------------------
    ekf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(mpl_dir, 'launch', 'ekf.launch.py')
        )
    )

    # --------------------------------------------------------------------------
    # 4. Definición de Nodos Principales
    # --------------------------------------------------------------------------
    # Bridge de comunicación con hardware / SDK
    bridge_node = Node(
        package='earth_rovers_sdk',
        executable='earth_rover_bridge',
        name='earth_rover_bridge',
        output='screen',
        parameters=[{
            'sdk_url': sdk_url,
            'gps_position_covariance': [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 4.0,
            ],
            'odom_twist_covariance': [
                2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.5, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.5, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.5, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.5,
            ],
        }],
        additional_env=node_env,
    )

    # Percepción y Planificación Local BEV (SAM-TP + GeNIE Path Bank)
    planner_node = Node(
        package='er_planning',
        executable='bev_planner_node',
        name='bev_planner_node',
        output='screen',
        parameters=[os.path.join(er_planning_dir, 'config', 'planner_params.yaml')],
        additional_env=node_env,
    )

    # Mapa Persistente Global (Opcional - Fase 4)
    persistent_map_node = Node(
        package='er_planning',
        executable='persistent_map_node',
        name='persistent_map_node',
        output='screen',
        parameters=[os.path.join(er_planning_dir, 'config', 'persistent_map_params.yaml')],
        condition=IfCondition(enable_global_planning),
        additional_env=node_env,
    )

    # Planificador Global Incremental D* Lite (Opcional - Fase 4)
    global_planner_node = Node(
        package='er_planning',
        executable='global_planner_node',
        name='global_planner_node',
        output='screen',
        parameters=[os.path.join(er_planning_dir, 'config', 'global_planner_params.yaml')],
        condition=IfCondition(enable_global_planning),
        additional_env=node_env,
    )

    # Controlador Motriz Reactivo y Seguimiento de Rutas BEV
    navigation_node = Node(
        package='er_navigation',
        executable='gps_waypoint_controller',
        name='gps_waypoint_controller',
        output='screen',
        parameters=[os.path.join(er_navigation_dir, 'config', 'navigation_params.yaml')],
        additional_env=node_env,
    )

    # Orquestador de Misión de Alto Nivel (Stateful Manager)
    mission_node = Node(
        package='er_mission',
        executable='mission_manager_node',
        name='mission_manager_node',
        output='screen',
        parameters=[
            os.path.join(er_mission_dir, 'config', 'mission_params.yaml'),
            {'sdk_url': sdk_url},
        ],
        remappings=[('earth_rover/gps', 'gps/filtered')],
        additional_env=node_env,
    )

    # --------------------------------------------------------------------------
    # 5. Grafo de Ejecución Orquestado
    # --------------------------------------------------------------------------
    return LaunchDescription([
        # Argumentos
        enable_global_planning_arg,
        mission_slug_arg,
        bot_slug_arg,
        sdk_url_arg,

        LogInfo(msg="[MISSION MANAGER] [FASE 1] Inicializando Bridge SDK, Filtros EKF y Percepción BEV..."),

        # Fase 1: Infraestructura base, EKF y Percepción (inmediatos)
        bridge_node,
        ekf_launch,
        planner_node,
        persistent_map_node,
        global_planner_node,

        # Fase 2: Control motriz y orquestador de misión (tras estabilización de 4s)
        TimerAction(
            period=4.0,
            actions=[
                LogInfo(msg="[MISSION MANAGER] [FASE 2] Sensores estabilizados. Arrancando Control de Navegación y Gestor de Misión..."),
                navigation_node,
                mission_node,
            ],
        ),
    ])
