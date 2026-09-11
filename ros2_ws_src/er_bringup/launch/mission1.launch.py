from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
import subprocess
import sys

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


def check_duplicate_nodes(context, *args, **kwargs):
    skip = LaunchConfiguration('skip_guard').perform(context).lower() in ('true', '1', 'yes')
    if skip or os.environ.get('ALLOW_DUPLICATE_STACK') == '1':
        return []

    conflicting = {
        'gps_waypoint_controller',
        'ekf_filter_node_map',
        'ekf_filter_node_odom',
        'ekf_heading_bridge',
        'earth_rover_bridge',
        'mission_manager_node',
        'udp_heading_exporter',
    }
    try:
        res = subprocess.run(['ros2', 'node', 'list'], capture_output=True, text=True, timeout=2)
        active = {line.strip().lstrip('/') for line in res.stdout.splitlines() if line.strip()}
        matches = active.intersection(conflicting)
        if matches:
            formatted_nodes = "\n".join(f"  - /{node}" for node in sorted(matches))
            raise RuntimeError(
                f"\n{'='*80}\n"
                f"[ERROR FATAL] GUARDA DE ARRANQUE: STACK DUPLICADO DETECTADO EN DDS\n"
                f"{'='*80}\n"
                f"Ya existen los siguientes nodos activos en el dominio (ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}):\n"
                f"{formatted_nodes}\n\n"
                f"No se puede iniciar mission1.launch.py mientras otro stack esté activo.\n"
                f"Detenga el stack previo o use skip_guard:=true si se trata de un test aislado.\n"
                f"{'='*80}"
            )
    except FileNotFoundError:
        pass
    return []


def generate_launch_description():
    er_navigation_dir = get_package_share_directory('er_navigation')
    er_planning_dir = get_package_share_directory('er_planning')
    er_mission_dir = get_package_share_directory('er_mission')
    mpl_dir = get_package_share_directory('mini_plus_localization')
    
    # 1. Gestión de Entornos (Conda)
    node_env = conda_python_env()

    # 2. Argumentos de Lanzamiento
    declare_skip_guard = DeclareLaunchArgument(
        'skip_guard',
        default_value='false',
        description='Omitir verificación de colisión de nodos duplicados en DDS',
    )

    enable_global_planning = LaunchConfiguration('enable_global_planning')
    declare_enable_global_planning = DeclareLaunchArgument(
        'enable_global_planning',
        default_value='true',
        description='Lanza persistent_map_node y global_planner_node junto a la misión',
    )

    seed_map_path = LaunchConfiguration('seed_map_path')
    declare_seed_map_path = DeclareLaunchArgument(
        'seed_map_path',
        default_value='',
        description='Ruta absoluta al mapa semilla .npy (vacio = sin precarga)',
    )

    # 3. Inclusión del EKF
    ekf_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(mpl_dir, 'launch', 'ekf.launch.py')
        )
    )

    # ==========================================================
    # DEFINICIÓN DE NODOS
    # ==========================================================
    
    bridge_node = Node(
        package='earth_rovers_sdk',
        executable='earth_rover_bridge',
        name='earth_rover_bridge',
        output='screen',
        parameters=[{
            'sdk_url': 'http://127.0.0.1:8000',
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

    # NODO: Cerebro Visual y Planificador BEV (SAM-TP + GeNIE)
    planner_node = Node(
        package='er_planning',
        executable='bev_planner_node',
        name='bev_planner_node',
        output='screen',
        parameters=[os.path.join(er_planning_dir, 'config', 'planner_params.yaml')],
        additional_env=node_env,
    )

    # NODO: Mapa Persistente Global (Acumulador con Confianza y Decaimiento)
    persistent_map_node = Node(
        package='er_planning',
        executable='persistent_map_node',
        name='persistent_map_node',
        output='screen',
        parameters=[
            os.path.join(er_planning_dir, 'config', 'persistent_map_params.yaml'),
            {'seed_map_path': seed_map_path},
        ],
        condition=IfCondition(enable_global_planning),
        additional_env=node_env,
    )

    # NODO: Planificación Global Incremental D* Lite
    global_planner_node = Node(
        package='er_planning',
        executable='global_planner_node',
        name='global_planner_node',
        output='screen',
        parameters=[os.path.join(er_planning_dir, 'config', 'global_planner_params.yaml')],
        condition=IfCondition(enable_global_planning),
        additional_env=node_env,
    )

    navigation_node = Node(
        package='er_navigation',
        executable='gps_waypoint_controller',
        name='gps_waypoint_controller',
        output='screen',
        parameters=[os.path.join(er_navigation_dir, 'config', 'navigation_params.yaml')],
        additional_env=node_env,
    )

    mission_node = Node(
        package='er_mission',
        executable='mission_manager_node',
        name='mission_manager_node',
        output='screen',
        parameters=[os.path.join(er_mission_dir, 'config', 'mission_params.yaml')],
        remappings=[('earth_rover/gps', 'gps/filtered')],
        additional_env=node_env,
    )

    # ==========================================================
    # GRAFO DE EJECUCIÓN ORQUESTADO (Fases)
    # ==========================================================
    return LaunchDescription([
        declare_skip_guard,
        declare_enable_global_planning,
        declare_seed_map_path,
        OpaqueFunction(function=check_duplicate_nodes),
        LogInfo(msg="[FASE 1] Inicializando Hardware SDK, Filtros EKF y Planificador BEV (PyTorch/GeNIE)..."),
        
        # Arrancan de inmediato:
        bridge_node,
        ekf_launch,
        planner_node,
        persistent_map_node,
        global_planner_node,

        # Arrancan con retraso de 5 segundos para evitar la saturación de CPU de PyTorch:
        TimerAction(
            period=5.0,
            actions=[
                LogInfo(msg="[FASE 2] Sistema estabilizado. Iniciando Control Motriz y Misión Híbrida..."),
                navigation_node,
                mission_node
            ]
        )
    ])