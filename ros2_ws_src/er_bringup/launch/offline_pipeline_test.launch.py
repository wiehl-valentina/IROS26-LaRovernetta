import os
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
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
    er_navigation_dir = get_package_share_directory('er_navigation')
    er_planning_dir = get_package_share_directory('er_planning')

    node_env = conda_python_env()

    # Argumentos de Configuración
    enable_stimulus = LaunchConfiguration('enable_stimulus')
    declare_enable_stimulus = DeclareLaunchArgument(
        'enable_stimulus',
        default_value='true',
        description='Lanza publicadores sintéticos de cámara y GPS/heading/target',
    )

    image_path = LaunchConfiguration('image_path')
    declare_image_path = DeclareLaunchArgument(
        'image_path',
        default_value='',
        description='Ruta a imagen de prueba (vacío = sintética)',
    )

    seed_map_path = LaunchConfiguration('seed_map_path')
    declare_seed_map_path = DeclareLaunchArgument(
        'seed_map_path',
        default_value='',
        description='Ruta a mapa semilla .npy (vacío = sin semilla)',
    )

    target_dist = LaunchConfiguration('target_distance_m')
    declare_target_dist = DeclareLaunchArgument(
        'target_distance_m',
        default_value='25.0',
        description='Distancia inicial al target en metros',
    )

    target_bearing = LaunchConfiguration('target_bearing_deg')
    declare_target_bearing = DeclareLaunchArgument(
        'target_bearing_deg',
        default_value='0.0',
        description='Azimut inicial al target en grados (0=Norte, 90=Este)',
    )

    initial_heading = LaunchConfiguration('initial_heading_deg')
    declare_initial_heading = DeclareLaunchArgument(
        'initial_heading_deg',
        default_value='90.0',
        description='Rumbo inicial del robot en grados (90=Este, 0=Norte)',
    )

    sim_kinematics = LaunchConfiguration('simulate_kinematics')
    declare_sim_kinematics = DeclareLaunchArgument(
        'simulate_kinematics',
        default_value='true',
        description='Habilita integración cinemática closed-loop desde cmd_vel',
    )

    # --------------------------------------------------------------------------
    # 1. Nodos de Producción (Pipeline Completo Offline)
    # --------------------------------------------------------------------------
    # NODO 1: Cerebro Visual BEV (SAM-TP + GeNIE)
    bev_planner_node = Node(
        package='er_planning',
        executable='bev_planner_node',
        name='bev_planner_node',
        output='screen',
        parameters=[
            os.path.join(er_planning_dir, 'config', 'planner_params.yaml'),
        ],
        additional_env=node_env,
    )

    # NODO 2: Mapa Persistente Global (Integrador Bayesiano + Decaimiento)
    persistent_map_node = Node(
        package='er_planning',
        executable='persistent_map_node',
        name='persistent_map_node',
        output='screen',
        parameters=[
            os.path.join(er_planning_dir, 'config', 'persistent_map_params.yaml'),
            {'seed_map_path': seed_map_path},
        ],
        additional_env=node_env,
    )

    # NODO 3: Planificador Global Incremental (D* Lite Optimizado)
    global_planner_node = Node(
        package='er_planning',
        executable='global_planner_node',
        name='global_planner_node',
        output='screen',
        parameters=[
            os.path.join(er_planning_dir, 'config', 'global_planner_params.yaml'),
        ],
        additional_env=node_env,
    )

    # NODO 4: Controlador Híbrido GPS/Rumbo (ALIGN / DRIVE)
    gps_controller_node = Node(
        package='er_navigation',
        executable='gps_waypoint_controller',
        name='gps_waypoint_controller',
        output='screen',
        parameters=[
            os.path.join(er_navigation_dir, 'config', 'navigation_params.yaml'),
        ],
        additional_env=node_env,
    )

    # --------------------------------------------------------------------------
    # 2. Nodos de Estímulo / Simulación Sintética (Subcarpeta testing/)
    # --------------------------------------------------------------------------
    test_image_node = Node(
        package='er_perception',
        executable='test_image_publisher',
        name='test_image_publisher',
        output='screen',
        parameters=[{
            'image_path': image_path,
            'publish_rate_hz': 10.0,
        }],
        condition=IfCondition(enable_stimulus),
        additional_env=node_env,
    )

    test_nav_node = Node(
        package='er_navigation',
        executable='test_nav_stimulus',
        name='test_nav_stimulus',
        output='screen',
        parameters=[{
            'publish_rate_hz': 10.0,
            'target_distance_m': target_dist,
            'target_bearing_deg': target_bearing,
            'initial_heading_deg': initial_heading,
            'simulate_kinematics': sim_kinematics,
            'broadcast_tf': True,
        }],
        condition=IfCondition(enable_stimulus),
        additional_env=node_env,
    )

    return LaunchDescription([
        declare_enable_stimulus,
        declare_image_path,
        declare_seed_map_path,
        declare_target_dist,
        declare_target_bearing,
        declare_initial_heading,
        declare_sim_kinematics,
        LogInfo(msg="[OFFLINE PIPELINE TEST] Iniciando Pipeline Completo (BEV, Persistent Map, D* Lite, Controller)..."),
        bev_planner_node,
        persistent_map_node,
        global_planner_node,
        gps_controller_node,
        test_image_node,
        test_nav_node,
    ])
