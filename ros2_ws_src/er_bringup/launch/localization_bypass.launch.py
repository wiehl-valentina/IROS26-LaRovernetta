import os
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def check_duplicate_nodes(context, *args, **kwargs):
    skip = LaunchConfiguration('skip_guard').perform(context).lower() in ('true', '1', 'yes')
    if skip or os.environ.get('ALLOW_DUPLICATE_STACK') == '1':
        return []

    conflicting = {
        'ekf_filter_node_map',
        'ekf_filter_node_odom',
        'ekf_heading_bridge',
        'earth_rover_bridge',
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
                f"[ERROR FATAL] GUARDA DE ARRANQUE: STACK EKF DUPLICADO DETECTADO EN DDS\n"
                f"{'='*80}\n"
                f"Ya existen los siguientes nodos activos en el dominio (ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}):\n"
                f"{formatted_nodes}\n\n"
                f"No se puede iniciar localization_bypass.launch.py mientras otro stack esté activo.\n"
                f"Detenga el stack previo o use skip_guard:=true si se trata de un test aislado.\n"
                f"{'='*80}"
            )
    except FileNotFoundError:
        pass
    return []

def generate_launch_description():
    localization_dir = get_package_share_directory('mini_plus_localization')

    declare_skip_guard = DeclareLaunchArgument(
        'skip_guard',
        default_value='false',
        description='Omitir verificación de colisión de nodos duplicados en DDS',
    )
    sdk_url_arg = DeclareLaunchArgument(
        'sdk_url',
        default_value=os.environ.get('SDK_URL', 'http://localhost:8000'),
        description='URL del servidor HTTP del SDK'
    )
    target_ip_arg = DeclareLaunchArgument(
        'target_ip',
        default_value=os.environ.get('TARGET_IP', '127.0.0.1'),
        description='IP de destino para exportar el socket UDP'
    )
    target_port_arg = DeclareLaunchArgument(
        'target_port',
        default_value=os.environ.get('TARGET_PORT', '9876'),
        description='Puerto UDP de destino'
    )

    return LaunchDescription([
        declare_skip_guard,
        OpaqueFunction(function=check_duplicate_nodes),
        sdk_url_arg,
        target_ip_arg,
        target_port_arg,
        # 1. Bridge para obtener datos del SDK (sin control motriz de retorno activo porque no hay gps_waypoint_controller)
        Node(
            package='earth_rovers_sdk',
            executable='bridge_node',
            name='earth_rover_bridge',
            output='screen',
            parameters=[{
                'sdk_url': LaunchConfiguration('sdk_url'),
            }],
        ),
        # 2. Stack de EKF (local, global y heading bridge)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(localization_dir, 'launch', 'ekf.launch.py')
            )
        ),
        # 3. Nuestro Exportador UDP
        Node(
            package='mini_plus_localization',
            executable='udp_heading_exporter.py',
            name='udp_heading_exporter',
            output='screen',
            parameters=[{
                'target_ip': LaunchConfiguration('target_ip'),
                'target_port': LaunchConfiguration('target_port'),
            }],
        ),
    ])
