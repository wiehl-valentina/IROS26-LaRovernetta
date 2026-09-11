import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    localization_dir = get_package_share_directory('mini_plus_localization')

    sdk_url_arg = DeclareLaunchArgument(
        'sdk_url',
        default_value='http://host.docker.internal:8000',
        description='URL del servidor HTTP del SDK'
    )
    target_ip_arg = DeclareLaunchArgument(
        'target_ip',
        default_value='127.0.0.1',
        description='IP de destino para exportar el socket UDP'
    )
    target_port_arg = DeclareLaunchArgument(
        'target_port',
        default_value='9876',
        description='Puerto UDP de destino'
    )

    return LaunchDescription([
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
