from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node
import os


def generate_launch_description():
    mpl_dir = get_package_share_directory('mini_plus_localization')
    ekf_yaml = os.path.join(mpl_dir, 'config', 'ekf.yaml')

    # Carga de parámetros en capas para navsat_transform:
    # 1. ekf.yaml (base)
    # 2. datum_resolved.yaml (si existe, sobreescribe datum y wait_for_datum sin tocar el resto)
    # Búsqueda robusta en share directory y en árbol fuente sin depender de '..' relativos:
    datum_resolved_yaml = os.path.join(mpl_dir, 'config', 'datum_resolved.yaml')
    if not os.path.isfile(datum_resolved_yaml):
        for base in [os.getcwd(), os.environ.get('ROS_WORKSPACE', '')]:
            if not base:
                continue
            src_candidate = os.path.join(base, 'src', 'mini_plus_localization', 'config', 'datum_resolved.yaml')
            if os.path.isfile(src_candidate):
                datum_resolved_yaml = src_candidate
                break

    datum_found = os.path.isfile(datum_resolved_yaml)
    navsat_parameters = [ekf_yaml]
    if datum_found:
        navsat_parameters.append(datum_resolved_yaml)
        datum_log = LogInfo(
            msg=f"[ekf.launch.py] DATUM RESUELTO CARGADO: '{datum_resolved_yaml}' (anclaje determinista activado)"
        )
    else:
        datum_log = LogInfo(
            msg="[ekf.launch.py] DATUM NO ENCONTRADO: navsat_transform usará comportamiento default (primer fix de GPS)"
        )

    return LaunchDescription([
        datum_log,
        # TF estatico base_link -> earth_rover_gps (asume antena en el centro
        # del robot; ajusta --x/--y/--z si la antena esta desplazada).
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_gps',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--yaw', '0', '--pitch', '0', '--roll', '0',
                '--frame-id', 'base_link', '--child-frame-id', 'earth_rover_gps',
            ],
        ),

        # EKF local (odom frame): funde /wheel_odom (vx,vy) + /imu/data (yaw, vyaw)
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_odom',
            output='screen',
            parameters=[ekf_yaml],
            remappings=[('odometry/filtered', 'odometry/local')],
        ),

        # EKF global (map frame): funde wheel_odom + imu + /odometry/gps
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node_map',
            output='screen',
            parameters=[ekf_yaml],
            remappings=[('odometry/filtered', 'odometry/global')],
        ),

        # navsat_transform: proyecta /gps/fix a /odometry/gps y devuelve el
        # GPS ya fusionado directo en earth_rover/gps (sin tocar er_navigation)

        # navsat_transform: Proyecta /gps/fix a /odometry/gps y genera /gps/filtered
        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform',
            output='screen',
            parameters=navsat_parameters,
            remappings=[
                ('imu', '/imu/data'),
                ('gps/fix', '/gps/fix'),
                ('odometry/filtered', 'odometry/global'),
                # CONTRATO ESTRICTO: Salida oficial limpia
                ('gps/filtered', 'gps/filtered'),
            ],
        ),

        # Traduce el yaw fusionado (map, ENU) a heading en grados (compass),
        # publicado en earth_rover/heading -- mismo topic que ya consumía
        # gps_waypoint_controller.
        Node(
            package='mini_plus_localization',
            executable='ekf_heading_bridge.py',
            name='ekf_heading_bridge',
            output='screen',
        ),

        # Anclaje de rumbo absoluto por curso GNSS (Fase 2)
        Node(
            package='mini_plus_localization',
            executable='gps_course_heading_bridge.py',
            name='gps_course_heading_bridge',
            output='screen',
        ),
    ])