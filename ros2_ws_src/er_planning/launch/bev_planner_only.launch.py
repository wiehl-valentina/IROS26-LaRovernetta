import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

# Test Docker layer cache validation: trivial modification outside AI stack
def generate_launch_description():
    er_planning_dir = get_package_share_directory('er_planning')
    planner_params = os.path.join(er_planning_dir, 'config', 'planner_params.yaml')

    planner_node = Node(
        package='er_planning',
        executable='bev_planner_node',
        name='bev_planner_node',
        output='screen',
        parameters=[planner_params],
    )

    return LaunchDescription([
        planner_node
    ])
