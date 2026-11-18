"""HW4: detector + localizer (known map, observations while driving).
Start motor_control and the camera launch separately (as before), then the controller in its own
terminal once the localizer is up:
    ros2 launch landmark_nav hw4_launch.py
    ros2 run landmark_nav hw4_control --ros-args -p mode:=short      # or mode:=safe
Each run of hw4_control starts from the map's start pose, so restart this launch (the localizer)
before every run.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_map = os.path.join(get_package_share_directory('landmark_nav'), 'config', 'map_hw4.yaml')
    m = LaunchConfiguration('map_file')
    return LaunchDescription([
        DeclareLaunchArgument('map_file', default_value=default_map),
        DeclareLaunchArgument('cam_offset', default_value='0.08'),   # m camera ahead of the robot centre
        Node(package='landmark_nav', executable='detector', name='landmark_detector', output='screen',
             parameters=[{'landmarks_file': m}]),                    # tag_size from the map
        Node(package='landmark_nav', executable='localizer', name='landmark_localizer', output='screen',
             parameters=[{'map_file': m,
                          'accept_moving': True,
                          'sigma_th_per_rad': 0.15,
                          'cam_offset': ParameterValue(LaunchConfiguration('cam_offset'), value_type=float)}]),
    ])
