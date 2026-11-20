"""HW4: the localizer only (known map, observations while driving, every still frame used at once).
Run the rest in their own terminals, so you can watch each one:
    ros2 run robot_control motor_control
    ros2 launch robot_vision_camera robot_vision_camera.launch.py
    ros2 run landmark_nav detector                      # prints the tags it sees
    ros2 launch landmark_nav hw4_launch.py              # this file
    ros2 run landmark_nav hw4_control --ros-args -p mode:=safe     # or mode:=short
The localizer starts from the map's start pose, so restart this launch before every run.
map_file is the INSTALLED copy: run colcon build after editing config/map_hw4.yaml.
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
    return LaunchDescription([
        DeclareLaunchArgument('map_file', default_value=default_map),
        DeclareLaunchArgument('cam_offset', default_value='0.08'),   # m camera ahead of the robot centre
        Node(package='landmark_nav', executable='localizer', name='landmark_localizer', output='screen',
             parameters=[{'map_file': LaunchConfiguration('map_file'),
                          'accept_moving': True,
                          'fuse_frames': 1,                      # use each still frame at once
                          'sigma_th_per_rad': 0.15,
                          'cam_offset': ParameterValue(LaunchConfiguration('cam_offset'), value_type=float)}]),
    ])
