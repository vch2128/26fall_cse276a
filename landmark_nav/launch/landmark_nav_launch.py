"""Detector + localizer. Start motor_control and the camera launch separately (as in HW1),
then run the controller in its own terminal:
    ros2 launch landmark_nav hw2_launch.py
    ros2 run landmark_nav closed_loop_control waypoints_hw2.txt
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_map = os.path.join(get_package_share_directory('landmark_nav'), 'config', 'landmarks.yaml')
    lm = LaunchConfiguration('landmarks_file')
    model = LaunchConfiguration('model')
    return LaunchDescription([
        DeclareLaunchArgument('landmarks_file', default_value=default_map),
        DeclareLaunchArgument('model', default_value='yolo11n.pt'),
        Node(package='landmark_nav', executable='detector', name='landmark_detector', output='screen',
             parameters=[{'landmarks_file': lm, 'model': model, 'imgsz': 320}]),
        Node(package='landmark_nav', executable='localizer', name='landmark_localizer', output='screen',
             parameters=[{'landmarks_file': lm, 'start_pose': [0.0, 0.0, 0.0],
                          'cam_offset': 0.0}]),
    ])
