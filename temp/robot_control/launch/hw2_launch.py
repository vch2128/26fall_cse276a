"""HW2: motor_control + camera (+ Foxglove) + landmark_detector.
Start velocity_control yourself in a second terminal when the robot is placed at the start:
    ros2 launch robot_control hw2_launch.py
    ros2 run robot_control velocity_control
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('robot_vision_camera'), 'launch', 'robot_vision_camera.launch.py'])),
    )
    motor_node = Node(
        package='robot_control',
        executable='motor_control',
        name='motor_controller_node',
        output='screen',
    )
    detector_node = Node(
        package='robot_control',
        executable='landmark_detector',
        name='landmark_detector',
        output='screen',
    )
    return LaunchDescription([camera, motor_node, detector_node])
