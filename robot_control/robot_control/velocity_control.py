#!/usr/bin/env python3
"""
CSE 276A HW1 - drive through the waypoints in waypoints.txt.

The motion primitives (drive / turn / stop and the turn tables) live in
robot_control/motion.py.

For every waypoint (x, y, theta):
    1. turn to face the waypoint
    2. drive straight to it
    3. turn to theta

Kinematic model: after each motion we update where we think the robot is
    x = x + distance * cos(heading)
    y = y + distance * sin(heading)
    heading = heading + turned angle

Run (motor_control must be running in another terminal):
    ros2 run robot_control velocity_control
    ros2 run robot_control velocity_control test_left.txt
"""
import math
import sys

import rclpy

from robot_control.motion import Motion, angle_diff, read_waypoints


def main():
    rclpy.init()
    node = rclpy.create_node('velocity_control')
    motion = Motion(node)

    name = sys.argv[1] if len(sys.argv) > 1 else 'waypoints.txt'
    waypoints = read_waypoints(name)
    x, y, heading = waypoints[0]          # the robot starts at the first waypoint
    motion.stop(1.0)

    try:
        for goal_x, goal_y, goal_theta in waypoints[1:]:
            distance = math.hypot(goal_x - x, goal_y - y)

            # 1. turn to face the waypoint
            direction = math.atan2(goal_y - y, goal_x - x)
            motion.turn(angle_diff(direction, heading))
            heading = direction
            # 2. drive to it  (kinematic model: update x, y)
            motion.drive(distance)
            x = x + distance * math.cos(heading)
            y = y + distance * math.sin(heading)

            # 3. turn to the waypoint's theta
            motion.turn(angle_diff(goal_theta, heading))
            heading = goal_theta

            print(f'reached ({goal_x}, {goal_y}, {goal_theta})')
            motion.stop(1.0)
        print('Done!')
    except KeyboardInterrupt:
        pass
    finally:
        try:
            motion.stop(0.2)              # always stop the robot
        except Exception:
            pass                          # ROS may already be shut down after Ctrl-C;
                                          # motor_control stops on its own after 0.15 s
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
