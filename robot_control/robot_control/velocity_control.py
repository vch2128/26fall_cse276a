#!/usr/bin/env python3
"""
CSE 276A HW1 - drive through the waypoints in waypoints.txt.

The robot can only do two things:
    drive(meters)   -> send [0.25, 0.25]  for  meters * SECONDS_PER_METER
    turn(radians)   -> send [-0.35, 0.35] for  (radians / 90 deg) * SECONDS_PER_90_DEG

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
import os
import sys
import time

import rclpy
from std_msgs.msg import Float32MultiArray

# ---- measure these two numbers on your robot ----
SECONDS_PER_METER = 3.6     # how many seconds of [0.25, 0.25] drive 1 meter
SECONDS_PER_90_DEG = 0.8    # how many seconds of [-0.35, 0.35] turn 90 degrees

FOLDER = os.path.expanduser('~/ros2_ws/src/26fall_cse276a/robot_control/config')


def send(left, right, seconds):
    """Send [left, right] to motor_control for some seconds, then stop."""
    end = time.time() + seconds
    while time.time() < end:
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        pub.publish(msg)
        time.sleep(0.05)          # motor_control needs a message at least every 0.15 s
    stop(0.5)


def stop(seconds):
    end = time.time() + seconds
    while time.time() < end:
        msg = Float32MultiArray()
        msg.data = [0.0, 0.0]
        pub.publish(msg)
        time.sleep(0.05)


def drive(meters): # go straight
    send(0.25, 0.25, meters * SECONDS_PER_METER)


def turn(radians):
    seconds = abs(radians) / (math.pi / 2) * SECONDS_PER_90_DEG
    if radians > 0:     # turn left
        send(-0.35, 0.35, seconds)
    else:               # turn right
        send(0.35, -0.35, seconds)


def angle_diff(a, b):
    """Smallest angle to turn from b to a (between -pi and pi)."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def read_waypoints(name):
    points = []
    for line in open(os.path.join(FOLDER, name)):
        if line.strip() and not line.startswith('#'):
            x, y, theta = line.split(',')
            points.append((float(x), float(y), float(theta)))
    return points


def main():
    global pub
    rclpy.init()
    node = rclpy.create_node('velocity_control')
    pub = node.create_publisher(Float32MultiArray, 'motor_commands', 10)

    name = sys.argv[1] if len(sys.argv) > 1 else 'waypoints.txt'
    waypoints = read_waypoints(name)
    x, y, heading = waypoints[0]          # the robot starts at the first waypoint
    stop(1.0)

    try:
        for goal_x, goal_y, goal_theta in waypoints[1:]:
            distance = math.hypot(goal_x - x, goal_y - y)

            if distance > 0.02:
                # 1. turn to face the waypoint
                direction = math.atan2(goal_y - y, goal_x - x)
                turn(angle_diff(direction, heading))
                heading = direction
                # 2. drive to it  (kinematic model: update x, y)
                drive(distance)
                x = x + distance * math.cos(heading)
                y = y + distance * math.sin(heading)

            # 3. turn to the waypoint's theta
            turn(angle_diff(goal_theta, heading))
            heading = goal_theta

            print(f'reached ({goal_x}, {goal_y}, {goal_theta})')
            stop(1.0)
        print('Done!')
    except KeyboardInterrupt:
        pass
    finally:
        stop(0.2)                         # always stop the robot
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()