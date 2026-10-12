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
SECONDS_PER_METER = 3.2     # how many seconds of [0.25, 0.25] drive 1 meter
L_SECONDS_PER_90_DEG = 0.95    # how many seconds of [-0.35, 0.35] turn 90 degrees
R_SECONDS_PER_90_DEG = 1 
L_fwd = 0.93
L_rev = 0.94
R_fwd = 1.08
R_rev = 1


BASE_TURN = 0.3

# Measured in the turn-timing experiment (seconds to turn that angle)
CCW_SECONDS_PER_45_DEG = 0.6   # left turn, 45
CCW_SECONDS_PER_90_DEG = 0.95   # left turn, 90°
CW_SECONDS_PER_45_DEG  = 0.61   # right turn, 45
CW_SECONDS_PER_90_DEG  = 1.0   # right turn, 90°

MIN_TURN_DEG = 3.0 

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


def turn_seconds(deg, s45, s90):
    """Time to turn `deg` degrees (deg >= 0).

    Straight line through the two measured points (45°, s45) and (90°, s90):
        t(θ) = s45 + (θ - 45°) · (s90 - s45) / 45°
    which is the same as t = t_dead + θ / ω, with
        ω      = 45° / (s90 - s45)        turn rate once spinning (deg/s)
        t_dead = (3·s45 - s90) / 2        spin-up delay (s)
    """
    return max(0.0, s45 + (deg - 45.0) * (s90 - s45) / 45.0)


def turn(radians):
    # take the short way round: wrap the angle to (-180°, 180°]
    radians = math.atan2(math.sin(radians), math.cos(radians))
    deg = math.degrees(abs(radians))
    if deg < MIN_TURN_DEG:
        return

    if radians > 0:     # turn left (CCW): left wheel back, right wheel forward
        seconds = turn_seconds(deg, CCW_SECONDS_PER_45_DEG, CCW_SECONDS_PER_90_DEG)
        send(-BASE_TURN * L_rev, BASE_TURN * R_fwd, seconds)
    else:               # turn right (CW): left wheel forward, right wheel back
        seconds = turn_seconds(deg, CW_SECONDS_PER_45_DEG, CW_SECONDS_PER_90_DEG)
        send(BASE_TURN * L_fwd, -BASE_TURN * R_rev, seconds)


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