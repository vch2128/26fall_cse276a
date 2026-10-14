#!/usr/bin/env python3
"""
CSE 276A HW1 - drive through the waypoints in waypoints.txt.

The robot can only do two things:
    drive(meters)   -> send [0.25, 0.25] for  meters * SECONDS_PER_METER
    turn(radians)   -> timed spin in place, duration looked up in a table of
                       measured (angle, seconds) points (CCW_TABLE / CW_TABLE)

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

# ---- driving straight ----
SECONDS_PER_METER = 3.2     # how many seconds of [0.25, 0.25] drive 1 meter

# ---- wheel gains (multiply BASE_TURN for each wheel / direction) ----
L_fwd = 0.93
L_rev = 0.94
R_fwd = 1.08
R_rev = 1

# ---- turning ----
BASE_TURN = 0.3             # command for the timed turn

# Measured (degrees, seconds) at BASE_TURN with the gains above.
# Between points: straight-line interpolation; outside: the end segment is extended.
CCW_TABLE = [(45, 0.68), (90, 1.08), (135, 1.48), (180, 1.88)]   # left turns
CW_TABLE = [(45, 0.7), (90, 1.04), (135, 1.38), (180, 1.95)]    # right turns
# NOTE: CW 90 and 135 sit on opposite sides of a straight line by ~0.07 s; re-measure.

MIN_TURN_DEG = 3.0          # smaller turns are skipped

# ---- motion ----
STOP_SECONDS = 0.5          # pause after every motion so the robot is fully still
RAMP_SECONDS = 0.15         # ramp every command up from 0 over this time

FOLDER = os.path.expanduser('~/ros2_ws/src/26fall_cse276a/robot_control/config')

node = None
pub = None


def spin_for(seconds):
    """Wait while letting ROS process its work (spin_once instead of time.sleep)."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.01)


def publish(left, right):
    msg = Float32MultiArray()
    msg.data = [float(left), float(right)]
    pub.publish(msg)


def stop(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        publish(0.0, 0.0)
        spin_for(0.05)


def ramp_factor(t0):
    """0 -> 1 over RAMP_SECONDS after t0 (same formula as calibrate.py)."""
    if RAMP_SECONDS <= 0.0:
        return 1.0
    return min(1.0, (time.monotonic() - t0) / RAMP_SECONDS)


def send(left, right, seconds):
    """Send [left, right] to motor_control for some seconds (ramped up at the start),
    then stop."""
    t0 = time.monotonic()
    end = t0 + seconds
    while time.monotonic() < end:
        k = ramp_factor(t0)
        publish(left * k, right * k)
        spin_for(min(0.05, max(0.0, end - time.monotonic())))  # motor_control needs a msg every 0.15 s
    stop(STOP_SECONDS)


def drive(meters):  # go straight
    send(0.25, 0.25, meters * SECONDS_PER_METER)


def table_time(deg, table):
    """Seconds to turn `deg` degrees: piecewise-linear through the measured points,
    with the first / last segment extended for angles outside the table."""
    pts = sorted(table)
    if deg <= pts[0][0]:
        (a0, t0), (a1, t1) = pts[0], pts[1]
    elif deg >= pts[-1][0]:
        (a0, t0), (a1, t1) = pts[-2], pts[-1]
    else:
        (a0, t0), (a1, t1) = next(p for p in zip(pts, pts[1:]) if p[0][0] <= deg <= p[1][0])
    return max(0.0, t0 + (deg - a0) * (t1 - t0) / (a1 - a0))


def spin_commands(sign, speed):
    """Wheel commands for spinning in place. sign > 0: left (CCW), sign < 0: right (CW)."""
    if sign > 0:
        return -speed * L_rev, speed * R_fwd
    return speed * L_fwd, -speed * R_rev


def rotate(sign, seconds, speed=BASE_TURN):
    """Spin in place for `seconds` (ramped up at the start), then stop."""
    left, right = spin_commands(sign, speed)
    send(left, right, seconds)


def turn(radians):
    # take the short way round: wrap the angle to (-180°, 180°]
    radians = math.atan2(math.sin(radians), math.cos(radians))
    target = math.degrees(radians)            # signed, + = left (CCW)
    if abs(target) < MIN_TURN_DEG:
        return
    sign = 1 if target > 0 else -1
    table = CCW_TABLE if sign > 0 else CW_TABLE
    seconds = table_time(abs(target), table)
    print(f'  turn {target:+6.1f} deg: {seconds:.3f} s')
    rotate(sign, seconds)


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
    global node, pub
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
        try:
            stop(0.2)                     # always stop the robot
        except Exception:
            pass                          # ROS may already be shut down after Ctrl-C;
                                          # motor_control stops on its own after 0.15 s
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()