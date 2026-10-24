"""
Open-loop motion primitives shared by the waypoint controllers.

The robot can only do two things:
    drive(meters)   -> send [0.25, 0.25] for  meters * SECONDS_PER_METER
    turn(radians)   -> timed spin in place, duration looked up in a table of
                       measured (angle, seconds) points (CCW_TABLE / CW_TABLE)

Usage (from any node):
    from robot_control.motion import Motion
    motion = Motion(node)      # creates the motor_commands publisher on `node`
    motion.turn(math.pi / 2)
    motion.drive(0.5)
"""
import math
import os
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


class Motion:
    """Timed drive / turn / stop on top of a node's motor_commands publisher."""

    def __init__(self, node, topic='motor_commands'):
        self.node = node
        self.pub = node.create_publisher(Float32MultiArray, topic, 10)

    def spin_for(self, seconds):
        """Wait while letting ROS process its work (spin_once instead of time.sleep)."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def publish(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.pub.publish(msg)

    def stop(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.publish(0.0, 0.0)
            self.spin_for(0.05)

    def send(self, left, right, seconds):
        """Send [left, right] to motor_control for some seconds (ramped up at the start),
        then stop."""
        t0 = time.monotonic()
        end = t0 + seconds
        while time.monotonic() < end:
            k = 1.0 if RAMP_SECONDS <= 0.0 else min(1.0, (time.monotonic() - t0) / RAMP_SECONDS)
            self.publish(left * k, right * k)
            self.spin_for(min(0.05, max(0.0, end - time.monotonic())))  # motor_control needs a msg every 0.15 s
        self.stop(STOP_SECONDS)

    def drive(self, meters):  # go straight
        self.send(0.25, 0.25, meters * SECONDS_PER_METER)

    def rotate(self, sign, seconds, speed=BASE_TURN):
        """Spin in place for `seconds` (ramped up at the start), then stop."""
        left, right = spin_commands(sign, speed)
        self.send(left, right, seconds)

    def turn(self, radians):
        # take the short way round: wrap the angle to (-180°, 180°]
        radians = math.atan2(math.sin(radians), math.cos(radians))
        target = math.degrees(radians)            # signed, + = left (CCW)
        if abs(target) < MIN_TURN_DEG:
            return
        sign = 1 if target > 0 else -1
        table = CCW_TABLE if sign > 0 else CW_TABLE
        seconds = table_time(abs(target), table)
        print(f'  turn {target:+6.1f} deg: {seconds:.3f} s')
        self.rotate(sign, seconds)
