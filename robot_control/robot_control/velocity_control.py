#!/usr/bin/env python3
"""
CSE 276A HW1 - drive the Waverover through the waypoints in waypoints.txt.

Algorithm, for every waypoint (x, y, theta):
    1. turn in place to face the waypoint      (closed loop on IMU yaw)
    2. drive straight to it                    (distance from the speed model)
    3. turn in place to the waypoint's theta   (closed loop on IMU yaw)

Kinematic model (differential drive / unicycle):
    v = (v_R + v_L) / 2,   omega = (v_R - v_L) / track_width
    x' = v cos(theta),     y' = v sin(theta),     theta' = omega
The robot has no wheel encoders, so the position is dead-reckoned:
    x += v cos(theta) dt,  y += v sin(theta) dt
with v = DRIVE_SPEED (calibrated) while driving, 0 while turning in place,
and theta read from the IMU.

Run (motor_control must be running in another terminal):
    ros2 run robot_control velocity_control                 # waypoints.txt
    ros2 run robot_control velocity_control test_turn.txt   # another file in config/
"""
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu

# ---------------- calibration (measure these on your robot!) ----------------
DRIVE_CMD = 0.25      # wheel command used for driving straight
DRIVE_SPEED = 0.20    # [m/s] measured speed at DRIVE_CMD  (distance / time)
IMU_SIGN = 1.0        # +1 if IMU yaw increases when turning left, else -1
TURN_MIN = 0.28       # smallest command that reliably rotates the robot in place
TURN_MAX = 0.45       # fastest turning command
K_TURN = 0.5          # turning P gain   [command per rad]
K_HEADING = 0.6       # steering P gain while driving [command per rad]
ANGLE_TOL = 0.05      # [rad] ~3 deg, turn is done inside this
PAUSE = 0.5           # [s] stop between motions (lets the robot settle)
RATE = 20.0           # [Hz] control loop (motor_control stops after 0.15 s without commands)


def wrap(a):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def find_waypoint_file(name):
    """Look for the file as given, then in robot_control/config/."""
    here = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    for path in (name, os.path.join(here, 'config', name),
                 os.path.expanduser('~/ros2_ws/src/26fall_cse276a/robot_control/config/' + name)):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(name)


def load_waypoints(path):
    pts = []
    for line in open(path):
        line = line.split('#')[0].strip()
        if line:
            x, y, th = (float(v) for v in line.split(','))
            pts.append((x, y, th))
    return pts


class WaypointNavigator(Node):
    def __init__(self, waypoint_file):
        super().__init__('velocity_control')
        self.waypoints = load_waypoints(waypoint_file)
        self.get_logger().info(f'{len(self.waypoints)} waypoints from {waypoint_file}')

        self.pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.create_subscription(Imu, 'imu_data', self.imu_callback, 10)

        # robot state (dead reckoning); the first waypoint is the start pose
        self.x, self.y, self.yaw = self.waypoints[0]
        self.imu_yaw = None       # latest raw IMU yaw
        self.yaw_offset = None    # makes yaw = theta of the first waypoint at start

        # list of motions still to do: ('turn', angle) or ('drive', distance)
        self.wp_index = 0
        self.tasks = []
        self.task = None
        self.task_start = 0.0
        self.driven = 0.0
        self.done = False
        self.last_time = time.time()
        self.timer = self.create_timer(1.0 / RATE, self.loop)

    # ---- sensors ----
    def imu_callback(self, msg):
        q = msg.orientation
        self.imu_yaw = IMU_SIGN * math.atan2(2 * (q.w * q.z + q.x * q.y),
                                             1 - 2 * (q.y * q.y + q.z * q.z))

    def send(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.pub.publish(msg)

    # ---- planning ----
    def plan_next_waypoint(self):
        """Turn toward the next waypoint, drive to it, turn to its theta."""
        self.wp_index += 1
        if self.wp_index >= len(self.waypoints):
            return False
        tx, ty, tth = self.waypoints[self.wp_index]
        dist = math.hypot(tx - self.x, ty - self.y)
        if dist > 0.02:
            self.tasks.append(('turn', math.atan2(ty - self.y, tx - self.x)))
            self.tasks.append(('drive', dist))
        self.tasks.append(('turn', tth))
        self.get_logger().info(f'--> waypoint {self.wp_index}: ({tx}, {ty}, {tth}), {dist:.2f} m away')
        return True

    # ---- control loop (runs at RATE Hz) ----
    def loop(self):
        now = time.time()
        dt = now - self.last_time
        self.last_time = now

        # wait for the IMU, then define yaw = theta of the first waypoint
        if self.yaw_offset is None:
            self.send(0, 0)
            if self.imu_yaw is not None:
                self.yaw_offset = self.yaw - self.imu_yaw
                self.get_logger().info('IMU ready, starting')
            return
        self.yaw = wrap(self.imu_yaw + self.yaw_offset)

        if self.done:
            self.send(0, 0)
            return

        # pick the next motion
        if self.task is None:
            if not self.tasks:
                if self.wp_index > 0:
                    self.get_logger().info(
                        f'    reached waypoint {self.wp_index}, estimate '
                        f'x={self.x:.2f} y={self.y:.2f} yaw={self.yaw:.2f}')
                if not self.plan_next_waypoint():
                    self.get_logger().info('All waypoints done!')
                    self.done = True
                    self.send(0, 0)
                    return
            self.task = self.tasks.pop(0)
            self.task_start = now + PAUSE      # stand still for PAUSE seconds first
            self.driven = 0.0
            self.turn_dir = None

        if now < self.task_start:
            self.send(0, 0)
            return

        kind, value = self.task
        if kind == 'turn':
            error = wrap(value - self.yaw)
            if self.turn_dir is None:
                self.turn_dir = math.copysign(1, error)
            # done when close enough, or when we overshoot (error changed sign)
            if abs(error) < ANGLE_TOL or math.copysign(1, error) != self.turn_dir:
                self.task = None
                self.send(0, 0)
                return
            cmd = min(max(K_TURN * abs(error), TURN_MIN), TURN_MAX)
            cmd = math.copysign(cmd, error)            # positive = turn left
            self.send(-cmd, cmd)

        elif kind == 'drive':
            if self.driven >= value:
                self.task = None
                self.send(0, 0)
                return
            # kinematic model: integrate the position with the measured speed
            self.x += DRIVE_SPEED * math.cos(self.yaw) * dt
            self.y += DRIVE_SPEED * math.sin(self.yaw) * dt
            self.driven += DRIVE_SPEED * dt
            # keep pointing at the waypoint while driving
            tx, ty, _ = self.waypoints[self.wp_index]
            heading_error = wrap(math.atan2(ty - self.y, tx - self.x) - self.yaw)
            if value - self.driven < 0.15:             # too close to steer reliably
                heading_error = 0.0
            steer = max(-0.1, min(0.1, K_HEADING * heading_error))
            self.send(DRIVE_CMD - steer, DRIVE_CMD + steer)


def main(args=None):
    rclpy.init(args=args)
    extra = rclpy.utilities.remove_ros_args(sys.argv)[1:]
    node = WaypointNavigator(find_waypoint_file(extra[0] if extra else 'waypoints.txt'))
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.send(0, 0)          # always stop the robot
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
