#!/usr/bin/env python3
"""
CSE 276A HW1 - calibration helper for the Waverover.

Run motor_control in another terminal first, then:

  # 1) IMU check: turn the robot LEFT (counter-clockwise) by hand ~90 deg.
  #    The printed yaw should go UP to about +90. If it goes DOWN -> IMU_SIGN = -1.0
  ros2 run robot_control calibrate --ros-args -p mode:=imu

  # 2) Straight line: drive at a fixed command for a fixed time, then
  #    measure the distance with a tape measure.  DRIVE_SPEED = distance / duration
  ros2 run robot_control calibrate --ros-args -p mode:=straight -p cmd:=0.25 -p duration:=3.0

  # 3) Spin in place: prints the yaw change measured by the IMU.
  #    Positive cmd should turn LEFT (positive yaw change).
  ros2 run robot_control calibrate --ros-args -p mode:=spin -p cmd:=0.35 -p duration:=2.0

  # 4) Fit the linear model  v = slope * (cmd - deadzone)  from several runs:
  ros2 run robot_control calibrate --ros-args -p mode:=fit \
      -p cmds:="[0.15, 0.25, 0.35]" -p values:="[0.10, 0.25, 0.40]"
"""

import math
import time

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def quat_to_yaw(w, x, y, z):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def fit_line(xs, ys):
    """Least squares y = slope * x + b  ->  returns slope, deadzone (= -b/slope)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    b = my - slope * mx
    return slope, -b / slope


class Calibrate(Node):
    def __init__(self):
        super().__init__('calibrate')
        dyn = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter('mode', 'imu', dyn)
        self.declare_parameter('cmd', 0.25, dyn)
        self.declare_parameter('duration', 3.0, dyn)
        self.declare_parameter('imu_yaw_sign', 1.0, dyn)
        self.declare_parameter('cmds', [0.0], dyn)
        self.declare_parameter('values', [0.0], dyn)
        self.mode = str(self.get_parameter('mode').value)
        self.cmd = float(self.get_parameter('cmd').value)
        self.duration = float(self.get_parameter('duration').value)
        self.sign = float(self.get_parameter('imu_yaw_sign').value)

        self.pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.create_subscription(Imu, 'imu_data', self.imu_cb, 50)
        self.yaw = None
        self.yaw_start = None
        self.yaw_total = 0.0         # unwrapped yaw change
        self.last_yaw = None
        self.t0 = time.monotonic()
        self.t_move = None
        self.finished = False
        self.last_print = 0.0

        if self.mode == 'fit':
            self.do_fit()
            self.finished = True
            return
        if self.mode not in ('imu', 'straight', 'spin'):
            self.get_logger().error(f'unknown mode {self.mode!r} (imu | straight | spin | fit)')
            self.finished = True
            return
        self.create_timer(0.05, self.loop)
        self.get_logger().info(f'mode={self.mode} cmd={self.cmd} duration={self.duration}s')
        if self.mode != 'imu':
            self.get_logger().info('Starting in 2 s - make sure the robot has room!')

    def imu_cb(self, msg):
        q = msg.orientation
        y = self.sign * quat_to_yaw(q.w, q.x, q.y, q.z)
        if self.last_yaw is not None and self.t_move is not None:
            self.yaw_total += wrap(y - self.last_yaw)
        self.last_yaw = y
        self.yaw = y

    def send(self, L, R):
        m = Float32MultiArray()
        m.data = [float(L), float(R)]
        self.pub.publish(m)

    def loop(self):
        now = time.monotonic()
        if self.mode == 'imu':
            self.send(0.0, 0.0)
            if self.yaw is None:
                if now - self.t0 > 3.0 and now - self.last_print > 2.0:
                    self.get_logger().warn('No /imu_data yet - is motor_control running?')
                    self.last_print = now
                return
            if self.yaw_start is None:
                self.yaw_start = self.yaw
                self.t_move = now          # start unwrapping
            if now - self.last_print > 0.2:
                rel = self.yaw_total
                print(f'\rrelative yaw: {math.degrees(rel):+7.1f} deg  ({rel:+.3f} rad)   '
                      f'raw: {math.degrees(self.yaw):+7.1f} deg     ', end='', flush=True)
                self.last_print = now
            return

        # straight / spin: wait 2 s, then move for `duration` seconds
        if now - self.t0 < 2.0:
            self.send(0.0, 0.0)
            return
        if self.t_move is None:
            self.t_move = now
            self.yaw_total = 0.0
            self.get_logger().info('GO')
        elapsed = now - self.t_move
        if elapsed < self.duration:
            if self.mode == 'straight':
                self.send(self.cmd, self.cmd)
            else:
                self.send(-self.cmd, self.cmd)
            return
        self.send(0.0, 0.0)
        if elapsed < self.duration + 1.0:   # let it coast to a stop, keep reading the IMU
            return
        deg = math.degrees(self.yaw_total) if self.yaw is not None else float('nan')
        if self.mode == 'straight':
            self.get_logger().info(
                f'DONE. cmd={self.cmd} for {self.duration:.2f} s. Heading drift = {deg:+.1f} deg.\n'
                f'  -> measure the distance d [m];  DRIVE_SPEED = d / {self.duration:.2f}\n'
                f'  -> drift > 0 means it curved LEFT, < 0 curved RIGHT')
        else:
            rate = math.radians(deg) / self.duration
            self.get_logger().info(
                f'DONE. spin cmd={self.cmd} for {self.duration:.2f} s.\n'
                f'  IMU yaw change = {deg:+.1f} deg  ->  yaw rate = {rate:+.2f} rad/s\n'
                f'  (positive cmd should give a POSITIVE change; if negative, set IMU_SIGN = -1.0)')
        self.finished = True

    def do_fit(self):
        xs = [float(v) for v in self.get_parameter('cmds').value]
        ys = [float(v) for v in self.get_parameter('values').value]
        if len(xs) != len(ys) or len(xs) < 2:
            self.get_logger().error('cmds and values must have the same length (>= 2)')
            return
        slope, dz = fit_line(xs, ys)
        self.get_logger().info(f'fit: value = {slope:.3f} * (cmd - {dz:.3f})')
        self.get_logger().info(f'  straight runs -> v_slope: {slope:.3f}   v_deadzone: {dz:.3f}')
        self.get_logger().info(f'  spin runs     -> w_slope: {slope:.3f}   w_deadzone: {dz:.3f}')


def main(args=None):
    rclpy.init(args=args)
    node = Calibrate()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            for _ in range(3):
                node.send(0.0, 0.0)
                time.sleep(0.02)
            print()
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
