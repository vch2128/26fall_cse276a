#!/usr/bin/env python3
"""
Timed calibration run for the robot.

Sends a constant [L, R] wheel command on /motor_commands for an exact
duration, stops, and reports how much the robot turned according to the IMU.
Optionally asks for the distance you measured by hand, and appends every run
to a CSV log so you can fit the calibration constants afterwards.

Start motor_control first, then for example:

  # straight line
  ros2 run robot_control calibrate --ros-args -p L:=0.3 -p R:=0.3 -p duration:=2.0
  # spin in place (counter-clockwise)
  ros2 run robot_control calibrate --ros-args -p L:=-0.4 -p R:=0.4 -p duration:=3.0 -p label:=spin_ccw

Parameters
  L, R          wheel commands (motor_control clips them to +/-0.5)
  duration      seconds to drive                         (default 2.0)
  rate          publish rate in Hz                        (default 20.0)
  start_delay   seconds to wait before moving             (default 1.0)
  settle        seconds to keep reading the IMU after the stop, so the
                coasting is included in the yaw change    (default 0.5)
  label         free text saved in the log, e.g. "fwd_0.3_trial2"
  ask_distance  prompt for the measured distance after the run (default true)
  log_file      CSV log path                              (default ~/calibration_log.csv)
"""
import csv
import math
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu

MAX_CMD = 0.5  # motor_control clips every command to this range


def quat_to_yaw(q):
    """Yaw (rad) from a geometry_msgs Quaternion."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(angle):
    """Wrap an angle to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')

        self.declare_parameter('L', 0.3)
        self.declare_parameter('R', 0.3)
        self.declare_parameter('duration', 2.0)
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('start_delay', 1.0)
        self.declare_parameter('settle', 0.5)
        self.declare_parameter('label', '')
        self.declare_parameter('ask_distance', True)
        self.declare_parameter('log_file', '~/ros2_ws/logs/calibration_log.csv')

        self.pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.create_subscription(Imu, 'imu_data', self.imu_callback, 50)

        # Yaw is accumulated across the +/-180 deg wrap, so multi-turn spins
        # are counted correctly.
        self.last_yaw = None
        self.total_yaw = 0.0
        self.imu_count = 0
        # Diagnostics: integrated gyro z (raw units * s), to cross-check the
        # board's fused yaw and to find out the gyro's units.
        self.gyro_z_int = 0.0
        self.last_imu_time = None

    # ------------------------------------------------------------------ helpers
    def imu_callback(self, msg):
        yaw = quat_to_yaw(msg.orientation)
        if self.last_yaw is not None:
            self.total_yaw += wrap(yaw - self.last_yaw)
        self.last_yaw = yaw
        self.imu_count += 1
        now = time.monotonic()
        if self.last_imu_time is not None:
            self.gyro_z_int += msg.angular_velocity.z * (now - self.last_imu_time)
        self.last_imu_time = now

    def send(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.pub.publish(msg)

    def spin_for(self, seconds):
        """Process callbacks (IMU) for a while without sending commands."""
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def param(self, name):
        return self.get_parameter(name).value

    # --------------------------------------------------------------------- run
    def run(self):
        left = float(self.param('L'))
        right = float(self.param('R'))
        duration = float(self.param('duration'))
        rate = float(self.param('rate'))
        label = str(self.param('label'))

        for name, value in (('L', left), ('R', right)):
            if abs(value) > MAX_CMD:
                self.get_logger().warn(
                    f'{name}={value} is outside +/-{MAX_CMD}; motor_control will clip it.')
        left = max(-MAX_CMD, min(MAX_CMD, left))
        right = max(-MAX_CMD, min(MAX_CMD, right))

        # 1. Make sure motor_control is listening.
        self.get_logger().info('Waiting for motor_control to subscribe to /motor_commands ...')
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and self.count_subscribers('motor_commands') == 0:
            if time.monotonic() > deadline:
                self.get_logger().error(
                    'Nothing is subscribed to /motor_commands. '
                    'Start it first: ros2 run robot_control motor_control')
                return
            rclpy.spin_once(self, timeout_sec=0.1)

        # 2. Let the IMU stream in and the robot sit still.
        self.get_logger().info(
            f'Run: L={left:.3f}  R={right:.3f}  for {duration:.2f} s  '
            f'(starting in {self.param("start_delay"):.1f} s)')
        self.spin_for(float(self.param('start_delay')))
        if self.imu_count == 0:
            self.get_logger().warn('No /imu_data received - yaw change will not be measured.')
        yaw_before = self.total_yaw
        gyro_before = self.gyro_z_int
        count_before = self.imu_count
        raw_yaw_before = self.last_yaw

        # 3. Drive for exactly `duration` seconds at `rate` Hz.
        period = 1.0 / rate
        t0 = time.monotonic()
        next_pub = t0
        while rclpy.ok():
            now = time.monotonic()
            if now - t0 >= duration:
                break
            if now >= next_pub:
                self.send(left, right)
                next_pub += period
            rclpy.spin_once(self, timeout_sec=0.002)
        actual = time.monotonic() - t0

        # 4. Stop (sent several times in case one is dropped), then let it coast.
        for _ in range(5):
            self.send(0.0, 0.0)
            self.spin_for(0.02)
        self.spin_for(float(self.param('settle')))
        yaw_change = self.total_yaw - yaw_before
        gyro_change = self.gyro_z_int - gyro_before
        imu_msgs = self.imu_count - count_before

        # 5. Report.
        imu_ok = self.imu_count > 0
        yaw_deg = math.degrees(yaw_change)
        omega = yaw_change / actual if actual > 0 else 0.0
        print('\n================ calibration result ================')
        print(f' label            : {label or "-"}')
        print(f' command  L, R    : {left:.3f}, {right:.3f}')
        print(f' drive time       : {actual:.3f} s (requested {duration:.3f} s)')
        if imu_ok:
            print(f' IMU yaw change   : {yaw_deg:+.1f} deg  ({yaw_change:+.3f} rad, '
                  f'{yaw_deg / 360.0:+.2f} turns; + = counter-clockwise)')
            print(f' avg turn rate    : {math.degrees(omega):+.1f} deg/s  ({omega:+.3f} rad/s)')
        else:
            print(' IMU yaw change   : n/a (no IMU data)')
        # --- diagnostics ---
        expected = int((actual + float(self.param('settle'))) * 50)
        print(f' [diag] IMU msgs during run : {imu_msgs} (expect ~{expected} at 50 Hz)')
        if raw_yaw_before is not None and self.last_yaw is not None:
            print(f' [diag] raw yaw start -> end : {math.degrees(raw_yaw_before):+.1f} -> '
                  f'{math.degrees(self.last_yaw):+.1f} deg')
        print(f' [diag] integrated gyro z    : {gyro_change:+.2f} (gyro units x s; '
              f'if gyro is deg/s this is degrees)')

        # 6. Optional hand-measured distance.
        distance = None
        if self.param('ask_distance'):
            try:
                raw = input(' measured distance in meters (Enter to skip): ').strip()
                if raw:
                    distance = float(raw)
            except (EOFError, ValueError):
                print(' (no valid distance entered)')
        if distance is not None and actual > 0:
            print(f' avg speed        : {distance / actual:.3f} m/s')
        print('====================================================\n')

        self.append_log(label, left, right, duration, actual,
                        yaw_change if imu_ok else None, omega if imu_ok else None, distance)

    def append_log(self, label, left, right, duration, actual, yaw, omega, distance):
        path = os.path.expanduser(str(self.param('log_file')))
        new_file = not os.path.exists(path)
        try:
            with open(path, 'a', newline='') as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow(['timestamp', 'label', 'L', 'R', 'duration_requested_s',
                                     'duration_actual_s', 'yaw_change_deg', 'omega_rad_s',
                                     'distance_m', 'speed_m_s'])

                def fmt(x, nd=4):
                    return '' if x is None else f'{x:.{nd}f}'

                writer.writerow([
                    datetime.now().isoformat(timespec='seconds'), label,
                    fmt(left, 3), fmt(right, 3), fmt(duration, 3), fmt(actual, 3),
                    fmt(None if yaw is None else math.degrees(yaw), 2), fmt(omega),
                    fmt(distance), fmt(None if distance is None else distance / actual),
                ])
            self.get_logger().info(f'Logged to {path}')
        except OSError as e:
            self.get_logger().error(f'Could not write log {path}: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationNode()
    try:
        node.run()
    except KeyboardInterrupt:
        # If ROS has already shut down, the stop below may not go out, but
        # motor_control stops the wheels on its own after 0.15 s without commands.
        pass
    finally:
        try:
            for _ in range(3):
                node.send(0.0, 0.0)
                time.sleep(0.02)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
