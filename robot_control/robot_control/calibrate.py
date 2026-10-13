#!/usr/bin/env python3
"""
Calibration run for the robot.

Sends an [L, R] wheel command on /motor_commands for exactly `duration` 
seconds, stops, and reports how much the robot turned according to the
(bias-corrected, integrated) gyro z. 
The gyro only measures; it never decides when to stop.

Start motor_control first, then for example:

  # straight line
  ros2 run robot_control calibrate --ros-args -p L:=0.25 -p R:=0.25 -p duration:=3.2
  # timed CCW spin (left wheel back, right wheel forward)
  ros2 run robot_control calibrate --ros-args -p L:=-0.3 -p R:=0.3 -p duration:=1.0 -p label:=spin_ccw
  # CCW spin with the left reverse gain reduced
  ros2 run robot_control calibrate --ros-args -p L:=-0.3 -p R:=0.3 -p duration:=1.0 \
      -p gain_l_rev:=0.94 -p label:=spin_ccw_gain094

Parameters
  L, R             wheel commands before gains (motor_control clips to +/-0.5)
  duration         seconds to drive                                (default 2.0)
  gyro_units       'deg' or 'rad' (units of angular_velocity.z)   (default 'deg')
  estimate_bias    average gyro z while still to remove its bias   (default true)
  gain_l_fwd, gain_l_rev, gain_r_fwd, gain_r_rev
                   multiplier per wheel and direction               (default 1.0)
  ramp             seconds to ramp the command up from 0           (default 0.15)
  rate             publish rate in Hz                               (default 20.0)
  start_delay      seconds to sit still before moving (bias window) (default 1.0)
  settle           seconds to keep reading the IMU after the stop   (default 0.5)
  label            free text saved in the log
  ask_distance     prompt for distance (straight) / drift (spin)    (default true)
  log_file         CSV log path
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


def clip(x):
    return max(-MAX_CMD, min(MAX_CMD, x))


def apply_gain(cmd, fwd_gain, rev_gain):
    """Scale a wheel command by its forward or reverse gain."""
    return cmd * (fwd_gain if cmd >= 0.0 else rev_gain)


class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')

        self.declare_parameter('L', 0.3)
        self.declare_parameter('R', 0.3)
        self.declare_parameter('duration', 2.0)
        self.declare_parameter('gyro_units', 'deg')
        self.declare_parameter('estimate_bias', True)
        self.declare_parameter('gain_l_fwd', 1.0)
        self.declare_parameter('gain_l_rev', 1.0)
        self.declare_parameter('gain_r_fwd', 1.0)
        self.declare_parameter('gain_r_rev', 1.0)
        self.declare_parameter('ramp', 0.15)
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('start_delay', 1.0)
        self.declare_parameter('settle', 0.5)
        self.declare_parameter('label', '')
        self.declare_parameter('ask_distance', True)
        # New file name because the columns changed (board yaw column removed).
        self.declare_parameter('log_file', '~/ros2_ws/logs/calibration_log_v3.csv')

        self.pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.create_subscription(Imu, 'imu_data', self.imu_callback, 50)

        units = str(self.param('gyro_units')).lower()
        self.gyro_scale = math.radians(1.0) if units.startswith('deg') else 1.0

        # Gyro heading: bias-corrected, trapezoid-integrated gyro z, in rad.
        self.imu_count = 0
        self.gyro_bias = 0.0          # rad/s
        self.gyro_heading = 0.0       # rad
        self.bias_samples = None      # list while estimating bias
        self.last_imu_t = None
        self.last_rate = 0.0

    # ------------------------------------------------------------------ helpers
    def imu_callback(self, msg):
        self.imu_count += 1
        rate = msg.angular_velocity.z * self.gyro_scale  # rad/s
        if self.bias_samples is not None:
            self.bias_samples.append(rate)

        # Prefer the sensor timestamp; fall back to arrival time.
        st = msg.header.stamp
        t = st.sec + st.nanosec * 1e-9 if (st.sec or st.nanosec) else time.monotonic()
        if self.last_imu_t is not None:
            dt = t - self.last_imu_t
            if 0.0 < dt < 0.2:  # skip gaps and clock jumps
                self.gyro_heading += 0.5 * ((rate - self.gyro_bias) +
                                            (self.last_rate - self.gyro_bias)) * dt
        self.last_imu_t = t
        self.last_rate = rate

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

    def gains(self):
        return (float(self.param('gain_l_fwd')), float(self.param('gain_l_rev')),
                float(self.param('gain_r_fwd')), float(self.param('gain_r_rev')))

    def shaped(self, left, right):
        """Apply per-wheel gains and clip."""
        glf, glr, grf, grr = self.gains()
        return clip(apply_gain(left, glf, glr)), clip(apply_gain(right, grf, grr))

    # --------------------------------------------------------------------- run
    def run(self):
        left = float(self.param('L'))
        right = float(self.param('R'))
        duration = float(self.param('duration'))
        rate = float(self.param('rate'))
        label = str(self.param('label'))
        ramp = float(self.param('ramp'))

        spin = left * right < 0.0

        for name, value in (('L', left), ('R', right)):
            if abs(value) > MAX_CMD:
                self.get_logger().warn(
                    f'{name}={value} is outside +/-{MAX_CMD}; it will be clipped.')
        eff_l, eff_r = self.shaped(left, right)

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

        # 2. Sit still: let the IMU stream in and estimate the gyro bias.
        self.get_logger().info(
            f'Run: L={left:.3f} R={right:.3f} (after gains {eff_l:.3f}, {eff_r:.3f}) '
            f'for {duration:.2f} s '
            f'(starting in {self.param("start_delay"):.1f} s - keep the robot still)')
        self.bias_samples = []
        self.spin_for(float(self.param('start_delay')))
        if self.param('estimate_bias') and len(self.bias_samples) >= 10:
            self.gyro_bias = sum(self.bias_samples) / len(self.bias_samples)
        self.bias_samples = None
        if self.imu_count == 0:
            self.get_logger().warn('No /imu_data received - turning will not be measured.')

        heading_before = self.gyro_heading
        count_before = self.imu_count

        # 3. Drive.
        period = 1.0 / rate
        t0 = time.monotonic()
        next_pub = t0
        while rclpy.ok():
            now = time.monotonic()
            elapsed = now - t0
            if elapsed >= duration:
                break
            if now >= next_pub:
                k = min(1.0, elapsed / ramp) if ramp > 0.0 else 1.0
                self.send(*self.shaped(left * k, right * k))
                next_pub += period
            rclpy.spin_once(self, timeout_sec=0.002)
        actual = time.monotonic() - t0
        heading_at_stop = self.gyro_heading - heading_before

        # 4. Stop (sent several times in case one is dropped), then let it coast.
        for _ in range(5):
            self.send(0.0, 0.0)
            self.spin_for(0.02)
        self.spin_for(float(self.param('settle')))
        heading_total = self.gyro_heading - heading_before
        coast = heading_total - heading_at_stop
        imu_msgs = self.imu_count - count_before

        # 5. Report.
        imu_ok = self.imu_count > 0
        deg = math.degrees
        omega = heading_at_stop / actual if actual > 0 else 0.0
        glf, glr, grf, grr = self.gains()
        print('\n================ calibration result ================')
        print(f' label            : {label or "-"}')
        print(f' command  L, R    : {left:.3f}, {right:.3f}  -> sent {eff_l:.3f}, {eff_r:.3f}')
        print(f' gains            : L fwd {glf:.3f} rev {glr:.3f} | R fwd {grf:.3f} rev {grr:.3f}')
        print(f' drive time       : {actual:.3f} s')
        if imu_ok:
            print(f' gyro heading     : {deg(heading_total):+.1f} deg total '
                  f'(at stop {deg(heading_at_stop):+.1f}, coast {deg(coast):+.1f}; + = CCW)')
            print(f' avg turn rate    : {deg(omega):+.1f} deg/s while driving')
            if spin and abs(deg(heading_at_stop)) > 5.0:
                need = 90.0 - abs(deg(coast))
                print(f' hint             : for 90 deg try duration ~ '
                      f'{actual * need / abs(deg(heading_at_stop)):.3f} s')
        else:
            print(' gyro heading     : n/a (no IMU data)')
        expected = int((actual + float(self.param('settle'))) * 50)
        print(f' [diag] gyro bias           : {deg(self.gyro_bias):+.3f} deg/s')
        print(f' [diag] IMU msgs during run : {imu_msgs} (expect ~{expected} at 50 Hz)')

        # 6. Hand measurements: distance for straight runs, drift for spins.
        distance = drift_fwd = drift_left = None
        if self.param('ask_distance'):
            if spin:
                print(' Measure where the center of the drive axle ended up, relative to the')
                print(' start mark, in the robot\'s STARTING frame.')
                drift_fwd = self.ask_float(' drift forward(+)/back(-) in cm (Enter to skip): ')
                drift_left = self.ask_float(' drift left(+)/right(-) in cm (Enter to skip): ')
            else:
                distance = self.ask_float(' measured distance in meters (Enter to skip): ')
        if distance is not None and actual > 0:
            print(f' avg speed        : {distance / actual:.3f} m/s')
        print('====================================================\n')

        self.append_log(label, left, right, eff_l, eff_r, duration, actual,
                        heading_total if imu_ok else None, coast if imu_ok else None,
                        distance, drift_fwd, drift_left)

    @staticmethod
    def ask_float(prompt):
        try:
            raw = input(prompt).strip()
            return float(raw) if raw else None
        except (EOFError, ValueError):
            print(' (no valid number entered)')
            return None

    def append_log(self, label, left, right, eff_l, eff_r, duration, actual,
                   heading, coast, distance, drift_fwd, drift_left):
        path = os.path.expanduser(str(self.param('log_file')))
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        new_file = not os.path.exists(path)
        glf, glr, grf, grr = self.gains()

        def fmt(x, nd=4):
            return '' if x is None else f'{x:.{nd}f}'

        def d(x):
            return None if x is None else math.degrees(x)

        try:
            with open(path, 'a', newline='') as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow([
                        'timestamp', 'label', 'L', 'R', 'L_sent', 'R_sent',
                        'gain_l_fwd', 'gain_l_rev', 'gain_r_fwd', 'gain_r_rev',
                        'duration_requested_s', 'duration_actual_s',
                        'gyro_heading_deg', 'coast_deg',
                        'gyro_bias_deg_s', 'distance_m', 'speed_m_s',
                        'drift_fwd_cm', 'drift_left_cm'])
                writer.writerow([
                    datetime.now().isoformat(timespec='seconds'), label,
                    fmt(left, 3), fmt(right, 3), fmt(eff_l, 3), fmt(eff_r, 3),
                    fmt(glf, 3), fmt(glr, 3), fmt(grf, 3), fmt(grr, 3),
                    fmt(duration, 3), fmt(actual, 3),
                    fmt(d(heading), 2), fmt(d(coast), 2),
                    fmt(math.degrees(self.gyro_bias), 3),
                    fmt(distance), fmt(None if distance is None else distance / actual),
                    fmt(drift_fwd, 1), fmt(drift_left, 1),
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