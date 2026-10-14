#!/usr/bin/env python3
"""
Calibration run for the robot.

Sends an [L, R] wheel command on /motor_commands for exactly `duration`
seconds, then stops. Every run is appended to a CSV log.

Start motor_control first, then for example:

  # straight line
  ros2 run robot_control calibrate --ros-args -p L:=0.25 -p R:=0.25 -p duration:=3.2
  # timed CCW spin (left wheel back, right wheel forward)
  ros2 run robot_control calibrate --ros-args -p L:=-0.3 -p R:=0.3 -p duration:=1.0 -p label:=spin_ccw
  # CCW spin with the left reverse gain reduced
  ros2 run robot_control calibrate --ros-args -p L:=-0.3 -p R:=0.3 -p duration:=1.0 \
      -p L_rev:=0.94 -p label:=spin_ccw_gain094

Parameters
  L, R             wheel commands before gains (motor_control clips to +/-0.5)
  duration         seconds to drive                                (default 2.0)
  L_fwd, L_rev, R_fwd, R_rev
                   multiplier per wheel and direction               (default 1.0)
  ramp             seconds to ramp the command up from 0           (default 0.15)
  rate             publish rate in Hz                               (default 20.0)
  start_delay      seconds to wait before moving                   (default 1.0)
  settle           seconds to wait after the stop                  (default 0.5)
  label            free text saved in the log
  log_file         CSV log path
"""
import csv
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

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
        self.declare_parameter('L_fwd', 1.0)
        self.declare_parameter('L_rev', 1.0)
        self.declare_parameter('R_fwd', 1.0)
        self.declare_parameter('R_rev', 1.0)
        self.declare_parameter('ramp', 0.15)
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('start_delay', 1.0)
        self.declare_parameter('settle', 0.5)
        self.declare_parameter('label', '')
        self.declare_parameter('log_file', '~/ros2_ws/logs/calibration_log_v3.csv')

        self.pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)

    # ------------------------------------------------------------------ helpers
    def send(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self.pub.publish(msg)

    def spin_for(self, seconds):
        """Wait while letting ROS process its work, without sending commands."""
        end = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)

    def param(self, name):
        return self.get_parameter(name).value

    def gains(self):
        return (float(self.param('L_fwd')), float(self.param('L_rev')),
                float(self.param('R_fwd')), float(self.param('R_rev')))

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

        # 2. Short pause before moving.
        self.get_logger().info(
            f'Run: L={left:.3f} R={right:.3f} (after gains {eff_l:.3f}, {eff_r:.3f}) '
            f'for {duration:.2f} s (starting in {self.param("start_delay"):.1f} s)')
        self.spin_for(float(self.param('start_delay')))

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

        # 4. Stop (sent several times in case one is dropped), then let it coast.
        for _ in range(5):
            self.send(0.0, 0.0)
            self.spin_for(0.02)
        self.spin_for(float(self.param('settle')))

        # 5. Report.
        glf, glr, grf, grr = self.gains()
        print('\n================ calibration result ================')
        print(f' label            : {label or "-"}')
        print(f' command  L, R    : {left:.3f}, {right:.3f}  -> sent {eff_l:.3f}, {eff_r:.3f}')
        print(f' gains            : L_fwd {glf:.3f}  L_rev {glr:.3f} | R_fwd {grf:.3f}  R_rev {grr:.3f}')
        print(f' drive time       : {actual:.3f} s')
        print('====================================================\n')

        self.append_log(label, left, right, eff_l, eff_r, duration, actual)

    def append_log(self, label, left, right, eff_l, eff_r, duration, actual):
        path = os.path.expanduser(str(self.param('log_file')))
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        new_file = not os.path.exists(path)
        glf, glr, grf, grr = self.gains()

        def fmt(x, nd=4):
            return '' if x is None else f'{x:.{nd}f}'

        try:
            with open(path, 'a', newline='') as f:
                writer = csv.writer(f)
                if new_file:
                    writer.writerow([
                        'timestamp', 'label', 'L', 'R', 'L_sent', 'R_sent',
                        'L_fwd', 'L_rev', 'R_fwd', 'R_rev',
                        'duration_requested_s', 'duration_actual_s'])
                writer.writerow([
                    datetime.now().isoformat(timespec='seconds'), label,
                    fmt(left, 3), fmt(right, 3), fmt(eff_l, 3), fmt(eff_r, 3),
                    fmt(glf, 3), fmt(glr, 3), fmt(grf, 3), fmt(grr, 3),
                    fmt(duration, 3), fmt(actual, 3),
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