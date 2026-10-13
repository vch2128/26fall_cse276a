#!/usr/bin/env python3
"""
Read the gyro once per second, so you can turn the robot by hand and check
whether the gyro is accurate. Every line is also saved to ~/gyro_log.txt.

Run (motor_control must be running in another terminal, it publishes /imu_data):
    python3 ~/txt.py
Stop with Ctrl+C. Run it again to reset the angle to 0.

1. Keep the robot STILL for the first 10 seconds (the program measures the bias).
2. Write down "turned", turn the robot by hand (e.g. 90 deg), wait 2 s.
3. Write down "turned" again: (after - before) should be about +90 (left) or -90 (right).

Columns:
    rate    average turning speed in the last second (deg/s, ~0 when still)
    turned  total angle since the start (deg, + = left, - = right)
"""
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

BIAS_SECONDS = 10.0       # the gyro is noisy: average 10 s for a good zero point
LOG_FILE = os.path.expanduser('~/gyro_log.txt')


class GyroCheck(Node):
    def __init__(self):
        super().__init__('gyro_check')
        self.create_subscription(Imu, 'imu_data', self.callback, 50)
        self.create_timer(1.0, self.show)
        self.start = time.monotonic()
        self.bias_samples = []
        self.bias = None            # gyro z when standing still
        self.turned = 0.0           # [deg] gyro z added up over time
        self.last_time = None
        self.last_second = []       # gyro z values of the last second
        self.count = 0
        self.log = open(LOG_FILE, 'w')
        self.log.write('time_s, rate_deg_s, turned_deg\n')

    def callback(self, msg):
        now = time.monotonic()
        z = msg.angular_velocity.z          # deg/s on this robot
        self.count += 1

        if self.bias is None:               # still measuring the bias
            self.bias_samples.append(z)
            if now - self.start >= BIAS_SECONDS:
                self.bias = sum(self.bias_samples) / len(self.bias_samples)
                print(f'bias = {self.bias:+.3f} deg/s. Now turn the robot by hand.\n')
            self.last_time = now
            return

        self.last_second.append(z - self.bias)
        dt = now - self.last_time
        if dt < 0.2:                        # ignore long gaps
            self.turned += (z - self.bias) * dt
        self.last_time = now

    def show(self):
        t = time.monotonic() - self.start
        if self.count == 0:
            print('waiting for /imu_data ... is motor_control running?')
            return
        if self.bias is None:
            print(f'{t:4.0f} s  measuring bias, keep the robot still ... '
                  f'({BIAS_SECONDS - t:.0f} s left)')
            return
        rate = sum(self.last_second) / len(self.last_second) if self.last_second else 0.0
        self.last_second = []
        print(f'{t:4.0f} s | rate {rate:+7.1f} deg/s | turned {self.turned:+7.1f} deg')
        self.log.write(f'{t:.0f}, {rate:.1f}, {self.turned:.1f}\n')
        self.log.flush()


def main():
    rclpy.init()
    node = GyroCheck()
    print(f'Keep the robot still for {BIAS_SECONDS:.0f} s ... (log: {LOG_FILE})')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()