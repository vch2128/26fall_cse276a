#!/usr/bin/env python3
"""EKF localizer: dead reckoning (motor commands only, no IMU) fused with landmark observations.

Subscribes:
    motor_commands (std_msgs/Float32MultiArray [L, R])   -> forward speed + turn rate (HW1 calibration)
    landmark_obs   (std_msgs/String JSON, from detector) -> EKF updates
Publishes:
    robot_pose     (geometry_msgs/PoseWithCovarianceStamped, frame "map"), 20 Hz
Writes:
    log_file CSV at 2 Hz: t, x, y, theta, sx, sy, stheta, n_updates   (report item i)

Run:
    ros2 run landmark_nav localizer --ros-args -p landmarks_file:=/home/ubuntu/ros2_ws/src/26fall_cse276a/landmark_nav/config/landmarks.yaml
"""
import csv
import json
import math
import os
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from spatialmath import UnitQuaternion

from landmark_nav.ekf import PoseEKF
from robot_control import motion as cal        # the calibration the controllers use


class Localizer(Node):
    def __init__(self):
        super().__init__('landmark_localizer')
        p = self.declare_parameter
        x0 = [float(v) for v in p('start_pose', [0.0, 0.0, 0.0]).value]
        # Odometry from the same calibration the controllers use (robot_control/motion.py),
        # so re-calibrating there updates the localizer too. No IMU: turns are open-loop.
        self.v_per_cmd = 1.0 / (cal.SECONDS_PER_METER * cal.DRIVE_CMD)   # m/s per unit wheel command

        def turn_rate(table, gain_sum):
            cmd = 0.5 * cal.BASE_TURN * gain_sum                         # value of 0.5*(R-L) during that turn
            return math.radians(90) / cal.table_time(90, table) / cmd    # rad/s per unit of 0.5*(R-L)

        self.w_ccw = turn_rate(cal.CCW_TABLE, cal.L_rev + cal.R_fwd)    # left turns
        self.w_cw = turn_rate(cal.CW_TABLE, cal.L_fwd + cal.R_rev)      # right turns
        self.cam_offset = p('cam_offset', 0.08).value                     # camera ahead of robot center (m)
        # process noise (std devs), tune from repeated HW1 runs
        self.s_ds = p('sigma_ds_per_m', 0.05).value                      # 5 cm per meter driven
        self.s_th_turn = p('sigma_th_per_rad', 0.10).value               # heading error per rad turned (open-loop)
        self.s_th_drive = p('sigma_th_per_m', 0.03).value                # heading drift per meter driven
        # measurement handling
        self.min_sr = p('min_sigma_r', 0.03).value
        self.min_sb = p('min_sigma_b', math.radians(1.0)).value
        self.max_age = p('max_obs_age', 0.4).value                       # s; older images are stale
        self.max_w = p('max_turn_rate_for_update', 0.4).value            # rad/s; skip blurry frames
        self.use_range = p('use_range', True).value
        lm_file = p('landmarks_file', '').value
        log_file = os.path.expanduser(p('log_file', '~/ros2_ws/logs/trajectory.csv').value)

        lm = yaml.safe_load(open(lm_file))['landmarks']
        self.map = {name: (float(v['x']), float(v['y'])) for name, v in lm.items()}

        P0 = np.diag([0.02 ** 2, 0.02 ** 2, math.radians(2) ** 2])
        self.ekf = PoseEKF(x0, P0, self.map, self.cam_offset)

        self.cmd = (0.0, 0.0)
        self.cmd_time = 0.0
        self.w = 0.0                 # current turn rate estimate (rad/s)
        self.n_updates = 0
        self.n_rejected_in_row = 0
        self.last_t = time.monotonic()

        self.create_subscription(Float32MultiArray, 'motor_commands', self.on_cmd, 10)
        self.create_subscription(String, 'landmark_obs', self.on_obs, 10)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, 'robot_pose', 10)
        self.create_timer(0.05, self.on_timer)        # predict + publish, 20 Hz
        self.create_timer(0.5, self.on_log)           # trajectory dot every 0.5 s

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_f = open(log_file, 'w', newline='')
        self.log = csv.writer(self.log_f)
        self.log.writerow(['t', 'x', 'y', 'theta', 'sx', 'sy', 'stheta', 'n_updates'])
        self.t0 = time.monotonic()
        self.get_logger().info(f'start pose {x0}, landmarks {self.map}')

    # ------------------------------------------------------------------ inputs
    def on_cmd(self, msg):
        if len(msg.data) >= 2:
            self.cmd = (float(msg.data[0]), float(msg.data[1]))
            self.cmd_time = time.monotonic()

    # ----------------------------------------------------------------- predict
    def on_timer(self):
        now = time.monotonic()
        dt, self.last_t = now - self.last_t, now

        L, R = self.cmd
        if now - self.cmd_time > 0.15:               # motor_control stops after 0.15 s without a command
            L = R = 0.0
        driving = L * R > 0.0                          # same sign -> straight; opposite -> spin in place
        ds = 0.5 * (L + R) * self.v_per_cmd * dt if driving else 0.0

        w_per_cmd = self.w_ccw if R > L else self.w_cw
        dth = 0.0 if driving else 0.5 * (R - L) * w_per_cmd * dt
        self.w = dth / dt if dt > 0 else 0.0

        var_ds = (self.s_ds ** 2) * abs(ds) + 1e-8
        var_dth = (self.s_th_turn ** 2) * abs(dth) + (self.s_th_drive ** 2) * abs(ds) + 1e-8
        self.ekf.predict(ds, dth, var_ds, var_dth)
        self.publish_pose()

    # ------------------------------------------------------------------ update
    def on_obs(self, msg):
        data = json.loads(msg.data)
        age = self.get_clock().now().nanoseconds * 1e-9 - data['stamp']
        if age > self.max_age or abs(self.w) > self.max_w:
            return                                      # stale image, or robot spinning (blur, timing)
        for o in data['obs']:
            if o['id'] not in self.map:
                continue
            sr = max(o['sr'], self.min_sr)
            sb = max(o['sb'], self.min_sb)
            if not self.use_range:
                sr = 1e3                                # effectively bearing-only
            ok, d2 = self.ekf.update([o['r'], o['b']], np.diag([sr ** 2, sb ** 2]), o['id'])
            if ok:
                self.n_updates += 1
                self.n_rejected_in_row = 0
            else:
                self.n_rejected_in_row += 1
                self.get_logger().warn(f"rejected {o['id']} (d2={d2:.1f})", throttle_duration_sec=1.0)
                if self.n_rejected_in_row >= 10:        # we are probably lost: open up the uncertainty
                    self.ekf.P += np.diag([0.15 ** 2, 0.15 ** 2, math.radians(15) ** 2])
                    self.n_rejected_in_row = 0
                    self.get_logger().warn('10 rejections in a row -> inflating covariance')

    # ----------------------------------------------------------------- outputs
    def publish_pose(self):
        x, y, th = self.ekf.x
        P = self.ekf.P
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        q = UnitQuaternion.Rz(th)
        msg.pose.pose.orientation.w = float(q.s)
        msg.pose.pose.orientation.z = float(q.v[2])
        cov = [0.0] * 36                                # 6x6 over (x, y, z, roll, pitch, yaw)
        idx = [0, 1, 5]
        for i in range(3):
            for j in range(3):
                cov[idx[i] * 6 + idx[j]] = float(P[i, j])
        msg.pose.covariance = cov
        self.pub.publish(msg)

    def on_log(self):
        x, y, th = self.ekf.x
        sd = np.sqrt(np.diag(self.ekf.P))
        self.log.writerow([f'{time.monotonic() - self.t0:.2f}', f'{x:.4f}', f'{y:.4f}', f'{th:.4f}',
                           f'{sd[0]:.4f}', f'{sd[1]:.4f}', f'{sd[2]:.4f}', self.n_updates])
        self.log_f.flush()

    def destroy_node(self):
        self.log_f.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = Localizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
