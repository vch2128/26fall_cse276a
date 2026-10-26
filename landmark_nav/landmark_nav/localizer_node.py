#!/usr/bin/env python3
"""EKF localizer: dead reckoning (motor commands only, no IMU) fused with landmark observations.

Subscribes:
    motor_commands (std_msgs/Float32MultiArray [L, R])   -> odometry (HW1 calibration)
    landmark_obs   (std_msgs/String JSON, from detector) -> EKF updates
Publishes:
    robot_pose     (geometry_msgs/PoseWithCovarianceStamped, frame "map"), 20 Hz
Writes:
    log_file CSV at 2 Hz: t, x, y, theta, sx, sy, stheta, n_updates   (report item i)

Odometry uses the same calibration the controller uses to time its motions:
    drive: distance = command * v_per_cmd * time          (SECONDS_PER_METER)
    turn:  angle    = inverse of the HW1 turn table       (models the start-up delay)
Landmarks are only used from images taken after the robot has stood still for a while.

Run:
    ros2 run landmark_nav localizer --ros-args \
        -p landmarks_file:=/home/ubuntu/ros2_ws/src/26fall_cse276a/landmark_nav/config/landmarks.yaml \
        -p cam_offset:=0.10
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

CMD_TIMEOUT = 0.15          # motor_control stops the wheels after 0.15 s without a command


def table_angle(sec, table):
    """Inverse of cal.table_time(): degrees turned after `sec` seconds of a timed turn.

    Piecewise-linear through the measured (deg, sec) points, end segments extended,
    same as table_time(). Never negative: the first ~0.28 s (start-up) counts as 0 deg.
    """
    pts = sorted(table)                                   # [(deg, sec), ...]
    if sec <= pts[0][1]:
        (a0, t0), (a1, t1) = pts[0], pts[1]
    elif sec >= pts[-1][1]:
        (a0, t0), (a1, t1) = pts[-2], pts[-1]
    else:
        (a0, t0), (a1, t1) = next(p for p in zip(pts, pts[1:]) if p[0][1] <= sec <= p[1][1])
    return max(0.0, a0 + (sec - t0) * (a1 - a0) / (t1 - t0))


def mode_of(L, R):
    """What a wheel command does: 'stop', 'drive', 'ccw' (left turn) or 'cw' (right turn)."""
    if L == 0.0 and R == 0.0:
        return 'stop'
    if L * R > 0.0:
        return 'drive'                         # same sign -> straight
    return 'ccw' if R > L else 'cw'            # opposite sign -> spin in place


class Localizer(Node):
    def __init__(self):
        super().__init__('landmark_localizer')
        p = self.declare_parameter
        x0 = [float(v) for v in p('start_pose', [0.0, 0.0, 0.0]).value]
        # Odometry from the same calibration the controllers use (robot_control/motion.py),
        # so re-calibrating there updates the localizer too. No IMU: turns are open-loop.
        self.v_per_cmd = 1.0 / (cal.SECONDS_PER_METER * cal.DRIVE_CMD)   # m/s per unit wheel command
        self.cam_offset = p('cam_offset', 8.0).value                     # camera ahead of rotation center (m) - MEASURE
        # process noise (std devs), tune from repeated HW1 runs
        self.s_ds = p('sigma_ds_per_m', 0.05).value                      # 5 cm per meter driven
        self.s_th_turn = p('sigma_th_per_rad', 0.10).value               # heading error per rad turned
        self.s_th_drive = p('sigma_th_per_m', 0.03).value                # heading drift per meter driven
        # measurement handling
        self.min_sr = p('min_sigma_r', 0.03).value                       # never trust a range better than 3 cm
        self.min_sb = p('min_sigma_b', math.radians(1.0)).value          # never trust a bearing better than 1 deg
        self.max_age = p('max_obs_age', 2.0).value                       # s; YOLO on the Pi CPU is slow
        self.still = p('still_after_motion', 0.5).value                  # s; use images taken this long after stopping
        self.use_range = p('use_range', True).value                      # False = bearing-only updates
        lm_file = p('landmarks_file', '').value
        log_file = os.path.expanduser(p('log_file', '~/ros2_ws/logs/trajectory.csv').value)

        lm = yaml.safe_load(open(lm_file))['landmarks']
        self.map = {name: (float(v['x']), float(v['y'])) for name, v in lm.items()}

        P0 = np.diag([0.02 ** 2, 0.02 ** 2, math.radians(2) ** 2])     # start pose known to 2 cm / 2 deg
        self.ekf = PoseEKF(x0, P0, self.map, self.cam_offset)

        self.cmd = (0.0, 0.0)         # latest wheel command [L, R]
        self.cmd_time = 0.0           # when it arrived (monotonic)
        self.mode = 'stop'            # what the wheels are doing: stop / drive / ccw / cw
        self.mode_start = 0.0         # when the current motion started (monotonic)
        self.mode_done = 0.0          # degrees of the current turn already passed to the EKF
        self.last_moving = 0.0        # ROS time of the last non-zero command (same clock as image stamps)
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
        self.get_logger().info(f'start pose {x0}, cam_offset {self.cam_offset}, landmarks {self.map}')

    def ros_now(self): 
        return self.get_clock().now().nanoseconds * 1e-9

    # ---------------------------------------------------------------- odometry
    def on_cmd(self, msg):
        if len(msg.data) < 2:
            return
        now = time.monotonic()
        self.advance(now)                              # integrate the previous command up to exactly now
        L, R = float(msg.data[0]), float(msg.data[1])
        new = mode_of(L, R)
        if new != self.mode or now - self.cmd_time > CMD_TIMEOUT:    # a new motion starts
            self.mode, self.mode_start, self.mode_done = new, now, 0.0
        self.cmd, self.cmd_time = (L, R), now
        if new != 'stop':
            self.last_moving = self.ros_now()

    def advance(self, now):
        """EKF predict for the motion between last_t and now."""
        t_beg, self.last_t = self.last_t, now
        t_end = min(now, self.cmd_time + CMD_TIMEOUT)  # wheels stop after the timeout
        L, R = self.cmd
        ds = dth = 0.0
        if self.mode == 'drive':
            ds = 0.5 * (L + R) * self.v_per_cmd * max(0.0, t_end - t_beg)
        elif self.mode in ('ccw', 'cw') and t_end > self.mode_start:
            table = cal.CCW_TABLE if self.mode == 'ccw' else cal.CW_TABLE
            total = table_angle(t_end - self.mode_start, table)       # degrees since the turn began
            step = max(0.0, total - self.mode_done)                    # minus what was already counted
            self.mode_done = max(total, self.mode_done)
            dth = math.radians(step) * (1.0 if self.mode == 'ccw' else -1.0)
        if ds == 0.0 and dth == 0.0:
            return
        var_ds = (self.s_ds ** 2) * abs(ds) + 1e-8
        var_dth = (self.s_th_turn ** 2) * abs(dth) + (self.s_th_drive ** 2) * abs(ds) + 1e-8
        self.ekf.predict(ds, dth, var_ds, var_dth)

    def on_timer(self):
        self.advance(time.monotonic())
        self.publish_pose()

    # ------------------------------------------------------------------ update
    def on_obs(self, msg):
        data = json.loads(msg.data)
        stamp = data['stamp']                           # when the image was taken (ROS time)
        if self.ros_now() - stamp > self.max_age:
            return                                      # image too old
        if stamp < self.last_moving + self.still:
            return                                      # taken while moving or right after (blur, wobble)
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
        q = UnitQuaternion.Rz(th)                       # rotation about z only
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