#!/usr/bin/env python3
"""HW4 navigation node: plans the path from the known map, then drives it smoothly.

Replaces closed_loop_control + localizer for HW4 (the detector stays). Structure follows the
instructor's HW2 solution: a fixed-rate loop keeps sending wheel commands and dead-reckons from
them, and tag observations correct the pose while driving. The logic lives in nav_core.py
(no ROS, tested with sim_hw4.py); this file only connects it to ROS.

Subscribes:  landmark_obs   (std_msgs/String JSON from detector_node.py)
Publishes:   motor_commands (std_msgs/Float32MultiArray [L, R])
             robot_pose     (geometry_msgs/PoseWithCovarianceStamped, frame "world")
Writes:      <log_dir>/hw4_<mode>_<time>.csv       t, x, y, theta, sx, sy, stheta, v, w, L, R, state, n_updates
             <log_dir>/hw4_<mode>_<time>_path.csv  the planned path (world frame)
             -> plot with: python3 plot_hw4_run.py <that csv>

Run (motor_control, camera and detector must be running; do NOT run localizer / closed_loop_control):
    ros2 run landmark_nav hw4_nav --ros-args -p mode:=short
    ros2 run landmark_nav hw4_nav --ros-args -p mode:=safe -p v_max:=0.12
Every number in nav_core.DEFAULTS is a parameter (-p name:=value).
"""
import csv
import json
import math
import os
import time

import rclpy
import yaml
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from landmark_nav import planner
from landmark_nav.nav_core import DEFAULTS, Navigator, make_config


class Hw4Nav(Node):
    def __init__(self):
        super().__init__('hw4_nav')
        p = self.declare_parameter
        mode = p('mode', 'short').value
        map_file = os.path.expanduser(p('map_file', planner.default_map()).value)
        log_dir = os.path.expanduser(p('log_dir', '~/ros2_ws/src/26fall_cse276a/traj_log').value)
        overrides = {}
        for k, v in DEFAULTS.items():
            val = p(k, list(v) if isinstance(v, tuple) else v).value
            overrides[k] = val
        self.cfg = make_config(overrides)
        if mode not in ('short', 'safe'):
            raise ValueError("mode must be 'short' or 'safe'")
        self.mode = mode

        with open(map_file) as f:
            m = yaml.safe_load(f)
        world = planner.World(m)
        res = planner.plan_visibility(world) if mode == 'short' else planner.plan_voronoi(world)
        self.path = [tuple(map(float, q)) for q in res['path']]
        tags = {str(k): (float(v['x']), float(v['y']), math.radians(float(v.get('yaw_deg', 0.0))))
                for k, v in (m.get('landmarks') or {}).items()}
        start = (world.start[0], world.start[1], world.start_th)
        self.nav = Navigator(self.path, start, tags, self.cfg)
        met = planner.metrics(world, res['path'])
        self.get_logger().info(
            f'{mode} path from {map_file}: {met["length"]:.2f} m, {met["waypoints"]} legs, '
            f'obstacle clearance {met["obstacle"]:.2f} m, {len(tags)} known tags\n  '
            + ' -> '.join(f'({x:.2f}, {y:.2f})' for x, y in self.path))

        self.pub = self.create_publisher(Float32MultiArray, 'motor_commands', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'robot_pose', 10)
        self.create_subscription(String, 'landmark_obs', self.on_obs, 10)

        stamp = time.strftime('%m%d_%H%M%S')
        os.makedirs(log_dir, exist_ok=True)
        self.log_name = os.path.join(log_dir, f'hw4_{mode}_{stamp}.csv')
        with open(self.log_name.replace('.csv', '_path.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['x', 'y'])
            w.writerows(self.path)
        self.log_f = open(self.log_name, 'w', newline='')
        self.log = csv.writer(self.log_f)
        self.log.writerow(['t', 'x', 'y', 'theta', 'sx', 'sy', 'stheta', 'v', 'w', 'L', 'R', 'state', 'n_updates'])

        self.detector_seen = False
        self.t0 = None                 # set when the run starts (first detector message)
        self.last = None
        self.finished_at = None
        self.create_timer(1.0 / self.cfg['rate'], self.on_timer)

    # ------------------------------------------------------------------ inputs
    def on_obs(self, msg):
        data = json.loads(msg.data)
        self.detector_seen = True
        if self.t0 is None or self.finished_at is not None:
            return
        age = self.get_clock().now().nanoseconds * 1e-9 - data['stamp']
        if data['obs']:
            self.nav.on_obs(data['obs'], max(age, 0.0))

    # ------------------------------------------------------------------ loop
    def send(self, L, R):
        msg = Float32MultiArray()
        msg.data = [float(L), float(R)]
        self.pub.publish(msg)

    def on_timer(self):
        now = time.monotonic()
        if self.t0 is None:                                  # wait for the detector before moving
            self.send(0.0, 0.0)
            if not self.detector_seen:
                self.get_logger().info('waiting for landmark_obs (is the detector running?)',
                                       throttle_duration_sec=2.0)
                return
            self.t0 = self.last = now
            self.get_logger().info('detector OK -> start')
            return
        if self.finished_at is not None:                     # done: hold still, then exit
            self.send(0.0, 0.0)
            if now - self.finished_at > 1.0:
                raise SystemExit
            return

        dt = min(max(now - self.last, 0.0), 0.2)
        self.last = now
        prev_state = self.nav.state
        L, R = self.nav.step(dt)
        self.send(L, R)
        if self.nav.state != prev_state:
            self.get_logger().info(f'{prev_state} -> {self.nav.state}')
        self.publish_pose()
        x, y, th = self.nav.loc.pose
        sx, sy, sth = self.nav.loc.sigma
        self.log.writerow([f'{now - self.t0:.3f}', f'{x:.4f}', f'{y:.4f}', f'{th:.4f}', f'{sx:.4f}',
                           f'{sy:.4f}', f'{sth:.4f}', f'{self.nav.v:.3f}', f'{self.nav.w:.3f}',
                           f'{L:.3f}', f'{R:.3f}', self.nav.state, self.nav.loc.n_updates])
        if self.nav.done:
            self.finished_at = now
            self.log_f.flush()
            gx, gy = self.path[-1]
            self.get_logger().info(
                f'== goal reached in {now - self.t0:.1f} s: est ({x:.3f}, {y:.3f}), '
                f'{math.hypot(gx - x, gy - y) * 100:.1f} cm from the goal (estimate), '
                f'{self.nav.loc.n_updates} tag updates, {self.nav.n_reloc} re-localizations\n'
                f'  log: {self.log_name}')

    def publish_pose(self):
        x, y, th = self.nav.loc.pose
        P = self.nav.loc.ekf.P
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = math.sin(th / 2)
        msg.pose.pose.orientation.w = math.cos(th / 2)
        cov = [0.0] * 36
        idx = [0, 1, 5]
        for i in range(3):
            for j in range(3):
                cov[idx[i] * 6 + idx[j]] = float(P[i, j])
        msg.pose.covariance = cov
        self.pose_pub.publish(msg)

    def destroy_node(self):
        try:
            self.log_f.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = Hw4Nav()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            for _ in range(3):
                node.send(0.0, 0.0)                         # always leave the robot stopped
                time.sleep(0.05)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
