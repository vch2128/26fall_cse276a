#!/usr/bin/env python3
"""HW3 controller: drives a square or octagon (several laps) while the SLAM node builds the map.

HW1's timed turn / drive motions, steered by the SLAM pose (robot_pose).
Waypoints are relative to the start pose (the SLAM map origin): the robot starts at
(0, 0) heading +x and turns left (CCW) at every corner. For each waypoint:

1. Leg (go_to): up to N_CHECKPOINTS + 1 drives (none shorter than MIN_SEGMENT).
   Before each: face the waypoint, drive remaining / drives_left, re-read pose.
2. Within NEAR: stop the leg and accept the residual error (no small-error turns).
3. Heading (face, only for waypoint files with a heading): up to MAX_TRIES turns until
   within ANG_TOL; errors below DAMP_BELOW are scaled by HEADING_GAIN to avoid oscillation.

Before every pose read (observe) the robot waits SETTLE s for a landmark observation;
if none came and the pose is uncertain, it turns CCW by SEARCH_STEP until a
landmark is seen (max one full circle). Set SEARCH = False to disable.

Run (motor_control, camera, detector and localizer must be running):
    ros2 run landmark_nav closed_loop_control square 1.2 2      # 1.2 m square, 2 laps
    ros2 run landmark_nav closed_loop_control octagon 0.5 2     # octagon with 0.5 m sides, 2 laps
    ros2 run landmark_nav closed_loop_control my_waypoints.txt  # x, y, theta per line (HW2 format)
"""
import math
import os
import sys
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Int32

import robot_control.motion as mo
from robot_control.motion import Motion, angle_diff

# ---- checkpoints
N_CHECKPOINTS = 2                  # stops (pose check + re-aim) between the start of a leg and the waypoint
MIN_SEGMENT = 0.25                 # m: never split a leg into drives shorter than this
# ---- tolerances / heading
NEAR = 0.20                        # m: inside this, end the leg and accept the residual error
ANG_TOL = math.radians(6)          # rad: close enough to the final heading
HEADING_GAIN = 0.7                 # fraction of a small heading error corrected per try
DAMP_BELOW = math.radians(30)      # heading errors larger than this are turned in full
MAX_TRIES = 3                      # heading-correction attempts per waypoint
SETTLE = 5                         # s standing still before reading the pose (detector needs a few frames)
# ---- landmark search
SEARCH = True                      # turn in place to look for a landmark when lost
SEARCH_STEP = math.radians(30)     # rad per search turn (CCW)
SEARCH_MAX_STEPS = 12              # 12 x 30 deg = one full circle
SEARCH_SIGMA_XY = 0.05             # m: only search when the pose is at least this uncertain ...
SEARCH_SIGMA_TH = math.radians(4)  # ... or the heading at least this uncertain

motion = None                      # Motion bound to this node (set in main)
pose = None                        # latest (x, y, theta, sigma_x, sigma_y, sigma_theta)
n_updates = None                   # landmark updates accepted by the localizer so far


def on_pose(msg):
    global pose
    q = msg.pose.pose.orientation
    c = msg.pose.covariance
    pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, 2.0 * math.atan2(q.z, q.w),
            math.sqrt(c[0]), math.sqrt(c[7]), math.sqrt(c[35]))


def on_updates(msg):
    global n_updates
    n_updates = msg.data


# ------------------------------------------------------------------ sensing
def wait_for_landmark(seconds):
    """Stand still for `seconds` (ROS keeps spinning); True if a landmark update came in."""
    n0 = n_updates
    motion.stop(seconds)
    return n_updates is not None and n0 is not None and n_updates > n0


def uncertain():
    _, _, _, sx, sy, sth = pose
    return max(sx, sy) > SEARCH_SIGMA_XY or sth > SEARCH_SIGMA_TH


def search():
    """Small turns in place until the localizer accepts a landmark (at most one full circle)."""
    for i in range(SEARCH_MAX_STEPS):
        print(f'  no landmark in view -> search turn {i + 1}/{SEARCH_MAX_STEPS}')
        motion.turn(SEARCH_STEP)
        if wait_for_landmark(SETTLE):
            print('  landmark found, pose corrected')
            return True
    print('  full circle without a landmark -> continuing on dead reckoning')
    return False


def observe():
    """Stop, let the localizer correct the pose (searching if needed), return (x, y, theta)."""
    seen = wait_for_landmark(SETTLE)
    if SEARCH and not seen and n_updates is not None and uncertain():
        print('no landmark in view and pose is uncertain, searching ...')
        search()
    return pose[:3]


# ------------------------------------------------------------------ motion helpers
def go_to(gx, gy):
    # leg with checkpoints: re-aim at the waypoint before each drive
    x, y, th = observe()
    dist = math.hypot(gx - x, gy - y)
    n_drives = max(1, min(N_CHECKPOINTS + 1, math.ceil(dist / MIN_SEGMENT)))
    for k in range(n_drives, 0, -1):               # k = drives left, including this one
        if dist < NEAR:
            break
        motion.turn(angle_diff(math.atan2(gy - y, gx - x), th))   # skipped if < MIN_TURN_DEG
        motion.drive(dist / k)
        x, y, th = observe()
        dist = math.hypot(gx - x, gy - y)
        print(f'  checkpoint: est=({x:.3f}, {y:.3f}, {math.degrees(th):+.1f}deg)  {dist:.3f} m to go')


def face(gth):
    for _ in range(MAX_TRIES):
        _, _, th = observe()
        err = angle_diff(gth, th)
        if abs(err) < ANG_TOL:
            return
        motion.turn(HEADING_GAIN * err if abs(err) < DAMP_BELOW else err)


# ------------------------------------------------------------------ main
def load_waypoints(name):
    """Look for `name` as a path, then in landmark_nav/config, then in robot_control/config (HW1)."""
    candidates = [os.path.expanduser(name),
                  os.path.join(get_package_share_directory('landmark_nav'), 'config', name),
                  os.path.join(mo.FOLDER, name)]
    path = next((c for c in candidates if os.path.isfile(c)), None)
    if path is None:
        raise FileNotFoundError(f'{name} not found in: {candidates}')
    points = []
    for line in open(path):
        if line.strip() and not line.startswith('#'):
            x, y, theta = line.split(',')
            points.append((float(x), float(y), float(theta)))
    print(f'waypoints from {path}: {points}')
    return points[1:], len(points) - 1                # first point = start; the whole file is one lap


def make_path(kind, size, laps):
    """Corners of a square (4 sides) or octagon (8 sides) with side `size` m, starting at (0, 0)
    heading +x and turning left, repeated `laps` times. Heading None = no face() at corners."""
    sides, turn = {'square': (4, math.pi / 2), 'octagon': (8, math.pi / 4)}[kind]
    points, x, y, th = [], 0.0, 0.0, 0.0
    for _ in range(laps * sides):
        x, y = x + size * math.cos(th), y + size * math.sin(th)
        points.append((round(x, 4) + 0.0, round(y, 4) + 0.0, None))
        th += turn
    print(f'{kind}: side {size} m ({size / 0.3048:.1f} ft), {laps} lap(s), {len(points)} waypoints')
    return points, sides


def main():
    global motion
    rclpy.init()
    node = rclpy.create_node('closed_loop_control')
    motion = Motion(node)
    node.create_subscription(PoseWithCovarianceStamped, 'robot_pose', on_pose, 10)
    node.create_subscription(Int32, 'landmark_updates', on_updates, 10)

    args = sys.argv[1:]
    if args and args[0] in ('square', 'octagon'):
        size = float(args[1]) if len(args) > 1 else 1.2
        laps = int(args[2]) if len(args) > 2 else 1
        waypoints, per_lap = make_path(args[0], size, laps)
    else:
        waypoints, per_lap = load_waypoints(args[0] if args else 'waypoints_hw3.txt')
    while (pose is None or n_updates is None) and rclpy.ok():
        node.get_logger().info('waiting for robot_pose and landmark_updates ...', throttle_duration_sec=2.0)
        motion.spin_for(0.1)

    t_start = time.monotonic()
    try:
        for k, (gx, gy, gth) in enumerate(waypoints, 1):
            print(f'--> waypoint {k}/{len(waypoints)} ({gx}, {gy})')
            go_to(gx, gy)
            if gth is not None:
                face(gth)
            if k % per_lap == 0:                   # end of a lap (back at the start corner)
                motion.stop(SETTLE)
                x, y, th, sx, sy, sth = pose
                print(f'== lap {k // per_lap} done at t = {time.monotonic() - t_start:.0f} s: '
                      f'est=({x:.3f}, {y:.3f}, {math.degrees(th):+.1f}deg)  '
                      f'sigma=({sx:.3f}, {sy:.3f}, {math.degrees(sth):.1f}deg)')
    except KeyboardInterrupt:
        pass
    finally:
        try:
            motion.stop(0.2)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
