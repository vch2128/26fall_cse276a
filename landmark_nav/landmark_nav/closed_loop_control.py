#!/usr/bin/env python3
"""HW3 controller: drives a square / octagon (or a waypoint file) while the SLAM node builds the map.

Timed HW1 motions steered by the SLAM pose. Waypoints are relative to the start pose
(map origin, heading +x, left turns at corners).

- scan_start: one full circle at the start to seed the map.
- go_to: up to N_CHECKPOINTS + 1 drives per leg, re-aiming and re-reading the pose
  before each; stops within NEAR.
- peek_known: after each leg, turn toward the closest mapped landmark that is near
  but out of view, to re-observe it.
- face: heading correction, only for waypoint files with a heading.
- observe: before each pose read, stand still SETTLE s; if no landmark and the pose is
  uncertain, search with CCW turns (SEARCH_AT_WAYPOINTS / SEARCH_AT_CHECKPOINTS).

Run (motor_control, camera, detector and localizer must be running):
    ros2 run landmark_nav closed_loop_control square 1.2 2      # 1.2 m sides, 2 laps
    ros2 run landmark_nav closed_loop_control octagon 0.5 2     # 0.5 m sides, 2 laps
    ros2 run landmark_nav closed_loop_control my_waypoints.txt  # x, y, theta per line
"""
import json
import math
import os
import sys
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Int32, String

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
SEARCH_AT_WAYPOINTS = True         # search at the waypoint: before leaving it, and in face()
SEARCH_AT_CHECKPOINTS = False      # search at the stop after each drive (incl. the arrival stop)
SEARCH_STEP = math.radians(45)     # rad per search turn (CCW)
SEARCH_MAX_STEPS = 8              # 8 x 45 deg = one full circle
SEARCH_SIGMA_XY = 0.05             # m: only search when the pose is at least this uncertain ...
SEARCH_SIGMA_TH = math.radians(4)  # ... or the heading at least this uncertain
# ---- active sensing with the SLAM map
SCAN_START = True                  # full circle at the start to seed the map
SCAN_STEP = math.radians(45)       # rad per scan turn (keep below the camera's full FOV)
PEEK = True                        # after each leg, look at a close mapped landmark that is out of view
HALF_FOV = math.radians(30)        # rad: half the camera's horizontal FOV = atan((W/2) / fx), minus a margin
PEEK_RANGE = (0.4, 2.0)            # m: closer gets cut off at the image edge, farther is too noisy
MAX_PEEKS = 1                      # peeks per corner

motion = None                      # Motion bound to this node (set in main)
pose = None                        # latest (x, y, theta, sigma_x, sigma_y, sigma_theta)
n_updates = None                   # landmark updates accepted by the localizer so far
landmap = {}                       # current SLAM map: id -> (x, y), from landmark_map


def on_pose(msg):
    global pose
    q = msg.pose.pose.orientation
    c = msg.pose.covariance
    pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, 2.0 * math.atan2(q.z, q.w),
            math.sqrt(c[0]), math.sqrt(c[7]), math.sqrt(c[35]))


def on_updates(msg):
    global n_updates
    n_updates = msg.data


def on_map(msg):
    global landmap
    landmap = {lid: (v[0], v[1]) for lid, v in json.loads(msg.data).items()}


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


def observe(allow_search):
    """Stop, let the localizer correct the pose (searching if needed and allowed), return (x, y, theta)."""
    seen = wait_for_landmark(SETTLE)
    if allow_search and not seen and n_updates is not None and uncertain():
        print('no landmark in view and pose is uncertain, searching ...')
        search()
    return pose[:3]


def scan_start():
    """Full circle in place at the start (P ~ 0): landmarks added here are the best-placed anchors."""
    steps = round(2 * math.pi / SCAN_STEP)
    print(f'start scan: {steps} x {math.degrees(SCAN_STEP):.0f} deg')
    wait_for_landmark(SETTLE)                      # what is in view before the first turn
    for i in range(steps):
        motion.turn(SCAN_STEP)
        wait_for_landmark(SETTLE)
        print(f'  scan {i + 1}/{steps}: {len(landmap)} landmark(s) in the map')


def peek_known():
    """Turn toward the closest mapped landmark that is near but out of view, and re-observe it."""
    for _ in range(MAX_PEEKS):
        x, y, th = pose[:3]
        cands = []
        for lid, (lx, ly) in landmap.items():
            r = math.hypot(lx - x, ly - y)
            b = angle_diff(math.atan2(ly - y, lx - x), th)    # bearing relative to the robot, + = left
            if PEEK_RANGE[0] < r < PEEK_RANGE[1] and abs(b) > HALF_FOV:
                cands.append((r, b, lid))
        if not cands:
            return
        r, b, lid = min(cands)                     # closest first: best range, most reliable detection
        print(f'  peek at {lid} ({r:.2f} m, {math.degrees(b):+.0f} deg)')
        motion.turn(b)
        seen = wait_for_landmark(SETTLE)
        print(f'  peek at {lid}: {"seen" if seen else "nothing accepted"}')


# ------------------------------------------------------------------ motion helpers
def go_to(gx, gy):
    # leg with checkpoints: re-aim at the waypoint before each drive
    x, y, th = observe(allow_search=SEARCH_AT_WAYPOINTS)        # at the previous waypoint (after the peek)
    dist = math.hypot(gx - x, gy - y)
    n_drives = max(1, min(N_CHECKPOINTS + 1, math.ceil(dist / MIN_SEGMENT)))
    for k in range(n_drives, 0, -1):               # k = drives left, including this one
        if dist < NEAR:
            break
        motion.turn(angle_diff(math.atan2(gy - y, gx - x), th))   # skipped if < MIN_TURN_DEG
        motion.drive(dist / k)
        x, y, th = observe(allow_search=SEARCH_AT_CHECKPOINTS)   # checkpoint (or arrival) stop
        dist = math.hypot(gx - x, gy - y)
        print(f'  checkpoint: est=({x:.3f}, {y:.3f}, {math.degrees(th):+.1f}deg)  {dist:.3f} m to go')


def face(gth):
    for _ in range(MAX_TRIES):
        _, _, th = observe(allow_search=SEARCH_AT_WAYPOINTS)
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
    node.create_subscription(String, 'landmark_map', on_map, 10)

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
        if SCAN_START:
            scan_start()
        for k, (gx, gy, gth) in enumerate(waypoints, 1):
            print(f'--> waypoint {k}/{len(waypoints)} ({gx}, {gy})')
            go_to(gx, gy)
            if PEEK:
                peek_known()
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
