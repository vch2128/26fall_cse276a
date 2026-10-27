#!/usr/bin/env python3
"""HW2 closed-loop waypoint controller.

Uses the same open-loop motions as HW1 (robot_control.motion: timed turn / drive),
but steers with the localizer's pose estimate (robot_pose) instead of its own
dead reckoning. For every waypoint (x, y, theta):

1. Leg with checkpoints. The leg is split into up to N_CHECKPOINTS + 1 drives
   (each at least MIN_SEGMENT long). Before every drive the robot stops, reads
   the pose and re-aims at the waypoint, so drift is corrected along the way
   instead of only once the robot thinks it has arrived.
2. Gentle final approach. Within NEAR of the waypoint the robot no longer turns
   to face it (a 60 deg turn for a 5 cm error adds more heading error than it
   removes); it only drives forward / backward along its current heading and
   accepts the sideways part of the error.
3. Heading. Small heading errors are corrected by HEADING_GAIN * error, not the
   full error, so an imperfect turn does not make the robot swing back and forth.

Landmark search: every time the robot stops to read the pose, it waits SETTLE
seconds for a landmark update. If none arrives and the pose is uncertain, it
turns in place by SEARCH_STEP until a landmark is seen (at most one full
circle), so the localizer can correct the pose before the robot moves on.

Run (motor_control, camera, detector and localizer must be running):
    ros2 run landmark_nav closed_loop_control waypoints_hw2.txt
"""
import math
import os
import sys

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Int32

import robot_control.motion as mo
from robot_control.motion import Motion, angle_diff

# ---- checkpoints
N_CHECKPOINTS = 2                  # stops (pose check + re-aim) between the start of a leg and the waypoint
MIN_SEGMENT = 0.25                 # m: never split a leg into drives shorter than this
# ---- tolerances / gentleness
POS_TOL = 0.08                     # m: close enough to the waypoint
NEAR = 0.20                        # m: inside this, only drive forward/backward, never turn to face the goal
ANG_TOL = math.radians(6)          # rad: close enough to the final heading
HEADING_GAIN = 0.7                 # fraction of a small heading error corrected per try
DAMP_BELOW = math.radians(30)      # heading errors larger than this are turned in full
MAX_TRIES = 3                      # final-approach / heading attempts per waypoint
SETTLE = 5                         # s standing still before reading the pose (detector needs a few frames)
# ---- landmark search
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
    if not seen and n_updates is not None and uncertain():
        print('no landmark in view and pose is uncertain, searching ...')
        search()
    return pose[:3]


# ------------------------------------------------------------------ motion helpers
def drive_signed(d):
    """Drive d meters along the current heading; negative = backward (assumes reverse speed = forward)."""
    if d >= 0:
        motion.drive(d)
    else:
        motion.send(-mo.DRIVE_CMD, -mo.DRIVE_CMD, -d * mo.SECONDS_PER_METER)


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
    return points


def main():
    global motion
    rclpy.init()
    node = rclpy.create_node('closed_loop_control')
    motion = Motion(node)
    node.create_subscription(PoseWithCovarianceStamped, 'robot_pose', on_pose, 10)
    node.create_subscription(Int32, 'landmark_updates', on_updates, 10)

    waypoints =  load_waypoints(sys.argv[1] if len(sys.argv) > 1 else 'waypoints_hw2.txt')
    while (pose is None or n_updates is None) and rclpy.ok():
        node.get_logger().info('waiting for robot_pose and landmark_updates ...', throttle_duration_sec=2.0)
        motion.spin_for(0.1)

    results = []
    try:
        for gx, gy, gth in waypoints[1:]:
            print(f'--> waypoint ({gx}, {gy}, {gth:.2f})')
            go_to(gx, gy)
            face(gth)
            motion.stop(SETTLE)
            x, y, th, sx, sy, sth = pose
            results.append((gx, gy, gth, x, y, th))
            print(f'reached ({gx}, {gy}, {gth:.2f})  est=({x:.3f}, {y:.3f}, {th:.3f})  '
                  f'sigma=({sx:.3f}, {sy:.3f}, {math.degrees(sth):.1f}deg)')

        print('\nwaypoint  target -> estimated (also tape-measure the real pose!)')
        for gx, gy, gth, x, y, th in results:
            print(f'({gx}, {gy}, {gth:.2f}) -> ({x:.3f}, {y:.3f}, {th:.3f})  '
                  f'pos err {math.hypot(gx - x, gy - y):.3f} m, '
                  f'ang err {math.degrees(angle_diff(gth, th)):+.1f} deg')
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
