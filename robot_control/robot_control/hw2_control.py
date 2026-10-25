#!/usr/bin/env python3
"""
CSE 276A HW2 - drive through the waypoints, closed loop with landmark objects.

Stop and go, built on the HW1 motions (copied in below from velocity_control.py,
so this file does not import it):
    1. look():  stand still, take what landmark_detector sees, correct the pose (x, y, heading)
    2. closer than POS_TOL to the waypoint -> done, else
    3. turn to face the waypoint, drive at most STEP meters toward it, back to 1.
At the waypoint: turn to its theta, look, turn the rest if still off by more than ANGLE_TOL.

Pose from the landmarks (map = landmarks.txt):
    0 seen   keep the estimate from the motion commands (same as HW1)
    1+ seen  for every landmark: heading from the direction we see it in, position from
             how far away it is (from_one); average over the landmarks

Run (camera, motor_control and landmark_detector must be running):
    ros2 run robot_control hw2_control
    ros2 run robot_control hw2_control test_line.txt

Every run writes ~/ros2_ws/logs/hw2_<time>.csv: the pose estimate before / after every
motion and after every look (for the trajectory plot in the report).
"""
import csv
import json
import math
import os
import statistics
import sys
import time

import rclpy
from std_msgs.msg import Float32MultiArray, String


# =====================================================================================
# HW1 motions + calibration (from velocity_control.py)
# =====================================================================================

# ---- driving straight ----
SECONDS_PER_METER = 3.2     # how many seconds of [0.25, 0.25] drive 1 meter

# ---- wheel gains (multiply BASE_TURN for each wheel / direction) ----
L_fwd = 0.93
L_rev = 0.94
R_fwd = 1.08
R_rev = 1

# ---- turning ----
BASE_TURN = 0.3             # command for the timed turn

# Measured (degrees, seconds) at BASE_TURN with the gains above.
# Between points: straight-line interpolation; outside: the end segment is extended.
CCW_TABLE = [(45, 0.68), (90, 1.08), (135, 1.48), (180, 1.88)]   # left turns
CW_TABLE = [(45, 0.7), (90, 1.04), (135, 1.38), (180, 1.95)]    # right turns
# NOTE: CW 90 and 135 sit on opposite sides of a straight line by ~0.07 s; re-measure.

MIN_TURN_DEG = 3.0          # smaller turns are skipped

# ---- motion ----
STOP_SECONDS = 0.5          # pause after every motion so the robot is fully still
RAMP_SECONDS = 0.15         # ramp every command up from 0 over this time

FOLDER = os.path.expanduser('~/ros2_ws/src/26fall_cse276a/robot_control/config')


def spin_for(seconds):
    """Wait while letting ROS process its work (spin_once instead of time.sleep).
    This is also what delivers the landmark messages to on_landmarks during look()."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.01)


def publish(left, right):
    msg = Float32MultiArray()
    msg.data = [float(left), float(right)]
    pub.publish(msg)


def stop(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        publish(0.0, 0.0)
        spin_for(0.05)


def ramp_factor(t0):
    """0 -> 1 over RAMP_SECONDS after t0 (same formula as calibrate.py)."""
    if RAMP_SECONDS <= 0.0:
        return 1.0
    return min(1.0, (time.monotonic() - t0) / RAMP_SECONDS)


def send(left, right, seconds):
    """Send [left, right] to motor_control for some seconds (ramped up at the start),
    then stop."""
    t0 = time.monotonic()
    end = t0 + seconds
    while time.monotonic() < end:
        k = ramp_factor(t0)
        publish(left * k, right * k)
        spin_for(min(0.05, max(0.0, end - time.monotonic())))  # motor_control needs a msg every 0.15 s
    stop(STOP_SECONDS)


def table_time(deg, table):
    """Seconds to turn `deg` degrees: piecewise-linear through the measured points,
    with the first / last segment extended for angles outside the table."""
    pts = sorted(table)
    if deg <= pts[0][0]:
        (a0, t0), (a1, t1) = pts[0], pts[1]
    elif deg >= pts[-1][0]:
        (a0, t0), (a1, t1) = pts[-2], pts[-1]
    else:
        (a0, t0), (a1, t1) = next(p for p in zip(pts, pts[1:]) if p[0][0] <= deg <= p[1][0])
    return max(0.0, t0 + (deg - a0) * (t1 - t0) / (a1 - a0))


def spin_commands(sign, speed):
    """Wheel commands for spinning in place. sign > 0: left (CCW), sign < 0: right (CW)."""
    if sign > 0:
        return -speed * L_rev, speed * R_fwd
    return speed * L_fwd, -speed * R_rev


def rotate(sign, seconds, speed=BASE_TURN):
    """Spin in place for `seconds` (ramped up at the start), then stop."""
    left, right = spin_commands(sign, speed)
    send(left, right, seconds)


def timed_turn(radians):
    """HW1 turn: wrap to (-180, 180], skip tiny turns, look the duration up in the table."""
    radians = math.atan2(math.sin(radians), math.cos(radians))
    target = math.degrees(radians)            # signed, + = left (CCW)
    if abs(target) < MIN_TURN_DEG:
        return
    sign = 1 if target > 0 else -1
    table = CCW_TABLE if sign > 0 else CW_TABLE
    seconds = table_time(abs(target), table)
    print(f'  turn {target:+6.1f} deg: {seconds:.3f} s')
    rotate(sign, seconds)


def timed_drive(meters):
    """HW1 drive: straight for meters * SECONDS_PER_METER, backwards if meters < 0."""
    if meters >= 0:
        send(0.25, 0.25, meters * SECONDS_PER_METER)
    else:
        send(-0.25, -0.25, -meters * SECONDS_PER_METER)


def read_waypoints(name):
    points = []
    for line in open(os.path.join(FOLDER, name)):
        if line.strip() and not line.startswith('#'):
            x, y, theta = line.split(',')
            points.append((float(x), float(y), float(theta)))
    return points


# =====================================================================================
# HW2: closed loop with the landmarks
# =====================================================================================

STEP = 0.4                      # m, drive at most this far, then stop and look again
POS_TOL = 0.08                  # m, close enough to the waypoint
ANGLE_TOL = math.radians(5)     # close enough to the waypoint heading
LOOK_SECONDS = 1.0              # stand still this long to collect detections
MAX_ROUNDS = 10                 # safety: at most this many look / turn / drive per waypoint

LOG_FOLDER = os.path.expanduser('~/ros2_ws/logs')

node = None
pub = None
pose = [0.0, 0.0, 0.0]          # x, y, heading estimate (world frame)
landmarks = {}                  # name -> (x, y, height)
seen = {}                       # name -> [(forward, left), ...] collected during look()
look_start = None               # time look() started; older images are ignored
log = None
t_start = 0.0


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def read_landmarks(name='landmarks.txt'):
    """landmarks.txt: name, x, y, height  ->  {name: (x, y, height)}"""
    marks = {}
    for line in open(os.path.join(FOLDER, name)):
        if line.strip() and not line.startswith('#'):
            n, x, y, h = [s.strip() for s in line.split(',')]
            marks[n] = (float(x), float(y), float(h))
    return marks


def log_pose(event):
    log.writerow([f'{time.monotonic() - t_start:.2f}',
                  f'{pose[0]:.4f}', f'{pose[1]:.4f}', f'{pose[2]:.4f}', event])


def on_landmarks(msg):
    data = json.loads(msg.data)
    if look_start is None or data['stamp'] < look_start:
        return                  # image taken while moving: blurred, and the robot was elsewhere
    for name, p in data['objects'].items():
        seen.setdefault(name, []).append(p)


def from_one(p, q):
    """One landmark. p = (forward, left) where the robot sees it, q = (x, y) on the map.
    Keep the direction landmark -> robot of the current estimate, put the robot at the
    measured distance, and set the heading so the landmark is where the camera sees it."""
    away = math.atan2(pose[1] - q[1], pose[0] - q[0])
    r = math.hypot(p[0], p[1])
    x, y = q[0] + r * math.cos(away), q[1] + r * math.sin(away)
    heading = math.atan2(q[1] - y, q[0] - x) - math.atan2(p[1], p[0])
    return x, y, wrap(heading)


def look():
    """Stand still, collect detections, correct the pose with them."""
    global look_start
    seen.clear()
    look_start = node.get_clock().now().nanoseconds * 1e-9
    stop(LOOK_SECONDS)
    look_start = None

    # median over the frames of where each landmark was seen -> (robot frame, map)
    pairs = []
    for name, ps in seen.items():
        if name in landmarks:
            p = (statistics.median(f for f, _ in ps), statistics.median(l for _, l in ps))
            pairs.append((p, landmarks[name][:2]))
    if pairs:
        # every landmark alone gives a corrected pose; average them
        ests = [from_one(p, q) for p, q in pairs]
        pose[:] = (sum(e[0] for e in ests) / len(ests),
                   sum(e[1] for e in ests) / len(ests),
                   math.atan2(sum(math.sin(e[2]) for e in ests),
                              sum(math.cos(e[2]) for e in ests)))

    names = '+'.join(seen) if pairs else 'nothing'
    print(f'  look: {names:30s} -> ({pose[0]:.2f}, {pose[1]:.2f}, '
          f'{math.degrees(pose[2]):.0f} deg)')
    log_pose(f'look {names}')


def turn(angle):
    """Timed turn; the estimate turns with it."""
    angle = wrap(angle)
    log_pose('turn start')
    timed_turn(angle)
    if abs(math.degrees(angle)) >= MIN_TURN_DEG:    # timed_turn skips tiny turns
        pose[2] = wrap(pose[2] + angle)
    log_pose('turn end')


def drive(meters):
    """Timed drive (backwards if meters < 0); the estimate moves with it."""
    log_pose('drive start')
    timed_drive(meters)
    pose[0] += meters * math.cos(pose[2])
    pose[1] += meters * math.sin(pose[2])
    log_pose('drive end')


def main():
    global node, pub, log, t_start
    rclpy.init()
    node = rclpy.create_node('hw2_control')
    pub = node.create_publisher(Float32MultiArray, 'motor_commands', 10)
    node.create_subscription(String, 'landmarks', on_landmarks, 10)

    name = sys.argv[1] if len(sys.argv) > 1 else 'hw2_waypoints.txt'
    waypoints = read_waypoints(name)
    landmarks.update(read_landmarks())
    pose[:] = waypoints[0]          # the robot starts at the first waypoint

    os.makedirs(LOG_FOLDER, exist_ok=True)
    log_file = open(os.path.join(LOG_FOLDER, time.strftime('hw2_%Y%m%d_%H%M%S.csv')), 'w',
                    newline='')
    log = csv.writer(log_file)
    log.writerow(['t', 'x', 'y', 'theta', 'event'])
    t_start = time.monotonic()
    stop(1.0)

    try:
        for goal_x, goal_y, goal_theta in waypoints[1:]:
            print(f'--> ({goal_x}, {goal_y}, {goal_theta})')

            # 1-3. small steps toward (goal_x, goal_y), look after every step
            for _ in range(MAX_ROUNDS):
                look()
                dx, dy = goal_x - pose[0], goal_y - pose[1]
                dist = math.hypot(dx, dy)
                if dist < POS_TOL:
                    break
                angle = wrap(math.atan2(dy, dx) - pose[2])
                step = min(dist, STEP)
                if abs(angle) > math.pi / 2 and dist < STEP:   # just past it: back up
                    angle, step = wrap(angle + math.pi), -step
                turn(angle)
                drive(step)

            # turn to goal_theta, check it with the landmarks
            for _ in range(MAX_ROUNDS):
                error = wrap(goal_theta - pose[2])
                if abs(error) < ANGLE_TOL:
                    break
                turn(error)
                look()

            log_pose('waypoint')
            log_file.flush()
            print(f'reached ({goal_x}, {goal_y}, {goal_theta}): estimate '
                  f'({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f})')
            stop(2.0)                   # pause at every waypoint (visible in the video)
        print('Done!')
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop(0.2)                   # always stop the robot
        except Exception:
            pass                        # ROS may already be shut down after Ctrl-C;
                                        # motor_control stops on its own after 0.15 s
        log_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()