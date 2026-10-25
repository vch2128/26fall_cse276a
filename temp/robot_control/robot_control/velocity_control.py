#!/usr/bin/env python3
"""
CSE 276A HW2 - drive through the waypoints, closed loop with landmark feedback.

Same motions as HW1:
    drive(meters)   -> send [0.25, 0.25] for  |meters| * SECONDS_PER_METER  (meters < 0: backwards)
    turn(radians)   -> timed spin in place, duration looked up in a table of
                       measured (angle, seconds) points (CCW_TABLE / CW_TABLE)
New in HW2:
    look()          -> stand still for LOOK_SECONDS and correct the pose estimate with the
                       landmarks landmark_detector sees (/landmarks: class, range, bearing)

Pose estimate (x, y, heading) = EKF
    predict: the HW1 kinematic model, spread over every motion
        x = x + distance * cos(heading)
        y = y + distance * sin(heading)
        heading = heading + turned angle
        (the uncertainty grows with every meter driven / radian turned)
    update : range + bearing to every detected landmark (positions in landmarks.txt);
             a detection that does not fit the estimate is dropped (Mahalanobis gate)
    Only images taken while standing still are used (no blur, no latency problem).
    No landmark in view -> only predict, like HW1.

For every waypoint (x, y, theta):
    1. look, stop if closer than POS_TOL
    2. turn to face the waypoint (close and behind: face away and back up)
    3. drive straight to it, at most SEGMENT meters, then back to 1.
    4. turn to theta, look, turn the rest if the heading is still off

Run (camera, motor_control and landmark_detector must be running, see hw2_launch.py):
    ros2 run robot_control velocity_control
    ros2 run robot_control velocity_control test_line.txt

Every run writes to ~/ros2_ws/logs/ (for tools/plot_trajectory.py):
    hw2_<time>_traj.csv       estimated pose every 0.5 s
    hw2_<time>_waypoints.csv  estimated pose at each waypoint
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from std_msgs.msg import Float32MultiArray, String

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

# ---- closed loop ----
SEGMENT = 0.5               # drive at most this far, then look again
POS_TOL = 0.05              # m, waypoint reached
ANG_TOL_DEG = 4.0           # final heading good enough
REVERSE_DIST = 0.3          # a closer waypoint behind the robot is reached by backing up
MAX_TRIES = 8               # look / turn / drive rounds per waypoint
LOOK_SECONDS = 1.0          # stand still this long to collect detections
WAYPOINT_PAUSE = 2.0        # extra stop at every waypoint (visible in the video)
LOG_PERIOD = 0.5            # s, trajectory log

# ---- EKF noise ----
# odometry error grows like a random walk: variance = SIG^2 * (distance or angle moved)
SIG_TRANS = 0.15            # m after 1 m driven
SIG_ROT = 0.20              # rad after 1 rad turned
SIG_ROT_DRIVE = 0.10        # rad after 1 m driven (unequal wheels)
SIG_R_REL = 0.10            # range noise = 10 % of the range (box height is noisy)
SIG_R_MIN = 0.05            # m
SIG_B = math.radians(3)     # bearing noise
GATE = 9.21                 # chi^2, 2 DOF, 99 %
MAX_REJECTS = 6             # this many rejected in a row -> the estimate is wrong: inflate P
INFLATE = np.diag([0.2 ** 2, 0.2 ** 2, 0.6 ** 2])

FOLDER = os.path.expanduser('~/ros2_ws/src/26fall_cse276a/robot_control/config')
LOG_FOLDER = os.path.expanduser('~/ros2_ws/logs')

node = None
pub = None
ekf = None
landmarks = []              # (class, x, y)
look_start = None           # ROS time look() started; detections are only used while looking
n_used = 0                  # detections used since the last trajectory log line
rejects = 0                 # detections rejected in a row
traj = None
t_start = 0.0
last_log = 0.0


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class EKF:
    def __init__(self, x, y, heading):
        self.x = np.array([x, y, heading], dtype=float)
        self.P = np.diag([0.01 ** 2, 0.01 ** 2, math.radians(2) ** 2])

    def predict(self, d, dth):
        """Kinematic model: drive d meters along the heading, then turn dth."""
        th = self.x[2]
        self.x += [d * math.cos(th), d * math.sin(th), dth]
        self.x[2] = wrap(self.x[2])
        F = np.array([[1, 0, -d * math.sin(th)],
                      [0, 1, d * math.cos(th)],
                      [0, 0, 1]])
        var_d = SIG_TRANS ** 2 * abs(d) + 1e-8
        var_th = SIG_ROT ** 2 * abs(dth) + SIG_ROT_DRIVE ** 2 * abs(d) + 1e-8
        self.P = F @ self.P @ F.T + np.diag([var_d, var_d, var_th])

    def innovation(self, r, b, lx, ly):
        """Measurement residual, its covariance and Jacobian for landmark (lx, ly)."""
        dx, dy = lx - self.x[0], ly - self.x[1]
        q = max(dx * dx + dy * dy, 1e-6)
        rp = math.sqrt(q)
        y = np.array([r - rp, wrap(b - (math.atan2(dy, dx) - self.x[2]))])
        H = np.array([[-dx / rp, -dy / rp, 0],
                      [dy / q, -dx / q, -1]])
        R = np.diag([max(SIG_R_REL * r, SIG_R_MIN) ** 2, SIG_B ** 2])
        return y, H @ self.P @ H.T + R, H

    def update(self, y, S, H):
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P


def ros_now():
    return node.get_clock().now().nanoseconds * 1e-9


def log_pose():
    """One trajectory line every LOG_PERIOD seconds."""
    global last_log, n_used
    t = time.monotonic()
    if t - last_log < LOG_PERIOD:
        return
    last_log = t
    x, y, th = ekf.x
    sx, sy, sth = np.sqrt(np.diag(ekf.P))
    traj.writerow([f'{t - t_start:.2f}', f'{x:.4f}', f'{y:.4f}', f'{th:.4f}',
                   f'{sx:.4f}', f'{sy:.4f}', f'{sth:.4f}', n_used])
    n_used = 0


def on_landmarks(msg):
    global n_used, rejects
    data = json.loads(msg.data)
    if look_start is None or data['stamp'] < look_start:
        return                                # image taken while moving: blurred / stale
    for det in data['detections']:
        best = None
        for cls, lx, ly in landmarks:         # same class, the one that fits best
            if cls == det['cls']:
                y, S, H = ekf.innovation(det['r'], det['b'], lx, ly)
                d2 = float(y @ np.linalg.solve(S, y))
                if best is None or d2 < best[0]:
                    best = (d2, y, S, H)
        if best is None:
            continue
        if best[0] < GATE:
            ekf.update(*best[1:])
            n_used += 1
            rejects = 0
        else:
            rejects += 1
            if rejects >= MAX_REJECTS:
                print('  landmarks disagree with the estimate - relocalizing')
                ekf.P += INFLATE
                rejects = 0


def spin_for(seconds):
    """Wait while letting ROS process its work (spin_once instead of time.sleep)."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.01)
        log_pose()


def publish(left, right):
    msg = Float32MultiArray()
    msg.data = [float(left), float(right)]
    pub.publish(msg)


def stop(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        publish(0.0, 0.0)
        spin_for(0.05)


def look():
    """Stand still and correct the estimate with the landmarks in view."""
    global look_start
    look_start = ros_now()
    stop(LOOK_SECONDS)
    look_start = None


def ramp_factor(t0):
    """0 -> 1 over RAMP_SECONDS after t0 (same formula as calibrate.py)."""
    if RAMP_SECONDS <= 0.0:
        return 1.0
    return min(1.0, (time.monotonic() - t0) / RAMP_SECONDS)


def send(left, right, seconds, distance=0.0, angle=0.0):
    """Send [left, right] to motor_control for some seconds (ramped up at the start),
    then stop. The estimate moves `distance` / turns `angle`, spread over the motion."""
    if seconds <= 0.0:
        return
    t0 = time.monotonic()
    end = t0 + seconds
    done = 0.0                                # part of the motion already predicted
    while time.monotonic() < end:
        k = ramp_factor(t0)
        publish(left * k, right * k)
        spin_for(min(0.05, max(0.0, end - time.monotonic())))  # motor_control needs a msg every 0.15 s
        f = min(1.0, (time.monotonic() - t0) / seconds)
        ekf.predict((f - done) * distance, (f - done) * angle)
        done = f
    ekf.predict((1.0 - done) * distance, (1.0 - done) * angle)
    stop(STOP_SECONDS)


def drive(meters):  # go straight, backwards if meters < 0
    speed = math.copysign(0.25, meters)
    send(speed, speed, abs(meters) * SECONDS_PER_METER, distance=meters)


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


def rotate(sign, seconds, angle, speed=BASE_TURN):
    """Spin in place for `seconds` (ramped up at the start), then stop."""
    left, right = spin_commands(sign, speed)
    send(left, right, seconds, angle=angle)


def turn(radians):
    # take the short way round: wrap the angle to (-180°, 180°]
    radians = math.atan2(math.sin(radians), math.cos(radians))
    target = math.degrees(radians)            # signed, + = left (CCW)
    if abs(target) < MIN_TURN_DEG:
        return
    sign = 1 if target > 0 else -1
    table = CCW_TABLE if sign > 0 else CW_TABLE
    seconds = table_time(abs(target), table)
    print(f'  turn {target:+6.1f} deg: {seconds:.3f} s')
    rotate(sign, seconds, radians)


def angle_diff(a, b):
    """Smallest angle to turn from b to a (between -pi and pi)."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def read_waypoints(name):
    points = []
    for line in open(os.path.join(FOLDER, name)):
        if line.strip() and not line.startswith('#'):
            x, y, theta = line.split(',')
            points.append((float(x), float(y), float(theta)))
    return points


def read_landmarks(name):
    """landmarks.txt: class, x, y, height_m"""
    marks = []
    for line in open(os.path.join(FOLDER, name)):
        if line.strip() and not line.startswith('#'):
            cls, x, y, _h = [s.strip() for s in line.split(',')]
            marks.append((cls, float(x), float(y)))
    return marks


def main():
    global node, pub, ekf, landmarks, traj, t_start
    rclpy.init()
    node = rclpy.create_node('velocity_control')
    pub = node.create_publisher(Float32MultiArray, 'motor_commands', 10)
    node.create_subscription(String, 'landmarks', on_landmarks, 10)

    name = sys.argv[1] if len(sys.argv) > 1 else 'hw2_waypoints.txt'
    waypoints = read_waypoints(name)
    landmarks = read_landmarks('landmarks.txt')
    ekf = EKF(*waypoints[0])              # the robot starts at the first waypoint

    os.makedirs(LOG_FOLDER, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    traj_file = open(os.path.join(LOG_FOLDER, f'hw2_{stamp}_traj.csv'), 'w', newline='')
    wp_file = open(os.path.join(LOG_FOLDER, f'hw2_{stamp}_waypoints.csv'), 'w', newline='')
    traj = csv.writer(traj_file)
    wp = csv.writer(wp_file)
    traj.writerow(['t', 'x', 'y', 'theta', 'std_x', 'std_y', 'std_theta', 'n_meas'])
    wp.writerow(['i', 'goal_x', 'goal_y', 'goal_theta', 'est_x', 'est_y', 'est_theta',
                 'std_x', 'std_y', 'std_theta'])
    t_start = time.monotonic()
    stop(1.0)

    try:
        for i, (goal_x, goal_y, goal_theta) in enumerate(waypoints[1:], 1):
            for _ in range(MAX_TRIES):
                # 1. look where we are
                look()
                x, y, heading = ekf.x
                distance = math.hypot(goal_x - x, goal_y - y)
                if distance < POS_TOL:
                    break
                # 2. turn to face the waypoint
                direction = math.atan2(goal_y - y, goal_x - x)
                sign = 1.0
                if distance < REVERSE_DIST and abs(angle_diff(direction, heading)) > math.pi / 2:
                    direction, sign = direction + math.pi, -1.0   # close and behind: back up
                turn(angle_diff(direction, heading))
                # 3. drive towards it, one segment at a time
                drive(sign * min(distance, SEGMENT))

            # 4. turn to the waypoint's theta, check it with the landmarks
            for _ in range(MAX_TRIES):
                turn(angle_diff(goal_theta, ekf.x[2]))
                look()
                if abs(angle_diff(goal_theta, ekf.x[2])) < math.radians(ANG_TOL_DEG):
                    break

            x, y, heading = ekf.x
            sx, sy, sth = np.sqrt(np.diag(ekf.P))
            wp.writerow([i, goal_x, goal_y, goal_theta, f'{x:.4f}', f'{y:.4f}', f'{heading:.4f}',
                         f'{sx:.4f}', f'{sy:.4f}', f'{sth:.4f}'])
            wp_file.flush()
            print(f'reached ({goal_x}, {goal_y}, {goal_theta}): '
                  f'estimate ({x:.3f}, {y:.3f}, {heading:.3f})')
            stop(WAYPOINT_PAUSE)
        print('Done!')
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop(0.2)                     # always stop the robot
        except Exception:
            pass                          # ROS may already be shut down after Ctrl-C;
                                          # motor_control stops on its own after 0.15 s
        traj_file.close()
        wp_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
