#!/usr/bin/env python3
"""
CSE 276A HW1 - drive through the waypoints in waypoints.txt.

The robot can only do two things:
    drive(meters)   -> send [0.25, 0.25] for  meters * SECONDS_PER_METER
    turn(radians)   -> 1. timed spin in place, duration looked up in a table of
                          measured (angle, seconds) points (CCW_TABLE / CW_TABLE)
                       2. the gyro measures how far the robot actually turned;
                          if it is off by more than TOL_DEG, one slow correction
                          spin is made that stops when the gyro reaches the target

For every waypoint (x, y, theta):
    1. turn to face the waypoint
    2. drive straight to it
    3. turn to theta

Kinematic model: after each motion we update where we think the robot is
    x = x + distance * cos(heading)
    y = y + distance * sin(heading)
    heading = heading + turned angle

Run (motor_control must be running in another terminal):
    ros2 run robot_control velocity_control
    ros2 run robot_control velocity_control test_left.txt

Every turn is logged to ~/ros2_ws/logs/turn_log.csv (target, angle after the
timed turn, angle after the correction) for the report.
"""
import csv
import math
import os
import sys
import time
from datetime import datetime

import rclpy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

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
CCW_TABLE = [(45, 0.65), (90, 1.065), (135, 1.40), (180, 1.85)]   # left turns
CW_TABLE = [(45, 0.555), (90, 1.03), (135, 1.28), (180, 1.75)]    # right turns
# NOTE: CW 90 and 135 sit on opposite sides of a straight line by ~0.07 s; re-measure.

MIN_TURN_DEG = 3.0          # smaller turns are skipped

# ---- gyro correction ----
USE_GYRO_CORRECTION = True
SLOW_TURN = 0.22            # command for the correction spin (must be above the dead zone)
TOL_DEG = 3.0               # accept the timed turn if it is within this many degrees
COAST_DEG = 2.0             # stop the correction this much early (robot coasts after stop)
CORRECTION_TIMEOUT = 1.5    # seconds; the correction never runs longer than this
MAX_CORRECTION_DEG = 20.0   # a larger measured error is treated as bad gyro data, not corrected
MAX_GAP_S = 0.2             # IMU interval >= this is a gap: not integrated (same as calibrate.py)
BIAS_SECONDS = 1.0          # still time to average the gyro bias (= calibrate.py start_delay)
MIN_BIAS_SAMPLES = 10       # fewer readings than this -> no bias, no correction
STOP_SECONDS = 0.5          # pause after every motion so the robot is fully still
RAMP_SECONDS = 0.15         # ramp every command up from 0 over this time
GYRO_UNITS = 'deg'          # units of angular_velocity.z: the board reports deg/s
                            # (motor_control does not convert it)

FOLDER = os.path.expanduser('~/ros2_ws/src/26fall_cse276a/robot_control/config')
TURN_LOG = os.path.expanduser('~/ros2_ws/logs/turn_log.csv')

node = None
pub = None


class Gyro:
    """Gyro z heading, computed exactly as in calibrate.py:
      * rate converted to rad/s (GYRO_UNITS)
      * dt from the message timestamp (fallback: arrival time)
      * trapezoid integration of (rate - bias)
      * intervals outside 0 < dt < MAX_GAP_S are skipped (gaps / clock jumps)
      * bias = mean rate over BIAS_SECONDS while still, needs MIN_BIAS_SAMPLES
    `deg` is the heading in degrees (+ = CCW). The gap / repeat counters are extra
    health checks used to decide whether to trust a turn; they do not change the
    measured angle."""

    def __init__(self):
        self.scale = math.radians(1.0) if GYRO_UNITS.startswith('deg') else 1.0
        self.heading = 0.0         # rad
        self.bias = 0.0            # rad/s
        self.count = 0
        self._last_t = None        # timestamp of the previous message
        self._last_rate = 0.0
        self._last_arrival = None  # monotonic arrival time, for stale()
        self._bias_samples = None  # list while estimating bias
        self.gaps = 0              # intervals skipped as gaps
        self.repeats = 0           # messages identical to the previous one (stale data)

    @property
    def deg(self):
        return math.degrees(self.heading)

    def callback(self, msg):
        self.count += 1
        self._last_arrival = time.monotonic()
        rate = msg.angular_velocity.z * self.scale      # rad/s
        if self._bias_samples is not None:
            self._bias_samples.append(rate)

        # Prefer the sensor timestamp; fall back to arrival time.
        st = msg.header.stamp
        t = st.sec + st.nanosec * 1e-9 if (st.sec or st.nanosec) else time.monotonic()
        if self._last_t is not None:
            dt = t - self._last_t
            if 0.0 < dt < MAX_GAP_S:                    # skip gaps and clock jumps
                self.heading += 0.5 * ((rate - self.bias) +
                                       (self._last_rate - self.bias)) * dt
            else:
                self.gaps += 1
            if rate == self._last_rate:                 # real noise never repeats exactly
                self.repeats += 1
        self._last_t = t
        self._last_rate = rate

    def snapshot(self):
        return (time.monotonic(), self.count, self.gaps, self.repeats)

    def healthy_since(self, snap):
        """Was the gyro stream usable since `snap`? Needs >= 60 % of the expected
        50 Hz messages, no gaps, and mostly fresh (non-repeated) values."""
        t0, count0, gaps0, rep0 = snap
        n = self.count - count0
        expected = (time.monotonic() - t0) * 50.0
        return (n >= 0.6 * expected and self.gaps == gaps0
                and (self.repeats - rep0) <= 0.5 * max(n, 1))

    def stale(self):
        """True if no IMU message arrived recently."""
        return (self._last_arrival is None
                or time.monotonic() - self._last_arrival > MAX_GAP_S)

    def update_bias(self, seconds=BIAS_SECONDS):
        """Average the rate while the robot is still (as calibrate.py does during
        start_delay). Returns False if fewer than MIN_BIAS_SAMPLES readings arrived;
        the old bias is then kept and the caller skips the correction."""
        self._bias_samples = []
        spin_for(seconds)
        samples, self._bias_samples = self._bias_samples, None
        if len(samples) < MIN_BIAS_SAMPLES:
            return False
        self.bias = sum(samples) / len(samples)
        return True


gyro = Gyro()


def spin_for(seconds):
    """Wait while still processing incoming messages (IMU). Never use time.sleep here:
    callbacks only run inside spin_once."""
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


def drive(meters):  # go straight
    send(0.25, 0.25, meters * SECONDS_PER_METER)


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


def rotate(sign, seconds, speed=BASE_TURN, stop_at_deg=None):
    """Spin in place for `seconds`, or until the gyro has measured `stop_at_deg` degrees.
    Then stop and wait for the robot to settle.
    Returns (signed angle turned in deg, + = CCW;  True if the gyro data was usable).
    If the gyro is steering (stop_at_deg given) and its data stops, the spin is aborted."""
    left, right = spin_commands(sign, speed)
    start = gyro.deg
    snap = gyro.snapshot()
    ok = True
    t0 = time.monotonic()
    end = t0 + seconds                        # time limit always applies
    while time.monotonic() < end:
        if stop_at_deg is not None:
            if abs(gyro.deg - start) >= stop_at_deg:
                break
            if time.monotonic() - snap[0] > 0.2 and gyro.stale():
                ok = False                    # gyro went silent: stop instead of guessing
                break
        k = ramp_factor(t0)
        publish(left * k, right * k)
        spin_for(min(0.05, max(0.0, end - time.monotonic())))
    stop(STOP_SECONDS)                        # keeps integrating while it coasts
    ok = ok and gyro.healthy_since(snap)
    return gyro.deg - start, ok


def log_turn(target, timed, final, corrected, gyro_ok):
    try:
        os.makedirs(os.path.dirname(TURN_LOG), exist_ok=True)
        new = not os.path.exists(TURN_LOG)
        with open(TURN_LOG, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['timestamp', 'target_deg', 'after_timed_deg',
                            'after_correction_deg', 'corrected', 'gyro_ok'])
            w.writerow([datetime.now().isoformat(timespec='seconds'), f'{target:.1f}',
                        f'{timed:.1f}', f'{final:.1f}', int(corrected), int(gyro_ok)])
    except OSError as e:
        print(f'  (could not write {TURN_LOG}: {e})')


def turn(radians):
    # take the short way round: wrap the angle to (-180°, 180°]
    radians = math.atan2(math.sin(radians), math.cos(radians))
    target = math.degrees(radians)            # signed, + = left (CCW)
    if abs(target) < MIN_TURN_DEG:
        return
    sign = 1 if target > 0 else -1
    table = CCW_TABLE if sign > 0 else CW_TABLE

    have_imu = USE_GYRO_CORRECTION and gyro.update_bias()   # robot is still here

    # 1. timed turn (the gyro only measures)
    turned, gyro_ok = rotate(sign, table_time(abs(target), table))
    timed = turned

    # 2. one slow correction, stopped by the gyro, with a time limit.
    #    Only if the gyro data during the timed turn was usable AND the error it
    #    reports is believable; otherwise keep the timed turn (never guess).
    corrected = False
    gyro_ok = have_imu and gyro_ok
    err = target - turned                     # signed: + means turn more to the left
    if not USE_GYRO_CORRECTION:
        pass
    elif not have_imu:
        print('  no IMU data before the turn - timed turn only')
    elif not gyro_ok:
        print('  IMU data dropped out or froze during the turn - timed turn only')
    elif abs(err) > MAX_CORRECTION_DEG:
        gyro_ok = False
        print(f'  gyro says {turned:+.1f} deg for a {target:+.1f} deg turn (error {err:+.1f}) '
              '- not believable, timed turn only')
    elif abs(err) > TOL_DEG:
        step, corr_ok = rotate(1 if err > 0 else -1, CORRECTION_TIMEOUT,
                               speed=SLOW_TURN, stop_at_deg=abs(err) - COAST_DEG)
        turned += step
        corrected = True
        if not corr_ok:
            gyro_ok = False
            print('  IMU data dropped out during the correction - correction stopped early')

    print(f'  turn {target:+6.1f} deg: timed {timed:+6.1f}'
          + (f', corrected {turned:+6.1f}' if corrected else '')
          + ('' if gyro_ok else '   (gyro value not trusted)'))
    log_turn(target, timed, turned, corrected, gyro_ok)


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


def main():
    global node, pub
    rclpy.init()
    node = rclpy.create_node('velocity_control')
    pub = node.create_publisher(Float32MultiArray, 'motor_commands', 10)
    node.create_subscription(Imu, 'imu_data', gyro.callback, 50)

    name = sys.argv[1] if len(sys.argv) > 1 else 'waypoints.txt'
    waypoints = read_waypoints(name)
    x, y, heading = waypoints[0]          # the robot starts at the first waypoint
    stop(1.0)
    if gyro.count == 0:
        print('WARNING: no /imu_data received - turns will be timed only')

    try:
        for goal_x, goal_y, goal_theta in waypoints[1:]:
            distance = math.hypot(goal_x - x, goal_y - y)

            if distance > 0.02:
                # 1. turn to face the waypoint
                direction = math.atan2(goal_y - y, goal_x - x)
                turn(angle_diff(direction, heading))
                heading = direction
                # 2. drive to it  (kinematic model: update x, y)
                drive(distance)
                x = x + distance * math.cos(heading)
                y = y + distance * math.sin(heading)

            # 3. turn to the waypoint's theta
            turn(angle_diff(goal_theta, heading))
            heading = goal_theta

            print(f'reached ({goal_x}, {goal_y}, {goal_theta})')
            stop(1.0)
        print('Done!')
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop(0.2)                     # always stop the robot
        except Exception:
            pass                          # ROS may already be shut down after Ctrl-C;
                                          # motor_control stops on its own after 0.15 s
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()