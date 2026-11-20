#!/usr/bin/env python3
"""HW4 controller: plans the path (planner.py) and drives it the motion.py way, without stopping.

Same split as HW3:
    detector_node   camera -> landmark_obs                              (unchanged)
    localizer_node  motor_commands + landmark_obs -> robot_pose (EKF)   (map_file + accept_moving on)
    hw4_control     robot_pose -> motor_commands                        (this file; replaces closed_loop_control)

Driving uses only the commands calibrated in robot_control/motion.py:
    - a leg is driven straight at DRIVE_CMD and NOT cut into 10 cm steps: the command keeps going
      while robot_pose is watched, until the waypoint is reached or passed;
    - a corner is a timed turn in place (motion.table_time on CCW_TABLE / CW_TABLE, with the same
      ramp as motion.send), and the next leg starts right after it (no STOP_SECONDS, no SETTLE);
    - if the heading drifts more than realign_tol while driving, a short timed turn re-aims it.
The localizer does the odometry from these commands with the same tables, and corrects it with
tags while the robot drives straight.

Start: the robot stands still until the localizer has used a tag (landmark_updates went up) and
the pose is certain, so it never sets off on an unchecked pose. After the first turn it stands
still once more and looks (until a new tag fix, at most look_max s) before driving.

States:  settle (start, waiting for a tag fix) -> spin (timed turn) -> look (first turn only)
         -> drive -> spin -> drive ... -> done

Run (motor_control, camera, detector and localizer must be running):
    ros2 run landmark_nav hw4_control --ros-args -p mode:=short
    ros2 run landmark_nav hw4_control --ros-args -p mode:=safe
Writes <log_dir>/hw4_<mode>_<time>.csv and ..._path.csv  ->  python3 plot_hw4_run.py <csv>
"""
import csv
import math
import os
import time

import numpy as np

try:
    from landmark_nav import planner
except ImportError:                                   # sim_hw4.py from this folder
    import planner

DEFAULTS = {
    'rate': 20.0,                 # Hz (motor_control stops after 0.15 s without a command)
    # ---- start: stand still until the localizer has a tag fix
    'start_wait': 1.0,            # s standing still at the start, at least ...
    'start_min_fixes': 0,         # ... until the localizer has used this many tag observations ...
    'start_sigma_xy': 0.05,       # ... and the pose is this certain (m) ...
    'start_sigma_th_deg': 6.0,    # ... (deg)
    'start_max_wait': 15.0,       # s: give up waiting and go anyway (no tag in view?)
    # ---- look once after the first turn
    'look_after_first_turn': False,
    'look_min': 0.5,              # s standing still, at least ...
    'look_max': 2.0,              # s ... and at most, waiting for a new tag fix
    # ---- turning
    'aim_tol_deg': 15.0,           # turn before a leg if the heading error is larger (smaller turns
                                  #   are poorly modelled by the turn tables: they are extrapolated)
    'max_turns_in_row': 3,        # after this many turns without driving `turn_reset_dist` in between,
    'turn_reset_dist': 0.10,      #   drive anyway (stops back-and-forth turning if the turn tables are off)
    'realign_tol_deg': 18.0,      # while driving: re-aim if the error grows past this ...
    'realign_min_dist': 0.25,     # m ... and the waypoint is still this far
    # ---- waypoints
    'near': 0.06,                 # m waypoint reached (also when it is passed)
}


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class PivotController:
    """No ROS: tick(t, pose, sigma, n_updates) -> (L, R).
    `mo` is robot_control.motion (calibration + helpers); n_updates = localizer's landmark_updates."""

    def __init__(self, path, cfg, mo):
        self.path = np.array(path, dtype=float)
        self.c = cfg
        self.mo = mo
        self.i = 1                                         # waypoint we are driving to
        self.state, self.t_state = 'settle', None
        self.t_motion = 0.0                                # start of the current drive / turn (ramp)
        self.spin_sign, self.spin_end, self.after_spin = 0, 0.0, 'drive'
        self.last_xy = None
        self.turns_in_row = 0
        self.since_turn = 0.0                              # m driven since the last turn
        self.n0 = None                                     # localizer tag updates when we started
        self.look_n0 = None                                # ... when the look began
        self.first_turn_done = False
        self.start_note = ''                               # why the start wait ended (for the log)

    # ---------------------------------------------------------------- helpers
    def _set(self, state, t):
        self.state, self.t_state = state, t

    def _ramp(self, t):
        """motion.send's ramp. Counted from one tick BEFORE the motion starts, so the first command
        is already non-zero and the localizer times the motion exactly as long as it was planned."""
        r = self.mo.RAMP_SECONDS
        return 1.0 if r <= 0 else min(1.0, (t - self.t_motion + 1.0 / self.c['rate']) / r)

    def _advance(self, p):
        """Skip waypoints that are reached or passed. False when the goal is done."""
        while self.i < len(self.path):
            wp, prev = self.path[self.i], self.path[self.i - 1]
            if np.linalg.norm(wp - p) < self.c['near'] or np.dot(wp - p, wp - prev) < 0.0:
                self.i += 1
            else:
                return True
        return False

    def _heading_err(self, pose):
        wp = self.path[self.i]
        return wrap(math.atan2(wp[1] - pose[1], wp[0] - pose[0]) - pose[2])

    def _start_spin(self, t, err, after):
        """Timed turn in place by err (motion.turn), then `after`. False if below MIN_TURN_DEG."""
        deg = abs(math.degrees(err))
        if deg < self.mo.MIN_TURN_DEG or (after == 'drive' and self.turns_in_row >= self.c['max_turns_in_row']):
            return False
        self.turns_in_row += 1
        self.since_turn = 0.0
        self.spin_sign = 1 if err > 0 else -1
        table = self.mo.CCW_TABLE if self.spin_sign > 0 else self.mo.CW_TABLE
        self.spin_end = t + self.mo.table_time(deg, table)
        self.after_spin = after
        self.t_motion = t
        self._set('spin', t)
        return True

    def _start_leg(self, t, pose):
        """At a waypoint (or after a turn / a look): turn first if needed, else drive."""
        p = np.asarray(pose[:2])
        if not self._advance(p):
            self._set('done', t)
            return
        err = self._heading_err(pose)
        after = 'drive'
        if self.c['look_after_first_turn'] and not self.first_turn_done:
            after = 'look'                                 # first turn: look before driving
        if abs(err) > math.radians(self.c['aim_tol_deg']) and self._start_spin(t, err, after):
            self.first_turn_done = True
            return
        self.first_turn_done = True
        if self.state != 'drive':
            self.t_motion = t
        self._set('drive', t)

    def _drove(self, step):
        self.since_turn += step
        if self.since_turn > self.c['turn_reset_dist']:
            self.turns_in_row = 0

    def _start_ready(self, t, sigma, n_updates):
        """Leave the start only after the localizer has used a tag and the pose is certain."""
        c = self.c
        waited = t - self.t_state
        if waited < c['start_wait']:
            return False
        if n_updates is None:                              # no counter available: plain wait
            self.start_note = 'no landmark_updates topic'
            return True
        if self.n0 is None:
            self.n0 = n_updates
        fixes = n_updates - self.n0
        good = (fixes >= c['start_min_fixes'] and max(sigma[0], sigma[1]) <= c['start_sigma_xy']
                and sigma[2] <= math.radians(c['start_sigma_th_deg']))
        if good:
            self.start_note = f'{fixes} tag fix(es) after {waited:.1f} s'
            return True
        if waited >= c['start_max_wait']:
            self.start_note = (f'NO good tag fix after {waited:.0f} s ({fixes} fixes): going anyway. '
                               'Is the detector seeing tags?')
            return True
        return False

    # ---------------------------------------------------------------- main
    def tick(self, t, pose, sigma, n_updates=None):
        c, mo = self.c, self.mo
        p = np.asarray(pose[:2])
        if self.t_state is None:
            self.t_state = t
        if self.last_xy is not None and self.state == 'drive':
            self._drove(float(np.linalg.norm(p - self.last_xy)))
        self.last_xy = p

        if self.state == 'settle':
            if self.n0 is None and n_updates is not None:
                self.n0 = n_updates
            if self._start_ready(t, sigma, n_updates):
                self._start_leg(t, pose)
        elif self.state == 'spin' and t >= self.spin_end:
            if self.after_spin == 'look':
                self._set('look', t)
                self.look_n0 = n_updates
            else:
                self._start_leg(t, pose)
        elif self.state == 'look':
            new_fix = n_updates is None or self.look_n0 is None or n_updates > self.look_n0
            looked = t - self.t_state
            if (looked >= c['look_min'] and new_fix) or looked >= c['look_max']:
                self._start_leg(t, pose)
        elif self.state == 'drive':
            i_before = self.i
            if not self._advance(p):
                self._set('done', t)
            elif self.i != i_before:                       # corner: turn if needed, else drive on
                self._start_leg(t, pose)
            else:
                err = self._heading_err(pose)
                d = float(np.linalg.norm(self.path[self.i] - p))
                if abs(err) > math.radians(c['realign_tol_deg']) and d > c['realign_min_dist']:
                    self._start_spin(t, err, 'drive')

        if self.state == 'drive':
            k = self._ramp(t)
            return mo.DRIVE_CMD * k, mo.DRIVE_CMD * k
        if self.state == 'spin':
            L, R = mo.spin_commands(self.spin_sign, mo.BASE_TURN)
            k = self._ramp(t)
            return L * k, R * k
        return 0.0, 0.0

    @property
    def done(self):
        return self.state == 'done'


def load_map(map_file, mode):
    """World, planned path (list of (x, y)) and the path metrics."""
    import yaml
    with open(map_file) as f:
        m = yaml.safe_load(f)
    world = planner.World(m)
    res = planner.plan_visibility(world) if mode == 'short' else planner.plan_voronoi(world)
    path = [tuple(map(float, q)) for q in res['path']]
    return world, path, planner.metrics(world, res['path'])


# ===================================================================== ROS node
def main():
    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from std_msgs.msg import Float32MultiArray, Int32

    import robot_control.motion as mo

    rclpy.init()
    node = rclpy.create_node('hw4_control')
    p = node.declare_parameter
    mode = p('mode', 'short').value
    if mode not in ('short', 'safe'):
        raise ValueError("mode must be 'short' or 'safe'")
    map_file = os.path.expanduser(p('map_file', planner.default_map()).value)
    log_dir = os.path.expanduser(p('log_dir', '~/ros2_ws/src/26fall_cse276a/traj_log').value)
    cfg = {k: p(k, v).value for k, v in DEFAULTS.items()}
    log = node.get_logger()

    world, path, met = load_map(map_file, mode)
    ctl = PivotController(path, cfg, mo)
    log.info(f'{mode} path ({met["length"]:.2f} m, {met["waypoints"]} legs, obstacle clearance '
             f'{met["obstacle"]:.2f} m): ' + ' -> '.join(f'({x:.2f}, {y:.2f})' for x, y in path))

    pub = node.create_publisher(Float32MultiArray, 'motor_commands', 10)
    state = {'pose': None, 'sigma': None, 'updates': None}

    def on_pose(msg):
        q, cov = msg.pose.pose.orientation, msg.pose.covariance
        state['pose'] = (msg.pose.pose.position.x, msg.pose.pose.position.y, 2.0 * math.atan2(q.z, q.w))
        state['sigma'] = tuple(math.sqrt(max(0.0, cov[i])) for i in (0, 7, 35))

    def on_updates(msg):
        state['updates'] = msg.data                        # tag observations the localizer has used

    node.create_subscription(PoseWithCovarianceStamped, 'robot_pose', on_pose, 10)
    node.create_subscription(Int32, 'landmark_updates', on_updates, 10)

    def send(L, R):
        msg = Float32MultiArray()
        msg.data = [float(L), float(R)]
        pub.publish(msg)

    os.makedirs(log_dir, exist_ok=True)
    name = os.path.join(log_dir, f'hw4_{mode}_{time.strftime("%m%d_%H%M%S")}.csv')
    with open(name.replace('.csv', '_path.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['x', 'y'])
        w.writerows(path)
    log_f = open(name, 'w', newline='')
    out = csv.writer(log_f)
    out.writerow(['t', 'x', 'y', 'theta', 'sx', 'sy', 'stheta', 'L', 'R', 'state'])

    # the start pose in map_hw4.yaml must be the localizer's start pose (same map_file)
    while rclpy.ok() and state['pose'] is None:
        log.info('waiting for robot_pose (is the localizer running with map_file?)', throttle_duration_sec=2.0)
        rclpy.spin_once(node, timeout_sec=0.1)
    x0, y0, _ = state['pose']
    if math.hypot(x0 - world.start[0], y0 - world.start[1]) > 0.2:
        log.warn(f'robot_pose starts at ({x0:.2f}, {y0:.2f}), not at the map start: '
                 'is the localizer running with map_file? (restart it before every run)')
    log.info('standing still until the localizer has a tag fix ...')

    t0 = time.monotonic()
    period = 1.0 / cfg['rate']
    prev = None
    try:
        while rclpy.ok() and not ctl.done:
            t_next = time.monotonic() + period
            t = time.monotonic() - t0
            L, R = ctl.tick(t, state['pose'], state['sigma'], state['updates'])
            send(L, R)
            if ctl.state != prev:
                if prev == 'settle':
                    x, y, th = state['pose']
                    log.info(f'start: {ctl.start_note}; pose ({x:.2f}, {y:.2f}, {math.degrees(th):.0f} deg)')
                log.info(f'{prev} -> {ctl.state}  (waypoint {min(ctl.i, len(path) - 1)}/{len(path) - 1})')
                prev = ctl.state
            out.writerow([f'{t:.3f}', *(f'{v:.4f}' for v in state['pose']), *(f'{v:.4f}' for v in state['sigma']),
                          f'{L:.3f}', f'{R:.3f}', ctl.state])
            while time.monotonic() < t_next:
                rclpy.spin_once(node, timeout_sec=max(0.0, t_next - time.monotonic()))
        if ctl.done:
            x, y, _ = state['pose']
            gx, gy = path[-1]
            log.info(f'== goal reached in {time.monotonic() - t0:.1f} s, estimate '
                     f'{math.hypot(gx - x, gy - y) * 100:.1f} cm from the goal. log: {name}')
    except KeyboardInterrupt:
        pass
    finally:
        log_f.close()
        end = time.monotonic() + 1.0
        while time.monotonic() < end:                      # leave the robot stopped
            try:
                send(0.0, 0.0)
            except Exception:
                break
            time.sleep(0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
