"""HW4 navigation core (no ROS imports, so sim_hw4.py can run it on a laptop).

Same idea as the instructor's HW2 solution (a fixed-rate loop that keeps sending velocity
commands, dead reckoning from the commanded velocities, AprilTag fixes while driving), with:

    Localizer     the HW3 EKF (ekf.py) given the KNOWN tag map: the tags are put in the state
                  with a small covariance, so it localizes instead of building a map. Tags are
                  used while driving; their noise is raised with speed x image age (latency).
    WheelModel    (v, w) -> wheel commands [L, R]. Driving: differential drive + the instructor's
                  deadzone / slope motor model. Turning in place: your calibrated spin command.
    PathFollower  pure pursuit on the planned polyline: steer toward a point `lookahead` m ahead
                  on the path, so corners become arcs and intermediate waypoints are never stops.
                  Speed: ramps up, slows on sharp curvature, slows down to stop at the goal.
                  Turns in place only if the heading error is large (e.g. the start).

All tunable numbers are in DEFAULTS; the ROS node (hw4_nav.py) exposes each as a parameter.
"""
import math

import numpy as np

try:
    from landmark_nav.ekf import SlamEKF, wrap
except ImportError:                               # running from this folder (sim_hw4.py)
    from ekf import SlamEKF, wrap

DEFAULTS = {
    # ---- loop
    'rate': 20.0,                 # Hz control / predict loop (motor_control stops after 0.15 s silence)
    # ---- path following
    'v_max': 0.15,                # m/s cruise speed (start slow: less blur, more tag fixes)
    'v_min': 0.04,                # m/s never crawl slower than this while tracking (except the final stop)
    'accel': 0.25,                # m/s^2 speed ramp-up
    'decel': 0.20,                # m/s^2 used for the stopping profile at the goal
    'lookahead': 0.25,            # m pure-pursuit lookahead (bigger = smoother, cuts corners more)
    'corner_lookahead': 0.10,     # m: never look further than this past the next corner (less cutting
                                  #    toward the obstacle on the shortest path)
    'w_max': 1.2,                 # rad/s max turn rate while driving
    'spin_enter': math.radians(60),   # heading error that triggers a turn in place
    'spin_exit': math.radians(15),    # turn in place until the error is below this
    'goal_tol': 0.06,             # m: done inside this distance of the goal
    # ---- wheels (instructor's HW2 motor model; re-check on your robot)
    'wheel_base': 0.127,          # m
    'deadzone_l': 0.09, 'deadzone_r': 0.09,   # command where the wheel starts to turn
    'slope_l': 2.5, 'slope_r': 2.5,           # (m/s) per unit command above the deadzone
    'trim_l': 1.0, 'trim_r': 1.0,             # scale one side if the robot curves when told to go straight
    'cmd_limit': 0.5,
    # turning in place: your HW1 calibration (motion.py BASE_TURN and wheel gains)
    'spin_cmd': 0.35,
    'gain_l_fwd': 0.93, 'gain_l_rev': 0.94, 'gain_r_fwd': 1.08, 'gain_r_rev': 1.0,
    'spin_rate_ccw': 1.85,        # rad/s actual rate at spin_cmd (slope of motion.CCW_TABLE)
    'spin_rate_cw': 2.20,         # rad/s (slope of motion.CW_TABLE)
    # ---- localization
    'cam_offset': 0.08,           # m camera ahead of the robot centre
    'start_sigma_xy': 0.03,       # m how well the robot is placed at the start
    'start_sigma_th': math.radians(3),
    'landmark_sigma': 0.02,       # m how well the tag positions are measured
    'sigma_ds_per_m': 0.05,       # process noise (same meaning as localizer_node.py)
    'sigma_th_per_rad': 0.3,
    'sigma_th_per_m': 0.25,       # no gyro: heading drifts while driving arcs
    'min_sigma_r': 0.03, 'min_sigma_b': math.radians(1.0),
    'meas_scale': 2.0,            # inflate the detector sigmas
    'max_range': 3.0,             # m ignore farther tags
    'max_obs_age': 0.4,           # s ignore older images
    'gate': 9.21,                 # Mahalanobis gate (99 %, 2 dof)
    # ---- start / re-localization (no gyro: if the pose gets too uncertain, stop and look at a tag)
    'start_wait': 1.0,            # s standing still at the start to get a first fix
    'reloc_sigma_xy': 0.08,       # m   re-localize when the position sigma exceeds this ...
    'reloc_sigma_th': math.radians(10),   # ... or the heading sigma exceeds this
    'reloc_wait': 0.8,            # s standing still facing a tag
    'reloc_aim_tol': math.radians(8),
    'reloc_range': (0.4, 2.5),    # m usable tag distance
    'reloc_view': math.radians(60),   # tag must be seen within this angle of its normal
    'reloc_max_tries': 4,
    'reloc_cooldown': 0.4,        # m to drive after a re-localization before the next one
    'reloc_min_remaining': 0.35,  # m: this close to the goal, just finish
}


def make_config(overrides=None):
    cfg = dict(DEFAULTS)
    for k, v in (overrides or {}).items():
        if k not in cfg:
            raise KeyError(f'unknown setting {k}')
        cfg[k] = tuple(v) if isinstance(cfg[k], tuple) else type(cfg[k])(v)
    return cfg


# ===================================================================== wheels
class WheelModel:
    def __init__(self, cfg):
        self.c = cfg

    def _wheel(self, vel, dz, slope, trim):
        if abs(vel) < 1e-4:
            return 0.0
        return math.copysign(dz + abs(vel) / slope, vel) * trim

    def drive(self, v, w):
        """Body velocity (m/s, rad/s) -> [L, R] motor commands."""
        c = self.c
        half = 0.5 * c['wheel_base']
        L = self._wheel(v - half * w, c['deadzone_l'], c['slope_l'], c['trim_l'])
        R = self._wheel(v + half * w, c['deadzone_r'], c['slope_r'], c['trim_r'])
        lim = c['cmd_limit']
        return float(np.clip(L, -lim, lim)), float(np.clip(R, -lim, lim))

    def spin(self, sign):
        """Turn in place: [L, R] and the rate (rad/s, signed) used for dead reckoning."""
        c = self.c
        s = c['spin_cmd']
        if sign > 0:                                       # CCW (left)
            return (-s * c['gain_l_rev'], s * c['gain_r_fwd']), c['spin_rate_ccw']
        return (s * c['gain_l_fwd'], -s * c['gain_r_rev']), -c['spin_rate_cw']


# ===================================================================== localization
class Localizer:
    """EKF localization in the WORLD frame with the known tag map."""

    def __init__(self, start_pose, landmarks, cfg):
        self.c = cfg
        self.ekf = SlamEKF(cfg['cam_offset'], P0_robot=(cfg['start_sigma_xy'] ** 2,
                                                        cfg['start_sigma_xy'] ** 2,
                                                        cfg['start_sigma_th'] ** 2))
        self.ekf.x[:3] = start_pose
        for lid, (x, y) in landmarks.items():
            self._add_known(lid, x, y, cfg['landmark_sigma'] ** 2)
        self.n_updates = 0
        self.n_rejected = 0
        self.rejected_in_row = 0

    def _add_known(self, lid, x, y, var):
        e = self.ekf
        n = len(e.x)
        P = np.zeros((n + 2, n + 2))
        P[:n, :n] = e.P
        P[n, n] = P[n + 1, n + 1] = var
        e.x = np.concatenate([e.x, [x, y]])
        e.P = P
        e.ids.append(lid)

    @property
    def pose(self):
        return np.array(self.ekf.x[:3])

    @property
    def sigma(self):
        return np.sqrt(np.maximum(np.diag(self.ekf.P)[:3], 0.0))

    def predict(self, v, w, dt):
        c = self.c
        ds, dth = v * dt, w * dt
        var_ds = c['sigma_ds_per_m'] ** 2 * abs(ds) + 1e-9
        var_dth = c['sigma_th_per_rad'] ** 2 * abs(dth) + c['sigma_th_per_m'] ** 2 * abs(ds) + 1e-9
        self.ekf.predict(ds, dth, var_ds, var_dth)

    def correct(self, obs, age, v, w):
        """obs: detector dicts {id, r, b, sr, sb}; age: s since the image; v, w: current motion.
        Returns the number of accepted tags."""
        c = self.c
        if age > c['max_obs_age']:
            return 0
        used = 0
        for o in obs:
            lid = o['id']
            if lid not in self.ekf.ids or o['r'] > c['max_range']:    # unknown tag: never add it
                continue
            # the robot moved since the image was taken: that error grows with speed x age
            sr = math.hypot(max(o['sr'], c['min_sigma_r']) * c['meas_scale'], abs(v) * age)
            sb = math.hypot(max(o['sb'], c['min_sigma_b']) * c['meas_scale'], abs(w) * age)
            result, _ = self.ekf.update(lid, [o['r'], o['b']], np.diag([sr ** 2, sb ** 2]), gate=c['gate'])
            if result == 'ok':
                used += 1
                self.n_updates += 1
                self.rejected_in_row = 0
            else:
                self.n_rejected += 1
                self.rejected_in_row += 1
                if self.rejected_in_row >= 10:                # probably lost: open the robot covariance
                    self.ekf.inflate_robot(0.10 ** 2, math.radians(10) ** 2)
                    self.rejected_in_row = 0
        return used


# ===================================================================== path following
class PathFollower:
    def __init__(self, path, cfg):
        self.c = cfg
        self.path = np.array(path, dtype=float)
        seg = np.diff(self.path, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.s_at = np.concatenate([[0.0], np.cumsum(self.seg_len)])   # arc length at each vertex
        self.total = float(self.s_at[-1])
        self.s = 0.0                                       # progress along the path (never goes back)
        self.v = 0.0                                       # speed actually commanded last tick
        self.spinning = False
        self.done = False
        self.goal = self.path[-1]

    def point_at(self, s):
        s = min(max(s, 0.0), self.total)
        i = int(np.searchsorted(self.s_at, s, side='right') - 1)
        i = min(i, len(self.seg_len) - 1)
        t = (s - self.s_at[i]) / max(self.seg_len[i], 1e-9)
        return self.path[i] + t * (self.path[i + 1] - self.path[i])

    def project(self, p, window=0.6):
        """Closest point on the path between s and s + window (forward only)."""
        best, best_s = math.inf, self.s
        for i in range(len(self.seg_len)):
            if self.s_at[i + 1] < self.s - 1e-9 or self.s_at[i] > self.s + window:
                continue
            a, b = self.path[i], self.path[i + 1]
            t = float(np.clip(np.dot(p - a, b - a) / max(self.seg_len[i] ** 2, 1e-12), 0.0, 1.0))
            s = self.s_at[i] + t * self.seg_len[i]
            if s < self.s:
                continue
            d = float(np.linalg.norm(p - (a + t * (b - a))))
            if d < best:
                best, best_s = d, s
        self.s = best_s
        return best                                         # cross-track distance

    def step(self, pose, dt):
        """Returns ('drive', v, w), ('spin', sign) or ('stop',)."""
        c = self.c
        p, th = np.asarray(pose[:2]), pose[2]
        d_goal = float(np.linalg.norm(self.goal - p))
        if self.done or d_goal < c['goal_tol'] or (self.s > self.total - 0.02 and d_goal < 2 * c['goal_tol']):
            self.done, self.v = True, 0.0
            return ('stop',)
        self.project(p)
        remaining = self.total - self.s
        nxt = self.s_at[self.s_at > self.s + 1e-6]           # arc length of the next corner(s)
        ahead = c['lookahead']
        if len(nxt) > 1:                                    # a corner (not the goal) is coming
            ahead = min(ahead, max(nxt[0] - self.s, 0.0) + c['corner_lookahead'])
        ahead = max(ahead, 0.08)
        target = self.goal if remaining < ahead else self.point_at(self.s + ahead)
        to_t = target - p
        alpha = wrap(math.atan2(to_t[1], to_t[0]) - th)

        # ---- large heading error: turn in place (start, or after a big correction)
        if self.spinning and abs(alpha) < c['spin_exit']:
            self.spinning = False
        if not self.spinning and abs(alpha) > c['spin_enter']:
            self.spinning = True
        if self.spinning:
            self.v = 0.0
            return ('spin', 1 if alpha > 0 else -1)

        # ---- pure pursuit
        Ld = max(float(np.linalg.norm(to_t)), 0.05)
        kappa = 2.0 * math.sin(alpha) / Ld                 # curvature of the arc through the target
        v_des = c['v_max'] * max(0.3, math.cos(alpha))
        if abs(kappa) > 1e-6:
            v_des = min(v_des, c['w_max'] / abs(kappa))     # sharp arcs: slow down
        v_stop = math.sqrt(2.0 * c['decel'] * max(d_goal - 0.5 * c['goal_tol'], 0.0))
        v_des = min(v_des, v_stop)                          # stopping profile at the goal
        v_des = max(v_des, min(c['v_min'], v_stop))
        v = min(v_des, self.v + c['accel'] * dt)            # ramp up; slow down at once
        self.v = v
        w = float(np.clip(v * kappa, -c['w_max'], c['w_max']))
        return ('drive', v, w)


# ===================================================================== everything together
class Navigator:
    """One object per run: call on_obs() for every detector message and step(dt) at `rate` Hz;
    step() returns the wheel commands [L, R]. States:

        settle   standing still at the start to get a first fix
        track    pure pursuit along the path
        aim      (re-localization) turning in place toward a known tag
        look     (re-localization) standing still, facing that tag
        done     at the goal
    """

    def __init__(self, path, start_pose, landmarks, cfg):
        """landmarks: {id: (x, y, facing_rad)} in the world frame."""
        self.c = cfg
        self.tags = landmarks
        self.loc = Localizer(start_pose, {k: v[:2] for k, v in landmarks.items()}, cfg)
        self.fol = PathFollower(path, cfg)
        self.wheels = WheelModel(cfg)
        self.state, self.t_state = 'settle', 0.0
        self.v, self.w = 0.0, 0.0                       # motion assumed this tick (dead reckoning)
        self.tried = set()
        self.aim_at = None
        self.n_reloc = 0
        self.driven = math.inf                          # m driven since the last re-localization

    # ---------------------------------------------------------------- inputs
    def on_obs(self, obs, age):
        return self.loc.correct(obs, age, self.v, self.w)

    # ---------------------------------------------------------------- helpers
    def _uncertain(self):
        s = self.loc.sigma
        return max(s[0], s[1]) > self.c['reloc_sigma_xy'] or s[2] > self.c['reloc_sigma_th']

    def _pick_tag(self):
        """Known tag at a usable range, facing the robot, needing the smallest turn."""
        x, y, th = self.loc.pose
        best = None
        for lid, (tx, ty, facing) in self.tags.items():
            if lid in self.tried:
                continue
            r = math.hypot(tx - x, ty - y)
            if not self.c['reloc_range'][0] < r < self.c['reloc_range'][1]:
                continue
            if abs(wrap(math.atan2(y - ty, x - tx) - facing)) > self.c['reloc_view']:
                continue
            b = wrap(math.atan2(ty - y, tx - x) - th)
            if best is None or abs(b) < abs(best[0]):
                best = (b, lid)
        return best

    def _set(self, state):
        self.state, self.t_state = state, 0.0

    def _start_reloc(self):
        pick = self._pick_tag()
        if pick is None or len(self.tried) >= self.c['reloc_max_tries']:
            self.tried.clear()
            self.driven = 0.0
            self._set('track')                          # nothing to look at: keep driving
            return
        b, lid = pick
        self.tried.add(lid)
        self.aim_at = wrap(self.loc.pose[2] + b)
        self.fol.v = 0.0
        self.n_reloc += 1
        self._set('aim')

    # ---------------------------------------------------------------- main step
    def step(self, dt):
        c = self.c
        self.t_state += dt
        cmd = (0.0, 0.0)
        self.v = self.w = 0.0
        if self.state == 'settle':
            if self.t_state >= c['start_wait']:
                self._set('track')
        elif self.state == 'aim':
            err = wrap(self.aim_at - self.loc.pose[2])
            if abs(err) < c['reloc_aim_tol']:
                self._set('look')
            else:
                cmd, self.w = self.wheels.spin(1 if err > 0 else -1)
        elif self.state == 'look':
            if not self._uncertain() and self.t_state > 0.3:
                self.tried.clear()
                self.driven = 0.0
                self._set('track')
            elif self.t_state >= c['reloc_wait']:
                self._start_reloc()                     # that tag did not help: try another one
        if self.state == 'track':
            may_reloc = (self.driven > c['reloc_cooldown']
                         and self.fol.total - self.fol.s > c['reloc_min_remaining'])
            if may_reloc and self._uncertain():
                self._start_reloc()
            else:
                out = self.fol.step(self.loc.pose, dt)
                if out[0] == 'stop':
                    self._set('done')
                elif out[0] == 'spin':
                    cmd, self.w = self.wheels.spin(out[1])
                else:
                    _, self.v, self.w = out
                    cmd = self.wheels.drive(self.v, self.w)
        self.loc.predict(self.v, self.w, dt)
        self.driven += abs(self.v) * dt
        return cmd

    @property
    def done(self):
        return self.state == 'done'
