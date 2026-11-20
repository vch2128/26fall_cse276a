#!/usr/bin/env python3
"""Laptop simulator for HW4: runs the REAL localizer_node.py (+ ekf.py) and hw4_control.py
(+ planner.py) against a simulated robot and camera, like sim_slam.py did for HW3. No ROS needed:
rclpy / messages are small stubs and the clock is simulated.

Simulated robot: it moves like your calibration tables say (DRIVE_TABLE / CCW_TABLE / CW_TABLE in
robot_control/motion.py), but each run gets its own errors: drive distance scale, turn scales and
a slight curve while driving straight. Simulated camera: fps, latency, field of view, range, tags
only seen from the front, the box hides tags behind it, range / bearing noise. The numbers in TRUTH
are GUESSES: replace them with what you measure.

    python3 sim_hw4.py short                      # one run, saves sim_hw4_short.png
    python3 sim_hw4.py safe --seed 4
    python3 sim_hw4.py both --runs 20             # statistics only
    python3 sim_hw4.py short --set realign_tol_deg=20        # hw4_control setting
    python3 sim_hw4.py short --loc meas_scale=2.0            # localizer parameter
    python3 sim_hw4.py short --truth fps=4 --truth latency=0.3
"""
import argparse
import json
import math
import os
import sys
import tempfile
import types

import numpy as np

VERBOSE = False                                            # -v: print hw4_control's console output

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, os.path.join(REPO, 'landmark_nav'), os.path.join(REPO, 'robot_control')]

TRUTH = {
    'drive_err': 0.03,          # std of the true / table distance ratio (per run)
    'turn_err': 0.06,           # std of the true / table angle ratio, per direction (per run)
    'curve': 0.06,              # std of the heading drift while driving straight (rad per m, per run)
    'fps': 8.0,                 # detector output rate
    'latency': 0.15,            # s photo -> landmark_obs
    'cam_offset': 0.08,         # m camera ahead of the robot centre
    'half_fov': math.radians(31),
    'min_range': 0.25, 'max_range': 3.0,
    'max_view': math.radians(65),   # tag seen only within this angle of its normal
    'sr_per_m': 0.02, 'sb': math.radians(1.0),
    'start_err_xy': 0.03, 'start_err_th': math.radians(3),
}
LOCALIZER = {                   # what hw4_launch.py gives the localizer
    'accept_moving': True,
    'sigma_th_per_rad': 0.15,
    'cam_offset': 0.08,
}


# ===================================================================== ROS stubs + simulated clock
class Clock:
    t = 0.0


def install_stubs(params, quiet=True):
    class Value:
        def __init__(self, v):
            self.value = v

    class Time:
        def __init__(self):
            self.nanoseconds = int(Clock.t * 1e9)

        def to_msg(self):
            return None

    class Logger:
        def info(self, m, **k):
            if not quiet:
                print('[info]', m)

        def warn(self, m, **k):
            if not quiet:
                print('[warn]', m)

    class Pub:
        def __init__(self):
            self.last = None

        def publish(self, m):
            self.last = m

    class Node:
        def __init__(self, name):
            self.subs, self.timers, self.pubs = {}, [], {}

        def declare_parameter(self, k, v):
            return Value(params.get(k, v))

        def create_subscription(self, typ, topic, cb, q):
            self.subs[topic] = cb

        def create_publisher(self, typ, topic, q):
            self.pubs[topic] = Pub()
            return self.pubs[topic]

        def create_timer(self, period, cb):
            self.timers.append((period, cb))

        def get_logger(self):
            return Logger()

        def get_clock(self):
            return types.SimpleNamespace(now=Time)

        def destroy_node(self):
            pass

    def msg_class(*fields):
        class M:
            def __init__(self, **k):
                for f in fields:
                    setattr(self, f, None)
                self.__dict__.update(k)
        return M

    class Pose:
        def __init__(self):
            ns = types.SimpleNamespace
            self.header = ns(stamp=None, frame_id='')
            self.pose = ns(pose=ns(position=ns(x=0.0, y=0.0, z=0.0),
                                   orientation=ns(x=0.0, y=0.0, z=0.0, w=1.0)), covariance=None)

    mods = {
        'rclpy': types.SimpleNamespace(init=lambda *a: None, ok=lambda: True, shutdown=lambda: None),
        'rclpy.node': types.SimpleNamespace(Node=Node),
        'geometry_msgs': types.ModuleType('geometry_msgs'),
        'geometry_msgs.msg': types.SimpleNamespace(PoseWithCovarianceStamped=Pose),
        'std_msgs': types.ModuleType('std_msgs'),
        'std_msgs.msg': types.SimpleNamespace(Float32MultiArray=msg_class('data'), Int32=msg_class('data'),
                                              String=msg_class('data')),
    }
    for k, v in mods.items():
        sys.modules[k] = v
    try:
        import spatialmath  # noqa: F401
    except ImportError:
        class UQ:
            def __init__(self, th):
                self.s, self.v = math.cos(th / 2), [0.0, 0.0, math.sin(th / 2)]
        sys.modules['spatialmath'] = types.SimpleNamespace(UnitQuaternion=types.SimpleNamespace(Rz=UQ))


# ===================================================================== simulation
def visible(world, tags, cam, th, T, rng):
    import planner
    obs = []
    for lid, (tx, ty, facing) in tags.items():
        d = np.array([tx, ty]) - cam
        r = float(np.linalg.norm(d))
        if not T['min_range'] < r < T['max_range']:
            continue
        b = math.atan2(math.sin(math.atan2(d[1], d[0]) - th), math.cos(math.atan2(d[1], d[0]) - th))
        if abs(b) > T['half_fov']:
            continue
        view = math.atan2(-d[1], -d[0]) - facing
        if abs(math.atan2(math.sin(view), math.cos(view))) > T['max_view']:
            continue
        if any(planner.segment_enters(cam, np.array([tx, ty]), poly) for poly in world.obstacles):
            continue
        sr, sb = T['sr_per_m'] * r, T['sb']
        obs.append({'id': lid, 'r': round(r + rng.normal(0, sr), 4), 'b': round(b + rng.normal(0, sb), 5),
                    'sr': round(sr, 4), 'sb': round(sb, 5), 'conf': 50.0})
    return obs


def run(mode, map_file, ctl_set, loc_set, T, seed, t_max=180.0):
    import hw4_control
    import localizer_node
    import planner
    import robot_control.motion as mo

    rng = np.random.default_rng(seed)
    tmp = tempfile.mkdtemp()
    localizer_node.time = types.SimpleNamespace(monotonic=lambda: Clock.t)   # simulated clock
    Clock.t = 0.0
    params = dict(LOCALIZER, map_file=map_file, log_file=os.path.join(tmp, 'traj.csv'),
                  map_log_file=os.path.join(tmp, 'map.csv'))
    params.update(loc_set)
    sys.modules['rclpy.node'].Node.declare_parameter = \
        lambda self, k, v: types.SimpleNamespace(value=params.get(k, v))
    loc = localizer_node.Localizer()
    world, path, tags, met = hw4_control.load_map(map_file, mode)
    cfg = dict(hw4_control.DEFAULTS, **ctl_set)
    ctl = hw4_control.PivotController(path, tags, cfg, mo,
                                      say=(lambda m: print(f'[{Clock.t:6.2f}] {m}')) if VERBOSE else None)
    String = sys.modules['std_msgs.msg'].String
    F32 = sys.modules['std_msgs.msg'].Float32MultiArray

    def est():
        m = loc.pub.last
        q, cov = m.pose.pose.orientation, m.pose.covariance
        return ((m.pose.pose.position.x, m.pose.pose.position.y, 2 * math.atan2(q.z, q.w)),
                tuple(math.sqrt(max(0.0, cov[i])) for i in (0, 7, 35)))

    # true robot: moves like the tables say, with this run's errors
    true = np.array([world.start[0], world.start[1], world.start_th])
    true += [rng.normal(0, T['start_err_xy']), rng.normal(0, T['start_err_xy']), rng.normal(0, T['start_err_th'])]
    k_drive = 1 + rng.normal(0, T['drive_err'])
    k_turn = {1: 1 + rng.normal(0, T['turn_err']), -1: 1 + rng.normal(0, T['turn_err'])}
    curve = rng.normal(0, T['curve'])
    seg, seg_done = None, 0.0                              # (kind, sign, start), amount done

    dt, ctl_dt = 0.01, 1.0 / cfg['rate']
    next_ctl = next_frame = 0.0
    pending, log = [], []
    loc.on_timer()
    L = R = 0.0
    while Clock.t < t_max and not ctl.done:
        Clock.t = round(Clock.t + dt, 6)
        if Clock.t >= next_ctl - 1e-9:                     # 20 Hz: localizer predict, then controller
            next_ctl += ctl_dt
            loc.on_timer()
            ctl.n_updates = loc.n_updates                  # = the landmark_updates topic (hw4_control 'verify')
            pose, sigma = est()
            L, R = ctl.tick(Clock.t, pose, sigma)
            loc.on_cmd(F32(data=[L, R]))
            kind = localizer_node.classify(L, R)
            if (kind and (kind[0], kind[1])) != (seg and (seg[0], seg[1])):
                seg, seg_done = ((kind[0], kind[1], Clock.t) if kind else None), 0.0
            log.append((Clock.t, *true, *pose, *sigma, L, R, ctl.state))
        if seg:                                            # true motion this physics step
            kind, sign, t0 = seg
            if kind == 'drive':
                total = localizer_node.table_inverse(Clock.t - t0, mo.DRIVE_TABLE) * k_drive
                ds, seg_done = total - seg_done, total
                true[0] += ds * math.cos(true[2])
                true[1] += ds * math.sin(true[2])
                true[2] += curve * ds
            else:
                table = mo.CCW_TABLE if sign > 0 else mo.CW_TABLE
                total = localizer_node.table_inverse(Clock.t - t0, table) * k_turn[sign]
                da, seg_done = total - seg_done, total
                true[2] += sign * math.radians(da)
        if Clock.t >= next_frame - 1e-9:                   # camera
            next_frame += 1.0 / T['fps']
            cam = true[:2] + T['cam_offset'] * np.array([math.cos(true[2]), math.sin(true[2])])
            pending.append((Clock.t + T['latency'], Clock.t, visible(world, tags, cam, true[2], T, rng)))
        while pending and pending[0][0] <= Clock.t:
            _, stamp, obs = pending.pop(0)
            loc.on_obs(String(data=json.dumps({'stamp': stamp, 'obs': obs})))
    loc.destroy_node()

    arr = np.array([r[:10] for r in log], dtype=float)
    states = [r[12] for r in log]
    true_xy, est_xy = arr[:, 1:3], arr[:, 4:6]
    pts = np.array(path)
    cross = [min(planner.point_segment_dist(q, a, b) for a, b in zip(pts, pts[1:])) for q in true_xy]
    moving = np.array([s in ('drive',) for s in states])
    return {'log': arr, 'LR': np.array([r[10:12] for r in log], dtype=float), 'states': states,
            'path': pts, 'world': world, 'tags': tags, 'time': Clock.t, 'reached': ctl.done,
            'goal_err': float(np.linalg.norm(true_xy[-1] - pts[-1])),
            'max_cross': float(max(cross)),
            'est_err': float(np.max(np.linalg.norm(true_xy - est_xy, axis=1))),
            'gap': float(min(world.obstacle_clearance(q) for q in true_xy) - world.radius),
            'not_driving_s': float((~moving).sum() * ctl_dt), 'reloc': ctl.n_reloc,
            'updates': loc.n_updates}


def plot(res, mode, fn):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import planner
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1.1, 1]})
    ax = axes[0]
    planner.draw_world(ax, res['world'], f'{mode}: planned vs simulated run')
    log = res['log']
    ax.plot(res['path'][:, 0], res['path'][:, 1], '--', color='gray', lw=1.5, label='planned path')
    ax.plot(log[:, 1], log[:, 2], color='#1f6fb4', lw=2, label='true')
    ax.plot(log[:, 4], log[:, 5], color='#d0521b', lw=1, label='estimated (localizer)')
    ax.legend(loc='lower left', fontsize=8)
    ax = axes[1]
    t = log[:, 0]
    ax.plot(t, res['LR'][:, 0], label='L command')
    ax.plot(t, res['LR'][:, 1], label='R command')
    ax.fill_between(t, 0, 1, where=np.array([s != 'drive' for s in res['states']]), color='gray', alpha=0.15,
                    transform=ax.get_xaxis_transform(), label='not driving (turn / look / start)')
    ax.set_xlabel('t (s)')
    ax.set_title(f'commands  ({res["time"]:.1f} s, goal error {res["goal_err"] * 100:.1f} cm)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fn, dpi=130)
    plt.close(fig)


def kv(items):
    out = {}
    for s in items or []:
        k, v = s.split('=', 1)
        try:
            out[k] = eval(v, {'math': math})
        except Exception:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['short', 'safe', 'both'])
    ap.add_argument('--map', default=os.path.join(HERE, '..', 'config', 'map_hw4.yaml'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--runs', type=int, default=1)
    ap.add_argument('--set', action='append', help='hw4_control setting, e.g. realign_tol_deg=20')
    ap.add_argument('--loc', action='append', help='localizer parameter, e.g. meas_scale=2.0')
    ap.add_argument('--truth', action='append', help='simulated world, e.g. fps=5')
    ap.add_argument('-v', '--verbose', action='store_true', help="print hw4_control's console output")
    args = ap.parse_args()
    global VERBOSE
    VERBOSE = args.verbose
    install_stubs({})
    T = dict(TRUTH, **kv(args.truth))
    map_file = os.path.abspath(args.map)
    keys = [('time', 1, 's'), ('goal_err', 100, 'cm'), ('max_cross', 100, 'cm'), ('est_err', 100, 'cm'),
            ('gap', 100, 'cm'), ('not_driving_s', 1, 's'), ('reloc', 1, '')]
    for mode in (['short', 'safe'] if args.mode == 'both' else [args.mode]):
        rs = [run(mode, map_file, kv(args.set), kv(args.loc), T, args.seed + i) for i in range(args.runs)]
        if args.runs == 1:
            r = rs[0]
            print(f'{mode}: {"reached" if r["reached"] else "NOT reached"} in {r["time"]:.1f} s, goal error '
                  f'{r["goal_err"] * 100:.1f} cm, max off-path {r["max_cross"] * 100:.1f} cm, max localizer '
                  f'error {r["est_err"] * 100:.1f} cm, min body-box gap {r["gap"] * 100:.1f} cm, not driving '
                  f'{r["not_driving_s"]:.1f} s, re-localizations {r["reloc"]}, tag updates {r["updates"]}')
            fn = os.path.join(os.getcwd(), f'sim_hw4_{mode}.png')
            plot(r, mode, fn)
            print(f'  saved {fn}')
        else:
            print(f'{mode}: reached {sum(r["reached"] for r in rs)}/{len(rs)}   (mean / worst)')
            for k, scale, unit in keys:
                a = np.array([r[k] for r in rs], dtype=float)
                worst = a.min() if k == 'gap' else a.max()
                print(f'  {k:14s} {a.mean() * scale:7.1f} / {worst * scale:7.1f} {unit}')


if __name__ == '__main__':
    main()
