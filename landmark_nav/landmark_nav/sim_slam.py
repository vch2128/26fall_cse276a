#!/usr/bin/env python3
"""Laptop simulator for HW3 EKF-SLAM: runs the REAL closed_loop_control.py and ekf.py (unchanged)
against a simulated 10 ft x 10 ft arena with AprilTags on the walls. No ROS needed.

How it works
    - ROS / robot_control are replaced by small stubs, so closed_loop_control.main() runs as is.
      Its motion.drive / turn / stop calls go to SimMotion below instead of the motors.
    - SimMotion moves a TRUE robot with noisy motion, feeds the EKF the COMMANDED motion
      (what the localizer gets from the timing tables), and while standing still simulates
      detector frames (FOV, range, viewing angle, noise) and applies them the way
      localizer_node.on_obs does (max range, minimum sigmas, gate, inflate after 10 rejections).
    - The estimated map (map frame = start pose) is transformed into the world frame with the
      TRUE start pose, like you will do with the measured start pose, and compared with the
      true landmarks.

Usage (from this folder):
    python3 sim_slam.py square 1.2 2                   # one run, prints errors, saves sim_square_1.2.png
    python3 sim_slam.py octagon 0.5 2 --seed 3 --show
    python3 sim_slam.py square 1.2 2 --set NEAR=0.05 --set "AIM_TOL=math.radians(5)"
    python3 sim_slam.py sweep square 0.6,0.9,1.2,1.5,1.8 --laps 3 --runs 20
    python3 sim_slam.py sweep octagon 0.3,0.4,0.5,0.6 --laps 3 --runs 20
    python3 sim_slam.py square 1.2 2 --landmarks my_landmarks.csv   # id,x,y,facing_deg  (world frame, m)

--set changes a controller constant (closed_loop_control.py); --truth / --filter / --cam change
the simulated world (see the dicts below; e.g. --truth drive_scale=0.05). Everything printed
by the controller is shown for single runs and hidden in sweeps.

The truth noise below is a GUESS until you measure it on the robot (drive 1 m and turn 90 deg a
few times; read r, b of a tag at a known spot). Put those numbers in TRUTH before trusting a sweep.
"""
import argparse
import contextlib
import csv
import importlib
import io
import math
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FT = 0.3048


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ============================================================== world / noise parameters
ARENA = 10 * FT                       # m, square arena side; origin at a corner, walls along x and y

# TRUE robot and sensor behaviour (what really happens; GUESSES until measured)
TRUTH = dict(
    drive_scale=0.0,                  # systematic: robot drives (1 + this) x the commanded distance
    turn_scale=0.0,                   # systematic: robot turns (1 + this) x the commanded angle
    drive_curve=0.0,                  # systematic: rad of heading drift per m driven (one wheel faster)
    sigma_ds_per_m=0.03,              # random: std of distance error after 1 m (variance ~ distance)
    sigma_th_per_rad=0.15,            # random: std of heading error per rad turned
    sigma_th_per_m=0.05,              # random: std of heading drift per m driven
    min_turn_deg=2.0,                 # robot_control.motion: turns below this are skipped
    sigma_r_per_m=0.02,               # detector range noise std, x range
    sigma_b=math.radians(1.0),        # detector bearing noise std
    range_scale=0.0,                  # systematic: r_measured = (1 + this) r  (wrong tag_size)
    frame_corr=0.9,                   # share of the measurement error that stays the same for every
                                      # frame of one stop (static scene: frames are NOT independent)
    p_detect=0.9,                     # chance a visible tag is detected in one frame
)
# what the FILTER assumes (copied from localizer_node.py / detector_node.py defaults)
FILTER = dict(
    sigma_ds_per_m=0.03, sigma_th_per_rad=0.3, sigma_th_per_m=0.18,
    det_sigma_r_per_m=0.02, det_sigma_b=math.radians(1.0),
    min_sigma_r=0.03, min_sigma_b=math.radians(1.0),
    max_range=4.0, still_margin=0.3, gate=9.21,
    fuse_frames=3,                    # one update per tag per stop from the mean of its first N frames
                                      # (0 = every frame is its own update, the old localizer)
    meas_scale=3.0,                   # multiplies the measurement sigmas the filter assumes
)
# camera / detector
CAM = dict(
    offset=0.08,                      # m, camera ahead of the robot centre (localizer cam_offset)
    half_fov=math.radians(31),        # rad, half the usable horizontal FOV (after the border_px cut)
    min_range=0.25,                   # m, closer -> tag cut off at the top/bottom of the image
    max_range=3.5,                    # m, farther -> tag too small to decode
    max_view=math.radians(65),        # rad, tag seen this far off its normal at most
    rate=5.0,                         # Hz, detector frames processed while standing still
    robot_radius=0.12,                # m, for the wall-collision check
)
SPEED = dict(drive=0.15, turn=math.radians(90))   # m/s, rad/s: only for the run-time estimate


def default_landmarks():
    """Two tags per wall at 1/3 and 2/3 of the side, facing into the arena: id -> (x, y, facing)."""
    a, s = ARENA, ARENA / 3
    return {'tag0': (s, 0, math.pi / 2), 'tag1': (2 * s, 0, math.pi / 2),
            'tag2': (a, s, math.pi), 'tag3': (a, 2 * s, math.pi),
            'tag4': (2 * s, a, -math.pi / 2), 'tag5': (s, a, -math.pi / 2),
            'tag6': (0, 2 * s, 0.0), 'tag7': (0, s, 0.0)}


def load_landmarks(path):
    out = {}
    for row in csv.reader(open(path)):
        if row and not row[0].strip().startswith('#'):
            lid, x, y, f = row[:4]
            out[lid.strip()] = (float(x), float(y), math.radians(float(f)))
    return out


# ============================================================== ROS / robot_control stubs
SIM = None                            # the SimMotion of the current run


def install_stubs():
    def mod(name, **kw):
        m = types.ModuleType(name)
        m.__dict__.update(kw)
        sys.modules[name] = m
        return m

    class _Node:
        def create_subscription(self, *a, **k): pass
        def destroy_node(self): pass
        def get_logger(self): return types.SimpleNamespace(info=lambda *a, **k: None,
                                                          warn=lambda *a, **k: None)
    mod('rclpy', init=lambda *a, **k: None, ok=lambda: True, shutdown=lambda: None,
        create_node=lambda *a, **k: _Node())
    mod('ament_index_python')
    mod('ament_index_python.packages', get_package_share_directory=lambda p: HERE)
    mod('geometry_msgs')
    mod('geometry_msgs.msg', PoseWithCovarianceStamped=object)
    mod('std_msgs')
    mod('std_msgs.msg', Int32=object, String=object)
    mod('robot_control')
    sys.modules['robot_control'].motion = mod(
        'robot_control.motion', Motion=lambda node: SIM, FOLDER=HERE,
        angle_diff=lambda a, b: wrap(a - b))


def load_controller():
    install_stubs()
    sys.path.insert(0, HERE)
    ccl = importlib.import_module('closed_loop_control')
    ekf = importlib.import_module('ekf')
    return ccl, ekf


# ============================================================== simulated robot + localizer
class SimMotion:
    """Stands in for robot_control.motion.Motion AND the localizer node."""

    def __init__(self, ccl, ekf_mod, start, landmarks, rng):
        self.ccl, self.rng, self.lms = ccl, rng, landmarks
        self.true = np.array(start, dtype=float)          # world frame
        self.ekf = ekf_mod.SlamEKF(CAM['offset'])
        self.n_updates, self.rej_in_row = 0, 0
        self.t = 0.0                                      # simulated seconds
        self.true_path, self.est_path = [self.true.copy()], [self.ekf.x[:3].copy()]
        self.stops = 0
        self.collided = False
        self.publish()

    # ---- what closed_loop_control reads
    def publish(self):
        P = self.ekf.P
        x, y, th = self.ekf.x[:3]
        sd = np.sqrt(np.maximum(np.diag(P)[:3], 0.0))
        self.ccl.pose = (float(x), float(y), float(th), sd[0], sd[1], sd[2])
        self.ccl.n_updates = self.n_updates

    def spin_for(self, s):
        self.publish()

    # ---- motion
    def predict(self, ds, dth):
        F = FILTER
        var_ds = F['sigma_ds_per_m'] ** 2 * abs(ds) + 1e-8
        var_dth = F['sigma_th_per_rad'] ** 2 * abs(dth) + F['sigma_th_per_m'] ** 2 * abs(ds) + 1e-8
        self.ekf.predict(ds, dth, var_ds, var_dth)

    def drive(self, d):
        T, n = TRUTH, max(1, int(math.ceil(abs(d) / 0.01)))           # 1 cm pieces (localizer: 20 Hz)
        scale = 1 + T['drive_scale'] + self.rng.normal(0, T['sigma_ds_per_m'] / math.sqrt(max(abs(d), 1e-6)))
        drift = T['drive_curve'] * abs(d) + self.rng.normal(0, T['sigma_th_per_m'] * math.sqrt(abs(d)))
        for _ in range(n):
            x, y, th = self.true
            step = d / n * scale
            th_mid = th + drift / n / 2
            self.true = np.array([x + step * math.cos(th_mid), y + step * math.sin(th_mid),
                                  wrap(th + drift / n)])
            self.predict(d / n, 0.0)
            self.check_walls()
        self.t += abs(d) / SPEED['drive']
        self.record()

    def turn(self, a):
        T = TRUTH
        if abs(math.degrees(a)) < T['min_turn_deg']:
            return
        true_a = a * (1 + T['turn_scale']) + self.rng.normal(0, T['sigma_th_per_rad'] * math.sqrt(abs(a)))
        n = max(1, int(math.ceil(abs(a) / math.radians(5))))
        for _ in range(n):
            self.true[2] = wrap(self.true[2] + true_a / n)
            self.predict(0.0, a / n)
        self.t += abs(a) / SPEED['turn']
        self.record()

    def stop(self, seconds):
        """Stand still: detector frames arrive after still_margin; each is fed to the EKF."""
        self.t += seconds
        self.stops += 1
        frames = int(max(0.0, seconds - FILTER['still_margin']) * CAM['rate'])
        if frames:
            visible = self.visible()
            T = TRUTH
            c = T['frame_corr']
            fixed = {lid: self.rng.normal(0, 1, 2) for lid in visible}   # same for every frame of this stop
            seen = {}
            for _ in range(frames):
                for lid, (r, b) in visible.items():
                    if self.rng.random() > T['p_detect']:
                        continue
                    e = math.sqrt(c) * fixed[lid] + math.sqrt(1 - c) * self.rng.normal(0, 1, 2)
                    zr = r * (1 + T['range_scale']) + T['sigma_r_per_m'] * r * e[0]
                    zb = wrap(b + T['sigma_b'] * e[1])
                    if FILTER['fuse_frames'] <= 0:
                        self.observe(lid, zr, zb)
                    elif len(seen.setdefault(lid, [])) < FILTER['fuse_frames']:
                        seen[lid].append((zr, zb))
            for lid, zs in seen.items():             # one update per tag per stop (localizer fuse())
                zr = float(np.mean([z[0] for z in zs]))
                zb = math.atan2(np.mean([math.sin(z[1]) for z in zs]), np.mean([math.cos(z[1]) for z in zs]))
                self.observe(lid, zr, zb)
        self.publish()

    # ---- sensing
    def camera(self):
        x, y, th = self.true
        return x + CAM['offset'] * math.cos(th), y + CAM['offset'] * math.sin(th), th

    def visible(self):
        cx, cy, th = self.camera()
        out = {}
        for lid, (lx, ly, face) in self.lms.items():
            dx, dy = lx - cx, ly - cy
            r = math.hypot(dx, dy)
            b = wrap(math.atan2(dy, dx) - th)
            view = abs(wrap(math.atan2(-dy, -dx) - face))              # angle off the tag's normal
            if CAM['min_range'] < r < CAM['max_range'] and abs(b) < CAM['half_fov'] and view < CAM['max_view']:
                out[lid] = (r, b)
        return out

    def observe(self, lid, r, b):
        """Same handling as localizer_node.on_obs."""
        F = FILTER
        if r > F['max_range']:
            return
        sr = max(F['det_sigma_r_per_m'] * r, F['min_sigma_r']) * F['meas_scale']
        sb = max(F['det_sigma_b'], F['min_sigma_b']) * F['meas_scale']
        res, _ = self.ekf.update(lid, [r, b], np.diag([sr ** 2, sb ** 2]), gate=F['gate'])
        if res in ('new', 'ok'):
            self.n_updates += 1
            if res == 'ok':
                self.rej_in_row = 0
        else:
            self.rej_in_row += 1
            if self.rej_in_row >= 10:
                self.ekf.inflate_robot(0.15 ** 2, math.radians(15) ** 2)
                self.rej_in_row = 0

    def check_walls(self):
        x, y, _ = self.true
        m = CAM['robot_radius']
        if not (m < x < ARENA - m and m < y < ARENA - m):
            self.collided = True

    def record(self):
        self.true_path.append(self.true.copy())
        self.est_path.append(self.ekf.x[:3].copy())


# ============================================================== one run
def plan_start(kind, size):
    """World start pose that centres the trajectory's bounding box in the arena, heading +x."""
    sides, turn = {'square': (4, math.pi / 2), 'octagon': (8, math.pi / 4)}[kind]
    pts, x, y, th = [(0.0, 0.0)], 0.0, 0.0, 0.0
    for _ in range(sides):
        x, y = x + size * math.cos(th), y + size * math.sin(th)
        pts.append((x, y))
        th += turn
    xs, ys = zip(*pts)
    return (ARENA / 2 - (min(xs) + max(xs)) / 2, ARENA / 2 - (min(ys) + max(ys)) / 2, 0.0)


def to_world(start, x, y, C=None):
    x0, y0, t0 = start
    c, s = math.cos(t0), math.sin(t0)
    R = np.array([[c, -s], [s, c]])
    p = R @ np.array([x, y]) + [x0, y0]
    return (p, None) if C is None else (p, R @ C @ R.T)


def evaluate(sim, start, landmarks):
    """Errors of the estimated map in the world frame (via the true start pose)."""
    rows = []
    for lid, (x, y, C) in sim.ekf.landmarks().items():
        p, Cw = to_world(start, x, y, C)
        e = p - np.array(landmarks[lid][:2])
        nees = float(e @ np.linalg.solve(Cw, e)) if lid in landmarks else float('nan')
        rows.append((lid, p, Cw, float(np.linalg.norm(e)), nees))
    errs = [r[3] for r in rows]
    nees = [r[4] for r in rows]
    # shape-only error: best rigid fit of the estimated to the true map (removes start-pose error)
    fit = float('nan')
    if len(rows) >= 2:
        A = np.array([r[1] for r in rows])
        B = np.array([landmarks[r[0]][:2] for r in rows])
        Ac, Bc = A - A.mean(0), B - B.mean(0)
        U, _, Vt = np.linalg.svd(Ac.T @ Bc)
        D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
        Rf = Vt.T @ D @ U.T
        fit = float(np.mean(np.linalg.norm((Rf @ Ac.T).T - Bc, axis=1)))
    return dict(rows=rows, n=len(rows), n_true=len(landmarks),
                mean_err=float(np.mean(errs)) if errs else float('nan'),
                max_err=float(np.max(errs)) if errs else float('nan'),
                fit_err=fit,
                in95=float(np.mean([v < 5.99 for v in nees])) if nees else float('nan'),
                mean_nees=float(np.mean(nees)) if nees else float('nan'))


def run(ccl, ekf_mod, kind, size, laps, seed, landmarks, quiet=False):
    global SIM
    rng = np.random.default_rng(seed)
    start = plan_start(kind, size)
    ccl.pose, ccl.n_updates = None, None
    SIM = SimMotion(ccl, ekf_mod, start, landmarks, rng)
    sim = SIM

    per_lap = {'square': 4, 'octagon': 8}[kind]
    laps_out, count = [], [0]
    real_go_to = ccl.go_to

    def go_to(gx, gy):                                   # snapshot the map after every lap
        real_go_to(gx, gy)
        count[0] += 1
        if count[0] % per_lap == 0:
            ev = evaluate(sim, start, landmarks)
            ev['t'] = sim.t
            laps_out.append(ev)

    ccl.go_to = go_to
    argv = sys.argv
    sys.argv = ['closed_loop_control', kind, str(size), str(laps)]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
            ccl.main()
    finally:
        sys.argv, ccl.go_to = argv, real_go_to
    final = evaluate(sim, start, landmarks)
    x, y, th = sim.true
    ex, _ = to_world(start, *sim.ekf.x[:2])
    final.update(t=sim.t, stops=sim.stops, collided=sim.collided, laps=laps_out,
                 pose_err=float(np.hypot(*(ex - [x, y]))),
                 heading_err=math.degrees(abs(wrap(sim.ekf.x[2] + start[2] - th))),
                 back_err=float(np.hypot(x - start[0], y - start[1])),
                 back_heading=math.degrees(abs(wrap(th - start[2]))))
    return sim, start, final


# ============================================================== output
def ellipse(C, k=2.0, n=60):
    w, V = np.linalg.eigh(C)
    t = np.linspace(0, 2 * np.pi, n)
    return (V @ np.diag(k * np.sqrt(np.maximum(w, 0))) @ np.array([np.cos(t), np.sin(t)])).T


def plot_run(sim, start, ev, landmarks, title, path, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, ARENA, ARENA, 0, 0], [0, 0, ARENA, ARENA, 0], 'k-', lw=1)
    tp = np.array(sim.true_path)
    ep = np.array([to_world(start, *p[:2])[0] for p in sim.est_path])
    ax.plot(tp[:, 0], tp[:, 1], '-', color='0.6', lw=1, label='true trajectory')
    ax.plot(ep[:, 0], ep[:, 1], '-', color='tab:blue', lw=1, label='estimated trajectory')
    ax.plot(*start[:2], 'k^', ms=9, label='start')
    for lid, (lx, ly, f) in landmarks.items():
        ax.plot(lx, ly, 'ks', ms=7)
        ax.annotate(lid, (lx, ly), textcoords='offset points', xytext=(5, 5), fontsize=8)
    first = True
    for lid, p, Cw, err, _ in ev['rows']:
        e = ellipse(Cw) + p
        ax.plot(e[:, 0], e[:, 1], '-', color='tab:red', lw=1, label='2-sigma' if first else None)
        ax.plot(*p, 'x', color='tab:red', ms=8, label='estimated landmark' if first else None)
        ax.plot([p[0], landmarks[lid][0]], [p[1], landmarks[lid][1]], ':', color='tab:red', lw=1)
        first = False
    ax.set_aspect('equal')
    ax.set_xlim(-0.3, ARENA + 0.3)
    ax.set_ylim(-0.3, ARENA + 0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title, fontsize=10)
    ax.legend(loc='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'plot: {path}')
    if show:
        plt.show()


def summary(ev):
    s = (f"map {ev['n']}/{ev['n_true']} landmarks | mean err {100 * ev['mean_err']:.1f} cm "
         f"(max {100 * ev['max_err']:.1f}, shape-only {100 * ev['fit_err']:.1f}) | "
         f"inside 95% ellipse {100 * ev['in95']:.0f}%")
    return s


def single(args, ccl, ekf_mod, landmarks):
    sim, start, ev = run(ccl, ekf_mod, args.kind, args.size, args.laps, args.seed, landmarks,
                         quiet=args.quiet)
    print('\n================ simulation result')
    print(f"{args.kind} {args.size} m x {args.laps} lap(s), start {start[0]:.2f}, {start[1]:.2f} (world)")
    for i, lap in enumerate(ev['laps'], 1):
        print(f"  after lap {i}: {summary(lap)}  [t = {lap['t'] / 60:.1f} min]")
    print(f"  final      : {summary(ev)}")
    print(f"  robot: est. pose error {100 * ev['pose_err']:.1f} cm / {ev['heading_err']:.1f} deg; "
          f"true end {100 * ev['back_err']:.1f} cm / {ev['back_heading']:.1f} deg from the start pose")
    print(f"  {ev['stops']} stops, ~{ev['t'] / 60:.1f} min" + ('   !! robot hit a wall' if ev['collided'] else ''))
    for lid, p, Cw, err, nees in sorted(ev['rows']):
        print(f"    {lid}: err {100 * err:5.1f} cm   2-sigma {200 * math.sqrt(max(np.linalg.eigvalsh(Cw))):5.1f} cm")
    missing = sorted(set(landmarks) - {r[0] for r in ev['rows']})
    if missing:
        print(f"    never seen: {', '.join(missing)}")
    path = args.out or os.path.join(os.getcwd(), f'sim_{args.kind}_{args.size}.png')
    plot_run(sim, start, ev, landmarks, f'{args.kind} {args.size} m, {args.laps} lap(s), seed {args.seed}\n'
             + summary(ev), path, args.show)


def sweep(args, ccl, ekf_mod, landmarks):
    sizes = [float(s) for s in args.size.split(',')]
    rows = []
    for size in sizes:
        res = []
        for k in range(args.runs):
            _, _, ev = run(ccl, ekf_mod, args.kind, size, args.laps, args.seed + k, landmarks, quiet=True)
            res.append(ev)
        for lap in range(args.laps):
            L = [r['laps'][lap] for r in res if len(r['laps']) > lap]
            if not L:
                continue
            rows.append(dict(size=size, lap=lap + 1,
                             mean_err=np.mean([l['mean_err'] for l in L]),
                             std_err=np.std([l['mean_err'] for l in L]),
                             fit_err=np.mean([l['fit_err'] for l in L]),
                             n_lm=np.mean([l['n'] for l in L]),
                             in95=np.mean([l['in95'] for l in L]),
                             minutes=np.mean([l['t'] for l in L]) / 60,
                             collisions=sum(r['collided'] for r in res)))
        print(f"size {size}: " + ', '.join(f"lap {r['lap']} {100 * r['mean_err']:.1f}+-{100 * r['std_err']:.1f} cm"
                                           for r in rows if r['size'] == size)
              + f"  | {np.mean([r['n'] for r in res]):.1f} landmarks, "
              f"{np.mean([r['t'] for r in res]) / 60:.0f} min, {sum(r['collided'] for r in res)} wall hits")
    stem = args.out or os.path.join(os.getcwd(), f'sweep_{args.kind}')
    stem = stem[:-4] if stem.endswith('.png') else stem
    with open(stem + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for lap in range(1, args.laps + 1):
        R = [r for r in rows if r['lap'] == lap]
        s = [r['size'] for r in R]
        axs[0].errorbar(s, [100 * r['mean_err'] for r in R], [100 * r['std_err'] for r in R],
                        marker='o', capsize=3, label=f'after lap {lap}')
        axs[1].plot(s, [100 * r['in95'] for r in R], 'o-', label=f'after lap {lap}')
    R = [r for r in rows if r['lap'] == args.laps]
    axs[2].plot([r['size'] for r in R], [r['n_lm'] for r in R], 'o-', label='landmarks mapped')
    ax2 = axs[2].twinx()
    ax2.plot([r['size'] for r in R], [r['minutes'] for r in R], 's--', color='tab:gray', label='run time')
    axs[0].set_ylabel('mean landmark error (cm)')
    axs[1].set_ylabel('landmarks inside 95% ellipse (%)')
    axs[2].set_ylabel(f'landmarks mapped (of {len(landmarks)})')
    axs[2].set_ylim(0, len(landmarks) + 0.5)
    ax2.set_ylabel(f'minutes for {args.laps} lap(s)')
    for ax in axs:
        ax.set_xlabel(f'{args.kind} side (m)')
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    axs[1].legend(fontsize=8)
    fig.suptitle(f'{args.kind}: {args.runs} runs per size', fontsize=10)
    fig.tight_layout()
    fig.savefig(stem + '.png', dpi=150)
    print(f'table: {stem}.csv\nplot: {stem}.png')
    if args.show:
        plt.show()


def apply_sets(ccl, pairs, target, label):
    env = {'math': math}
    for kv in pairs or []:
        k, v = kv.split('=', 1)
        k = k.strip()
        if isinstance(target, dict):
            if k not in target:
                sys.exit(f'unknown {label} key {k}; choose from {sorted(target)}')
            target[k] = float(eval(v, env))
        else:
            if not hasattr(target, k):
                sys.exit(f'closed_loop_control has no constant {k}')
            setattr(target, k, eval(v, env))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind', help="square | octagon | sweep")
    ap.add_argument('rest', nargs='*', help='single: SIZE LAPS   sweep: KIND SIZES')
    ap.add_argument('--laps', type=int, default=None, help='sweep: laps per run (default 3)')
    ap.add_argument('--runs', type=int, default=10, help='sweep: runs (random seeds) per size')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--landmarks', help='CSV id,x,y,facing_deg (world frame, m); default: 2 per wall')
    ap.add_argument('--set', action='append', help='controller constant, e.g. NEAR=0.05')
    ap.add_argument('--truth', action='append', help='true noise, e.g. drive_scale=0.05')
    ap.add_argument('--filter', action='append', help='filter assumption, e.g. sigma_th_per_rad=0.2')
    ap.add_argument('--cam', action='append', help='camera, e.g. "half_fov=math.radians(25)"')
    ap.add_argument('--out', help='output file (single) or file stem (sweep)')
    ap.add_argument('--show', action='store_true', help='open the plot window')
    ap.add_argument('--quiet', action='store_true', help='hide the controller prints (single run)')
    args = ap.parse_args()

    ccl, ekf_mod = load_controller()
    apply_sets(ccl, args.set, ccl, 'controller')
    apply_sets(ccl, args.truth, TRUTH, 'truth')
    apply_sets(ccl, args.filter, FILTER, 'filter')
    apply_sets(ccl, args.cam, CAM, 'cam')
    landmarks = load_landmarks(args.landmarks) if args.landmarks else default_landmarks()

    if args.kind == 'sweep':
        if len(args.rest) != 2:
            sys.exit('usage: sim_slam.py sweep square|octagon SIZE1,SIZE2,...')
        args.kind, args.size = args.rest
        args.laps = args.laps or 3
        sweep(args, ccl, ekf_mod, landmarks)
    else:
        args.size = float(args.rest[0]) if args.rest else {'square': 1.2, 'octagon': 0.5}[args.kind]
        args.laps = int(args.rest[1]) if len(args.rest) > 1 else (args.laps or 1)
        single(args, ccl, ekf_mod, landmarks)


if __name__ == '__main__':
    main()
