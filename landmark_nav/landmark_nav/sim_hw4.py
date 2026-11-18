#!/usr/bin/env python3
"""Laptop simulator for HW4: runs the REAL planner.py + nav_core.py (what hw4_nav.py runs on the
robot) against a simulated robot and camera. No ROS needed.

Simulated robot: each wheel has its own speed error (the robot curves a little when told to go
straight), a 0.1 s motor lag, and the in-place turn rate is off by a few %. Simulated camera:
fps, latency, field of view, range limit, tags only seen from the front, the obstacle hides tags
behind it, range / bearing noise. These are GUESSES: replace them with what you measure.

    python3 sim_hw4.py short                 # one run of the shortest path, saves sim_hw4_short.png
    python3 sim_hw4.py safe --seed 4
    python3 sim_hw4.py both --runs 20        # many runs, prints statistics only
    python3 sim_hw4.py short --set v_max=0.2 --set lookahead=0.35
    python3 sim_hw4.py short --truth fps=5 --truth latency=0.25
"""
import argparse
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planner                                            # noqa: E402
from nav_core import DEFAULTS, Navigator, make_config, wrap   # noqa: E402

TRUTH = {
    'wheel_err': 0.03,          # std of each wheel's speed scale error after trimming (per run)
    'spin_err': 0.08,           # std of the in-place turn rate error (per run)
    'motor_tau': 0.10,          # s first-order motor lag
    'fps': 8.0,                 # detector output rate
    'latency': 0.15,            # s image -> observation
    'half_fov': math.radians(31),
    'min_range': 0.25, 'max_range': 3.0,
    'max_view': math.radians(65),   # tag seen only if viewed within this angle of its normal
    'sr_per_m': 0.02, 'sb': math.radians(1.0),
    'start_err_xy': 0.03, 'start_err_th': math.radians(3),
}


def load(map_file):
    with open(map_file) as f:
        cfg = yaml.safe_load(f)
    world = planner.World(cfg)
    tags = {str(k): (float(v['x']), float(v['y']), math.radians(float(v.get('yaw_deg', 0.0))))
            for k, v in (cfg.get('landmarks') or {}).items()}
    return world, tags


def visible(world, tags, cam, th, T):
    """Detector output for a camera at cam with heading th (true pose), with noise."""
    rng = T['rng']
    obs = []
    for lid, (tx, ty, tyaw) in tags.items():
        d = np.array([tx, ty]) - cam
        r = float(np.linalg.norm(d))
        if not T['min_range'] < r < T['max_range']:
            continue
        b = wrap(math.atan2(d[1], d[0]) - th)
        if abs(b) > T['half_fov']:
            continue
        view = wrap(math.atan2(-d[1], -d[0]) - tyaw)   # direction tag -> camera vs tag normal
        if abs(view) > T['max_view']:
            continue
        if any(planner.segment_enters(cam, np.array([tx, ty]), poly) for poly in world.obstacles):
            continue
        sr, sb = T['sr_per_m'] * r, T['sb']
        obs.append({'id': lid, 'r': r + rng.normal(0, sr), 'b': b + rng.normal(0, sb), 'sr': sr, 'sb': sb})
    return obs


def run(world, tags, mode, cfg, T, seed, t_max=150.0):
    rng = np.random.default_rng(seed)
    T = dict(T, rng=rng)
    path = (planner.plan_visibility(world) if mode == 'short' else planner.plan_voronoi(world))['path']
    start = np.array([world.start[0], world.start[1], world.start_th])
    nav = Navigator(path, start, tags, cfg)

    true = start + np.array([rng.normal(0, T['start_err_xy']), rng.normal(0, T['start_err_xy']),
                             rng.normal(0, T['start_err_th'])])
    kl, kr = 1 + rng.normal(0, T['wheel_err']), 1 + rng.normal(0, T['wheel_err'])
    ks = 1 + rng.normal(0, T['spin_err'])
    b = cfg['wheel_base']
    vl = vr = 0.0                                          # true wheel speeds (with lag)
    dt = 1.0 / cfg['rate']
    next_frame, pending = 0.0, []                          # (deliver time, capture time, obs)
    log = []
    t = 0.0
    while t < t_max and not nav.done:
        nav.step(dt)                                       # wheel commands; nav.v / nav.w = assumed motion
        spinning = nav.v == 0.0 and nav.w != 0.0
        if spinning:                                       # true turn rate is off by ks
            tl, tr = -0.5 * b * nav.w * ks, 0.5 * b * nav.w * ks
        else:                                              # each wheel off by its own scale
            tl, tr = (nav.v - 0.5 * b * nav.w) * kl, (nav.v + 0.5 * b * nav.w) * kr
        a = dt / (T['motor_tau'] + dt)
        vl, vr = vl + a * (tl - vl), vr + a * (tr - vr)
        v_t, w_t = 0.5 * (vl + vr), (vr - vl) / b
        true = true + np.array([v_t * math.cos(true[2]) * dt, v_t * math.sin(true[2]) * dt, w_t * dt])
        true[2] = wrap(true[2])
        t += dt
        if t >= next_frame:                                # camera captures now
            cam = true[:2] + cfg['cam_offset'] * np.array([math.cos(true[2]), math.sin(true[2])])
            pending.append((t + T['latency'], t, visible(world, tags, cam, true[2], T)))
            next_frame = t + 1.0 / T['fps']
        while pending and pending[0][0] <= t:
            _, t_cap, obs = pending.pop(0)
            nav.on_obs(obs, t - t_cap)
        log.append((t, *true, *nav.loc.pose, nav.v, nav.w, spinning or nav.state in ('look', 'settle')))
    log = np.array(log)
    true_xy, est_xy = log[:, 1:3], log[:, 4:6]
    pts = np.array(path)
    cross = [min(planner.point_segment_dist(p, a, b) for a, b in zip(pts, pts[1:])) for p in true_xy]
    clear = min(world.obstacle_clearance(p) for p in true_xy) - world.radius
    return {'log': log, 'path': pts, 'time': t, 'reached': nav.done,
            'goal_err': float(np.linalg.norm(true_xy[-1] - world.goal)),
            'max_cross': float(max(cross)), 'est_err': float(np.max(np.linalg.norm(true_xy - est_xy, axis=1))),
            'gap': float(clear), 'updates': nav.loc.n_updates, 'rejected': nav.loc.n_rejected,
            'stopped_s': float(np.sum(log[:, 9]) * dt), 'reloc': nav.n_reloc}


def plot(world, tags, res, mode, fn):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1.1, 1]})
    ax = axes[0]
    planner.draw_world(ax, world, f'{mode}: planned vs simulated run')
    log = res['log']
    ax.plot(res['path'][:, 0], res['path'][:, 1], '--', color='gray', lw=1.5, label='planned path')
    ax.plot(log[:, 1], log[:, 2], color='#1f6fb4', lw=2, label='true')
    ax.plot(log[:, 4], log[:, 5], color='#d0521b', lw=1, label='estimated (EKF)')
    ax.legend(loc='lower left', fontsize=8)
    ax = axes[1]
    ax.plot(log[:, 0], log[:, 7], label='v (m/s)')
    ax.fill_between(log[:, 0], 0, 1, where=log[:, 9] > 0, color='gray', alpha=0.15,
                    transform=ax.get_xaxis_transform(), label='turning / standing')
    ax.plot(log[:, 0], log[:, 8], label='w (rad/s)')
    ax.set_xlabel('t (s)')
    ax.set_title(f'commands  ({res["time"]:.1f} s, goal error {res["goal_err"] * 100:.1f} cm)', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fn, dpi=130)
    plt.close(fig)


def kv(items, base):
    out = {}
    for s in items or []:
        k, v = s.split('=', 1)
        out[k] = eval(v, {'math': math}) if k in base else v
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['short', 'safe', 'both'])
    ap.add_argument('--map', default=os.path.join(here, '..', 'config', 'map_hw4.yaml'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--runs', type=int, default=1)
    ap.add_argument('--set', action='append', help='nav_core setting, e.g. v_max=0.2')
    ap.add_argument('--truth', action='append', help='simulated world, e.g. fps=5')
    args = ap.parse_args()

    world, tags = load(args.map)
    cfg = make_config(kv(args.set, DEFAULTS))
    T = dict(TRUTH, **kv(args.truth, TRUTH))
    modes = ['short', 'safe'] if args.mode == 'both' else [args.mode]
    keys = ['time', 'goal_err', 'max_cross', 'est_err', 'gap', 'stopped_s', 'reloc']
    for mode in modes:
        rs = [run(world, tags, mode, cfg, T, args.seed + i) for i in range(args.runs)]
        if args.runs == 1:
            r = rs[0]
            print(f'{mode}: {"reached" if r["reached"] else "NOT reached"} in {r["time"]:.1f} s, '
                  f'goal error {r["goal_err"] * 100:.1f} cm, max off-path {r["max_cross"] * 100:.1f} cm, '
                  f'max EKF error {r["est_err"] * 100:.1f} cm, min body-obstacle gap {r["gap"] * 100:.1f} cm, '
                  f'turning/standing {r["stopped_s"]:.1f} s, re-localizations {r["reloc"]}, '
                  f'tags used {r["updates"]} (rejected {r["rejected"]})')
            fn = os.path.join(os.getcwd(), f'sim_hw4_{mode}.png')
            plot(world, tags, r, mode, fn)
            print(f'  saved {fn}')
        else:
            ok = sum(r['reached'] for r in rs)
            print(f'{mode}: reached {ok}/{len(rs)}   (mean / worst)')
            for k in keys:
                a = np.array([r[k] for r in rs])
                worst = a.min() if k == 'gap' else a.max()
                scale, unit = {'time': (1, 's'), 'stopped_s': (1, 's'), 'reloc': (1, '')}.get(k, (100, 'cm'))
                print(f'  {k:10s} {a.mean() * scale:7.1f} / {worst * scale:7.1f} {unit}')


if __name__ == '__main__':
    main()
