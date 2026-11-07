#!/usr/bin/env python3
"""HW3 evaluation (report items 4 and 7) for the SLAM node's logs.

Reads the two CSVs written by localizer_node.py (SLAM version):
    trajectory.csv  t, x, y, theta, sx, sy, stheta, n_updates, n_landmarks
    map.csv         t, id, x, y, sxx, sxy, syy        (whole map every 0.5 s)
and a ground-truth YAML (tape-measured, room frame, meters; extra keys like height are ignored):
    landmarks:
      cup:        {x: 0.5, y: 2.0}
      wine glass: {x: 1.4, y: 0.0}

What it does
1. Transforms the SLAM map / trajectory from the START-pose frame into the room frame with the
   measured start pose:  p_room = R(th0) p_slam + [x0, y0]   (covariance: R C R^T).
   If the ground truth was measured in the start frame, leave --start at 0 0 0.
2. Matches landmarks by id, prints per-landmark and AVERAGE error, missing / extra landmarks.
3. Splits the run into laps (robot back at the start corner, or --lap-times) and prints the
   map error after every lap -> "does the map improve if you drive multiple times?".
4. Saves <prefix>_map.png (trajectory, landmarks with 2-sigma ellipses, ground truth) and
   <prefix>_laps.png (error and filter sigma per lap) next to map.csv.

Usage:
    python3 evaluate.py LOG_DIR GROUND_TRUTH.yaml --start X0 Y0 TH0_DEG --title "Square 1.2 m, 2 laps"
    python3 evaluate.py --traj run1/trajectory.csv --map run1/map.csv gt.yaml --start 0.9 0.9 0
Only needs numpy, matplotlib, pyyaml (runs on a laptop, no ROS).
"""
import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Ellipse


# ------------------------------------------------------------------ loading
def load_traj(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r['t']) for r in rows])
    xyz = np.array([[float(r['x']), float(r['y']), float(r['theta'])] for r in rows])
    return t, xyz


def load_map_log(path):
    """{t: {id: (mean (2,), cov (2,2))}} with times sorted."""
    snaps = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        C = np.array([[float(r['sxx']), float(r['sxy'])],
                      [float(r['sxy']), float(r['syy'])]])
        snaps[float(r['t'])][r['id']] = (np.array([float(r['x']), float(r['y'])]), C)
    return dict(sorted(snaps.items()))


def map_at(snaps, t):
    """The map as it was at time t (last snapshot at or before t)."""
    best = None
    for ts, m in snaps.items():
        if ts <= t + 1e-6:
            best = m
        else:
            break
    return best or {}


def load_gt(path):
    lm = yaml.safe_load(open(path))['landmarks']
    return {name: np.array([float(v['x']), float(v['y'])]) for name, v in lm.items()}


# ------------------------------------------------------------------ geometry
def transform(start):
    x0, y0, th0 = start
    c, s = math.cos(th0), math.sin(th0)
    Rw = np.array([[c, -s], [s, c]])
    return lambda p: Rw @ np.asarray(p) + [x0, y0], lambda C: Rw @ C @ Rw.T, Rw


def map_errors(m, gt, to_room):
    return {n: float(np.linalg.norm(to_room(mu) - gt[n])) for n, (mu, _) in m.items() if n in gt}


def mean_sigma(m):
    return float(np.mean([math.sqrt(np.trace(C) / 2) for _, C in m.values()])) if m else float('nan')


def find_laps(t, xyz, radius, min_away):
    """Lap ends = the LAST log time of each visit back near the start (0, 0), after the robot has
    been at least `min_away` m away. The controller stands still there, so the map is settled."""
    ends, away, near_since_away, last_near = [], False, False, None
    for ti, (x, y, _) in zip(t, xyz):
        d = math.hypot(x, y)
        if d > min_away:
            if near_since_away and last_near is not None:
                ends.append(last_near)            # left the start corner again -> that lap ended
            away, near_since_away = True, False
        elif d < radius and away:
            near_since_away, last_near = True, ti
    if near_since_away and last_near is not None:
        ends.append(t[-1])                        # final lap: end of the log
    return ends


def ellipse(ax, mean, cov, k=2.0, **kw):
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mean, 2 * k * math.sqrt(vals[1]), 2 * k * math.sqrt(vals[0]), angle=ang, **kw))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log_dir', nargs='?', help='folder with trajectory.csv and map.csv')
    ap.add_argument('ground_truth', help='YAML with landmarks: {id: {x, y}} in the room frame')
    ap.add_argument('--traj', help='trajectory CSV (default LOG_DIR/trajectory.csv)')
    ap.add_argument('--map', dest='map_csv', help='map CSV (default LOG_DIR/map.csv)')
    ap.add_argument('--start', nargs=3, type=float, default=[0.0, 0.0, 0.0],
                    metavar=('X0', 'Y0', 'TH0_DEG'), help='measured start pose in the room frame')
    ap.add_argument('--lap-times', type=float, nargs='+',
                    help='node times t (s) at which each lap ended (default: detected automatically)')
    ap.add_argument('--lap-radius', type=float, default=0.30, help='m: "back at the start" radius')
    ap.add_argument('--min-away', type=float, default=0.50, help='m: must get this far to count a lap')
    ap.add_argument('--area', type=float, nargs=4, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
                    help='draw the square area boundary (room frame, m)')
    ap.add_argument('--title', default='EKF-SLAM')
    ap.add_argument('--out', help='output prefix (default: next to map.csv)')
    a = ap.parse_args()

    traj_path = a.traj or os.path.join(a.log_dir, 'trajectory.csv')
    map_path = a.map_csv or os.path.join(a.log_dir, 'map.csv')
    prefix = a.out or os.path.join(os.path.dirname(os.path.abspath(map_path)), 'eval')
    start = (a.start[0], a.start[1], math.radians(a.start[2]))
    to_room, cov_room, _ = transform(start)

    t, xyz = load_traj(traj_path)
    snaps = load_map_log(map_path)
    gt = load_gt(a.ground_truth)
    if not snaps:
        raise SystemExit(f'{map_path} has no landmarks')
    final = list(snaps.values())[-1]

    # ---------------------------------------------------------- final map vs ground truth
    err = map_errors(final, gt, to_room)
    print(f'final map (t = {list(snaps)[-1]:.1f} s): {len(final)} landmarks; ground truth: {len(gt)}')
    print(f'  {"id":14s} {"x":>7s} {"y":>7s} {"sx":>6s} {"sy":>6s}   error')
    for n in sorted(final):
        mu, C = final[n]
        p, sd = to_room(mu), np.sqrt(np.diag(cov_room(C)))
        e = f'{err[n]:.3f} m' if n in err else 'not in ground truth (false detection?)'
        print(f'  {n:14s} {p[0]:7.3f} {p[1]:7.3f} {sd[0]:6.3f} {sd[1]:6.3f}   {e}')
    missing = [n for n in gt if n not in final]
    if missing:
        print(f'  never mapped: {", ".join(missing)}')
    if err:
        print(f'AVERAGE landmark error: {np.mean(list(err.values())):.3f} m  '
              f'(median {np.median(list(err.values())):.3f}, max {max(err.values()):.3f}, '
              f'{len(err)} landmarks)')

    # ---------------------------------------------------------- per lap
    lap_ends = a.lap_times or find_laps(t, xyz, a.lap_radius, a.min_away)
    lap_rows = []
    if lap_ends:
        print(f'\nlaps ({"given" if a.lap_times else "detected"}): end times {[round(float(x), 1) for x in lap_ends]} s')
        print('  lap   t_end   #landmarks   mean error   mean sigma   robot drift at start')
        for k, te in enumerate(lap_ends, 1):
            m = map_at(snaps, te)
            e = map_errors(m, gt, to_room)
            i = int(np.searchsorted(t, te, side='right')) - 1
            drift = math.hypot(*xyz[max(i, 0), :2])
            row = (k, te, len(m), np.mean(list(e.values())) if e else float('nan'), mean_sigma(m))
            lap_rows.append(row)
            print(f'  {k:3d} {te:7.1f} {row[2]:9d}      {row[3]:7.3f} m    {row[4]:7.3f} m    {drift:.3f} m')
    else:
        print('\nno laps detected (robot never returned near the start); use --lap-times')

    # ---------------------------------------------------------- map plot
    fig, ax = plt.subplots(figsize=(7, 7))
    if a.area:
        x0, y0, x1, y1 = a.area
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color='0.7', lw=1, label='area boundary')
    tr = np.array([to_room(p) for p in xyz[:, :2]])
    ax.plot(tr[:, 0], tr[:, 1], '-', color='tab:blue', lw=1.2, label='estimated trajectory')
    ax.plot(*tr[0], 'o', color='tab:blue', ms=8)
    ax.annotate('start', tr[0], textcoords='offset points', xytext=(6, 6))
    for k, (n, p) in enumerate(gt.items()):
        ax.plot(*p, 'k^', ms=9, label='ground truth' if k == 0 else None)
    for k, (n, (mu, C)) in enumerate(final.items()):
        p = to_room(mu)
        ax.plot(*p, 'x', color='tab:red', ms=9, mew=2, label='SLAM landmark' if k == 0 else None)
        ellipse(ax, p, cov_room(C), fill=False, color='tab:red', lw=1,
                label='2σ covariance' if k == 0 else None)
        ax.annotate(n, p, textcoords='offset points', xytext=(5, -12), fontsize=8)
        if n in gt:
            ax.plot([p[0], gt[n][0]], [p[1], gt[n][1]], ':', color='0.4', lw=0.8)
    if err:
        a.title += f' — mean landmark error {np.mean(list(err.values())):.2f} m'
    ax.set_title(a.title)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    fig.savefig(prefix + '_map.png', dpi=150, bbox_inches='tight')
    print(f'\nplot -> {prefix}_map.png')

    # ---------------------------------------------------------- lap plot
    if lap_rows:
        k, _, _, e, s = zip(*lap_rows)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(k, e, 'o-', label='mean error vs ground truth')
        ax.plot(k, s, 's--', label='mean landmark σ (filter)')
        ax.set_xlabel('lap')
        ax.set_ylabel('m')
        ax.set_xticks(k)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.savefig(prefix + '_laps.png', dpi=150, bbox_inches='tight')
        print(f'plot -> {prefix}_laps.png')


if __name__ == '__main__':
    main()
