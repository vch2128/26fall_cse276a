#!/usr/bin/env python3
"""Plot a SLAM run: robot trajectory + landmark map with covariance ellipses.

Reads the two CSV files the localizer writes:
    trajectory.csv : t, x, y, theta, sx, sy, stheta, n_updates, n_landmarks   (robot, every 0.5 s)
    map.csv        : t, id, x, y, sxx, sxy, syy                             (every landmark, every 0.5 s)
Everything is in the SLAM frame: the robot starts at (0, 0) facing +x.

Saves next to this script (in test_plot/):
    <name>_map.png          trajectory dots + final landmark estimates with 95% ellipses
    <name>_uncertainty.png  each landmark's position uncertainty over time (shrinks with laps)

Usage:
    python3 plot_slam.py                                   # uses ~/ros2_ws/logs/map.csv and trajectory.csv
    python3 plot_slam.py --map run1/map.csv --traj run1/trajectory.csv --name square_run1
    python3 plot_slam.py --at 120                          # the map as it was at t = 120 s (e.g. end of lap 1)
"""
import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')                                   # no screen needed (works over SSH)
import matplotlib.pyplot as plt                        # noqa: E402
import numpy as np                                     # noqa: E402
from matplotlib.patches import Ellipse                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.expanduser('~/ros2_ws/logs')
CHI2_95 = 5.991                                         # 95% region for 2 DOF


def load(map_csv, traj_csv):
    snaps = defaultdict(dict)                           # t -> {id: (x, y, 2x2 cov)}
    for r in csv.DictReader(open(map_csv)):
        C = np.array([[float(r['sxx']), float(r['sxy'])], [float(r['sxy']), float(r['syy'])]])
        snaps[float(r['t'])][r['id']] = (float(r['x']), float(r['y']), C)
    traj = np.array([[float(r['t']), float(r['x']), float(r['y']), float(r['theta'])]
                     for r in csv.DictReader(open(traj_csv))])
    return snaps, traj


def ellipse(ax, x, y, C, color):
    w, v = np.linalg.eigh(C)
    w = np.clip(w, 0, None)
    ang = math.degrees(math.atan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse((x, y), 2 * math.sqrt(CHI2_95 * w[1]), 2 * math.sqrt(CHI2_95 * w[0]),
                         angle=ang, fill=False, color=color, lw=1.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default=os.path.join(LOGS, 'map.csv'))
    ap.add_argument('--traj', default=os.path.join(LOGS, 'trajectory.csv'))
    ap.add_argument('--at', type=float, default=None, help='plot the map at this time (s); default: end')
    ap.add_argument('--name', default='slam', help='prefix of the saved pictures')
    args = ap.parse_args()

    snaps, traj = load(args.map, args.traj)
    times = sorted(snaps)
    if not times:
        raise SystemExit('map.csv has no landmarks yet')
    t_end = times[-1] if args.at is None else max(t for t in times if t <= args.at)
    final = snaps[t_end]
    traj = traj[traj[:, 0] <= t_end]
    ids = sorted({lid for s in snaps.values() for lid in s},
                 key=lambda s: (len(s), s))             # tag2 before tag10
    colors = {lid: plt.cm.tab10(i % 10) for i, lid in enumerate(ids)}

    # ------------------------------------------------------------------ map
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(traj[:, 1], traj[:, 2], '-', color='0.8', lw=1, zorder=1)
    ax.plot(traj[:, 1], traj[:, 2], '.', color='tab:blue', ms=4, zorder=2,
            label='robot (EKF estimate, every 0.5 s)')
    ax.plot(0, 0, 'k^', ms=11, zorder=4, label='start (0, 0), facing +x')
    ax.annotate('', xy=(0.25, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=2), zorder=4)
    ax.plot(traj[-1, 1], traj[-1, 2], 'ks', ms=8, mfc='none', zorder=4, label='end')

    print(f'map at t = {t_end:.1f} s: {len(final)} landmarks')
    for lid in ids:
        if lid not in final:
            continue
        x, y, C = final[lid]
        ellipse(ax, x, y, C, colors[lid])
        ax.plot(x, y, '+', color=colors[lid], ms=12, mew=2, zorder=3)
        ax.text(x + 0.06, y + 0.06, lid, color=colors[lid], fontsize=10, weight='bold')
        print(f'  {lid:12s} ({x:+.3f}, {y:+.3f}) m   sigma = ({100 * math.sqrt(C[0, 0]):.1f}, '
              f'{100 * math.sqrt(C[1, 1]):.1f}) cm')

    ax.plot([], [], '+', color='k', ms=10, mew=2, label='landmark estimate (95% ellipse)')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'SLAM map and trajectory (t = {t_end:.0f} s, {len(final)} landmarks)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8)
    ax.margins(0.15)
    fig.tight_layout()
    out1 = os.path.join(HERE, f'{args.name}_map.png')
    fig.savefig(out1, dpi=200)

    # ------------------------------------------------------------------ uncertainty over time
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    for lid in ids:
        ts = [t for t in times if t <= t_end and lid in snaps[t]]
        sd = [100 * math.sqrt(max(np.linalg.eigvalsh(snaps[t][lid][2]))) for t in ts]
        ax2.plot(ts, sd, color=colors[lid], label=lid)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('landmark sigma, largest axis (cm)')
    ax2.set_yscale('log')
    ax2.grid(alpha=0.3, which='both')
    ax2.legend(ncol=4, fontsize=8)
    ax2.set_title('Landmark uncertainty over time')
    fig2.tight_layout()
    out2 = os.path.join(HERE, f'{args.name}_uncertainty.png')
    fig2.savefig(out2, dpi=200)
    print(f'saved {out1}\nsaved {out2}')


if __name__ == '__main__':
    main()