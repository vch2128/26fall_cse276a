#!/usr/bin/env python3
"""Plot a real HW4 run (the CSV written by hw4_control) over the map: planned path vs the
localizer's trajectory, plus the wheel commands over time. For the report.

    python3 plot_hw4_run.py ~/ros2_ws/src/26fall_cse276a/traj_log/hw4_short_1118_153012.csv
    python3 plot_hw4_run.py run_short.csv run_safe.csv          # several runs on one map
"""
import argparse
import csv
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planner                                            # noqa: E402

COLORS = ['#d0521b', '#1f6fb4', '#2a7a2a', '#7b3fa0']


def read(fn):
    with open(fn) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--map', default=planner.default_map())
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(args.map) as f:
        world = planner.World(yaml.safe_load(f))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1.1, 1]})
    planner.draw_world(axes[0], world, 'Planned path vs EKF trajectory')
    for i, fn in enumerate(args.logs):
        c = COLORS[i % len(COLORS)]
        log = read(fn)
        name = os.path.basename(fn).replace('.csv', '')
        path = read(fn.replace('.csv', '_path.csv'))
        t = log['t'].astype(float)
        x, y = log['x'].astype(float), log['y'].astype(float)
        axes[0].plot(path['x'].astype(float), path['y'].astype(float), '--o', color=c, lw=1.2, ms=3, alpha=0.6)
        axes[0].plot(x, y, color=c, lw=2, label=f'{name} ({t[-1]:.1f} s)')
        axes[1].plot(t, log['L'].astype(float), color=c, label=f'{name} L')
        axes[1].plot(t, log['R'].astype(float), color=c, ls=':', label=f'{name} R')
    axes[0].legend(loc='lower left', fontsize=8)
    axes[1].set_xlabel('t (s)')
    axes[1].set_title('wheel commands', fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out = args.out or args.logs[0].replace('.csv', '.png')
    fig.savefig(out, dpi=150)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
