#!/usr/bin/env python3
"""Plot a real HW4 run (the CSV written by hw4_control) on the map: planned path vs the
localizer's (EKF) trajectory. For the report.

    python3 plot_hw4_run.py traj_log/hw4_rrt_1120_044847.csv
    python3 plot_hw4_run.py run_safe.csv run_rrt.csv --out compare.png     # several runs on one map
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import planner                                            # noqa: E402

COLORS = ['#d0521b', '#1f6fb4', '#2a7a2a', '#7b3fa0']
TAG_COLOR = 'black'                                   # teal: not used by the paths / box


def read(fn):
    with open(fn) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0] if k != 'state'}


def draw_map(ax, cfg, world):
    """Arena walls, the box (with its size), and every tag as a flat piece of the wall."""
    from matplotlib.patches import Polygon, Rectangle
    ax.add_patch(Rectangle((0, 0), world.W, world.H, fill=False, lw=2, color='black'))

    for o, poly in zip(cfg.get('obstacles', []), world.obstacles):
        ax.add_patch(Polygon(poly, closed=True, color='#8c6d46'))            # cardboard
        ax.text(float(o['cx']), float(o['cy']), 'box', ha='center', va='center', fontsize=9, color='white')
        ax.text(float(o['cx']), poly[:, 1].min() - 0.04, f'{o["w"] * 100:.1f} x {o["h"] * 100:.1f} cm',
                ha='center', va='top', fontsize=8, color='#5a4630')

    size = float(cfg.get('tag_size', 0.165))
    for name, v in (cfg.get('landmarks') or {}).items():
        x, y, yaw = float(v['x']), float(v['y']), math.radians(float(v.get('yaw_deg', 0.0)))
        n = np.array([math.cos(yaw), math.sin(yaw)])                        # tag faces into the arena
        t = np.array([-n[1], n[0]]) * size / 2                              # along the wall
        p = np.array([x, y])
        ax.plot(*zip(p - t, p + t), color=TAG_COLOR, lw=7, solid_capstyle='butt')
        lx, ly = p + n * 0.09
        ax.text(lx, ly, name, ha='center', va='center', fontsize=8, color=TAG_COLOR)

    ax.plot(*world.start, marker='o', ms=9, color='#2a7a2a', ls='none', zorder=5)
    ax.plot(*world.goal, marker='*', ms=14, color='#c03030', ls='none', zorder=5)
    ax.annotate('start', world.start, textcoords='offset points', xytext=(8, -12), fontsize=9)
    ax.annotate('goal', world.goal, textcoords='offset points', xytext=(8, 6), fontsize=9)
    ax.set_xlim(-0.08, world.W + 0.08)
    ax.set_ylim(-0.08, world.H + 0.08)
    ax.set_aspect('equal')
    for side in ax.spines.values():                                       # one frame: the arena walls
        side.set_visible(False)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(alpha=0.2)


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
        cfg = yaml.safe_load(f)
    world = planner.World(cfg)
    fig, ax = plt.subplots(figsize=(6.5, 7.2))
    draw_map(ax, cfg, world)
    for i, fn in enumerate(args.logs):
        c = COLORS[i % len(COLORS)]
        log = read(fn)
        mode = os.path.basename(fn).split('_')[1] if os.path.basename(fn).startswith('hw4_') else 'run'
        path = read(fn.replace('.csv', '_path.csv'))
        ax.plot(path['x'], path['y'], '--', color=c, lw=1.3, alpha=0.7, label=f'{mode}: planned path')
        ax.plot(log['x'], log['y'], color=c, lw=2.2, label=f'{mode}: EKF trajectory ({log["t"][-1]:.1f} s)')
    ax.set_title('Planned path vs EKF trajectory', fontsize=11)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    out = args.out or args.logs[0].replace('.csv', '.png')
    fig.savefig(out, dpi=150)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
