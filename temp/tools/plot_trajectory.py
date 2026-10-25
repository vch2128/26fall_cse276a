#!/usr/bin/env python3
"""
HW2 report numbers from a waypoint_control run (runs on a laptop, needs matplotlib).

    scp 'ubuntu@<robot-ip>:~/ros2_ws/logs/hw2_*' logs/
    python3 tools/plot_trajectory.py logs/hw2_20261025_141500 --out trajectory.png
    python3 tools/plot_trajectory.py logs/hw2_20261025_141500 --truth truth.csv --unit 0.5

Prints
    total moving distance (translation + rotation) from the estimated poses
    estimated pose at every waypoint and its error to the goal
    with --truth: error between the estimate and the pose you measured on the floor
                  (truth.csv: one line "x,y,theta_deg" per reached waypoint, same order)
Draws one dot per logged pose (every 0.5 s), the waypoints and the landmarks.
"""
import argparse
import csv
import math
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
LANDMARKS = os.path.join(HERE, '..', 'robot_control', 'config', 'landmarks.txt')


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def read_csv(path):
    with open(path) as f:
        return [{k: (v if k == 'state' else float(v)) for k, v in row.items()}
                for row in csv.DictReader(f)]


def read_landmarks(path):
    marks = []
    for line in open(path):
        if line.strip() and not line.startswith('#'):
            cls, x, y, _h = [s.strip() for s in line.split(',')]
            marks.append((cls, float(x), float(y)))
    return marks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', help='log prefix, e.g. logs/hw2_20261025_141500')
    ap.add_argument('--truth', help='measured poses at the waypoints: x,y,theta_deg per line')
    ap.add_argument('--unit', type=float, default=1.0, help='report unit in meters (>= 0.5)')
    ap.add_argument('--out', default='trajectory.png')
    args = ap.parse_args()

    traj = read_csv(args.run + '_traj.csv')
    wps = read_csv(args.run + '_waypoints.csv')

    trans = sum(math.hypot(b['x'] - a['x'], b['y'] - a['y']) for a, b in zip(traj, traj[1:]))
    rot = sum(abs(wrap(b['theta'] - a['theta'])) for a, b in zip(traj, traj[1:]))
    print(f'total translation: {trans:.2f} m = {trans / args.unit:.2f} units')
    print(f'total rotation:    {rot:.2f} rad = {math.degrees(rot):.0f} deg')
    print(f'run time:          {traj[-1]["t"]:.1f} s')

    truth = []
    if args.truth:
        for line in open(args.truth):
            if line.strip() and not line.startswith('#'):
                x, y, th = map(float, line.split(','))
                truth.append((x, y, math.radians(th)))

    print('\n  i   goal (x, y, deg)        estimate (x, y, deg)     est-goal: pos  ang'
          + ('    truth-est: pos  ang' if truth else ''))
    for k, w in enumerate(wps):
        pos = math.hypot(w['est_x'] - w['goal_x'], w['est_y'] - w['goal_y'])
        ang = math.degrees(wrap(w['est_theta'] - w['goal_theta']))
        line = (f'{int(w["i"]):3d}  ({w["goal_x"]:5.2f}, {w["goal_y"]:5.2f}, '
                f'{math.degrees(w["goal_theta"]):6.1f})  ({w["est_x"]:5.2f}, {w["est_y"]:5.2f}, '
                f'{math.degrees(w["est_theta"]):6.1f})  {pos:7.3f} m {ang:+6.1f}')
        if k < len(truth):
            tx, ty, tth = truth[k]
            line += (f'   {math.hypot(tx - w["est_x"], ty - w["est_y"]):7.3f} m '
                     f'{math.degrees(wrap(tth - w["est_theta"])):+6.1f}')
        print(line)

    fig, ax = plt.subplots(figsize=(6, 7))
    ax.axhline(0, color='black', lw=1)
    ax.axvline(0, color='black', lw=1)
    ax.plot([p['x'] for p in traj], [p['y'] for p in traj], 'o', color='red', ms=4,
            label='estimated pose (every 0.5 s)')
    ax.plot([w['goal_x'] for w in wps], [w['goal_y'] for w in wps], 'x', color='blue',
            ms=12, mew=2, label='waypoint')
    if os.path.exists(LANDMARKS):
        for cls, x, y in read_landmarks(LANDMARKS):
            ax.plot(x, y, 's', color='green', ms=9)
            ax.annotate(cls, (x, y), textcoords='offset points', xytext=(6, 6))
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
