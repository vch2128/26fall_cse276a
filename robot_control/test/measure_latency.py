#!/usr/bin/env python3
"""Measure how long the detector takes: image capture -> landmark_obs published.

    python3 measure_latency.py          # listen for 30 s
    python3 measure_latency.py 60       # listen for 60 s

Keep the robot still with the landmarks in view. Run it once per detector setting
(imgsz, model) and compare. At the end it prints how often each landmark was seen
and suggests a SETTLE for closed_loop_control.
"""
import json
import statistics
import sys

import rclpy
from std_msgs.msg import String

STILL = 0.5                 # same as the localizer / controller: images taken before this are ignored

duration = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
rclpy.init()
node = rclpy.create_node('measure_latency')
lat, gaps, seen = [], [], {}
last = None


def now():
    return node.get_clock().now().nanoseconds * 1e-9


def on_obs(msg):
    global last
    t = now()
    data = json.loads(msg.data)
    lat.append(t - data['stamp'])                     # capture -> result
    if last is not None:
        gaps.append(t - last)                         # time between two results
    last = t
    for o in data['obs']:
        seen[o['id']] = seen.get(o['id'], 0) + 1
    names = ', '.join(f"{o['id']} r={o['r']:.2f}" for o in data['obs']) or '-'
    print(f'latency {lat[-1]:5.2f} s   {names}')


node.create_subscription(String, 'landmark_obs', on_obs, 10)
end = now() + duration
while rclpy.ok() and now() < end:
    rclpy.spin_once(node, timeout_sec=0.1)

n = len(lat)
print('\n' + '=' * 60)
if n < 3:
    print('fewer than 3 results: is the detector running?')
else:
    print(f'results: {n} in {duration:.0f} s  ->  {n / duration:.2f} Hz')
    print(f'latency (capture -> result): mean {statistics.mean(lat):.2f} s, max {max(lat):.2f} s')
    print(f'time between results:        mean {statistics.mean(gaps):.2f} s, max {max(gaps):.2f} s')
    if max(lat) > 5.0 or min(lat) < 0.0:
        print('!! latency looks wrong: the image stamps are probably not ROS time')
    for name, k in sorted(seen.items()):
        print(f'  {name:12s} seen in {k}/{n} results ({100 * k / n:.0f}%)')
    first = STILL + max(gaps) + max(lat)              # worst case until the first usable result
    print(f'suggested SETTLE: {first:.1f} s for >= 1 usable result per stop, '
          f'{first + max(gaps):.1f} s for >= 2')
node.destroy_node()
rclpy.shutdown()