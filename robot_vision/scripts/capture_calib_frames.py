#!/usr/bin/env python3
"""
Save camera frames for checkerboard calibration - runs on the robot over SSH, no GUI.

Watch the live image in Foxglove on your laptop (ws://<robot-ip>:8765, which
robot_vision_camera.launch.py already starts) to aim the board, then press Enter
in this terminal to save the current frame. Every saved frame is checked for the
checkerboard right away, and a 3x3 coverage grid shows which parts of the image
you have covered so far (it replaces the ROS calibrator's X / Y / Size bars).

Terminal 1:  ros2 launch robot_vision_camera robot_vision_camera.launch.py
Terminal 2:  python3 capture_calib_frames.py                  # Enter = save, q + Enter = quit
             python3 capture_calib_frames.py --auto 1.5       # save every 1.5 s instead
             python3 capture_calib_frames.py --cols 9 --rows 6 --out ~/calib_frames

--cols / --rows are INNER corners (a board of 10 x 7 squares has 9 x 6 inner corners).
Only frames where the board is found are kept, unless you pass --keep-all.
Copy the folder to your laptop afterwards, e.g.
    scp -r <user>@<robot-ip>:~/calib_frames .
"""
import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

GRID = 3  # coverage grid is GRID x GRID cells


def image_to_bgr(msg):
    """sensor_msgs/Image -> BGR numpy array (no cv_bridge needed)."""
    channels = {'rgb8': 3, 'bgr8': 3, 'mono8': 1}.get(msg.encoding)
    if channels is None:
        raise ValueError(f'unsupported encoding {msg.encoding!r}')
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if msg.encoding == 'rgb8':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if msg.encoding == 'mono8':
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def find_board(bgr, pattern):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    return corners.reshape(-1, 2) if found else None


def coverage_text(covered, sizes):
    rows = []
    for r in range(GRID):
        rows.append('  ' + ' '.join('#' if covered[r][c] else '.' for c in range(GRID)))
    small = sum(s < 0.05 for s in sizes)
    large = sum(s > 0.15 for s in sizes)
    rows.append(f'  board size: {small} far/small, {len(sizes) - small - large} medium, '
                f'{large} close/large  (aim for some of each, and tilt the board ~30-45 deg)')
    return '\n'.join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--topic', default='/camera/image_raw')
    p.add_argument('--out', default='~/calib_frames')
    p.add_argument('--cols', type=int, default=9, help='inner corners along the long side')
    p.add_argument('--rows', type=int, default=6, help='inner corners along the short side')
    p.add_argument('--auto', type=float, default=0.0, help='save every N seconds instead of on Enter')
    p.add_argument('--keep-all', action='store_true', help='also keep frames without a detected board')
    args = p.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    pattern = (args.cols, args.rows)

    rclpy.init()
    node = rclpy.create_node('capture_calib_frames')
    latest = {'msg': None}

    def on_image(msg):
        latest['msg'] = msg

    node.create_subscription(Image, args.topic, on_image, qos_profile_sensor_data)

    requests = []          # filled by the keyboard thread: 'save' or 'quit'
    if args.auto <= 0:
        def keyboard():
            for line in sys.stdin:
                requests.append('quit' if line.strip().lower() == 'q' else 'save')
            requests.append('quit')
        threading.Thread(target=keyboard, daemon=True).start()

    covered = [[False] * GRID for _ in range(GRID)]
    sizes = []
    count = len([f for f in os.listdir(out) if f.endswith('.png')])
    next_auto = time.monotonic() + args.auto

    print(f'Waiting for images on {args.topic} ...')
    while rclpy.ok() and latest['msg'] is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    m = latest['msg']
    print(f'Got {m.width}x{m.height} {m.encoding}. Saving to {out}  (board {pattern[0]}x{pattern[1]} inner corners)')
    print('Auto mode, Ctrl-C to stop.' if args.auto > 0 else 'Enter = save frame,  q + Enter = quit')

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if args.auto > 0:
                if time.monotonic() < next_auto:
                    continue
                next_auto = time.monotonic() + args.auto
                action = 'save'
            elif requests:
                action = requests.pop(0)
            else:
                continue
            if action == 'quit':
                break

            img = image_to_bgr(latest['msg'])
            corners = find_board(img, pattern)
            if corners is None:
                print('  board NOT found' + (' - saved anyway' if args.keep_all else ' - skipped'))
                if not args.keep_all:
                    continue
            else:
                h, w = img.shape[:2]
                for x, y in corners:
                    covered[min(int(y * GRID / h), GRID - 1)][min(int(x * GRID / w), GRID - 1)] = True
                hull = cv2.convexHull(corners.astype(np.float32))
                sizes.append(cv2.contourArea(hull) / (w * h))

            count += 1
            path = os.path.join(out, f'frame_{count:03d}.png')
            cv2.imwrite(path, img)       # PNG = lossless
            print(f'saved {path}   ({len(sizes)} frames with board)')
            print(coverage_text(covered, sizes))
    except KeyboardInterrupt:
        pass
    finally:
        print(f'\nDone: {len(sizes)} frames with the board in {out}. Aim for 20-40 varied ones.')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
