#!/usr/bin/env python3
"""Take one photo with the robot camera, run YOLO on it and save the picture with the boxes drawn.

Landmark classes (from landmarks.yaml) get a GREEN box with the range / bearing the detector
node would compute; boxes the detector node would DROP get a RED box with the reason
(touching the image border, too small, low confidence, or a second box of the same class).
Every other COCO class gets a thin GRAY box, so you can see what YOLO thinks things are
(e.g. if your suitcase is detected as "handbag").

The camera can only be used by one program: stop the ROS camera node first.

Usage (on the robot):
    python3 yolo_snapshot.py                     # take a photo, detect, save *_yolo.jpg
    python3 yolo_snapshot.py --image some.jpg    # detect on an existing photo instead
    python3 yolo_snapshot.py --conf 0.15         # try a lower confidence threshold
Output: focus_photos/snap_<time>.jpg and focus_photos/snap_<time>_yolo.jpg
"""
import argparse
import glob
import math
import os
import signal
import subprocess
import time

import cv2
import numpy as np
import yaml

SRC = os.path.expanduser('~/ros2_ws/src/26fall_cse276a')
# same calibration as camera_info (HW2)
K = np.array([[674.94, 0.0, 660.29],
              [0.0, 678.95, 304.12],
              [0.0, 0.0, 1.0]])
D = np.array([-0.32423, 0.10492, -0.00041866, -0.00025781, -0.015663])


def take_photo(path, seconds=3):
    """Record ~`seconds` of frames (auto-exposure needs a moment) and keep the last one."""
    tmp = path.replace('.jpg', '_frame%03d.jpg')
    cmd = ['gst-launch-1.0', '-e', 'qtiqmmfsrc', 'camera=0', '!',
           'video/x-raw,format=NV12,width=1280,height=720,framerate=10/1', '!',
           'jpegenc', '!', 'multifilesink', f'location={tmp}']
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(seconds)
    proc.send_signal(signal.SIGINT)          # like Ctrl+C; -e makes gst finish cleanly
    try:
        log, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        log, _ = proc.communicate()
    frames = sorted(glob.glob(path.replace('.jpg', '_frame*.jpg')))
    if not frames:
        print(log)
        raise RuntimeError('no photo saved (gst output above). Is the ROS camera node still running?')
    os.replace(frames[-1], path)                        # keep the newest frame
    for f in frames[:-1]:
        os.remove(f)


def measure(height, x1, y1, x2, y2):
    """Range / bearing exactly like detector_node.measure; returns (r, b_deg, h_px)."""
    uc = 0.5 * (x1 + x2)
    pts = np.array([[[uc, y1]], [[uc, y2]]], dtype=np.float64)
    (ut, vt), (ub, vb) = cv2.undistortPoints(pts, K, D, P=K).reshape(-1, 2)
    h = vb - vt
    Z = K[1, 1] * height / h
    X = (0.5 * (ut + ub) - K[0, 2]) * Z / K[0, 0]
    return math.hypot(Z, X), math.degrees(math.atan2(-X, Z)), h


def label(img, text, x, y, color):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(y, h + 6)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', help='use this photo instead of taking one')
    ap.add_argument('--out_dir', default=os.path.join(SRC, 'focus_photos'))
    ap.add_argument('--landmarks', default=os.path.join(SRC, 'landmark_nav/config/landmarks.yaml'))
    ap.add_argument('--model', default='yolo26n.pt')
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=0.25, help="detector node's threshold")
    ap.add_argument('--border_px', type=int, default=5)
    ap.add_argument('--min_box_h_px', type=float, default=12)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.image:
        photo = args.image
    else:
        photo = os.path.join(args.out_dir, f'snap_{time.strftime("%Y%m%d_%H%M%S")}.jpg')
        print('taking photo ...')
        take_photo(photo)
    img = cv2.imread(photo)
    H, W = img.shape[:2]
    heights = {k: float(v['height']) for k, v in yaml.safe_load(open(args.landmarks))['landmarks'].items()}

    from ultralytics import YOLO
    print('running YOLO ...')
    res = YOLO(args.model).predict(img, imgsz=args.imgsz, conf=0.05, verbose=False)[0]   # low conf: show everything

    boxes = sorted(zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()),
                   key=lambda b: -b[1])
    used = set()
    out = img.copy()
    print(f'\n{"class":12s} {"conf":>5s}  {"box h":>6s}  result')
    for (x1, y1, x2, y2), conf, cls in boxes:
        name = res.names[int(cls)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        if name not in heights:                                   # not a landmark class
            if conf >= args.conf:
                cv2.rectangle(out, p1, p2, (160, 160, 160), 1)
                label(out, f'{name} {conf:.2f}', p1[0], p1[1], (160, 160, 160))
            continue
        r, b, h = measure(heights[name], x1, y1, x2, y2)
        if conf < args.conf:
            why = f'conf < {args.conf}'
        elif x1 < args.border_px or y1 < args.border_px or x2 > W - args.border_px or y2 > H - args.border_px:
            why = 'touches border'
        elif h < args.min_box_h_px:
            why = f'box < {args.min_box_h_px:.0f}px'
        elif name in used:
            why = 'duplicate class'
        else:
            why = None
        if why is None:
            used.add(name)
            cv2.rectangle(out, p1, p2, (0, 200, 0), 3)
            label(out, f'{name} {conf:.2f} r={r:.2f}m b={b:+.0f}', p1[0], p1[1], (0, 200, 0))
            print(f'{name:12s} {conf:5.2f}  {h:5.0f}px  USED  r = {r:.2f} m, b = {b:+.1f} deg')
        else:
            cv2.rectangle(out, p1, p2, (0, 0, 255), 2)
            label(out, f'{name} {conf:.2f} DROP: {why}', p1[0], p1[1], (0, 0, 255))
            print(f'{name:12s} {conf:5.2f}  {h:5.0f}px  dropped ({why})')

    missing = [n for n in heights if n not in used]
    print(f'\nlandmarks used: {sorted(used) or "none"}')
    print(f'not used:       {missing}')
    dst = os.path.splitext(photo)[0] + '_yolo.jpg'
    if args.image:
        dst = os.path.join(args.out_dir, os.path.basename(dst))
    cv2.imwrite(dst, out)
    print(f'saved {dst}')


if __name__ == '__main__':
    main()