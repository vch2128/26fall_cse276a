#!/usr/bin/env python3
"""
Simple YOLO test: run yolo26n on every photo in test_photos/, print what it finds and
how far away it is, and save the photos with boxes drawn into results_photos/.

Distance comes from the known real height of the object:
    distance = real height / (height of the box after undistortion)

    python3 ~/ros2_ws/src/26fall_cse276a/robot_control/test/test_yolo.py
"""
import glob
import os
import shutil

import cv2
import numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(HERE, 'test_photos')
RESULTS_DIR = os.path.join(HERE, 'results_photos')
MODEL_PATH = os.path.join(HERE, 'models', 'yolo26n.pt')
CAMERA_YAML = os.path.expanduser(
    '~/ros2_ws/src/26fall_cse276a/robot_vision/config/camera_parameter.yaml')  # your calibration

# YOLO (COCO) class name -> real height in meters (measured)
HEIGHTS = {
    'bottle':     0.20,
    'keyboard':   0.15,    # standing up
    'wine glass': 0.215,
    'hair drier': 0.22,
    'cell phone': 0.15,    # standing up
    'book':       0.185,
}

# camera matrix K and distortion D from the calibration yaml
fs = cv2.FileStorage(CAMERA_YAML, cv2.FILE_STORAGE_READ)
K = fs.getNode('camera_matrix').mat()
D = fs.getNode('distortion_coefficients').mat()
fs.release()


def distance(box, height):
    """Distance (m) straight ahead from the top / bottom of the box, and left (m)."""
    x1, y1, x2, y2 = box
    xc = (x1 + x2) / 2
    pts = np.array([[[xc, y1]], [[xc, y2]]], dtype=np.float64)
    (xt, yt), (xb, yb) = cv2.undistortPoints(pts, K, D).reshape(-1, 2)   # pixels -> x/z, y/z
    z = height / (yb - yt)
    left = -(xt + xb) / 2 * z
    return z, left


# download the model into test/models/ the first time
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    YOLO('yolo26n.pt')                                   # downloads to the current folder
    shutil.move('yolo26n.pt', MODEL_PATH)

model = YOLO(MODEL_PATH)
os.makedirs(RESULTS_DIR, exist_ok=True)

for f in sorted(glob.glob(os.path.join(PHOTOS_DIR, '*.jpg'))):
    result = model(f, imgsz=1280, conf=0.25, verbose=False)[0]
    img_h = result.orig_shape[0]

    found = []
    for box in result.boxes:
        name = model.names[int(box.cls)]
        if name not in HEIGHTS:
            continue
        xyxy = box.xyxy[0].tolist()
        text = f'{name} {float(box.conf):.2f}'
        if xyxy[1] < 5 or xyxy[3] > img_h - 5:
            text += ' (cut off, no distance)'           # box touches top/bottom: height wrong
        else:
            z, left = distance(xyxy, HEIGHTS[name])
            text += f' {z:.2f} m ahead, {left:+.2f} m left'
        found.append(text)

    print(f'{os.path.basename(f)}:')
    for t in found or ['nothing']:
        print(f'    {t}')
    cv2.imwrite(os.path.join(RESULTS_DIR, os.path.basename(f)), result.plot())

print(f'\nphotos with boxes saved in {RESULTS_DIR}')