#!/usr/bin/env python3
"""
Calibrate the camera from checkerboard frames and write camera_parameter.yaml
in the format robot_vision_camera reads (OpenCV FileStorage, %YAML:1.0).

Follows the OpenCV tutorial
(https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html):
    findChessboardCorners -> cornerSubPix -> calibrateCamera
    -> getOptimalNewCameraMatrix -> undistort -> re-projection error (projectPoints)
and adds: per-image error, dropping bad images, and writing the YAML.

Runs anywhere with Python + OpenCV (no ROS):  pip install opencv-python numpy

    python3 calibrate_camera.py calib_frames                         # 9x6 inner corners, 25 mm squares
    python3 calibrate_camera.py calib_frames --cols 9 --rows 6 --square 0.025
    python3 calibrate_camera.py calib_frames --drop-above 1.0        # drop images worse than 1 px, recalibrate
    python3 calibrate_camera.py calib_frames --model rational        # 8 coefficients; only if plumb_bob
                                                                     # leaves edges curved / error high
    python3 calibrate_camera.py calib_frames --show                  # look at corners + undistorted images

Output:
    camera_parameter.yaml       copy over robot_vision/config/camera_parameter.yaml, rebuild, relaunch
    <frames>/undistorted/       every frame undistorted, to check by eye (straight edges must look straight)

About projection_matrix: robot_vision_camera rectifies with
    newK = getOptimalNewCameraMatrix(K, D, size, alpha=0)
but publishes P from this file, and apriltag_ros takes its intrinsics from P.
So P is written as [newK | 0] with alpha = 0, which matches the rectified image exactly.
"""
import argparse
import glob
import math
import os
import sys

import cv2
import numpy as np

# termination criteria for cornerSubPix (as in the tutorial)
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def close_windows():
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass                    # headless OpenCV build (no GUI) - nothing to close


def find_corners(paths, pattern, square, show):
    """Return object points, image points, used paths and the image size."""
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (cols-1,rows-1,0), times square size
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square

    objpoints, imgpoints, used, size = [], [], [], None
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f'  skip {os.path.basename(path)}: cannot read')
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]            # (width, height)
        elif gray.shape[::-1] != size:
            print(f'  skip {os.path.basename(path)}: size {gray.shape[::-1]} != {size}')
            continue

        ret, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not ret:
            print(f'  skip {os.path.basename(path)}: board not found')
            continue
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        objpoints.append(objp)
        imgpoints.append(corners2)
        used.append(path)

        if show:
            cv2.drawChessboardCorners(img, pattern, corners2, ret)
            cv2.imshow('corners (any key = next, q = stop showing)', img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                show = False
    close_windows()
    return objpoints, imgpoints, used, size


def per_image_errors(objpoints, imgpoints, K, D, rvecs, tvecs):
    """RMS re-projection error of every image, in pixels."""
    errs = []
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        d = imgpoints[i].reshape(-1, 2) - imgpoints2.reshape(-1, 2)
        errs.append(math.sqrt((d ** 2).sum(axis=1).mean()))
    return errs


def calibrate(objpoints, imgpoints, size, flags):
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None, flags=flags)
    # keep 5 (plumb_bob) or 8 (rational_polynomial) coefficients, the lengths ROS camera_info expects
    n = 8 if flags & cv2.CALIB_RATIONAL_MODEL else 5
    return rms, K, D.ravel()[:n].reshape(1, -1), rvecs, tvecs


def coverage_report(imgpoints, size):
    """How much of the image the corners cover (3x3 grid); distortion at the edges needs data there."""
    w, h = size
    grid = np.zeros((3, 3), int)
    for pts in imgpoints:
        for x, y in pts.reshape(-1, 2):
            grid[min(int(y * 3 / h), 2), min(int(x * 3 / w), 2)] += 1
    lines = ['  ' + ' '.join(f'{n:5d}' for n in row) for row in grid]
    empty = int((grid == 0).sum())
    if empty:
        lines.append(f'  WARNING: {empty} of 9 image regions have no corners - distortion there is a guess')
    return '\n'.join(lines)


def write_yaml(path, name, size, K, D, P, model):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write('image_width', int(size[0]))
    fs.write('image_height', int(size[1]))
    fs.write('camera_name', name)
    fs.write('camera_matrix', K)
    fs.write('distortion_model', model)
    fs.write('distortion_coefficients', D.reshape(1, -1))
    fs.write('rectification_matrix', np.eye(3))
    fs.write('projection_matrix', P)
    fs.release()
    # newer OpenCV writes '%YAML 1.2'; use the '%YAML:1.0' header the robot's (older) OpenCV writes itself
    with open(path) as f:
        text = f.read()
    if text.startswith('%YAML') and not text.startswith('%YAML:1.0'):
        text = '%YAML:1.0' + text[text.index('\n'):]
        with open(path, 'w') as f:
            f.write(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('frames', help='folder with the calibration images (png/jpg)')
    ap.add_argument('--cols', type=int, default=9, help='inner corners along the long side')
    ap.add_argument('--rows', type=int, default=6, help='inner corners along the short side')
    ap.add_argument('--square', type=float, default=0.025, help='square edge in meters')
    ap.add_argument('--model', choices=['plumb_bob', 'rational'], default='plumb_bob',
                    help='plumb_bob = k1 k2 p1 p2 k3 (standard); rational = 8 coefficients, for wide lenses')
    ap.add_argument('--drop-above', type=float, default=0.0,
                    help='drop images whose error is above this many px and calibrate again')
    ap.add_argument('--out', default='camera_parameter.yaml')
    ap.add_argument('--name', default='camera')
    ap.add_argument('--show', action='store_true', help='show detected corners and undistorted images')
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.frames, '*.png')) + glob.glob(os.path.join(args.frames, '*.jpg')))
    if not paths:
        sys.exit(f'no .png / .jpg images in {args.frames}')
    pattern = (args.cols, args.rows)

    print(f'1) finding {pattern[0]}x{pattern[1]} corners in {len(paths)} images')
    objpoints, imgpoints, used, size = find_corners(paths, pattern, args.square, args.show)
    print(f'   board found in {len(used)} / {len(paths)} images, image size {size[0]}x{size[1]}')
    if len(used) < 10:
        sys.exit('   need at least 10 good images (20-40 recommended)')
    print('   corners per image region (3x3 grid):')
    print(coverage_report(imgpoints, size))

    flags = cv2.CALIB_RATIONAL_MODEL if args.model == 'rational' else 0
    print('2) calibrating')
    rms, K, D, rvecs, tvecs = calibrate(objpoints, imgpoints, size, flags)
    errs = per_image_errors(objpoints, imgpoints, K, D, rvecs, tvecs)

    if args.drop_above > 0:
        keep = [i for i, e in enumerate(errs) if e <= args.drop_above]
        dropped = [os.path.basename(used[i]) for i in range(len(used)) if i not in keep]
        if dropped and len(keep) >= 10:
            print(f'   dropping {len(dropped)} images above {args.drop_above} px: {", ".join(dropped)}')
            objpoints = [objpoints[i] for i in keep]
            imgpoints = [imgpoints[i] for i in keep]
            used = [used[i] for i in keep]
            rms, K, D, rvecs, tvecs = calibrate(objpoints, imgpoints, size, flags)
            errs = per_image_errors(objpoints, imgpoints, K, D, rvecs, tvecs)
        elif dropped:
            print(f'   not dropping: only {len(keep)} images would be left')

    print('3) re-projection error per image (px):')
    for path, e in sorted(zip(used, errs), key=lambda t: -t[1]):
        flag = '   <- check this one' if e > 2 * np.median(errs) and e > 0.5 else ''
        print(f'   {os.path.basename(path):24s} {e:6.3f}{flag}')
    print(f'   overall RMS: {rms:.3f} px   (good: < 0.5, acceptable: < 1.0)')

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = D.ravel()
    print('4) result')
    print(f'   fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}')
    print(f'   distortion ({args.model}): {np.array2string(d, precision=5, suppress_small=True)}')
    print(f'   field of view: {math.degrees(2 * math.atan(size[0] / (2 * fx))):.1f} deg horizontal, '
          f'{math.degrees(2 * math.atan(size[1] / (2 * fy))):.1f} deg vertical (pinhole estimate)')
    if abs(d[2]) > 0.01 or abs(d[3]) > 0.01:
        print('   WARNING: tangential terms p1/p2 are large - usually means poor coverage or a bent board')
    if abs(cx - size[0] / 2) > 0.1 * size[0] or abs(cy - size[1] / 2) > 0.1 * size[1]:
        print('   WARNING: principal point is far from the image center - check coverage')

    # new camera matrix, exactly as robot_vision_camera computes it for rectification (alpha = 0)
    newK, roi = cv2.getOptimalNewCameraMatrix(K, D, size, 0, size)
    P = np.hstack([newK, np.zeros((3, 1))])
    write_yaml(args.out, args.name, size, K, D, P, 'rational_polynomial' if args.model == 'rational' else 'plumb_bob')
    print(f'5) wrote {args.out}  (projection_matrix = rectified intrinsics, alpha 0)')

    # undistort every frame for a visual check (tutorial: cv.undistort with newcameramtx)
    out_dir = os.path.join(args.frames, 'undistorted')
    os.makedirs(out_dir, exist_ok=True)
    for path in used:
        img = cv2.imread(path)
        dst = cv2.undistort(img, K, D, None, newK)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(path)), dst)
        if args.show:
            cv2.imshow('original | undistorted (any key = next, q = stop)', np.hstack([img, dst]))
            if cv2.waitKey(0) & 0xFF == ord('q'):
                args.show = False
    close_windows()
    print(f'6) undistorted frames in {out_dir} - straight edges near the image border should look straight')


if __name__ == '__main__':
    main()
