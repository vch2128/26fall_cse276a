#!/usr/bin/env python3
"""AprilTag landmark detector (drop-in replacement for the YOLO detector_node.py).

Same node file, same `ros2 run landmark_nav detector` command and the same output, so
setup.py, launch files, the localizer and the controller do not change.

Subscribes:  camera/image_raw (sensor_msgs/Image), camera/camera_info
Publishes:   landmark_obs (std_msgs/String, JSON):
    {"stamp": 1729800000.12,
     "obs": [{"id": "tag3", "r": 1.23, "b": 0.10,
              "sr": 0.03, "sb": 0.017, "conf": 85.2}, ...]}
    id     = "tag<ID>" (tag36h11 family by default)
    r      = horizontal distance camera -> tag center (m)
    b      = bearing, radians, + = tag is to the LEFT of the camera axis
    sr, sb = 1-sigma uncertainty of r and b
    conf   = AprilTag decision margin (higher = more certain; not used by the localizer)

Tag size: the side of the BLACK square of the printed tag, in meters (measure it after printing).
It is read from landmarks.yaml if that file has a top-level `tag_size:` line, otherwise from the
`tag_size` parameter. The YOLO parameters (model, imgsz, conf) are still accepted and ignored,
so old launch commands keep working.

Backend: OpenCV's ArUco module, which has the AprilTag 36h11 dictionary built in (no extra
install). `-p backend:=pupil` uses pupil-apriltags instead (pip install pupil-apriltags), but that
package crashed with memory errors in our tests. Either way the tag position comes from
cv2.solvePnP on the four corners of the undistorted image.

Run:
    ros2 run landmark_nav detector --ros-args -p tag_size:=0.16
"""
import json
import math

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


class LandmarkDetector(Node):
    def __init__(self):
        super().__init__('landmark_detector')
        p = self.declare_parameter
        self.tag_size = p('tag_size', 0.169).value                # m, black square side
        self.backend = p('backend', 'opencv').value              # 'opencv' (default) or 'pupil'
        self.min_margin = p('min_decision_margin', 25.0).value   # pupil only: weaker detections are often false
        self.border = p('border_px', 5).value                    # drop tags this close to the image edge
        self.sr_per_m = p('sigma_r_per_m', 0.02).value           # range noise grows with distance
        self.sb = p('sigma_b', math.radians(1.0)).value
        ids = p('tag_ids', [-1]).value                           # [-1] = accept every tag id
        self.tag_ids = None if list(ids) == [-1] else set(int(i) for i in ids)
        lm_file = p('landmarks_file', '').value
        for unused in ('model', 'imgsz', 'conf', 'min_box_h_px', 'sigma_px'):   # old YOLO parameters
            p(unused, '')

        if lm_file:                                              # optional: tag_size from landmarks.yaml
            cfg = yaml.safe_load(open(lm_file)) or {}
            if 'tag_size' in cfg:
                self.tag_size = float(cfg['tag_size'])
        self.get_logger().info(f'AprilTag 36h11 detector ({self.backend}): tag size {self.tag_size} m, '
                               f'ids {"all" if self.tag_ids is None else sorted(self.tag_ids)}')

        if self.backend == 'pupil':
            from pupil_apriltags import Detector
            self.det = Detector(families='tag36h11', nthreads=1)
        else:
            d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            prm = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, 'DetectorParameters')
                   else cv2.aruco.DetectorParameters_create())
            prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX   # sub-pixel corners -> better range
            if hasattr(cv2.aruco, 'ArucoDetector'):                       # OpenCV >= 4.7
                det = cv2.aruco.ArucoDetector(d, prm)
                self.aruco = det.detectMarkers
            else:                                                         # older OpenCV
                self.aruco = lambda img: cv2.aruco.detectMarkers(img, d, parameters=prm)
        h = self.tag_size / 2                                    # tag corners in the tag frame (IPPE_SQUARE order)
        self.obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)

        self.K = None
        self.maps = None                                         # undistortion maps, built once
        self.create_subscription(CameraInfo, 'camera/camera_info', self.on_info, 10)
        # depth 1: if detection is slower than the camera, we always process the newest frame
        self.create_subscription(Image, 'camera/image_raw', self.on_image, 1)
        self.pub = self.create_publisher(String, 'landmark_obs', 10)

    # ------------------------------------------------------------------ camera
    def on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            D = np.array(msg.d, dtype=np.float64)
            # our lens distortion is strong (k1 = -0.32): undistort the whole image, keeping K
            self.maps = cv2.initUndistortRectifyMap(self.K, D, None, self.K,
                                                    (msg.width, msg.height), cv2.CV_16SC2)
            self.get_logger().info(f'camera K =\n{self.K}\nD = {D}')

    @staticmethod
    def to_gray(msg):
        if msg.encoding in ('mono8', '8UC1'):
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        img = img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        code = cv2.COLOR_RGB2GRAY if msg.encoding == 'rgb8' else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(img, code)

    # --------------------------------------------------------------- detection
    def detect(self, gray):
        """[(tag_id, 4x2 corners TL TR BR BL, confidence)] in the undistorted image."""
        if self.backend == 'pupil':
            out = []
            for t in self.det.detect(gray):
                if t.hamming == 0 and t.decision_margin >= self.min_margin:   # weak decode: often a false tag
                    # pupil corners go counter-clockwise from bottom-left -> reorder to TL TR BR BL
                    c = t.corners
                    out.append((t.tag_id, c[[3, 2, 1, 0]], float(t.decision_margin)))
            return out
        corners, ids, _ = self.aruco(gray)
        if ids is None:
            return []
        return [(int(i), c.reshape(4, 2), 1.0) for c, i in zip(corners, ids.flatten())]

    def on_image(self, msg):
        if self.K is None:
            return
        gray = cv2.remap(self.to_gray(msg), *self.maps, cv2.INTER_LINEAR)
        H, W = gray.shape
        b = self.border
        obs = []
        for tag_id, c, conf in self.detect(gray):
            if self.tag_ids is not None and tag_id not in self.tag_ids:
                continue
            if c[:, 0].min() < b or c[:, 1].min() < b or c[:, 0].max() > W - b or c[:, 1].max() > H - b:
                continue                                         # cut off -> pose would be wrong
            ok, _, tvec = cv2.solvePnP(self.obj, c.astype(np.float64), self.K, None,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            X, _, Z = tvec.flatten()                             # camera frame: x right, y down, z forward
            r = math.hypot(X, Z)
            bearing = math.atan2(-X, Z)                          # + = left (matches the robot's CCW-positive frame)
            obs.append({'id': f'tag{tag_id}', 'r': round(r, 4), 'b': round(bearing, 5),
                        'sr': round(self.sr_per_m * r, 4), 'sb': round(self.sb, 5),
                        'conf': round(conf, 1)})

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        out = String()
        out.data = json.dumps({'stamp': stamp, 'obs': obs})
        self.pub.publish(out)
        if obs:
            self.get_logger().info(
                ' | '.join(f"{o['id']}: r={o['r']:.2f} b={math.degrees(o['b']):+.1f}deg" for o in obs),
                throttle_duration_sec=1.0)


def main():
    rclpy.init()
    node = LandmarkDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()