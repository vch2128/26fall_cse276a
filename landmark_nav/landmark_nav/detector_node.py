#!/usr/bin/env python3
"""YOLO landmark detector.

Subscribes:  camera/image_raw (sensor_msgs/Image), camera/camera_info
Publishes:   landmark_obs (std_msgs/String, JSON):
    {"stamp": 1729800000.12,
     "obs": [{"id": "teddy bear", "r": 1.23, "b": 0.10,
              "sr": 0.05, "sb": 0.006, "conf": 0.87}, ...]}
    r  = horizontal distance camera -> landmark (m)
    b  = bearing, radians, + = landmark is to the LEFT of the camera axis
    sr, sb = 1-sigma uncertainty of r and b

Only classes listed in landmarks.yaml are reported, at most one box per class
(the most confident one), so each class must be a unique object in the room.

Run:
    ros2 run landmark_nav detector --ros-args -p landmarks_file:=/home/ubuntu/ros2_ws/src/26fall_cse276a/landmark_nav/config/landmarks.yaml -p model:=yolo26n.pt
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
        self.model_path = p('model', 'yolo26n.pt').value
        self.imgsz = p('imgsz', 1280).value            # YOLO input size; 320 is much faster on the Pi
        self.conf_min = p('conf', 0.25).value
        self.border = p('border_px', 5).value          # drop boxes this close to the image edge
        self.min_h = p('min_box_h_px', 12).value       # drop tiny (far / unreliable) boxes
        self.sigma_px = p('sigma_px', 3.0).value       # pixel noise of a box edge (tune from data)
        self.image_is_rgb = p('image_is_rgb', False).value  # camera node says "rgb8"; YOLO wants BGR
        self.rectified = p('image_rectified', False).value  # true if the camera node already undistorts
        lm_file = p('landmarks_file', '').value

        lm = yaml.safe_load(open(lm_file))['landmarks']
        self.heights = {name: float(v['height']) for name, v in lm.items()}
        self.get_logger().info(f'landmark classes: {list(self.heights)}')

        from ultralytics import YOLO                   # imported here so the node starts fast on errors above
        self.model = YOLO(self.model_path)

        self.K = None
        self.D = None
        self.create_subscription(CameraInfo, 'camera/camera_info', self.on_info, 10)
        # depth 1: if YOLO is slower than the camera, we always process the newest frame
        self.create_subscription(Image, 'camera/image_raw', self.on_image, 1)
        self.pub = self.create_publisher(String, 'landmark_obs', 10)

    # ------------------------------------------------------------------ camera
    def on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.D = np.zeros(5) if self.rectified else np.array(msg.d, dtype=np.float64)
            self.get_logger().info(f'camera K =\n{self.K}\nD = {self.D}')

    @staticmethod
    def to_numpy(msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        return img[:, :msg.width * 3].reshape(msg.height, msg.width, 3)

    # --------------------------------------------------------------- detection
    def on_image(self, msg):
        if self.K is None:
            return
        img = self.to_numpy(msg)
        if self.image_is_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]

        # Run YOLO on the raw (distorted) image; only the box corners get undistorted.
        res = self.model.predict(img, imgsz=self.imgsz, conf=self.conf_min, verbose=False)[0]

        best = {}
        b = self.border
        for (x1, y1, x2, y2), conf, cls in zip(res.boxes.xyxy.tolist(),
                                               res.boxes.conf.tolist(),
                                               res.boxes.cls.tolist()):
            name = res.names[int(cls)]
            if name not in self.heights:
                continue
            if x1 < b or y1 < b or x2 > W - b or y2 > H - b:
                continue                                # cut off -> height (and range) would be wrong
            if name in best and best[name]['conf'] >= conf:
                continue
            obs = self.measure(name, x1, y1, x2, y2)
            if obs is not None:
                obs['conf'] = round(conf, 3)
                best[name] = obs

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        out = String()
        out.data = json.dumps({'stamp': stamp, 'obs': list(best.values())})
        self.pub.publish(out)
        if best:
            self.get_logger().info(
                ' | '.join(f"{o['id']}: r={o['r']:.2f} b={math.degrees(o['b']):+.1f}deg"
                           for o in best.values()),
                throttle_duration_sec=1.0)

    def measure(self, name, x1, y1, x2, y2):
        """Range + bearing from the box height, assuming a level camera."""
        uc = 0.5 * (x1 + x2)
        pts = np.array([[[uc, y1]], [[uc, y2]]], dtype=np.float64)       # top-center, bottom-center
        (ut, vt), (ub, vb) = cv2.undistortPoints(pts, self.K, self.D, P=self.K).reshape(-1, 2)
        h = vb - vt
        if h < self.min_h:
            return None
        fx, fy, cx = self.K[0, 0], self.K[1, 1], self.K[0, 2]
        u = 0.5 * (ut + ub)

        Z = fy * self.heights[name] / h          # depth along the optical axis (pinhole: h = fy*H/Z)
        X = (u - cx) * Z / fx                     # lateral offset, + = right in the image
        r = math.hypot(Z, X)
        bearing = math.atan2(-X, Z)               # + = left (matches the robot's CCW-positive frame)

        # Error propagation: r ~ 1/h, so sigma_r = r * sigma_h / h, with sigma_h from two edges.
        sr = r * math.sqrt(2.0) * self.sigma_px / h
        sb = self.sigma_px / fx
        return {'id': name, 'r': round(r, 4), 'b': round(bearing, 5),
                'sr': round(sr, 4), 'sb': round(sb, 5)}


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
