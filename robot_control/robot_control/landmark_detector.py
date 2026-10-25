#!/usr/bin/env python3
"""
CSE 276A HW2 - find the landmark objects in the camera image with YOLO.

camera/image_raw/compressed -> YOLO -> where each landmark is, seen from the robot -> landmarks

For every object in landmarks.txt the best box is used. Its distance comes from the known
height of the object (top / bottom of the box undistorted to normalized coordinates y/z):
    ahead    Z = height / (y_bottom - y_top)
    right    X = x_center * Z
    robot frame (x forward, y left, origin = rotation center): forward = CAM_X + Z, left = -X

Published on 'landmarks' as JSON text (std_msgs/String), name: [forward m, left m]:
    {"stamp": 1761300000.12, "objects": {"bottle": [1.52, 0.31], "book": [2.03, -0.40]}}

Run (camera must be running):
    ros2 run robot_control landmark_detector
"""
import json
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from ultralytics import YOLO

from robot_control.hw2_control import read_landmarks

CAMERA_YAML = os.path.expanduser(
    '~/ros2_ws/src/26fall_cse276a/robot_vision/config/camera_parameter.yaml')
MODEL = 'yolo11n.pt'    # COCO-pretrained, downloaded automatically the first time
IMGSZ = 640             # YOLO input size; 960 or 1280 finds farther objects but is slower
MIN_CONF = 0.4
CAM_X = 0.0             # m, camera in front of the rotation center   TODO: measure
EDGE = 5                # px, a box touching the top / bottom edge is cut off: wrong height


class LandmarkDetector(Node):
    def __init__(self):
        super().__init__('landmark_detector')
        self.heights = {name: h for name, (_x, _y, h) in read_landmarks().items()}

        fs = cv2.FileStorage(CAMERA_YAML, cv2.FILE_STORAGE_READ)
        self.K = fs.getNode('camera_matrix').mat()
        self.D = fs.getNode('distortion_coefficients').mat()
        fs.release()

        self.model = YOLO(MODEL)
        unknown = [n for n in self.heights if n not in self.model.names.values()]
        if unknown:
            self.get_logger().error(f'not a YOLO class name: {unknown}')

        self.pub = self.create_publisher(String, 'landmarks', 10)
        # depth 1: while YOLO runs, old frames are dropped instead of queued
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CompressedImage, 'camera/image_raw/compressed',
                                 self.on_image, qos)
        self.get_logger().info(f'looking for {list(self.heights)}')

    def on_image(self, msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        result = self.model(img, imgsz=IMGSZ, conf=MIN_CONF, verbose=False)[0]

        best = {}                                   # name -> (conf, box), one per landmark
        for box in result.boxes:
            name = self.model.names[int(box.cls)]
            conf = float(box.conf)
            if name in self.heights and conf > best.get(name, (0.0,))[0]:
                best[name] = (conf, box.xyxy[0].tolist())

        objects = {}
        for name, (conf, (x1, y1, x2, y2)) in best.items():
            if y1 < EDGE or y2 > img.shape[0] - EDGE:
                continue
            xc = (x1 + x2) / 2
            pts = np.array([[[xc, y1]], [[xc, y2]]], dtype=np.float64)
            (xt, yt), (xb, yb) = cv2.undistortPoints(pts, self.K, self.D).reshape(-1, 2)
            z = self.heights[name] / (yb - yt)
            forward, left = CAM_X + z, -(xt + xb) / 2 * z
            objects[name] = [round(forward, 3), round(left, 3)]
            self.get_logger().info(f'{name:11s} conf {conf:.2f}  forward {forward:.2f} m  '
                                   f'left {left:+.2f} m  box {y2 - y1:.0f} px')

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.pub.publish(String(data=json.dumps({'stamp': stamp, 'objects': objects})))


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
