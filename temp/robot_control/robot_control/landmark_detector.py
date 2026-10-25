#!/usr/bin/env python3
"""
CSE 276A HW2 - landmark detector (object detection, no AprilTags).

camera/image_raw/compressed -> YOLO (COCO classes) -> range + bearing of every landmark -> /landmarks

Only classes listed in landmarks.txt are used. Range comes from the known object height:
    the top and bottom of the box are undistorted to normalized image coordinates yn
    depth  Z = height / (yn_bottom - yn_top)
    side   X = xn_center * Z                      (camera frame: x right, z forward)
Robot frame (x forward, y left, origin = rotation center):
    forward = CAM_X + Z,  left = -X,  range = hypot,  bearing = atan2(left, forward)

/landmarks is a std_msgs/String with JSON:
    {"stamp": <image time, s>, "detections": [{"cls": "bottle", "r": 1.23, "b": 0.10, "conf": 0.81}]}
/landmarks/debug/compressed is the image with boxes (view it in Foxglove).

Run:
    ros2 run robot_control landmark_detector
    ros2 run robot_control landmark_detector --ros-args -p model:=yolov8n.pt -p imgsz:=416
"""
import json
import math
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from ultralytics import YOLO

FOLDER = os.path.expanduser('~/ros2_ws/src/26fall_cse276a/robot_control/config')
CAMERA_YAML = os.path.expanduser(
    '~/ros2_ws/src/26fall_cse276a/robot_vision/config/camera_parameter.yaml')

CAM_X = 0.08        # m, camera in front of the rotation center (TODO: measure)
BORDER = 5          # px, boxes touching the top / bottom edge are cut off -> wrong range


def read_heights(name):
    """landmarks.txt: class, x, y, height_m  ->  {class: height}"""
    heights = {}
    for line in open(os.path.join(FOLDER, name)):
        if line.strip() and not line.startswith('#'):
            cls, _x, _y, h = [s.strip() for s in line.split(',')]
            heights[cls] = float(h)
    return heights


class LandmarkDetector(Node):
    def __init__(self):
        super().__init__('landmark_detector')
        self.declare_parameter('model', 'yolov8n.pt')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('landmarks', 'landmarks.txt')
        self.declare_parameter('camera_yaml', CAMERA_YAML)
        self.declare_parameter('use_distortion', True)
        self.declare_parameter('debug_image', True)

        self.imgsz = self.get_parameter('imgsz').value
        self.conf = self.get_parameter('conf').value
        self.debug_image = self.get_parameter('debug_image').value
        self.heights = read_heights(self.get_parameter('landmarks').value)

        fs = cv2.FileStorage(self.get_parameter('camera_yaml').value, cv2.FILE_STORAGE_READ)
        self.K = fs.getNode('camera_matrix').mat()
        self.dist = fs.getNode('distortion_coefficients').mat()
        fs.release()
        if not self.get_parameter('use_distortion').value:
            self.dist = np.zeros(5)

        self.model = YOLO(self.get_parameter('model').value)
        unknown = [c for c in self.heights if c not in self.model.names.values()]
        if unknown:
            self.get_logger().error(f'not a class of this model: {unknown}')

        self.pub = self.create_publisher(String, 'landmarks', 10)
        self.debug_pub = self.create_publisher(CompressedImage, 'landmarks/debug/compressed', 1)
        # depth 1: while YOLO runs, old frames are dropped instead of queued
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CompressedImage, 'camera/image_raw/compressed',
                                 self.on_image, qos)
        self.get_logger().info(f'landmark classes: {self.heights}')

    def normalized(self, pts):
        """Pixels -> undistorted normalized coordinates (x/z, y/z)."""
        pts = np.array(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist).reshape(-1, 2)

    def on_image(self, msg):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        h_img = img.shape[0]
        result = self.model(img, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]

        detections = []
        for box in result.boxes:
            cls = self.model.names[int(box.cls)]
            if cls not in self.heights:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf)
            cut = y1 < BORDER or y2 > h_img - BORDER
            if not cut:
                xc = (x1 + x2) / 2
                (_, yt), (_, yb), (xn, _) = self.normalized([[xc, y1], [xc, y2], [xc, (y1 + y2) / 2]])
                z = self.heights[cls] / max(yb - yt, 1e-6)
                fwd, left = CAM_X + z, -xn * z
                r, b = math.hypot(fwd, left), math.atan2(left, fwd)
                detections.append({'cls': cls, 'r': round(r, 4), 'b': round(b, 4),
                                   'conf': round(conf, 3)})
                self.get_logger().info(f'{cls}: r={r:.2f} m  b={math.degrees(b):+.1f} deg  '
                                       f'box h={y2 - y1:.0f} px')
            if self.debug_image:
                color = (0, 0, 255) if cut else (0, 255, 0)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f'{cls} {conf:.2f}' + ('' if cut else f' r={r:.2f} b={math.degrees(b):+.0f}')
                cv2.putText(img, label, (int(x1), max(15, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        out = String()
        out.data = json.dumps({'stamp': stamp, 'detections': detections})
        self.pub.publish(out)

        if self.debug_image:
            dbg = CompressedImage()
            dbg.header = msg.header
            dbg.format = 'jpeg'
            dbg.data = cv2.imencode('.jpg', cv2.resize(img, None, fx=0.5, fy=0.5))[1].tobytes()
            self.debug_pub.publish(dbg)


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
