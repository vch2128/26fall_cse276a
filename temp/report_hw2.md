# CSE 276A – Homework 2: Closed-Loop Waypoint Navigation with Object Landmarks

Team: TODO names · Video: TODO unlisted YouTube URL

We use **option 2**: the landmarks are everyday objects recognized by a learned
object detector (YOLOv8n, COCO-pretrained). We use no AprilTags.

## Coordinate frames and start pose

* **World frame**: origin at the start position, x = the robot's initial heading,
  y = 90° to the left, z up. Units: meters, radians (CCW positive).
* **Vehicle frame**: origin at the rotation center (midpoint between the wheels), x forward, y left.
* **Sensor (camera) frame**: OpenCV convention, x right, y down, z along the optical axis.
  The camera sits `CAM_X` = TODO m in front of the rotation center and looks forward. A point
  (X, Y, Z) in the camera frame is at (CAM_X + Z, −X) in the vehicle frame.
* **Start pose**: (0, 0, 0) in the world frame, which is the first line of `hw2_waypoints.txt`.

## (a) Closed-loop controller

```
 reference             pose error         [L, R] wheel cmd                 robot motion
 waypoints ──▶(+)──▶ Waypoint controller ───────────────▶ motor_control ──▶ Robot ──┐
              ▲ −     (drive / turn / look)                                         │
              │ estimated pose (x, y, θ)                                            │ camera image
              │                                                                     ▼
             EKF  ◀── landmarks: (class, range, bearing) ── landmark_detector ◀── camera
              ▲
              └── commanded motion: distance driven / angle turned (prediction)
```

| Node | Input → output |
|---|---|
| `robot_vision_camera` | → 1280×720 JPEG images, 10 Hz (`/camera/image_raw/compressed`) |
| `landmark_detector` | image → class, range, and bearing for each visible landmark (`/landmarks`) |
| `velocity_control` (EKF) | commanded motions + landmark measurements + map → pose estimate and covariance |
| `velocity_control` (controller) | pose error to the current waypoint → `drive(d)` / `turn(α)` → wheel commands `[L, R]` (`/motor_commands`) |
| `motor_control` | wheel commands → serial commands to the motor board |

The controller is built on our HW1 motion primitives: `drive(d)` is a timed straight drive at
a calibrated speed, and `turn(α)` is a timed spin with its duration taken from a measured
angle–time table. HW2 closes the loop around them. For each waypoint, the controller repeats
the following until the estimate is within 5 cm of the waypoint:
1. **Look**: stand still for 1 s and update the pose estimate with the landmarks in view.
2. **Turn** to face the waypoint, using the estimated heading.
3. **Drive** toward the waypoint, at most 0.5 m at a time, then go back to step 1.

Then the robot turns to the waypoint heading θ, looks again, and turns the remaining
amount until the heading error is below 4°. When the waypoint is less than 0.3 m away and
behind the robot, the robot backs up to it instead of turning 180°.

## (b) Landmarks

The landmarks are TODO (e.g. bottle, chair, teddy bear, tv), all COCO classes. They sit at:

| Landmark | World position (m) | Seen during |
|---|---|---|
| TODO | (2, 0) | leg 1 (0,0)→(1,0), and the final heading 0 at (0,0) |
| TODO | (1, 3) | leg 2 (1,0)→(1,2) |
| TODO | (−1, 2) | the pose (1,2,π) |
| TODO | (−0.5, −1) | leg 3 (1,2)→(0,0) |

**Why this layout:**
* Each landmark lies roughly straight ahead on one leg, so at least one landmark is in the
  camera's field of view (about 105°) during every drive and at every waypoint.
* A landmark straight ahead gives the heading error with the most accurate measurement
  (bearing), and the range shrinks as the robot approaches. Range error is proportional to range.
* Every landmark is at least about 0.7 m from the nearest waypoint, so it is never cut off
  at the image border.
* Each landmark has a different class, so data association is unambiguous.

## (c) Pose estimation and noise

**Measurement.** YOLO returns a bounding box. We undistort the top and bottom midpoints of
the box with the calibrated intrinsics to get normalized coordinates y_t and y_b. The depth
is Z = H / (y_b − y_t), where H is the object's known (effective) height, and the lateral
offset is X = x_n·Z. Converting to the vehicle frame gives the range r and bearing b.
The effective height is calibrated by placing each object 1.0 m from the camera.

**Filter.** We use an extended Kalman filter on (x, y, θ):
* **Predict**: the HW1 kinematic model, x += d·cos θ, y += d·sin θ, θ += α, where d and α
  are the commanded drive distance and turn angle. Each motion is spread over its duration
  so the trajectory can be logged every 0.5 s. Odometry noise grows like a random walk
  with the distance driven and the angle turned.
* **Update**: for each detection, a range–bearing model to the landmark's mapped position.
  Range noise is σ_r = 10 % of r, because box height is noisy. Bearing noise is
  σ_b = 3°, because the box center is accurate.

**Handling noise and outliers.**
* **Mahalanobis gate** (χ², 99 %): drops false detections and boxes that don't fit the prediction.
* **Cut-off boxes**: boxes touching the top or bottom image border are discarded,
  because their range would be wrong.
* **Images taken while moving**: only images taken while the robot stands still (during
  `look()`) are used. Images taken while moving are blurred, and the detector's latency of
  about 0.3 s would put them at the wrong pose.
* **Re-localization**: six gated-out measurements in a row mean the estimate is wrong, not
  the camera. In that case the covariance is inflated so the landmarks are accepted again.

**Why an EKF.** Using a single landmark's pose directly is noisy, especially in range, and
fails when nothing is visible. The EKF weights each source by its uncertainty. It averages
many noisy detections while the robot is stopped. It fuses range-only-accurate and
bearing-accurate information from different landmarks. And it falls back to odometry
naturally, with honest uncertainty, when no landmark is visible.

## (d) No landmark in view

When no landmark is detected, the EKF only runs the prediction step. This is exactly
HW1's dead reckoning, and the covariance grows. The controller keeps working on that
estimate. Because the robot looks again after every 0.5 m segment and after every turn,
dead-reckoning error can only build up over short stretches. When a landmark reappears,
the larger covariance gives the measurement a large weight, so the estimate snaps back.
The landmark layout guarantees a landmark is visible at every waypoint. If the drift became
so large that the gate rejected everything, re-localization (see (c)) recovers.

## (e) Detection algorithm (option 2)

* **Motivation**: a pretrained detector recognizes ordinary objects without any markers,
  and YOLOv8n is small enough to run on the Rubik Pi CPU (TODO measured fps at imgsz TODO).
* **Details**: YOLOv8n trained on COCO (80 classes), with a confidence threshold of 0.4.
  Only classes in our map are used. Each frame is processed when it arrives. Frames that
  arrive during inference are dropped (queue depth 1), so we always use the newest image.
* **Robustness**: TODO. For example: detection rate per landmark at 0.5–3 m, false
  positives observed, and how lighting or viewing angle affects detection. The gate
  removes most false positives.
* **Accuracy**: TODO table of true distance vs. measured range and bearing
  (e.g. at 0.5, 1, 2, 3 m, and ±30° off-axis). Range error is about TODO %,
  bearing error is about TODO°.

## (f) Smoothness

TODO, based on the video and trajectory. By design, the robot moves stop-and-go: it drives
0.5 m segments and stops for 1 s to look after every segment and turn. This trades
smoothness and time for accuracy, because every measurement is taken from a sharp,
current image. Small heading corrections between segments show up as short turns.
Larger `SEGMENT` or shorter `LOOK_SECONDS` would make the motion smoother but less accurate.

## (g) Total moving distance

Unit = TODO m (distance between the first and second landmark, ≥ 0.5 m).
From `plot_trajectory.py`: translation = TODO m (TODO units), rotation = TODO rad.
Ideal path: 1 + √5 + 2.24 ≈ … (TODO).

## (h) Waypoint errors

Measured on the floor against the estimated pose and the goal:

| Waypoint | Goal | Estimated | True (measured) | Translation error | Rotation error |
|---|---|---|---|---|---|
| 1 | (1, 0, 0) | TODO | TODO | TODO | TODO |
| 2 | (1, 2, π) | TODO | TODO | TODO | TODO |
| 3 | (0, 0, 0) | TODO | TODO | TODO | TODO |

## (i) Trajectory

![trajectory](trajectory.png)

The red dots are the EKF estimate every 0.5 s, the blue crosses are waypoints, and the
green squares are landmarks.

## (j) Reflection

TODO. Points to discuss:
* Accuracy compared with HW1's open-loop control.
* Which part limits accuracy most: range from box height, calibration of the wheel model,
  or latency.
* Time spent on retries.
* What we would improve next, such as using the IMU yaw in the prediction, compensating
  for image latency, or a detector trained on our specific objects.
