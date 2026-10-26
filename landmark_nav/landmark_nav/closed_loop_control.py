#!/usr/bin/env python3
"""HW2 closed-loop waypoint controller.

Same turn -> drive -> turn motions as HW1 (it reuses turn()/drive() from
robot_control.motion unchanged), but instead of trusting its own
dead-reckoned (x, y, heading), it stops after each motion, reads the
localizer's pose estimate from robot_pose, and re-plans / corrects until the
waypoint is reached within tolerance.
If no landmark is seen for MAX_MISSES stops in a row, it turns in place
SEARCH_STEP at a time (at most one full circle) until one is in view again.

Run (motor_control, camera, detector and localizer must be running):
    ros2 run landmark_nav closed_loop_control waypoints_hw2.txt
"""
import json
import math
import sys

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String

from robot_control.motion import Motion, angle_diff, read_waypoints

POS_TOL = 0.08                 # m: close enough to the waypoint
ANG_TOL = math.radians(4)      # rad: close enough to the final heading (> MIN_TURN_DEG)
STEP = 0.25                    # m: drive at most this far, then stop and look again
MAX_STEPS = 30                 # safety: at most this many steps per waypoint
MAX_TRIES = 5                  # heading correction attempts per waypoint
SETTLE = 2.0                   # s standing still before reading the pose (detector gets several frames)
NEAR = 0.15                    # m: closer than this, correct along the heading only (no turning); must be > POS_TOL
MAX_MISSES = 2                 # stops in a row without seeing any landmark -> search
SEARCH_STEP = math.radians(45) # rad: search turn per look (45 deg is a measured point in the turn table)

node = None
motion = None                  # Motion bound to this node (set in main)
pose = None                    # latest (x, y, theta, sigma_x, sigma_y, sigma_theta)
last_seen = 0.0                # ROS time of the last image that had a landmark in it
seen = False                   # did the last settle_and_read() see a landmark?


def on_pose(msg):
    global pose
    q = msg.pose.pose.orientation
    c = msg.pose.covariance
    pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, 2.0 * math.atan2(q.z, q.w),
            math.sqrt(c[0]), math.sqrt(c[7]), math.sqrt(c[35]))


def on_obs(msg):
    global last_seen
    data = json.loads(msg.data)
    if data['obs']:
        last_seen = max(last_seen, data['stamp'])


def ros_now():
    return node.get_clock().now().nanoseconds * 1e-9


def settle_and_read():
    global seen
    t = ros_now()
    motion.stop(SETTLE)            # publishes zeros and spins ROS, so on_pose / on_obs keep running
    seen = last_seen > t + 0.5     # a landmark in an image taken after the robot settled
    return pose


def search():
    """No landmark for a while: turn SEARCH_STEP at a time and look, at most one full circle.
    Returns True when a landmark is in view again (the localizer then corrects the pose)."""
    turned = 0.0
    while turned < 2 * math.pi - 1e-6:
        motion.turn(SEARCH_STEP)
        turned += SEARCH_STEP
        settle_and_read()
        if seen:
            print(f'  search: landmark found after {math.degrees(turned):.0f} deg')
            return True
    print('  search: nothing in a full circle, continuing on dead reckoning')
    return False


def main():
    global node, motion
    rclpy.init()
    node = rclpy.create_node('closed_loop_control')
    motion = Motion(node)
    node.create_subscription(PoseWithCovarianceStamped, 'robot_pose', on_pose, 10)
    node.create_subscription(String, 'landmark_obs', on_obs, 10)

    name = sys.argv[1] if len(sys.argv) > 1 else 'waypoints_hw2.txt'
    waypoints = read_waypoints(name)
    while pose is None and rclpy.ok():
        node.get_logger().info('waiting for robot_pose ...', throttle_duration_sec=2.0)
        motion.spin_for(0.1)

    results = []
    try:
        for gx, gy, gth in waypoints[1:]:
            # --- position: small steps toward the goal, look after every step
            misses, gave_up = 0, False
            for _ in range(MAX_STEPS):
                x, y, th, *_ = settle_and_read()
                misses = 0 if seen else misses + 1
                if misses >= MAX_MISSES and not gave_up:     # lost: look around for a landmark
                    gave_up = not search()                     # nothing in a full circle -> don't search again here
                    misses = 0
                    x, y, th, *_ = settle_and_read()           # pose corrected by the landmark found
                dist = math.hypot(gx - x, gy - y)
                if dist < POS_TOL:
                    break
                ahead = (gx - x) * math.cos(th) + (gy - y) * math.sin(th)     # goal distance along the heading
                side = -(gx - x) * math.sin(th) + (gy - y) * math.cos(th)     # goal distance to the side
                if dist < NEAR and abs(side) < POS_TOL:
                    motion.drive(ahead)                                       # forward or backward, no turn
                    continue
                motion.turn(angle_diff(math.atan2(gy - y, gx - x), th))
                x, y, th, *_ = settle_and_read()                              # re-aim with a fresh estimate
                motion.turn(angle_diff(math.atan2(gy - y, gx - x), th))       # skipped if < MIN_TURN_DEG
                motion.drive(min(math.hypot(gx - x, gy - y), STEP))           # at most STEP, then look again

            # --- heading: turn to the waypoint's theta, re-check, repeat
            for _ in range(MAX_TRIES):
                x, y, th, *_ = settle_and_read()
                err = angle_diff(gth, th)
                if abs(err) < ANG_TOL:
                    break
                motion.turn(err)

            x, y, th, sx, sy, sth = settle_and_read()
            results.append((gx, gy, gth, x, y, th))
            print(f'reached ({gx}, {gy}, {gth:.2f})  est=({x:.3f}, {y:.3f}, {th:.3f})  '
                  f'sigma=({sx:.3f}, {sy:.3f}, {math.degrees(sth):.1f}deg)')

        print('\nwaypoint  target -> estimated (also tape-measure the real pose!)')
        for gx, gy, gth, x, y, th in results:
            print(f'({gx}, {gy}, {gth:.2f}) -> ({x:.3f}, {y:.3f}, {th:.3f})  '
                  f'pos err {math.hypot(gx - x, gy - y):.3f} m, '
                  f'ang err {math.degrees(angle_diff(gth, th)):+.1f} deg')
    except KeyboardInterrupt:
        pass
    finally:
        try:
            motion.stop(0.2)
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()