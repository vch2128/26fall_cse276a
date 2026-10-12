#!/usr/bin/env python3
"""
Offline test for velocity_control.py - no ROS, no robot needed (runs on a laptop).

It imports the real WaypointNavigator, replaces rclpy / the messages with tiny
stubs, and closes the loop with a simulated skid-steer robot:
    motor_commands [L, R]  ->  simulated robot  ->  imu_data (yaw quaternion)

The simulated robot has a deadzone, a first-order motor lag (so it coasts and
overshoots a little), and you can make its true speed differ from DRIVE_SPEED
to see what a bad calibration does.

    python3 tools/sim_waypoints.py                          # waypoints.txt
    python3 tools/sim_waypoints.py test_square.txt
    python3 tools/sim_waypoints.py --speed-error 0.1        # robot 10 % faster than calibrated
    python3 tools/sim_waypoints.py --imu-sign -1            # IMU mounted the other way
    python3 tools/sim_waypoints.py --plot sim.png           # save the path (needs matplotlib)
    python3 tools/sim_waypoints.py --set TURN_LEAD=0.1 --set K_TURN=0.8   # try other parameters
"""
import argparse
import math
import os
import sys
import types

HERE = os.path.dirname(os.path.realpath(__file__))
PKG = os.path.join(HERE, '..', 'robot_control')


# ---------------- stubs for rclpy and the message types ----------------
class Msg:
    def __init__(self):
        self.data = []
        self.orientation = types.SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0)


class Logger:
    def __init__(self, sim):
        self.sim = sim

    def info(self, text):
        print(f'[{self.sim.t:6.2f}s] {text}')

    warn = warning = error = info


class FakeNode:
    sim = None    # set before the node is created

    def __init__(self, name):
        self._logger = Logger(self.sim)

    def get_logger(self):
        return self._logger

    def create_publisher(self, msg_type, topic, qos):
        return types.SimpleNamespace(publish=self.sim.on_motor_command)

    def create_subscription(self, msg_type, topic, callback, qos):
        self.sim.imu_callback = callback

    def create_timer(self, period, callback):
        self.sim.timers.append((period, callback))


def install_stubs():
    rclpy = types.ModuleType('rclpy')
    rclpy_node = types.ModuleType('rclpy.node')
    rclpy_node.Node = FakeNode
    rclpy.node = rclpy_node
    std_msgs = types.ModuleType('std_msgs')
    std_msgs_msg = types.ModuleType('std_msgs.msg')
    std_msgs_msg.Float32MultiArray = Msg
    sensor_msgs = types.ModuleType('sensor_msgs')
    sensor_msgs_msg = types.ModuleType('sensor_msgs.msg')
    sensor_msgs_msg.Imu = Msg
    sys.modules.update({
        'rclpy': rclpy, 'rclpy.node': rclpy_node,
        'std_msgs': std_msgs, 'std_msgs.msg': std_msgs_msg,
        'sensor_msgs': sensor_msgs, 'sensor_msgs.msg': sensor_msgs_msg,
    })


# ---------------- simulated robot ----------------
class SimRobot:
    DT = 0.005          # physics step [s]
    TRACK = 0.40        # effective wheel separation [m] (skid steer -> much larger than real)
    TAU = 0.15          # motor time constant [s]
    DEADZONE = 0.08     # |cmd| below this does not move the wheel

    def __init__(self, vc, speed_error, imu_sign, start):
        self.vc = vc
        # wheel speed per command so that DRIVE_CMD gives (1 + speed_error) * DRIVE_SPEED
        self.gain = vc.DRIVE_SPEED * (1 + speed_error) / (vc.DRIVE_CMD - self.DEADZONE)
        self.imu_sign = imu_sign
        self.t = 0.0
        self.x, self.y, self.yaw = start
        self.yaw0 = 0.3     # the IMU's zero is NOT the start heading
        self.vl = self.vr = 0.0
        self.cmd = (0.0, 0.0)
        self.last_cmd_time = 0.0
        self.timers = []
        self.imu_callback = None
        self.path = [(self.x, self.y)]

    def on_motor_command(self, msg):
        # motor_control clips to +-0.5
        self.cmd = tuple(max(-0.5, min(0.5, v)) for v in msg.data[:2])
        self.last_cmd_time = self.t

    def wheel_speed(self, cmd):
        if abs(cmd) < self.DEADZONE:
            return 0.0
        return self.gain * (cmd - math.copysign(self.DEADZONE, cmd))

    def step(self):
        cmd = self.cmd if self.t - self.last_cmd_time < 0.15 else (0.0, 0.0)  # motor_control timeout
        a = self.DT / self.TAU
        self.vl += a * (self.wheel_speed(cmd[0]) - self.vl)
        self.vr += a * (self.wheel_speed(cmd[1]) - self.vr)
        v = (self.vl + self.vr) / 2
        w = (self.vr - self.vl) / self.TRACK
        self.x += v * math.cos(self.yaw) * self.DT
        self.y += v * math.sin(self.yaw) * self.DT
        self.yaw += w * self.DT
        self.t += self.DT
        self.path.append((self.x, self.y))

    def publish_imu(self):
        yaw = self.imu_sign * (self.yaw - self.yaw0)
        msg = Msg()
        msg.orientation = types.SimpleNamespace(w=math.cos(yaw / 2), x=0.0, y=0.0, z=math.sin(yaw / 2))
        self.imu_callback(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file', nargs='?', default='waypoints.txt')
    ap.add_argument('--speed-error', type=float, default=0.0,
                    help='true speed / calibrated speed - 1  (e.g. 0.1 = 10%% faster)')
    ap.add_argument('--imu-sign', type=float, default=1.0,
                    help='sign of the simulated IMU (set IMU_SIGN in the code to match)')
    ap.add_argument('--set', action='append', default=[], metavar='NAME=VALUE',
                    help='override a constant in velocity_control.py, e.g. TURN_LEAD=0.1')
    ap.add_argument('--timeout', type=float, default=180.0)
    ap.add_argument('--plot', help='save a picture of the path to this file')
    args = ap.parse_args()

    install_stubs()
    sys.path.insert(0, PKG)
    from robot_control import velocity_control as vc
    for item in args.set:
        name, value = item.split('=')
        assert hasattr(vc, name), f'velocity_control has no constant {name}'
        setattr(vc, name, float(value))

    path = args.file if os.path.isfile(args.file) else os.path.join(PKG, 'config', args.file)
    waypoints = vc.load_waypoints(path)
    sim = SimRobot(vc, args.speed_error, args.imu_sign, waypoints[0])
    FakeNode.sim = sim
    vc.time = types.SimpleNamespace(time=lambda: sim.t)    # the node uses the simulated clock

    node = vc.WaypointNavigator(path)
    period, loop = sim.timers[0]
    next_loop = next_imu = 0.0
    reached = {}
    done_at = {}         # waypoint -> time its last motion ended
    end = args.timeout
    while sim.t < end:
        if node.done and end == args.timeout:
            end = sim.t + 0.5                    # let the robot settle after the last one
        if sim.t >= next_imu:
            sim.publish_imu()
            next_imu += 0.02                     # IMU at 50 Hz
        if sim.t >= next_loop:
            loop()
            next_loop += period
            # a waypoint is finished when the node has no motion left for it
            if node.task is None and not node.tasks and node.wp_index > 0:
                done_at.setdefault(node.wp_index, sim.t)
        # measure the pose 0.5 s later, after the robot stopped coasting (= PAUSE)
        for i, t in done_at.items():
            if i not in reached and sim.t >= t + 0.5:
                reached[i] = (sim.x, sim.y, sim.yaw)
        sim.step()

    print('\nwaypoint   target (x, y, th)          true pose (x, y, th)        pos err   ang err')
    pos_errs = []
    for i, (tx, ty, tth) in enumerate(waypoints[1:], start=1):
        if i not in reached:
            print(f'{i:>4}   ({tx:5.2f}, {ty:5.2f}, {tth:5.2f})   NOT REACHED')
            continue
        x, y, th = reached[i]
        pe = math.hypot(x - tx, y - ty)
        ae = math.degrees(vc.wrap(th - tth))
        pos_errs.append(pe)
        print(f'{i:>4}   ({tx:5.2f}, {ty:5.2f}, {tth:5.2f})   ({x:5.2f}, {y:5.2f}, {vc.wrap(th):5.2f})'
              f'   {pe * 100:5.1f} cm  {ae:+6.1f} deg')
    if pos_errs:
        print(f'mean position error {sum(pos_errs) / len(pos_errs) * 100:.1f} cm, '
              f'total time {sim.t:.1f} s')
    if not node.done:
        print(f'TIMEOUT after {args.timeout} s - the robot got stuck (check signs / TURN_MIN)')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        xs, ys = zip(*sim.path)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(xs, ys, label='simulated robot')
        for i, (tx, ty, tth) in enumerate(waypoints):
            ax.annotate('', xy=(tx + 0.15 * math.cos(tth), ty + 0.15 * math.sin(tth)), xytext=(tx, ty),
                        arrowprops=dict(arrowstyle='->', color='C3'))
            ax.text(tx + 0.04, ty + 0.04, str(i), color='C3')
        ax.plot([w[0] for w in waypoints], [w[1] for w in waypoints], 'o', color='C3', label='waypoints')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f'saved {args.plot}')
    return 0 if node.done else 1


if __name__ == '__main__':
    sys.exit(main())
