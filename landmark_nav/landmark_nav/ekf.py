"""Pose EKF for a unicycle robot with range-bearing landmark measurements.

State: [x, y, theta] in the world (map) frame.

The motion and sensor models (and their Jacobians) come from Peter Corke's
Robotics Toolbox:
    Unicycle.f / Fx / Fv            x' = x + [dd cos(th), dd sin(th), dth]
    RangeBearingSensor.h / Hx       z  = [range, bearing] to a map landmark
rtb's own EKF class can't be used directly: its step() pulls odometry and
readings from a *simulated* robot and sensor, so this class runs the same
predict / update equations with our real ROS data instead.
No ROS imports, so it can be unit-tested and simulated on a laptop.
"""
import numpy as np
from roboticstoolbox.mobile import LandmarkMap, RangeBearingSensor, Unicycle
from spatialmath.base import wrap_mpi_pi


class PoseEKF:
    def __init__(self, x0, P0, landmarks, cam_offset=0.0):
        """landmarks: {name: (x, y)} in the world frame.
        cam_offset: camera is this many meters ahead of the robot center."""
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.a = cam_offset

        self.names = list(landmarks)                                   # name -> index in the rtb map
        self.robot = Unicycle()                                        # only its motion model is used
        self.map = LandmarkMap(np.array([landmarks[n] for n in self.names], dtype=float).T)
        self.sensor = RangeBearingSensor(self.robot, self.map)         # only its h / Hx are used

    def predict(self, ds, dtheta, var_ds, var_dtheta):
        """Motion step with odometry (ds forward, dtheta turned) and its variances V."""
        odo = (ds, dtheta)
        Fx = self.robot.Fx(self.x, odo)
        Fv = self.robot.Fv(self.x, odo)
        self.x = self.robot.f(self.x, odo)
        self.x[2] = wrap_mpi_pi(self.x[2])
        self.P = Fx @ self.P @ Fx.T + Fv @ np.diag([var_ds, var_dtheta]) @ Fv.T

    def expected(self, name, x=None):
        """Expected [range, bearing] to landmark `name`, and the Jacobian H w.r.t. the robot state.

        rtb's sensor sits at the robot center, so we evaluate it at the camera pose
        and chain-rule back to the robot pose (J = d camera_pose / d robot_pose).
        """
        x = self.x if x is None else x
        c, s = np.cos(x[2]), np.sin(x[2])
        cam = x + np.array([self.a * c, self.a * s, 0.0])
        J = np.array([[1.0, 0.0, -self.a * s],
                      [0.0, 1.0, self.a * c],
                      [0.0, 0.0, 1.0]])
        i = self.names.index(name)
        return self.sensor.h(cam, i), self.sensor.Hx(cam, i) @ J

    def update(self, z, R, name, gate=9.21):
        """Measurement step with one observation z = [range, bearing] of landmark `name`.

        gate: Mahalanobis threshold (9.21 = 99% for 2 DOF); observations further
        than this from what we expect are treated as outliers and ignored.
        Returns (accepted, mahalanobis_distance_squared).
        """
        z_hat, H = self.expected(name)
        nu = np.array([z[0] - z_hat[0], wrap_mpi_pi(z[1] - z_hat[1])])   # innovation
        S = H @ self.P @ H.T + R
        d2 = float(nu @ np.linalg.solve(S, nu))
        if d2 > gate:
            return False, d2
        K = self.P @ H.T @ np.linalg.inv(S)                              # Kalman gain
        self.x = self.x + K @ nu
        self.x[2] = wrap_mpi_pi(self.x[2])
        IKH = np.eye(3) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T                      # Joseph form, as in rtb's EKF
        return True, d2
