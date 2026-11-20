"""EKF for a unicycle robot with range-bearing measurements of KNOWN landmarks (AprilTags).

State  x = [xr, yr, thr, l1x, l1y, l2x, l2y, ...]   (robot pose, then one (x, y) per tag)
The tags are put in the state by the localizer from the map (map_hw4.yaml) with a small
covariance, so the filter localizes the robot in the known map.

Models (same as Peter Corke's Robotics Toolbox Unicycle / RangeBearingSensor,
written out here so F, H can be shown in the report and the file runs
without the toolbox):
    motion       xr' = xr + ds cos(thr),  yr' = yr + ds sin(thr),  thr' = thr + dth
    measurement  z = [r, b] from the CAMERA, which sits `a` m ahead of the robot centre:
                 cx = xr + a cos(thr), cy = yr + a sin(thr)
                 r = sqrt((lx-cx)^2 + (ly-cy)^2),  b = atan2(ly-cy, lx-cx) - thr

Tags are identified by the detector's id, so data association is known.
No ROS imports, so it can be tested on a laptop (sim_hw4.py).
"""
import math

import numpy as np


def wrap(a):
    """Angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class LandmarkEKF:
    def __init__(self, cam_offset=0.0):
        self.a = cam_offset
        self.x = np.zeros(3)                       # start pose and tags: set by the localizer (map)
        self.P = np.zeros((3, 3))
        self.ids = []                              # tag id -> position in self.ids

    # ------------------------------------------------------------------ helpers
    @property
    def n(self):
        """Number of tags in the state."""
        return len(self.ids)

    def idx(self, lid):
        """Index of tag lid's x in the state vector (y is idx + 1)."""
        return 3 + 2 * self.ids.index(lid)

    def inflate_robot(self, var_xy, var_th):
        """Add uncertainty to the robot pose only (used when the filter seems lost)."""
        self.P[0, 0] += var_xy
        self.P[1, 1] += var_xy
        self.P[2, 2] += var_th

    # ------------------------------------------------------------------ predict
    def predict(self, ds, dth, var_ds, var_dth):
        """Odometry step. Only the robot part of x changes; landmarks are static.

        F = d f / d x_robot  (3x3),  G = d f / d (ds, dth)  (3x2),  V = diag(var_ds, var_dth)
        P_rr <- F P_rr F^T + G V G^T,   P_rm <- F P_rm   (robot-landmark cross terms)
        """
        th = self.x[2]
        c, s = math.cos(th), math.sin(th)
        F = np.array([[1.0, 0.0, -ds * s],
                      [0.0, 1.0, ds * c],
                      [0.0, 0.0, 1.0]])
        G = np.array([[c, 0.0],
                      [s, 0.0],
                      [0.0, 1.0]])
        self.x[0] += ds * c
        self.x[1] += ds * s
        self.x[2] = wrap(th + dth)

        Prr = self.P[:3, :3]
        self.P[:3, :3] = F @ Prr @ F.T + G @ np.diag([var_ds, var_dth]) @ G.T
        if self.n:
            self.P[:3, 3:] = F @ self.P[:3, 3:]
            self.P[3:, :3] = self.P[:3, 3:].T

    # ------------------------------------------------------------------ measurement model
    def expected(self, lid):
        """Expected z = [r, b] of a known landmark, and H (2 x len(x)), non-zero only
        in the robot columns and that landmark's two columns."""
        xr, yr, th = self.x[:3]
        c, s = math.cos(th), math.sin(th)
        cx, cy = xr + self.a * c, yr + self.a * s
        i = self.idx(lid)
        dx, dy = self.x[i] - cx, self.x[i + 1] - cy
        q = dx * dx + dy * dy
        r = math.sqrt(q)
        z_hat = np.array([r, wrap(math.atan2(dy, dx) - th)])

        H = np.zeros((2, len(self.x)))
        # w.r.t. the robot (camera offset chained in through d(cx, cy)/d theta = (-a s, a c))
        H[0, 0:3] = [-dx / r, -dy / r, self.a * (dx * s - dy * c) / r]
        H[1, 0:3] = [dy / q, -dx / q, -self.a * (dx * c + dy * s) / q - 1.0]
        # w.r.t. the landmark
        H[0, i:i + 2] = [dx / r, dy / r]
        H[1, i:i + 2] = [-dy / q, dx / q]
        return z_hat, H

    # ------------------------------------------------------------------ update
    def update(self, lid, z, R, gate=9.21):
        """One observation z = [r, b] of tag lid (must be in self.ids).
        EKF update with a Mahalanobis gate (9.21 = 99%, 2 DOF): returns 'ok' or 'rejected', and d2.
        """
        z = np.asarray(z, dtype=float)
        R = np.asarray(R, dtype=float)
        z_hat, H = self.expected(lid)
        nu = np.array([z[0] - z_hat[0], wrap(z[1] - z_hat[1])])
        S = H @ self.P @ H.T + R
        d2 = float(nu @ np.linalg.solve(S, nu))
        if d2 > gate:
            return 'rejected', d2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[2] = wrap(self.x[2])
        IKH = np.eye(len(self.x)) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T               # Joseph form keeps P symmetric / PSD
        return 'ok', d2
