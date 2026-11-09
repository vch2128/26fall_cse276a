"""EKF-SLAM for a unicycle robot with range-bearing landmark measurements.

State  x = [xr, yr, thr, l1x, l1y, l2x, l2y, ...]   (robot pose, then one (x, y) per landmark)
Covariance P is (3 + 2n) x (3 + 2n) and grows when a new landmark is added.

The map frame is the robot's START pose: x0 = [0, 0, 0] with P0 = 0, because the
robot does not know where it starts. To compare with the measured map, transform
the estimate with the measured start pose afterwards (tools/evaluate_map.py).

Models (same as Peter Corke's Robotics Toolbox Unicycle / RangeBearingSensor,
written out here so F, G, H can be shown in the report and the file runs
without the toolbox):
    motion       xr' = xr + ds cos(thr),  yr' = yr + ds sin(thr),  thr' = thr + dth
    measurement  z = [r, b] from the CAMERA, which sits `a` m ahead of the robot centre:
                 cx = xr + a cos(thr), cy = yr + a sin(thr)
                 r = sqrt((lx-cx)^2 + (ly-cy)^2),  b = atan2(ly-cy, lx-cx) - thr
    new landmark lx = cx + r cos(thr + b),  ly = cy + r sin(thr + b)

Landmarks are identified by the detector's id (YOLO class or AprilTag id), so
data association is known; the filter does not assume how many there are.
No ROS imports, so it can be tested on a laptop (tools/sim_slam.py).
"""
import math

import numpy as np


def wrap(a):
    """Angle to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class SlamEKF:
    def __init__(self, cam_offset=0.0, P0_robot=(1e-6, 1e-6, 1e-6)):
        self.a = cam_offset
        self.x = np.zeros(3)                       # robot starts at the origin of its own map
        self.P = np.diag(P0_robot).astype(float)   # (almost) zero: the start pose defines the frame
        self.ids = []                              # landmark id -> position in self.ids

    # ------------------------------------------------------------------ helpers
    @property
    def n(self):
        """Number of landmarks in the map."""
        return len(self.ids)

    def idx(self, lid):
        """Index of landmark lid's x in the state vector (y is idx + 1)."""
        return 3 + 2 * self.ids.index(lid)

    def landmarks(self):
        """{id: (x, y, 2x2 covariance)} for every landmark in the map."""
        out = {}
        for lid in self.ids:
            i = self.idx(lid)
            out[lid] = (self.x[i], self.x[i + 1], self.P[i:i + 2, i:i + 2].copy())
        return out
    '''
    def camera(self):
        """Camera position in the map frame."""
        xr, yr, th = self.x[:3]
        return xr + self.a * math.cos(th), yr + self.a * math.sin(th)
    '''
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

    # ------------------------------------------------------------------ new landmark
    def add_landmark(self, lid, z, R):
        """First sighting: place the landmark from the current pose and z, and extend x and P.

        l = g(x_robot, z);  Gx = dg/dx_robot (2x3),  Gz = dg/dz (2x2)
        P_ll = Gx P_rr Gx^T + Gz R Gz^T,   P_l,* = Gx P_r,*
        """
        r, b = z
        xr, yr, th = self.x[:3]
        c, s = math.cos(th), math.sin(th)
        ca, sa = math.cos(th + b), math.sin(th + b)
        lx = xr + self.a * c + r * ca
        ly = yr + self.a * s + r * sa
        Gx = np.array([[1.0, 0.0, -self.a * s - r * sa],
                       [0.0, 1.0, self.a * c + r * ca]])
        Gz = np.array([[ca, -r * sa],
                       [sa, r * ca]])

        N = len(self.x)
        P_new = np.zeros((N + 2, N + 2))
        P_new[:N, :N] = self.P
        cross = Gx @ self.P[:3, :]                               # 2 x N
        P_new[N:, :N] = cross
        P_new[:N, N:] = cross.T
        P_new[N:, N:] = Gx @ self.P[:3, :3] @ Gx.T + Gz @ R @ Gz.T
        self.x = np.concatenate([self.x, [lx, ly]])
        self.P = P_new
        self.ids.append(lid)

    # ------------------------------------------------------------------ update
    def update(self, lid, z, R, gate=9.21):
        """One observation z = [r, b] of landmark lid.

        Unknown id -> add it to the map (returns 'new').
        Known id   -> EKF update with a Mahalanobis gate (9.21 = 99%, 2 DOF):
                      returns 'ok' or 'rejected'.  Also returns d2 (None for 'new').
        """
        z = np.asarray(z, dtype=float)
        R = np.asarray(R, dtype=float)
        if lid not in self.ids:
            self.add_landmark(lid, z, R)
            return 'new', None
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
