"""Offline checks for the EKF (no ROS needed):  python3 -m pytest test/test_ekf.py"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from spatialmath.base import wrap_mpi_pi as wrap  # noqa: E402

from landmark_nav.ekf import PoseEKF  # noqa: E402


def test_jacobian_matches_finite_difference():
    ekf = PoseEKF([0.3, -0.2, 0.7], np.eye(3), {'tv': (2.0, 1.0)}, cam_offset=0.08)
    _, H = ekf.expected('tv')
    eps = 1e-6
    for i in range(3):
        xp = ekf.x.copy(); xp[i] += eps
        xm = ekf.x.copy(); xm[i] -= eps
        zp, _ = ekf.expected('tv', xp)
        zm, _ = ekf.expected('tv', xm)
        num = np.array([zp[0] - zm[0], wrap(zp[1] - zm[1])]) / (2 * eps)
        assert np.allclose(H[:, i], num, atol=1e-5)


def test_zero_offset_matches_plain_rtb_sensor():
    """With the camera at the robot center, expected() is exactly rtb's h / Hx."""
    ekf = PoseEKF([0.3, -0.2, 0.7], np.eye(3), {'tv': (2.0, 1.0)}, cam_offset=0.0)
    z, H = ekf.expected('tv')
    assert np.allclose(z, ekf.sensor.h(ekf.x, 0))
    assert np.allclose(H, ekf.sensor.Hx(ekf.x, 0))


def test_landmarks_pull_drifted_estimate_back():
    """Drive the HW2 path with biased odometry; the EKF should end much closer than dead reckoning."""
    rng = np.random.default_rng(0)
    lms = {'a': (2.2, 0.3), 'b': (1.3, 3.0), 'c': (-1.0, 2.0), 'd': (-0.6, -1.2)}
    true = np.array([0.0, 0.0, 0.0])
    ekf = PoseEKF(true, np.diag([1e-4, 1e-4, 1e-3]), lms, cam_offset=0.0)
    dr = true.copy()
    fov = math.radians(35)

    def step(ds, dth):
        nonlocal true, dr
        true = np.array([true[0] + ds * math.cos(true[2] + dth / 2),
                         true[1] + ds * math.sin(true[2] + dth / 2), wrap(true[2] + dth)])
        mds, mdth = ds * 1.08, dth * 1.05 + 0.002      # biased odometry
        dr = np.array([dr[0] + mds * math.cos(dr[2] + mdth / 2),
                       dr[1] + mds * math.sin(dr[2] + mdth / 2), wrap(dr[2] + mdth)])
        ekf.predict(mds, mdth, 0.05 ** 2 * abs(mds) + 1e-8, 0.05 ** 2 * abs(mdth) + 0.03 ** 2 * abs(mds) + 1e-8)
        for name, (lx, ly) in lms.items():
            r = math.hypot(lx - true[0], ly - true[1])
            b = wrap(math.atan2(ly - true[1], lx - true[0]) - true[2])
            if abs(b) < fov and r < 3.5:
                z = [r * (1 + rng.normal(0, 0.04)), b + rng.normal(0, math.radians(1))]
                ekf.update(z, np.diag([(0.04 * r) ** 2, math.radians(1.5) ** 2]), name)

    path = [(0, 0, 0), (1, 0, 0), (1, 2, math.pi), (0, 0, 0)]
    for (gx, gy, gth) in path[1:]:
        face = math.atan2(gy - true[1], gx - true[0])
        for _ in range(20):
            step(0.0, wrap(face - true[2]) / 20)
        d = math.hypot(gx - true[0], gy - true[1])
        for _ in range(40):
            step(d / 40, 0.0)
        for _ in range(20):
            step(0.0, wrap(gth - true[2]) / 20)

    ekf_err = math.hypot(*(ekf.x[:2] - true[:2]))
    dr_err = math.hypot(*(dr[:2] - true[:2]))
    print(f'final position error: dead reckoning {dr_err:.3f} m, EKF {ekf_err:.3f} m')
    assert ekf_err < 0.5 * dr_err
    assert ekf_err < 0.15   # open-loop drive with 8% / 5% odometry bias; the real controller also corrects
