"""
Motion calibration shared by hw4_control (commands) and the localizer (odometry).

The robot can only do two things:
    drive straight  -> [DRIVE_CMD, DRIVE_CMD]; distance vs time from DRIVE_TABLE
    turn in place   -> spin_commands(sign, BASE_TURN) for a time looked up in a table of
                       measured (angle, seconds) points (CCW_TABLE / CW_TABLE)
"""

# ---- driving straight ----
DRIVE_CMD = 0.2           # wheel command for driving straight

# Measured (meters, seconds) at DRIVE_CMD.
# Between points: straight-line interpolation; outside: the line is extended.
DRIVE_TABLE = [(0.1, 0.55), (1.0, 3.9)]

# ---- wheel gains (multiply BASE_TURN for each wheel / direction) ----
L_fwd = 0.93
L_rev = 0.94
R_fwd = 1.08
R_rev = 1

# ---- turning ----
BASE_TURN = 0.35             # command for the timed turn

# Measured (degrees, seconds) at BASE_TURN with the gains above.
# Between points: straight-line interpolation; outside: the end segment is extended.
CCW_TABLE = [(45, 0.6), (90, 0.95), (135, 1.4), (180, 1.8)]   # left turns
CW_TABLE = [(45, 0.5), (90, 0.8), (135, 1.2), (180, 1.5)]    # right turns

# NOTE: CW 90 and 135 sit on opposite sides of a straight line by ~0.07 s; re-measure.

MIN_TURN_DEG = 3.0          # smaller turns are skipped

RAMP_SECONDS = 0.15         # ramp every command up from 0 over this time


def table_time(deg, table):
    """Seconds to turn `deg` degrees: piecewise-linear through the measured points,
    with the first / last segment extended for angles outside the table."""
    pts = sorted(table)
    if deg <= pts[0][0]:
        (a0, t0), (a1, t1) = pts[0], pts[1]
    elif deg >= pts[-1][0]:
        (a0, t0), (a1, t1) = pts[-2], pts[-1]
    else:
        (a0, t0), (a1, t1) = next(p for p in zip(pts, pts[1:]) if p[0][0] <= deg <= p[1][0])
    return max(0.0, t0 + (deg - a0) * (t1 - t0) / (a1 - a0))


def spin_commands(sign, speed):
    """Wheel commands for spinning in place. sign > 0: left (CCW), sign < 0: right (CW)."""
    if sign > 0:
        return -speed * L_rev, speed * R_fwd
    return speed * L_fwd, -speed * R_rev
