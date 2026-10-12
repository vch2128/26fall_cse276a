# CSE 276A – Homework 1: Waypoint Navigation

Team: TODO names · Video: TODO unlisted YouTube URL

## 1. Algorithm

The robot visits the waypoints in `waypoints.txt` in order. The first line is the
start pose. For every following waypoint `(x, y, θ)` the planner creates three motions:

1. **Turn in place** to face the waypoint, `φ = atan2(y − ŷ, x − x̂)`.
2. **Drive straight** for the distance `d = ‖(x, y) − (x̂, ŷ)‖`.
3. **Turn in place** to the requested heading `θ`.

The robot stops for 0.5 s between motions so it is at rest before the next one starts.

### Kinematic model

The Waverover is a skid-steer (differential-drive) vehicle, modelled as a unicycle:

```
v = (v_R + v_L) / 2          ω = (v_R − v_L) / b
ẋ = v cos θ                  ẏ = v sin θ                  θ̇ = ω
```

The robot has no wheel encoders. Position is dead-reckoned with the calibrated
forward speed (`x += v cos θ Δt`, `y += v sin θ Δt`, with `v = DRIVE_SPEED` while
driving and 0 while turning in place). Heading θ comes from the on-board IMU yaw,
offset so that it equals the start heading at t = 0.

## 2. Control

* **Turning** – proportional control on the IMU heading error,
  `u = clip(K_TURN·|e|, TURN_MIN, TURN_MAX)`, with wheel commands `(−u, +u)`.
  `TURN_MIN` is the smallest command that overcomes skid friction. The robot keeps
  rotating briefly after the motors stop, so each turn stops `TURN_LEAD` early.
  The robot then settles, re-measures the heading and corrects (up to
  `MAX_CORRECTIONS` times) until the error is below `ANGLE_TOL` (≈3°).
* **Driving** – constant forward command `DRIVE_CMD`, plus a small P steering term
  `K_HEADING·(bearing to waypoint − θ)`, limited to ±0.1. This corrects the drift
  caused by unequal motors. The motion ends when the integrated distance reaches `d`.
* The loop runs at 20 Hz and publishes `[L, R]` on `/motor_commands`. `motor_control`
  stops the motors if no command arrives for 0.15 s.

## 3. Code

| File | Purpose |
|---|---|
| `robot_control/robot_control/velocity_control.py` | waypoint planner + controller node (`ros2 run robot_control velocity_control [file]`) |
| `robot_control/robot_control/calibrate.py` | calibration node: IMU sign, straight-line speed, spin rate / coast |
| `robot_control/robot_control/motor_control.py` | provided serial driver (motor commands, IMU publisher) |
| `robot_control/config/*.txt` | `waypoints.txt` and test paths (`test_turn`, `test_line`, `test_square`) |
| `tools/sim_waypoints.py` | offline simulator: runs the real controller against a simulated robot with lag, deadzone and coasting |

## 4. Calibration

All of these were measured on our robot, on TODO floor surface.

| Parameter | How it was measured | Value |
|---|---|---|
| `IMU_SIGN` | `calibrate mode:=imu`, rotate the robot 90° CCW by hand, check the sign of the yaw change | TODO |
| `DRIVE_SPEED` at `DRIVE_CMD` | `calibrate mode:=straight cmd:=0.25 duration:=3`, tape-measure distance d, 3 runs, `v = d / t` | TODO (runs: …) |
| straight-line drift | heading drift printed by the same runs | TODO |
| `TURN_MIN` | smallest `spin` command that rotates reliably | TODO |
| `TURN_LEAD` | coast angle printed by `calibrate mode:=spin cmd:=TURN_MIN` | TODO |
| `K_TURN`, `K_HEADING` | tuned on `test_turn.txt` / `test_line.txt` | TODO |

## 5. Performance

TODO: record the final pose at each waypoint (tape measure, start point marked on the floor).

| # | target (x, y, θ) | measured (x, y, θ) | position error | heading error |
|---|---|---|---|---|
| 1 | (1, 0, 0) | | | |
| 2 | (1, 1, 1.57) | | | |
| 3 | (2, 1, 0) | | | |
| 4 | (2, 2, −1.57) | | | |
| 5 | (1, 1, −0.78) | | | |
| 6 | (0, 0, 0) | | | |

In simulation (`python3 tools/sim_waypoints.py --set TURN_LEAD=0.15`) the controller
reaches every waypoint within about 1 cm and 3°. A 15 % error in `DRIVE_SPEED` grows to
about 24 cm mean error, so the speed calibration dominates the position error.

Discussion points (TODO, fill in from the real runs):

* Position error comes mostly from open-loop distance. Speed varies with battery
  level and floor surface, and acceleration and coasting are not modelled.
  The error accumulates along the path because nothing corrects the position estimate.
* Heading is closed-loop on the IMU, so heading errors do not accumulate. They are
  limited by IMU yaw drift and by how much the robot coasts after a turn.
* Possible improvements: wheel encoders or visual landmarks (e.g. the AprilTags used
  in later homeworks) to correct the position, and a velocity ramp to reduce wheel slip.
