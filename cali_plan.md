Here's the updated plan. The main change from before is that it measures each wheel separately, in each direction, instead of assuming both wheels behave the same. It takes about 2.5–3 hours and about 140 runs, but most runs are short spins that don't need the robot reset to the start line.

**What you end up with**

| Constant | Meaning |
|---|---|
| `K[L/R][fwd/rev]` | m/s per unit of command, for each wheel and direction |
| `C0[L/R][fwd/rev]` | dead zone for each wheel and direction |
| `B_EFF` | effective wheel separation (includes slip when turning) |

Every number comes from a model of each wheel, **v_wheel = K·(|c| − C0)**, rather than from a single shared trim factor.

---

## Phase 0: Setup (10 min)
- [ ] Fully charge the battery and note the start time.
- [ ] Use the floor you'll be graded on: about 3 m of straight space plus a spot to spin.
- [ ] Put down a tape start line, and mark a **reference point** on the robot (midpoint between the wheel axles).
- [ ] Get a **box or stand** that lifts the wheels off the ground, a tape measure, and paper for sideways-drift notes.
- [ ] Open two SSH terminals. Terminal 1: `tmux` → `ros2 run robot_control motor_control`.

> Pass zeros as `0.0`, not `0`. The parameters are floats, and ROS rejects an integer.

## Phase 1: Basic checks (5 min)
**With the robot lifted:**
```bash
ros2 run robot_control calibrate --ros-args -p L:=0.3 -p R:=0.0 -p duration:=1.5 -p ask_distance:=false -p label:=check_L
```
- [ ] Only the **left** wheel turns, and it turns **forward**. Repeat with `R`. If either is wrong, figure out the swap before going any further.

**On the floor:**
```bash
ros2 run robot_control calibrate --ros-args -p L:=-0.4 -p R:=0.4 -p duration:=2.0 -p ask_distance:=false -p label:=check_ccw
```
- [ ] The robot turns counter-clockwise, and the printed yaw is **positive**.

## Phase 2: Dead zone for each wheel (15 min, 28 runs)
**Lifted.** Increase each wheel's command in each direction until it starts turning:
```bash
for w in L R; do for d in fwd rev; do for c in 0.03 0.05 0.07 0.09 0.11 0.13 0.15; do
  s=$c; [ $d = rev ] && s=-$c
  if [ $w = L ]; then L=$s; R=0.0; else L=0.0; R=$s; fi
  ros2 run robot_control calibrate --ros-args -p L:=$L -p R:=$R -p duration:=1.0 \
    -p start_delay:=0.3 -p ask_distance:=false -p label:=dz_${w}_${d}_$c
  read -p "did the $w wheel turn ($d)? note it, Enter"
done; done; done
```

for w in R; do for d in fwd; do for c in 0.03 0.05 0.07 0.09 0.11 0.13 0.15; do
  s=$c; [ $d = rev ] && s=-$c
  if [ $w = L ]; then L=$s; R=0.0; else L=0.0; R=$s; fi
  ros2 run robot_control calibrate --ros-args -p L:=$L -p R:=$R -p duration:=1.0 \
    -p start_delay:=0.3 -p ask_distance:=false -p label:=dz_${w}_${d}_$c
  read -p "did the $w wheel turn ($d)? note it, Enter"
done; done; done

- [ ] Write down four lifted dead zones: L fwd = 0.07, L rev = 0.05, R fwd=0.07, R rev=0.07.

**On the floor (2 min):** repeat the same increase with `L=R=c`.
- [ ] The floor dead zone will be **higher** than the lifted one, because the wheels now carry the robot's weight. The lifted numbers show the *difference between the wheels*. The final C0 values come from the fit in Phase 6.

## Phase 3: Pivot tests, one wheel at a time (20 min, 32 runs)
Drive one wheel while the other stays at 0. The robot pivots around the stopped wheel, and the IMU's turn rate equals that wheel's speed divided by b. **Comparing turn rates gives you each wheel's speed relative to the other without knowing b.** The robot stays in place, so you don't have to reset it to the start line.
```bash
for w in L R; do for d in fwd rev; do for c in 0.2 0.3 0.4 0.5; do for n in 1 2; do
  s=$c; [ $d = rev ] && s=-$c
  if [ $w = L ]; then L=$s; R=0.0; else L=0.0; R=$s; fi
  ros2 run robot_control calibrate --ros-args -p L:=$L -p R:=$R -p duration:=3.0 \
    -p ask_distance:=false -p label:=pivot${w}_${d}_${c}_t$n
  read -p "Enter for next"
done; done; done; done
```
- [ ] Quick look: compare the |turn rate| of `pivotL_fwd_0.3` with `pivotR_fwd_0.3`. The difference is the speed mismatch between the wheels at that command.

## Phase 4: Straight runs (35 min, 24 runs)
These give the **average speed** in meters per second. The yaw the script records adds a **second measure of the mismatch**.
```bash
for c in 0.2 0.3 0.4 0.5; do for t in 2 4; do for n in 1 2 3; do
  ros2 run robot_control calibrate --ros-args -p L:=$c -p R:=$c -p duration:=$t -p label:=fwd_${c}_${t}s_t$n
  read -p "reset robot to start line, Enter"
done; done; done
```
- [ ] Type in the **forward distance** when asked.
- [ ] On paper, write the **sideways drift** in cm (+ for left).
- [ ] If 0.5 for 4 s runs out of space, use 3 s and change the `4.0` in the fit script to match.

## Phase 5: Spins (20 min, 24 runs)
These give **B_EFF**, and they check that the reverse speeds are right.
```bash
for c in 0.2 0.3 0.4 0.5; do for d in ccw cw; do for n in 1 2 3; do
  if [ $d = ccw ]; then L=-$c; R=$c; else L=$c; R=-$c; fi
  ros2 run robot_control calibrate --ros-args -p L:=$L -p R:=$R -p duration:=3.0 \
    -p ask_distance:=false -p label:=spin_${d}_${c}_t$n
  read -p "Enter for next"
done; done; done
```
- [ ] For at least one run per speed, count turns against a tape mark to check the IMU.

## Phase 6: Fit (20 min, on your laptop)
```bash
scp ubuntu@<robot-ip>:~/calibration_log.csv .
```
```python
import numpy as np, pandas as pd

df = pd.read_csv('calibration_log.csv')
LEVELS = [0.2, 0.3, 0.4, 0.5]
LONG, SHORT = 4.0, 2.0          # straight-run durations used in Phase 4

# --- Phase 4: average forward speed, (d_long - d_short)/(T_long - T_short) cancels start/stop effects
fwd = df[df.label.str.startswith('fwd_')]
d = fwd.groupby(['L', 'duration_requested_s']).distance_m.mean()
v_avg = {c: (d[(c, LONG)] - d[(c, SHORT)]) / (LONG - SHORT) for c in LEVELS}

# --- Phase 3: pivot turn rate for one wheel = v_wheel / b
def pivot(prefix, c):
    g = df[df.label.str.startswith(prefix) & np.isclose(df.L.abs() + df.R.abs(), c)]
    return g.omega_rad_s.abs().mean()

rows = []
for c in LEVELS:
    pLf, pRf = pivot('pivotL_fwd', c), pivot('pivotR_fwd', c)
    pLr, pRr = pivot('pivotL_rev', c), pivot('pivotR_rev', c)
    r = pRf / pLf                                  # v_R / v_L going forward (b cancels)
    vLf = 2 * v_avg[c] / (1 + r); vRf = r * vLf    # split the average speed between the wheels
    vLr = vLf * pLr / pLf;        vRr = vRf * pRr / pRf   # reverse speed, same wheel
    rows.append(dict(c=c, L_fwd=vLf, R_fwd=vRf, L_rev=vLr, R_rev=vRr))
w = pd.DataFrame(rows).set_index('c')
print('wheel speeds (m/s):\n', w.round(3))

# --- per wheel, per direction: v = K*(c - C0)
for col in w:
    k, b0 = np.polyfit(w.index, w[col].values, 1)
    print(f'{col}:  K = {k:.3f} m/s per unit   C0 = {-b0 / k:.3f}')

# --- Phase 5: effective wheel separation
spin = df[df.label.str.startswith('spin_')].assign(c=lambda x: x.L.abs())
for dname, (a, bcol) in {'ccw': ('R_fwd', 'L_rev'), 'cw': ('L_fwd', 'R_rev')}.items():
    om = spin[spin.label.str.contains(dname)].groupby('c').omega_rad_s.mean().abs()
    print(f'{dname}: B_EFF = {((w[a] + w[bcol]) / om).mean():.3f} m')
```
**How it works:** the pivot ratio tells you how the average speed from the straight runs splits between the left and right wheels. Comparing each wheel's forward and reverse pivots gives its reverse speed. Spins then give B_EFF. None of this requires knowing b beforehand.

- [ ] **Check against the Phase 4 yaw:** the straight runs should have turned at about ω ≈ (v_R − v_L)/B_EFF. If they come close to the yaw you recorded, the model holds together.
- [ ] B_EFF from CW and CCW should agree within about 10%.
- [ ] If forward and reverse K for a wheel differ by less than about 5%, merge them into one value to keep things simple.
- [ ] For the report: plot speed against command with **four lines** (L/R × fwd/rev), plus a table of the pivot ratios.

## Phase 7: Check against new targets (20 min)
With the inverse kinematics plus the per-wheel constants:
- [ ] **Straight test:** desired v = 0.2 m/s and ω = 0 for 5 s. The robot should go straight with no extra trim: |yaw| < 2°, sideways drift < 3 cm, distance within 5 cm of 1 m.
- [ ] **Four 90° turns** in place: final heading within about 10° of the start.
- [ ] **1 m × 1 m square**, then **your actual `waypoints.txt`**. Measure how far the robot ends up from each waypoint.

If distances are off, scale both K_fwd values. If the robot curves, adjust the ratio between K_L and K_R. If turns are off, adjust B_EFF, or turn using IMU yaw.

## Phase 8: Battery check (5 min)
- [ ] Repeat `fwd_0.3_4s` and `pivotL_fwd_0.3` / `pivotR_fwd_0.3`. If speed dropped more than about 5% or the L/R ratio changed, note it under performance in your report.

---

## Paste into `velocity_control.py`
```python
import math

K  = {'L': {'fwd': ..., 'rev': ...}, 'R': {'fwd': ..., 'rev': ...}}   # m/s per unit cmd
C0 = {'L': {'fwd': ..., 'rev': ...}, 'R': {'fwd': ..., 'rev': ...}}   # dead zones
B_EFF = ...                                                           # m
MAX_CMD = 0.5

def wheel_cmd(v_wheel, side):
    """Desired wheel speed (m/s, signed) -> motor command for that wheel."""
    if abs(v_wheel) < 1e-3:
        return 0.0
    d = 'fwd' if v_wheel > 0 else 'rev'
    c = abs(v_wheel) / K[side][d] + C0[side][d]
    return math.copysign(min(c, MAX_CMD), v_wheel)

def body_to_cmd(v, omega):
    """Inverse kinematics: (v, ω) -> [L, R]."""
    v_l = v - omega * B_EFF / 2
    v_r = v + omega * B_EFF / 2
    return wheel_cmd(v_l, 'L'), wheel_cmd(v_r, 'R')
```

| Phase | Time | Runs |
|---|---|---|
| 0–1 setup and checks | 15 min | 4 |
| 2 dead zones | 15 min | 28 + a few |
| 3 pivots | 20 min | 32 |
| 4 straight | 35 min | 24 |
| 5 spins | 20 min | 24 |
| 6 fit | 20 min | (laptop) |
| 7–8 validation | 25 min | ~10 |

If you're short on time, cut trials before cutting command levels. The fits need all four levels, not three repeats of each.