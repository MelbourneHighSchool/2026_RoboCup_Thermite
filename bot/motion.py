"""Drive layer: speed model, keep-out guards (wall, enemy, teammate), jam and stand-off
recovery, carry/shield creep, pass race gate, and finishing-angle selection.

Functions here take explicit arguments (rx, ry, hdg, enemies, ...) rather than reading
bot.state; only _slew_drive reads the shared pose, for its gates.
"""

import math
import time

import bot.state as state
from bot.field import FieldModel, wrap_deg as _wrap_deg
from bot.hardware import Motor
from bot.perception import Perception
from bot.drive_config import base_speed, rush_speed, wall_slide_zone_mm

# flick range, shared with the drive-speed model below
flick_range_mm = 900.0

# Ball capture: direct-pursuit steering (_orbit_approach), shared by all three
# controllers.
capture_cone_half_deg = 20.0 # half-angle about the dwibbler direction
capture_cone_half_width_mm = 35.0 # lateral half-width (dwibbler mouth half-width)

# Terminal curve-in: a straight chase is right at range, but on the last few hundred mm it
# meets the ball off-centre and nudges it away before the mouth closes. Several
# high-placing robots fix that with an approach offset that grows as the ball nears, so
# the chassis curves in and scoops the ball mouth-aligned. Ours blends the ball bearing
# toward a point set back behind the ball on the goal line (so the curve ends
# goal-aligned), ramping from 0 at curve_in_near_mm to full at curve_in_full_mm.
curve_in_enabled = True
curve_in_near_mm = 350.0 # ramp start: at/above this range the approach is pure direct pursuit
curve_in_full_mm = 120.0 # ramp end: inside this the offset is at its full value
# full-strength offset: arcs the mouth onto the ball without spoiling the chase
curve_in_max_off_deg = 40.0

# Approach speed shaping (forward damping 0.1, side damping 0.35, close 250 / far 500 mm).
# Damps speed by how far off the target bearing the drive points, fading back to none by
# `far`: a close, off-axis request is asking the wheels for a sideways sprint they can't
# give. A cap, never a boost.
speed_bias_forward_damping = 0.1
speed_bias_side_damping = 0.35
speed_bias_close_mm = 250.0
speed_bias_far_mm = 500.0

# Facing-away recovery: after 110 consecutive ticks with the ball 120+ deg off our
# heading, stop steering onto the ball, mirror the drive to the far side of the field
# (-sign(x) * |direction|) and drop to 0.67 speed. An approach that grinds without ever
# facing the ball is better retried from the other side.
facing_away_recovery_enabled = True
facing_away_off_axis_deg = 120.0
facing_away_recovery_ticks = 110
facing_away_recovery_scale = 0.67

# Sideline carry route: when the ball is secured but the direct lane isn't winnable, gain
# depth along a flank instead of shielding in place. One mid waypoint out at the near
# sideline (outside the goal box), then the normal carry logic approaches from the flank.
# Armed at capture, retired on arrival; every other has_ball branch (aim found, flick,
# press) outranks it.
sideline_route_enabled = True
# this close to the waypoint the route is done; normal carry logic resumes
sideline_route_slow_mm = 250.0

# Drive speed model: normalised command <-> mm/s. One wheel at full scale turns at
# Motor.MAX_REV_PER_S and body speed is sqrt2 times rim speed. Re-check if the wheels
# change.
wheel_diameter_mm = 50.0
max_speed_cmd = 1.0 # full scale, in the normalised units
vmax_full_cmd_mms = (math.sqrt(2.0) * math.pi * wheel_diameter_mm
                      * Motor.MAX_REV_PER_S) # about 7346 mm/s, uncapped

# Bearing matters too: the achievable fraction of vmax is cap/(|cos| + |sin|), the full
# cap on a cardinal bearing and 0.707 of it on a diagonal.

# Acceleration slew cap: commanded |speed| may only rise this much per 50 Hz loop.
# Decreases, reversals and rot_speed are never capped, so this only smooths a jump
# straight to a big command: 0 -> base_speed (0.3) takes 5 loops (0.1 s), 0 ->
# rush_speed (0.5) about 9. overrideAcc bypasses it for calls that need an instant
# response (the final capture press, emergency pushback).
ACCEL_SLEW_PER_LOOP = 0.06
_last_slew_speed = 0.0


# Last-resort white-line gate, layered under _wall_guard. _wall_guard steers on a pose up
# to a revolution (~100 ms) old and only shapes velocity, so a fast approach or a
# localisation error can put the body over the line before it ever sees the keep band.
# This backstop runs inside _slew_drive (every drive passes through it) on the freshest
# position ("pose_live", else the raw fit), cancelling the into-wall component inside its
# band and blocking fully at body-line contact. Like every wall guard it ignores the goal
# mouth and back wall (active_walls only returns boundary and goal-flank segments), so
# driving into the goal is never blocked. Rotation is untouched: spinning in place can't
# cross a line.
white_gate_enabled = True
white_gate_slide_mm = 40.0 # width of the ramp band outside body contact
white_gate_max_pose_age_s = 0.3 # older than this the gate stands down

# Gate telemetry for the debug page's /calib panel; read as motion.gate_trim_events etc.,
# never a from-import.
gate_trim_events = 0 # lifetime count of commands the gate trimmed or fully blocked
gate_last_trim_frac = 0.0 # fraction of the last gated command's speed removed (0..1)
gate_last_block = False # True when the last gated command was fully cancelled


def reset_gate_calibration_counters():
    """zero the gate's calibration telemetry (the /calib panel's reset button)."""
    global gate_trim_events, gate_last_trim_frac, gate_last_block
    gate_trim_events = 0
    gate_last_trim_frac = 0.0
    gate_last_block = False


def _white_line_gate(rx, ry, hdg, bearing_rel, cmd, pose_age_s):
    """pure half of the gate (state access lives in _slew_drive): cancel the into-wall part of
    a translation command within white_gate_slide_mm of crossing a keep-out wall, scaling to
    a full block at body-line contact. A pose_age_s of None or past
    white_gate_max_pose_age_s stands the gate down.
    """
    global gate_last_trim_frac, gate_last_block
    gate_last_trim_frac, gate_last_block = 0.0, False
    if not white_gate_enabled or cmd <= 0.0:
        return bearing_rel, cmd
    if pose_age_s is None or pose_age_s > white_gate_max_pose_age_s:
        return bearing_rel, cmd
    body = Perception.robot_radius_mm
    walls = FieldModel.active_walls(rx, ry, body + white_gate_slide_mm)
    if not walls:
        return bearing_rel, cmd
    bf = math.radians(bearing_rel + hdg)
    vx, vy = cmd * math.sin(bf), cmd * math.cos(bf)
    for dist, nx, ny, keep in walls:
        if nx == 0.0 and ny == 0.0:
            continue
        vn = vx * nx + vy * ny # + = moving away from this wall
        if vn >= 0.0:
            continue
        # trim fraction: 0 at the band's outer edge, 1 (full cancel) at contact
        f = 1.0 - min(1.0, max(0.0, (dist - body) / white_gate_slide_mm))
        trim = vn * f
        vx -= trim * nx
        vy -= trim * ny
    new_cmd = math.hypot(vx, vy)
    # telemetry: how much of the command this pass removed, and whether it blocked
    # fully
    if new_cmd < cmd:
        global gate_trim_events
        gate_trim_events += 1
        gate_last_trim_frac = 1.0 - new_cmd / cmd
        gate_last_block = new_cmd < 1e-3
    if new_cmd < 1e-3:
        return bearing_rel, 0.0
    return _wrap_deg(math.degrees(math.atan2(vx, vy)) - hdg), min(new_cmd, max_speed_cmd)


# Wall-approach speed governor: a predictive cap on top of _wall_guard's reactive one. The
# guards only act inside their keep band, so a clean run at rush speed arrives with all its
# speed and sheds it at the line, which reads as a lurch. This projects the distance to each
# wall along the commanded direction ((distance - keep) / cos(angle to the wall normal),
# minimum over walls) and caps speed on a line from cruise at the zone edge down to zero at
# the line.
wall_approach_governor_enabled = True
# start slowing this far out along the commanded direction
wall_approach_slow_dist_mm = 600.0
wall_approach_min_frac = 0.0 # speed allowed at the keep-out line itself
# ...and the speed allowed at the zone's outer edge, as a ratio of our cruise. It lands
# between wall_safe_speed_cmd and base_speed, and halves again by 300 mm out.
wall_approach_edge_ratio = 800.0 / 1500.0
wall_approach_max_frac = wall_approach_edge_ratio * base_speed
# If the entry lurches, soften the step at the zone edge first.


def _wall_approach_governor(rx, ry, hdg, bearing_rel, cmd, pose_age_s):
    """pure half of the governor (state access lives in _slew_drive): cap speed so the robot
    bleeds it off before a wall, from the projected closing distance along the commanded
    direction. A command that isn't closing on a wall (parallel or away) is untouched, so
    sliding along a wall isn't penalised. Stands down on a stale pose, like the gate.
    """
    if not wall_approach_governor_enabled or cmd <= 0.0:
        return bearing_rel, cmd
    if pose_age_s is None or pose_age_s > white_gate_max_pose_age_s:
        return bearing_rel, cmd
    walls = FieldModel.active_walls(rx, ry, keep_min_mm + wall_approach_slow_dist_mm)
    if not walls:
        return bearing_rel, cmd
    bf = math.radians(bearing_rel + hdg)
    vx, vy = cmd * math.sin(bf), cmd * math.cos(bf)
    closing = None
    for dist, nx, ny, keep in walls:
        if nx == 0.0 and ny == 0.0:
            continue
        cos_to_wall = (vx * nx + vy * ny) / cmd # + = moving away from this wall
        if cos_to_wall >= -1e-6:
            continue # not closing on it, so it cannot slow us down
        # projected travel before the wall's keep line along this command; already
        # past it reads 0
        projected = max(1e-9, (dist - keep) / -cos_to_wall)
        if closing is None or projected < closing:
            closing = projected
    if closing is None or closing >= wall_approach_slow_dist_mm:
        return bearing_rel, cmd
    allowance = remap(closing, 0.0, wall_approach_slow_dist_mm,
                      wall_approach_min_frac, wall_approach_max_frac)
    return bearing_rel, min(cmd, max(0.0, allowance))


# Turn about the ball (a top international Open robot's get_turn_around_ball_movement).
# Every other carrying turn spins on the spot, sweeping the mouth sideways under the ball
# and walking it off line. This turns the body about the ball at 120 mm instead, so the
# mouth stays on it and the ball sees a pivot. The orbit comes from Motor.drive's
# rot_r/rot_theta.
turn_ball_radius_mm = 120.0


def _turn_about_ball(ball_rel_bearing, rot_speed):
    """arguments for a _slew_drive that rotates the body at rot_speed about the carried ball
    rather than its centre: _slew_drive(*_turn_about_ball(rel, spin)). ball_rel_bearing
    is the ball's robot-frame bearing (pass 0.0 when untracked: a mouth-held ball is dead
    ahead).
    """
    return (0.0, 0.0, turn_ball_radius_mm, ball_rel_bearing, rot_speed)


def _slew_drive(bearing, speed, rot_r=0.0, rot_theta=0.0, rot_speed=0.0, overrideAcc=False):
    """Motor.drive wrapper that caps how fast commanded |speed| rises per loop (never a
    decrease, a reversal or rot_speed). overrideAcc=True skips the slew cap only; the wall
    governor and the white-line gate always apply.
    """
    global _last_slew_speed
    speed = float(speed)
    if speed > 0.0 and (white_gate_enabled or wall_approach_governor_enabled):
        with state._lock:
            live = state._state["pose_live"]
            raw = state._state["pose"]
            pose_t = state._state["pose_t"]
        pose = None
        if live is not None:
            # the position PosePropagator extends at ~50 Hz; the gate's own
            # staleness check still applies
            pose = (live[0], live[1], live[2], time.monotonic() - live[3])
        elif raw is not None:
            pose = (raw[0], raw[1], raw[2],
                    None if pose_t is None else time.monotonic() - pose_t)
        if pose is not None:
            px, py, ph, age = pose
            # gate first: it can redirect the command, and the governor must
            # judge the direction actually driven
            if white_gate_enabled:
                bearing, speed = _white_line_gate(px, py, ph, bearing, speed, age)
            if wall_approach_governor_enabled:
                bearing, speed = _wall_approach_governor(px, py, ph, bearing, speed, age)
    if not overrideAcc and speed > _last_slew_speed:
        speed = min(speed, _last_slew_speed + ACCEL_SLEW_PER_LOOP)
    _last_slew_speed = speed
    Motor.drive(bearing, speed, rot_r=rot_r, rot_theta=rot_theta, rot_speed=rot_speed)


def _vmax_mms(bearing_deg):
    """top speed (mm/s) the chassis can reach along a robot-frame bearing: full command less
    the per-wheel cap.
    """
    rad = math.radians(bearing_deg)
    c, s = abs(math.cos(rad)), abs(math.sin(rad))
    peak = c + s # peak wheel command per unit of translation command
    return vmax_full_cmd_mms * Motor.MOTOR_CAP_FRAC / peak


def remap(value, low1, high1, low2, high2):
    """linearly map value from one range [low1, high1] onto another [low2, high2]."""
    if abs(high1 - low1) < 1e-9:
        return low2
    return low2 + (float(value) - low1) * (high2 - low2) / (high1 - low1)


# Heading-to-spin law: a 3 deg deadband, so a fix that close is held rather than hunted,
# then spin proportional to error, saturating at 1.0 at 60 deg. rot_speed is unslewed, so a
# hard error now commands a sharp spin: if that rings on the bench, walk back the shape, not
# a gain.
heading_deadband_deg = 3.0
heading_saturation_deg = 60.0

# Optional rate damping on the heading law, a PD (turn = p * err - d * measured_rate). The
# rate term subtracts the body's measured spin from the command, damping the approach
# without touching the deadband or saturation. Units are ours: command per deg/s. On by
# default; False restores the plain saturated-P law exactly (the term is additive).
heading_d_term_enabled = True
heading_d_gain_frac_per_dps = 0.006 # 0.006 * 270 deg/s: sheds the whole command at ~45 deg/s

# The rate for the D term comes from the compass itself (_imu_yaw_rate_dps), so the flag
# alone turns it on; no call site passes it. Two guards: a minimum gap between the
# differenced samples (back-to-back calls would divide quantisation by microseconds) and a
# maximum one (a stale sample must never read as a huge rate). The rate is cached per
# control tick so every call in a tick sees the same value.
imu_rate_min_gap_s = 0.015
imu_rate_max_gap_s = 0.2
_imu_rate_prev = None # (heading_deg, monotonic_t) of the previous sample pair member
_imu_rate_cache = None # (monotonic_t, rate_dps): this tick's derived rate


def _imu_yaw_rate_dps():
    """body yaw rate (deg/s, cw+) from consecutive imu_heading samples, or 0.0 with no usable
    pair (no compass, a sample inside the minimum gap or older than the maximum, or an
    unmoved reading).
    """
    global _imu_rate_prev, _imu_rate_cache
    now = time.monotonic()
    prev = _imu_rate_prev
    if prev is not None and now - prev[1] < imu_rate_min_gap_s:
        # a later call in the same control tick reuses this tick's rate
        return _imu_rate_cache[1] if _imu_rate_cache is not None else 0.0
    with state._lock:
        hdg = state._state["imu_heading"]
    rate = 0.0
    if hdg is not None:
        if prev is not None:
            ph, pt = prev
            dt = now - pt
            if dt <= imu_rate_max_gap_s:
                rate = _wrap_deg(hdg - ph) / dt
        _imu_rate_prev = (hdg, now)
    else:
        _imu_rate_prev = None # no compass read at all: no evidence, and no stale pair to carry
    _imu_rate_cache = (now, rate)
    return rate


def _heading_spin(rel_heading_deg, rate_dps=None):
    """heading error (deg, cw+) -> rot_speed fraction (-1..1).

    0 inside the deadband, error/60 up to saturation, signed beyond it. With
    heading_d_term_enabled the measured spin subtracts through the D gain. Leave rate_dps as
    None to use the compass-derived rate; pass a figure to override it, or 0.0 for the pure
    saturated-P law.
    """
    rel = float(rel_heading_deg)
    if abs(rel) <= heading_deadband_deg:
        return 0.0
    cmd = rel / heading_saturation_deg
    if heading_d_term_enabled:
        rate = _imu_yaw_rate_dps() if rate_dps is None else float(rate_dps)
        if rate:
            cmd -= rate * heading_d_gain_frac_per_dps
    return max(-1.0, min(1.0, cmd))


def _drive_speed_bias(bearing_deg, dist_mm, forward=None, side=None,
                      close_mm=None, far_mm=None):
    """speed multiplier (0..1) for driving at bearing_deg while dist_mm from the target;
    1.0 at and beyond far_mm.
    """
    forward = speed_bias_forward_damping if forward is None else forward
    side = speed_bias_side_damping if side is None else side
    close_mm = speed_bias_close_mm if close_mm is None else close_mm
    far_mm = speed_bias_far_mm if far_mm is None else far_mm
    a = abs(_wrap_deg(bearing_deg))
    f = 1.0 - forward / (1.0 + (0.02 * a) ** 4) # straight-line damping
    g = 1.0 - 0.5 * side * (1.0 - math.cos(math.radians(2.0 * a))) # sideways damping
    mapped = f * g
    if far_mm <= close_mm:
        return mapped
    fade = min(1.0, max(0.0, (float(dist_mm) - close_mm) / (far_mm - close_mm)))
    return mapped + (1.0 - mapped) * fade


def _facing_away(rel_bearing_deg):
    """True when the ball sits at least facing_away_off_axis_deg off our heading, the condition
    the recovery counts.
    """
    return abs(_wrap_deg(rel_bearing_deg)) >= facing_away_off_axis_deg


def _facing_away_mirror_angle(rel_bearing_deg, rx):
    """the bail-out direction: mirror the approach to the far side of the field centre, keeping
    its magnitude (-sign(x) * |direction|).
    """
    side = -1.0 if rx > FieldModel.cx else 1.0
    return _wrap_deg(math.copysign(abs(_wrap_deg(rel_bearing_deg)), side))


# Final-approach braking (_brake_speed_frac): inside brake_linear_mm the sqrt stopping
# curve becomes a linear ramp, so the slew and accel limits can't overshoot on top of the
# ball; inside brake_stop_dist_mm the drive is commanded to a full stop.
brake_stop_dist_mm = 10.0
brake_linear_mm = 100.0

# Deceleration the braking laws assume: the measured 6000 mm/s^2 the goalie law already
# used, rather than our own ~44 m/s^2 guess (Motor.DECEL_MAX_FRAC_PER_S, about 7x
# optimistic). False restores the guess; about 15 approach sites use the stricter curve.
brake_decel_measured = True
brake_decel_mms2 = 6000.0


def _brake_speed_frac(dist_mm):
    """max command fraction that can still brake to a stop within dist_mm. Stopping distance
    grows with speed squared, so easing off linearly with distance (as every "hold this
    point" drive used to) lets a fast approach arrive moving, overshoot, and correct back.
    This is the safe speed at a given distance.
    """
    if dist_mm <= brake_stop_dist_mm:
        return 0.0
    a_max = (brake_decel_mms2 if brake_decel_measured
             else Motor.DECEL_MAX_FRAC_PER_S * vmax_full_cmd_mms) # mm/s^2, see the block above
    sqrt_frac = math.sqrt(max(0.0, 2.0 * a_max * dist_mm)) / vmax_full_cmd_mms
    if dist_mm >= brake_linear_mm:
        return sqrt_frac
    # linear ramp from 0 at brake_stop_dist_mm to the sqrt curve's value at brake_linear_mm
    return remap(dist_mm, brake_stop_dist_mm, brake_linear_mm,
                 0.0, math.sqrt(max(0.0, 2.0 * a_max * brake_linear_mm)) / vmax_full_cmd_mms)

# Goalie positioning braking: speed = min(max_speed, sqrt(2 * accel * remaining)), remaining
# = distance - stop_distance, instead of a linear ease-off. Derived from a deceleration the
# chassis really admits, so the keeper is already slowing on the way in; overshooting a
# block line is how a keeper concedes. At full speed the taper starts 750 mm out (3000^2 /
# (2 * 6000)), plus the 10 mm dead stop.
goalie_max_speed_mms = 3000.0
goalie_braking_accel_mms2 = 6000.0
goalie_stop_dist_mm = 10.0
goalie_max_frac = goalie_max_speed_mms / vmax_full_cmd_mms # about 0.41


def _goalie_speed_frac(dist_mm):
    """command fraction for the keeper's positioning drive, from the stopping-distance law
    above. It tops out at goalie_max_frac, above base_speed: a keeper has to get across
    to a shot faster than the field players cruise.
    """
    remaining = max(0.0, dist_mm - goalie_stop_dist_mm)
    v = math.sqrt(2.0 * goalie_braking_accel_mms2 * remaining)
    return min(goalie_max_speed_mms, v) / vmax_full_cmd_mms

# Pass release model. A roller-released pass leaves at the roller's rim speed, so the
# eject command is the ball's launch speed, and the roll distance follows from it. A fixed
# gentle nudge (0.05, about 0.13 m/s) never reached anyone. Following the simulator's
# _pass_power(d), launch at pass_roll_overspeed times the speed that would only just reach
# the target, so the receiver still gets a rolling ball.
pass_roll_overspeed = 1.4 # launch this many times the "only just reaches" speed
pass_eject_min_frac = 0.22 # floor: below this the roller barely clears its mouth friction
pass_eject_max_frac = 0.80 # ceiling: a pass must never become a shot (full reverse is the shot)

# Roller rim speed at full command, from the roller's measured 1800 rpm (28 mm roller);
# the simulator's hardware bridge models releases with the same number.
dwibble_rim_mms_full = 2640.0

# Ball roll-down: speed decays first-order with this time constant, so a ball launched at
# v rolls about v * tau before stopping. From the simulator's ball model (0.96 retained
# per 60 Hz frame); a bench roll-out can pin it exactly (BENCH_TEST_CHECKLIST).
pass_roll_tau_s = 0.42


def pass_eject_frac(dist_mm):
    """reversed-roller command magnitude (positive; callers negate it) for a pass of dist_mm,
    clamped into the pass band.
    """
    v0 = pass_roll_overspeed * float(dist_mm) / pass_roll_tau_s
    return min(pass_eject_max_frac,
               max(pass_eject_min_frac, v0 / dwibble_rim_mms_full))


def pass_ball_speed_mms(dist_mm):
    """the launch speed (mm/s) the eject will use for dist_mm, so the race gate races the real
    pass.
    """
    return pass_eject_frac(dist_mm) * dwibble_rim_mms_full

# the furthest a pass can reach: the ceiling launch, rolled out. Further than this stops
# short, so the gates refuse it and the striker keeps driving
pass_reach_mm = pass_eject_max_frac * dwibble_rim_mms_full * pass_roll_tau_s

# Pass race gate: a worst-case race, matching the simulator's model. Enemy closing speed
# is the simulator's floor-calibrated bot speed (about 954 mm/s, rounded up). It used to
# be derived from vmax and base_speed, claiming opponents at up to 5.9 m/s, so every armed
# pass was blocked whenever any enemy was tracked.
pass_enemy_speed_mms = 1000.0
# receiver/target must beat every tracked enemy to the pass by at least this long.
pass_race_margin_s = 0.15


def _pass_race_open(passer_xy, target_xy, enemies, enemy_vel=None):
    """race half of the pass-lane gate: True unless some tracked enemy could reach target_xy at
    pass_enemy_speed_mms at least pass_race_margin_s before the ball (launched at
    pass_ball_speed_mms). With enemy_vel ({id: (vx, vy)}), each enemy is first led forward
    by the ball's travel time; enemies without an entry race from where they stand.
    """
    px, py = passer_xy
    tx, ty = target_xy
    dist_pt = math.hypot(tx - px, ty - py)
    if dist_pt < 1.0 or not enemies:
        return True
    t_ball = dist_pt / pass_ball_speed_mms(dist_pt)
    for e in enemies:
        ex, ey = e["x"], e["y"]
        if enemy_vel is not None:
            vx, vy = enemy_vel.get(e.get("id"), (0.0, 0.0))
            ex, ey = ex + vx * t_ball, ey + vy * t_ball
        t_enemy = math.hypot(ex - tx, ey - ty) / pass_enemy_speed_mms
        if t_enemy < t_ball + pass_race_margin_s:
            return False
    return True

# Finishing-angle selection: rather than always aiming at the goal centre, try several
# direct bearings across the mouth and drive at one with a clear, scoring path. A goal is
# a strike on the back wall (RCJA 5.5.1) and the goal is only 74 mm deep (2.3.2), so there
# is no bank shot: every candidate is direct. Bearings are atan2(dx, dy) (0 = +y), with y
# the depth axis (goal lines at y = 0 and field_y) and x across the mouth.
#
# _goal_shot_aim can return aim_found=False (every candidate blocked, or the ball already
# past the back wall). Callers treat that as "no opinion, use the fallback bearing", and
# never read a None aim_bearing.
finish_ball_radius_mm = 21.5 # ball_diameter_mm / 2
finish_post_half_width_mm = 5.0 # ball-to-post gap: half a 10 mm goal-line width
finish_post_clear_mm = finish_ball_radius_mm + finish_post_half_width_mm
# min angular clearance off a mouth wall: a grazing shot crosses the post diagonally and
# is fragile even when it lands
finish_wall_clearance_deg = 5.0


def _finish_robot_clear_mm():
    """enemy-corridor clearance for a shot path: robot radius plus ball radius (a function so
    it follows Perception.robot_radius_mm).
    """
    return Perception.robot_radius_mm + finish_ball_radius_mm


def _goal_depth_lines(goal_y_line):
    """(mouth_y, back_y) depth lines for the goal at goal_y_line (0 or field_y): the mouth is
    goal_depth in from the line, the back wall goal_depth - slot_depth.
    """
    direction = 1.0 if goal_y_line <= FieldModel.field_y / 2.0 else -1.0
    mouth_y = goal_y_line + direction * FieldModel.goal_depth
    back_y = goal_y_line + direction * (FieldModel.goal_depth - FieldModel.slot_depth)
    return mouth_y, back_y


def _finish_ray_hit(ball_x, ball_y, bearing_deg, wall_x=None, wall_y=None):
    """where a forward ray at bearing_deg (atan2(dx, dy)) meets x=wall_x or y=wall_y, as (x,
    y), or None.
    """
    rad = math.radians(bearing_deg)
    sin_b, cos_b = math.sin(rad), math.cos(rad)
    if wall_x is not None:
        if abs(sin_b) < 1e-9:
            return None
        t = (wall_x - ball_x) / sin_b
        if t <= 0:
            return None
        return wall_x, ball_y + t * cos_b
    if abs(cos_b) < 1e-9:
        return None
    t = (wall_y - ball_y) / cos_b
    if t <= 0:
        return None
    return ball_x + t * sin_b, wall_y


def _finish_clears_posts(ball_x, ball_y, bearing_deg, mouth_y):
    """Ball body must stay clear of both mouth posts along the kick ray."""
    min_dist = finish_post_clear_mm
    rad = math.radians(bearing_deg)
    ux, uy = math.sin(rad), math.cos(rad)
    for post_x in (FieldModel.sx0, FieldModel.sx1):
        wx, wy = post_x - ball_x, mouth_y - ball_y
        proj = wx * ux + wy * uy
        dist = math.hypot(wx, wy) if proj < 0 else abs(wx * uy - wy * ux)
        if dist + 1e-9 < min_dist:
            return False
    return True


def _finish_point_to_segment_distance(px, py, start, end):
    """Shortest distance from a point to a finite line segment."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return math.hypot(px - start[0], py - start[1])
    t = ((px - start[0]) * dx + (py - start[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (start[0] + t * dx), py - (start[1] + t * dy))


def _finish_wall_bearings(ball_x, ball_y, mouth_y):
    """bearings of the two mouth walls seen from the ball, ordered so the sector between them
    runs across the mouth from the near wall (the ball's side) to the far one.

    The order matters: the clearance gate is signed, and a fixed sx0 -> sx1 order
    degenerates with the ball level with the mouth line, where the walls are +-90 deg apart
    and wrap_deg turns the span into -180, refusing a shot at an open goal. That depth is
    exactly where the keeper releases from.
    """
    if ball_x >= FieldModel.cx:
        near_bearing = math.degrees(math.atan2(FieldModel.sx1 - ball_x, mouth_y - ball_y))
        far_bearing = math.degrees(math.atan2(FieldModel.sx0 - ball_x, mouth_y - ball_y))
    else:
        near_bearing = math.degrees(math.atan2(FieldModel.sx0 - ball_x, mouth_y - ball_y))
        far_bearing = math.degrees(math.atan2(FieldModel.sx1 - ball_x, mouth_y - ball_y))
    return near_bearing, far_bearing


def _finish_scoring_path(ball_x, ball_y, bearing_deg, back_y, mouth_y):
    """((start_x, start_y), (end_x, end_y)) of the direct path when bearing_deg enters the
    mouth with clearance off both walls and ends on the back wall between the posts, else
    None.
    """
    near_bearing, far_bearing = _finish_wall_bearings(ball_x, ball_y, mouth_y)
    span = _wrap_deg(far_bearing - near_bearing)
    from_near = _wrap_deg(bearing_deg - near_bearing)
    if abs(span) + 1e-9 < finish_wall_clearance_deg:
        return None
    if from_near * span <= 0:
        return None
    if abs(from_near) + 1e-9 < finish_wall_clearance_deg:
        return None
    if abs(from_near) > abs(span) + 1e-9:
        return None
    if not _finish_clears_posts(ball_x, ball_y, bearing_deg, mouth_y):
        return None

    back_hit = _finish_ray_hit(ball_x, ball_y, bearing_deg, wall_y=back_y)
    if back_hit is not None and FieldModel.sx0 <= back_hit[0] <= FieldModel.sx1:
        return (ball_x, ball_y), back_hit
    return None


def _finish_kick_scores(ball_x, ball_y, bearing_deg, back_y, mouth_y,
                         enemy_bot_positions=None):
    """True when bearing_deg ends on the back wall between the posts and the path clears every
    tracked enemy's robot-plus-ball corridor.
    """
    path = _finish_scoring_path(ball_x, ball_y, bearing_deg, back_y, mouth_y)
    if path is None:
        return False
    clearance = _finish_robot_clear_mm()
    for e in enemy_bot_positions or ():
        if _finish_point_to_segment_distance(e["x"], e["y"],
                                            path[0], path[1]) <= clearance:
            return False
    return True


def _shot_release_scores(rx, ry, hdg, goal_y_line, enemy_bot_positions=None):
    """True when a release this tick, in the direction the body already faces, would score.

    Cast the body's facing ray at the goal and ask whether it lands inside the mouth. It
    replaces a proxy (within shot_align_tol_deg = 15 of the planned aim), which refused
    releases that obviously score and allowed ones that don't. It runs on the finishing
    predicate, which also charges the ball radius, clears both posts and slot walls, and refuses a lane with
    a tracked enemy in its corridor, so every admitted release has been shown to reach the
    back wall.

    (rx, ry) stands in for the ball: a mouth-held ball lies on the same forward ray, and
    starting from the body only makes the clearance checks slightly stricter. A ray with no
    forward crossing (ball level with or past the back wall, or facing along the goal line)
    is False.
    """
    mouth_y, back_y = _goal_depth_lines(goal_y_line)
    return _finish_kick_scores(rx, ry, hdg, back_y, mouth_y, enemy_bot_positions)


# Shot-aim commitment. The finishing search re-runs every tick and its winner changes with
# every defender move, so the steering line would jump and the chassis wobble instead of
# closing the lane it had. The held lane is retried first; a new one is adopted only after
# the held one has been blocked for shot_blocked_hold_s, and a replacement must stay open
# for shot_open_hold_s before it is taken.
shot_blocked_hold_s = 0.3
shot_open_hold_s = 0.4


class ShotCommitment:
    """holds the chosen finishing lane across ticks.

    select(candidate_bearing, now) -> (steer_bearing, shot_ready): the line to drive, and
    whether it scores right now. While a hold runs they differ (keep closing the committed
    lane, but it isn't a shot). steer_bearing is None once the aim is given up, the caller's
    cue to use its drive.

    Pass None when no lane scores, and pass the previous aim back as the preferred candidate
    (_goal_shot_aim's preferred_bearing), so a lane that still scores comes back unchanged
    and the comparison is exact.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """forget the held lane; called when a new carry begins."""
        self.aim = None
        self.blocked_since = None
        self.open_since = None
        self.repositioning = False

    def select(self, candidate, now):
        if self.aim is not None and candidate != self.aim:
            candidate = None # never immediately reverse a live aim for a newly preferred lane
        if candidate is None:
            self.open_since = None
            if self.blocked_since is None:
                self.blocked_since = now
            if self.aim is None or now - self.blocked_since >= shot_blocked_hold_s:
                self.aim = None
                # given up; a replacement must now stay open for
                # shot_open_hold_s
                self.repositioning = True
            return self.aim, False
        self.blocked_since = None
        if self.repositioning:
            if self.open_since is None:
                self.open_since = now
            if now - self.open_since < shot_open_hold_s:
                return self.aim, False
        self.aim = candidate
        self.repositioning = False
        self.open_since = None
        return self.aim, True


def _goal_shot_aim(ball_x, ball_y, goal_y_line, enemy_bot_positions=None,
                   preferred_bearing=None):
    """(aim_bearing_deg, True) for the best direct finishing angle into the goal at
    goal_y_line, or (None, False) if nothing scores clear of every tracked enemy.
    preferred_bearing, a lane already committed to, is tried first and returned unchanged
    while it still scores.
    """
    mouth_y, back_y = _goal_depth_lines(goal_y_line)
    dx_back = back_y - ball_y
    if abs(dx_back) < 1e-6:
        return None, False

    # bail if the ball is at or past the back wall: the candidate math assumes the
    # ball still travels forward through the mouth, and would otherwise produce a
    # backward bearing. The caller's plain bearing is right there anyway.
    goal_direction = 1.0 if goal_y_line <= FieldModel.field_y / 2.0 else -1.0
    if (ball_y - back_y) * goal_direction < 0:
        return None, False

    mouth_x_min = FieldModel.sx0 + finish_ball_radius_mm
    mouth_x_max = FieldModel.sx1 - finish_ball_radius_mm
    aim_x_min, aim_x_max = FieldModel.sx0, FieldModel.sx1
    # Outside the mouth: clip the back-wall window to rays that pass through it.
    before_mouth = (dx_back > 0 and ball_y < mouth_y) or (dx_back < 0 and ball_y > mouth_y)
    if before_mouth and abs(mouth_y - ball_y) > 1e-6:
        scale = dx_back / (mouth_y - ball_y)
        x_lo = ball_x + (mouth_x_min - ball_x) * scale
        x_hi = ball_x + (mouth_x_max - ball_x) * scale
        visible_lo, visible_hi = min(x_lo, x_hi), max(x_lo, x_hi)
        aim_x_min = max(FieldModel.sx0, visible_lo)
        aim_x_max = min(FieldModel.sx1, visible_hi)

    candidates = [] if preferred_bearing is None else [preferred_bearing]
    if aim_x_min <= aim_x_max:
        # Prefer the centre, but try off-centre direct shots when a bot blocks it.
        for fraction in (0.5, 0.25, 0.75, 0.0, 1.0):
            aim_x = aim_x_min + (aim_x_max - aim_x_min) * fraction
            candidates.append(math.degrees(math.atan2(aim_x - ball_x, back_y - ball_y)))

    for aim in candidates:
        if _finish_kick_scores(ball_x, ball_y, aim, back_y, mouth_y,
                                enemy_bot_positions):
            return aim, True
    return None, False

# Carry/shield: once no pass, tap-in, shot or flick fired, has_ball used to push straight
# at goal whether or not the lane was winnable. Now it presses only when
# _carry_should_press says so, and otherwise holds a shield heading while translating
# independently of it (a creep).
carry_1v1_gap_mm = flick_range_mm * 1.2 # 2nd-nearest enemy must be at
                             # least this far off to call it a real 1v1, not a double-team.
carry_level_y_mm = Perception.robot_radius_mm * 2.5
carry_level_x_mm = 500.0
carry_lane_margin_mm = Perception.robot_radius_mm * 0.7
carry_engage_dist_mm = 700.0 # shield-facing kicks in once the nearest enemy
                             # is this close
carry_shield_goal_bias = 0.25 # toward goal, so it isn't facing straight backward
# Creep while shielding: translate away from the blocking defender, blended with a little
# toward goal, well below base_speed so it stays a creep that doesn't undermine the shield.
carry_creep_speed_frac = 0.5 # fraction of base_speed used for the creep
carry_creep_away_weight = 0.65 # blend: away-from-defender vs toward-goal
                              # (1.0 = pure sideways-away, 0.0 = pure upfield)


def _path_blocked(sx, sy, gx, gy, enemies, margin):
    """True if a tracked enemy's body sits in the straight corridor from (sx, sy) to (gx, gy)."""
    dx, dy = gx - sx, gy - sy
    length = math.hypot(dx, dy)
    if length < 1.0:
        return False
    ux, uy = dx / length, dy / length
    for e in enemies:
        ex, ey = e["x"], e["y"]
        proj = (ex - sx) * ux + (ey - sy) * uy
        if 0.0 < proj < length: # ahead of us, between here and the target
            perp = abs((ex - sx) * uy - (ey - sy) * ux)
            if perp < Perception.robot_radius_mm + margin:
                return True
    return False


# Retreat bound for the shield creep: with a defender dead centre on the lane,
# _carry_should_press stays False at every depth, and the creep used to retreat without
# limit (about 7 m in 12 s in the sim, into our own goal). Past carry_retreat_max_mm from
# the carry's anchor the away-from-goal part of the creep is zeroed and the robot holds
# its shield there; forcing a press instead just oscillated press/creep across the bound.
carry_retreat_max_mm = 350.0


def _carry_retreat_mm(rx, ry, gx, gy, carry_anchor):
    """how far (mm) we have retreated from the carry anchor toward our own goal (negative =
    advanced).
    """
    ax, ay = gx - carry_anchor[0], gy - carry_anchor[1]
    ax_d = math.hypot(ax, ay)
    if ax_d < 1.0:
        return 0.0
    return ((rx - carry_anchor[0]) * ax
            + (ry - carry_anchor[1]) * ay) / ax_d


def _carry_should_press(rx, ry, gx, gy, enemies):
    """True if there's a winnable central lane to goal right now; False means hold up and
    shield. Only a real 1v1, or a defender roughly level with us but too far across to
    cut the lane, is worth pressing.
    """
    if not enemies:
        return True
    if _path_blocked(rx, ry, gx, gy, enemies, carry_lane_margin_mm):
        return False
    nearest = min(enemies, key=lambda e: math.hypot(e["x"] - rx, e["y"] - ry))
    others = [e for e in enemies if e is not nearest]
    other_dist = min((math.hypot(e["x"] - rx, e["y"] - ry) for e in others),
                     default=float("inf"))
    genuine_1v1 = other_dist > carry_1v1_gap_mm
    level_and_far = (abs(nearest["y"] - ry) < carry_level_y_mm
                     and abs(nearest["x"] - rx) > carry_level_x_mm)
    return genuine_1v1 or level_and_far


def _shield_heading_deg(rx, ry, hdg, gx, gy, enemies,
                        engage_dist_mm=carry_engage_dist_mm):
    """relative heading (deg, feed straight to _heading_spin) that shields the front-carried
    ball from the nearest enemy: face away from it, biased carry_shield_goal_bias back
    toward goal. Beyond engage_dist_mm there's no pressure yet, so just face goal.
    """
    nearest = min(enemies, key=lambda e: math.hypot(e["x"] - rx, e["y"] - ry))
    de = math.hypot(nearest["x"] - rx, nearest["y"] - ry)
    goal_bearing = math.degrees(math.atan2(gx - rx, gy - ry))
    if de > engage_dist_mm:
        return _wrap_deg(goal_bearing - hdg)
    ang_away_abs = math.degrees(math.atan2(rx - nearest["x"], ry - nearest["y"]))
    target_abs = (ang_away_abs
                 + _wrap_deg(goal_bearing - ang_away_abs) * carry_shield_goal_bias)
    return _wrap_deg(target_abs - hdg)


def _carry_creep(rx, ry, hdg, gx, gy, enemies, carry_anchor=None):
    """bounded creep toward open space while shielding, as (drive_rel_deg, speed_cmd): mostly
    away from the nearest defender plus a little toward goal, never straight back into them.
    Past carry_retreat_max_mm from carry_anchor the away-from-goal part is zeroed so the
    retreat stops.
    """
    nearest = min(enemies, key=lambda e: math.hypot(e["x"] - rx, e["y"] - ry))
    away_x, away_y = rx - nearest["x"], ry - nearest["y"]
    away_d = math.hypot(away_x, away_y)
    if away_d < 1.0:
        away_x, away_y = 0.0, 1.0
    else:
        away_x, away_y = away_x / away_d, away_y / away_d
    goal_x, goal_y = gx - rx, gy - ry
    goal_d = math.hypot(goal_x, goal_y)
    if goal_d < 1.0:
        goal_x, goal_y = 0.0, 1.0
    else:
        goal_x, goal_y = goal_x / goal_d, goal_y / goal_d
    w = carry_creep_away_weight
    cx = away_x * w + goal_x * (1.0 - w)
    cy = away_y * w + goal_y * (1.0 - w)
    # retreat bound: past carry_retreat_max_mm, strip the away-from-goal part and hold
    # (the caller still turns the shield via rot_speed)
    if carry_anchor is not None:
        retreat = _carry_retreat_mm(rx, ry, gx, gy, carry_anchor)
        if retreat <= -carry_retreat_max_mm:
            gn = math.hypot(goal_x, goal_y)
            if gn >= 1.0:
                along = cx * goal_x / gn + cy * goal_y / gn
                if along < 0.0: # any retreat component -> remove it
                    cx -= along * goal_x / gn
                    cy -= along * goal_y / gn
    cn = math.hypot(cx, cy)
    if cn < 1e-6:
        return 0.0, 0.0
    cx, cy = cx / cn, cy / cn
    drive_rel = _wrap_deg(math.degrees(math.atan2(cx, cy)) - hdg)
    return drive_rel, base_speed * carry_creep_speed_frac

# Wall keep-out: decompose the command against each nearby wall's normal and damp only the
# into-wall part, so the robot can slide along a wall at full speed.
wall_keepout_enabled = True
keep_min_mm = 150.0 # centre-to-wall: never touch a wall
# goal-flank clearance: enough to stop the wheel cutouts wedging there without the full
# keep_min_mm standoff next to a goal
goal_flank_keep_mm = 50.0

# FieldModel.active_walls reads both as bot.field module globals; publish them there.
import bot.field as _field_module
_field_module.keep_min_mm = keep_min_mm
_field_module.goal_flank_keep_mm = goal_flank_keep_mm

# The slide band (wall_slide_zone_mm, from bot/drive_config) damps the into-wall component
# outside the keep line. It must cover one pose-staleness interval (~100 ms) of travel; it
# is a body-space distance, not scaled with the command limit, because traction caps real
# speed long before rush_speed does. Widen it only against a measured overshoot.

# wall_safe_speed_cmd: a deliberately slow speed for steering on a ~100 ms stale pose at a
# wall. Not rescaled with base_speed (see the slide band above).
wall_safe_speed_cmd = 0.083
wall_push_gain = wall_safe_speed_cmd / 60.0 # outward command per mm of penetration
wall_push_max = wall_safe_speed_cmd # cap on the restoring push


def _wall_guard(rx, ry, hdg, bearing_rel, cmd):
    """constrain a translation command to keep clear of the four boundary walls and each goal's
    flanks (the mouth and back wall stay exempt), each at its keep-out distance.

    Resolving each wall in turn used to crush a diagonal command into a concave corner (a
    wall meeting a goal flank, the ball tucked in the pocket) to (0, 0), though a path along
    the corner exists. Now the hard-zone clip runs to convergence over every violated wall,
    the slide-zone damp runs once, and if the clip still collapses a real command, it is
    bent onto the sum of every blocking wall's outward normal (non-negative against each of
    them) and clipped again.
    """
    if not wall_keepout_enabled:
        return bearing_rel, cmd
    walls = FieldModel.active_walls(rx, ry, keep_min_mm + wall_slide_zone_mm)
    if not walls:
        return bearing_rel, cmd

    bf = math.radians(bearing_rel + hdg)
    vx0, vy0 = cmd * math.sin(bf), cmd * math.cos(bf) # original commanded field-frame velocity
    cmd_mag = cmd

    # positional restoring push, summed over every currently-penetrated wall
    push_x = push_y = 0.0
    for dist, nx, ny, keep in walls:
        if nx == 0.0 and ny == 0.0:
            continue
        if dist <= keep:
            push = min(wall_push_max, wall_push_gain * (keep - dist))
            push_x += push * nx
            push_y += push * ny

    # Reactive anti-line steering: a purely subtractive guard never adds outward
    # motion, so a command grazing a wall keeps hugging it until drift puts the robot
    # over. A small repulsion along each nearby wall's outward normal, added before
    # the clip, peels it away: full inside the keep-out, fading to zero across the
    # slide band, and capped at wall_safe_speed_cmd so a wall-side target still
    # converges.
    bounce_x = bounce_y = 0.0
    for dist, nx, ny, keep in walls:
        if nx == 0.0 and ny == 0.0:
            continue
        if dist <= keep:
            bounce_mag = wall_safe_speed_cmd
        elif dist <= keep + wall_slide_zone_mm:
            bounce_mag = wall_safe_speed_cmd * (keep + wall_slide_zone_mm - dist) / wall_slide_zone_mm
        else:
            continue
        bounce_x += bounce_mag * nx
        bounce_y += bounce_mag * ny

    def _resolve(vx, vy):
        """hard-zone clip (to convergence) plus one slide-zone damp pass; returns the clipped
        velocity and the (nx, ny) of every wall that constrained it.
        """
        blocking = []
        for _ in range(3): # converge the hard-zone clip over every violated wall (a corner)
            changed = False
            for dist, nx, ny, keep in walls:
                if nx == 0.0 and ny == 0.0 or dist > keep:
                    continue
                vn = vx * nx + vy * ny # + = moving away from this wall
                if vn < 0.0: # cancel the into-wall component
                    vx -= vn * nx
                    vy -= vn * ny
                    blocking.append((nx, ny))
                    changed = True
            if not changed:
                break
        for dist, nx, ny, keep in walls: # slide-zone damp: a single pass, same as before
            if nx == 0.0 and ny == 0.0 or dist <= keep:
                continue
            vn = vx * nx + vy * ny
            if vn < 0.0: # in the slide band, heading inward
                factor = (dist - keep) / wall_slide_zone_mm # 0 at keep -> 1 at edge
                # the surviving into-wall speed is also capped at
                # wall_safe_speed_cmd
                closing = max(factor * vn, -wall_safe_speed_cmd)
                vx += (closing - vn) * nx
                vy += (closing - vn) * ny
                blocking.append((nx, ny))
        return vx, vy, blocking

    vx0, vy0 = vx0 + bounce_x, vy0 + bounce_y
    vx, vy, blocking = _resolve(vx0, vy0)

    # De-dup the blocking normals and keep the distinct directions. One wall's
    # normal alone isn't a corner: bending onto it would blast the robot straight off
    # the wall regardless of its goal, so trust the clip. Only two or more
    # non-parallel walls form a pocket worth bending around.
    distinct = []
    for nx, ny in blocking:
        if not any(nx * dnx + ny * dny > 0.98 for dnx, dny in distinct):
            distinct.append((nx, ny))

    # Also require a real drive request (well above wall_safe_speed_cmd) that
    # collapsed to nearly nothing in absolute terms. A slow creep near a wall (careful
    # positioning, a goalie on its line) is small on its own, and bending it onto a
    # full-magnitude escape would overshoot in a direction nobody asked for.
    if (len(distinct) >= 2 and cmd_mag > wall_safe_speed_cmd * 1.5
            and math.hypot(vx, vy) < 0.05):
        # The clip collapsed a real command: the bearing points into a concave
        # pocket. Bend the original command onto the walls' combined escape
        # direction and re-clip it, so the bend can never re-violate a wall.
        ex = sum(nx for nx, ny in distinct)
        ey = sum(ny for nx, ny in distinct)
        emag = math.hypot(ex, ey)
        if emag > 1e-6:
            bend_vx, bend_vy = cmd_mag * ex / emag, cmd_mag * ey / emag
            bend_vx, bend_vy, _ = _resolve(bend_vx, bend_vy)
            if math.hypot(bend_vx, bend_vy) > math.hypot(vx, vy):
                vx, vy = bend_vx, bend_vy

    vx += push_x
    vy += push_y

    new_cmd = math.hypot(vx, vy)
    if new_cmd < 1e-3:
        # compared against full scale this always lost, zeroing the command near a
        # wall instead of using the damped velocity
        return bearing_rel, 0.0
    new_cmd = min(new_cmd, max_speed_cmd)
    new_bear = _wrap_deg(math.degrees(math.atan2(vx, vy)) - hdg)
    return new_bear, new_cmd

# Enemy keep-out: the same decomposition as _wall_guard against each tracked enemy, so a
# chase steers around a blocking robot. Off once we're close enough to the ball to contest
# it: this stops us driving through an enemy on the way to a distant ball, not backing off
# a real 50/50.
enemy_avoid_enabled = True
# centre-to-centre before contact (~2x robot_radius_mm plus margin)
enemy_avoid_keep_mm = 260.0
# damp the into-enemy component within this band outside the keep ring
enemy_avoid_slide_zone_mm = 300.0
# This close to the ball, avoidance turns off. At 300 (just past the keep ring) the
# "avoid" and "contest" zones nearly coincided when an enemy stood on the ball, and we
# stalled short of it; 550 leaves room to commit to the contest before contact without
# bulldozing a merely nearby enemy on a normal chase.
enemy_contest_dist_mm = 550.0
# Committed approach speed inside enemy_contest_dist_mm when an enemy is holding the ball:
# skips _brake_speed_frac's soft stop, which just glides up beside a similar robot and
# never dispossesses it. Above base_speed so contact is real, below rush_speed since it's
# still a steered final approach. 0.4 was the match code's rush, which shoved defenders
# off the ball.
contest_speed_cmd = 0.4
# an enemy this close to the ball counts as holding it: its body radius plus room for a
# ball not dead-centre on it
contest_hold_dist_mm = Perception.robot_radius_mm + 180.0


def _enemy_holds_ball(bx, by, enemies):
    """True if some tracked enemy is close enough to (bx, by) to plausibly be holding the ball."""
    return any(math.hypot(e["x"] - bx, e["y"] - by) <= contest_hold_dist_mm
               for e in enemies)


def _approach_speed_cap(dist_mm, bx, by, enemies, fallback_cmd):
    """the speed cap for a ball approach: contest_speed_cmd inside enemy_contest_dist_mm with
    an enemy holding the ball, else fallback_cmd (normally _brake_speed_frac's cap).
    """
    if (dist_mm <= enemy_contest_dist_mm and enemies
            and _enemy_holds_ball(bx, by, enemies)):
        return max(fallback_cmd, contest_speed_cmd)
    return fallback_cmd


def _enemy_guard(rx, ry, hdg, bearing_rel, cmd, enemies, ball_dist_mm=None):
    """like _wall_guard for tracked enemies: damp/deflect the into-enemy component so the path
    slides around. ball_dist_mm is our distance to the ball: pass it when chasing a
    known ball so a close contest isn't steered away from; omit it (positioning, yield hold,
    lost ball) and avoidance always applies.
    """
    if not enemy_avoid_enabled or not enemies:
        return bearing_rel, cmd
    if ball_dist_mm is not None and ball_dist_mm <= enemy_contest_dist_mm:
        return bearing_rel, cmd

    bf = math.radians(bearing_rel + hdg)
    vx, vy = cmd * math.sin(bf), cmd * math.cos(bf) # field-frame velocity
    push_x = push_y = 0.0
    for e in enemies:
        dx, dy = rx - e["x"], ry - e["y"] # enemy -> robot
        dist = math.hypot(dx, dy)
        if dist < 1e-6 or dist > enemy_avoid_keep_mm + enemy_avoid_slide_zone_mm:
            continue
        nx, ny = dx / dist, dy / dist
        vn = vx * nx + vy * ny # + = moving away from this enemy
        if dist <= enemy_avoid_keep_mm:
            if vn < 0.0: # cancel the into-enemy component
                vx -= vn * nx
                vy -= vn * ny
            push = min(wall_push_max, wall_push_gain * (enemy_avoid_keep_mm - dist))
            push_x += push * nx
            push_y += push * ny
        elif vn < 0.0: # in the slide band, heading inward
            factor = (dist - enemy_avoid_keep_mm) / enemy_avoid_slide_zone_mm
            closing = max(factor * vn, -wall_safe_speed_cmd)
            vx += (closing - vn) * nx
            vy += (closing - vn) * ny
    vx += push_x
    vy += push_y

    new_cmd = math.hypot(vx, vy)
    if new_cmd < 1e-3:
        return bearing_rel, 0.0
    new_cmd = min(new_cmd, max_speed_cmd)
    new_bear = _wrap_deg(math.degrees(math.atan2(vx, vy)) - hdg)
    return new_bear, new_cmd


# Teammate keep-out: the same decomposition as _enemy_guard against the teammate's
# reported position (None when the link is stale, so a dropped packet means no guard,
# never a block). It exists because the lidar tracker deliberately drops the teammate from
# the obstacle list, so without it two of our robots converging on a loose ball meet
# face-first. Slide-around only, with no restoring push: pushing against a robot that is
# itself steering is the guard-vs-guard limit cycle.
teammate_avoid_enabled = True
teammate_avoid_keep_mm = 260.0 # centre-to-centre keep-out, same ring as the enemy guard
teammate_avoid_slide_zone_mm = 300.0 # inward-damp band outside the keep line


def _teammate_guard(rx, ry, hdg, bearing_rel, cmd, teammate_pos):
    """like _enemy_guard for our teammate's reported (x, y): cancel the into-teammate component
    and damp inward commands in the slide band, so converging paths slide around each other.
    None (stale link) stands the guard down. No push term by design.
    """
    if not teammate_avoid_enabled or teammate_pos is None:
        return bearing_rel, cmd
    tx, ty = teammate_pos
    dx, dy = rx - tx, ry - ty # teammate -> robot
    dist = math.hypot(dx, dy)
    if dist < 1e-6 or dist > teammate_avoid_keep_mm + teammate_avoid_slide_zone_mm:
        return bearing_rel, cmd
    nx, ny = dx / dist, dy / dist
    bf = math.radians(bearing_rel + hdg)
    vx, vy = cmd * math.sin(bf), cmd * math.cos(bf) # field-frame command velocity
    vn = vx * nx + vy * ny # + = moving away from the teammate
    if dist <= teammate_avoid_keep_mm:
        if vn < 0.0: # cancel the into-teammate component
            vx -= vn * nx
            vy -= vn * ny
    elif vn < 0.0: # slide band, heading inward: damp
        factor = (dist - teammate_avoid_keep_mm) / teammate_avoid_slide_zone_mm
        closing = max(factor * vn, -wall_safe_speed_cmd)
        vx += (closing - vn) * nx
        vy += (closing - vn) * ny
    new_cmd = math.hypot(vx, vy)
    if new_cmd < 1e-3:
        return bearing_rel, 0.0
    new_cmd = min(new_cmd, max_speed_cmd)
    new_bear = _wrap_deg(math.degrees(math.atan2(vx, vy)) - hdg)
    return new_bear, new_cmd


# Jam recovery: a body-contact stuck detector for a robot jammed against another robot
# (walls are handled by the wall guards). An enemy dead ahead in contact range while our
# displacement stalls, debounced with hysteresis, escapes with a fixed-dwell full-power
# push, half toward the target bearing, half dead ahead. A real 50/50 shoving match is
# expected contact, handled by the contest speed, so callers pass skip=True for it.
# both bots' radius plus a small touching margin
jam_contact_mm = 2.0 * Perception.robot_radius_mm + 20.0
jam_ahead_deg = 35.0 # "opponent dead ahead" half-angle gate
jam_speed_mm_s = 90.0 # "going nowhere" own-speed floor
# Conservative on purpose: a goalie holding its line or a slow final approach still
# produces real displacement; own speed rarely stays this low this long unless something
# is blocking.
jam_enter_s = 1.2
jam_hold_s = 0.5 # push-through dwell once declared
stall_window_s = 0.5 # commanding but not moving this long triggers the stall escape
stall_cmd_floor = wall_safe_speed_cmd # slew-level floor that counts as "commanding real motion"
stall_dt_gap_s = 0.2 # a longer tick gap resets stall evidence (a pause, not stuckness)


class JamRecovery:
    """hysteresis jam detector plus push-through override, one per controller. Call check()
    once near the top of tick() with this tick's target bearing; a non-None result is the
    (drive_angle, speed_frac) to drive instead of the normal steering. None means carry on
    as normal.

    skip=True suppresses new evidence this tick (an armed push-through still runs out its
    dwell): pass it for an expected, already-handled contact. check() also has a
    displacement-stall trigger (stall_window_s): commanding motion but not moving escapes
    the same way even with no enemy read.
    """

    def __init__(self):
        self._last_pos = None # (rx, ry, t) for the own-speed estimate
        self._jam_s = 0.0 # accumulated jam evidence (s)
        self._stall_s = 0.0 # displacement-stall evidence (s)
        self._push_until_t = None # set once declared; push-through lasts
                                      # until this monotonic time

    def reset(self):
        self._last_pos = None
        self._jam_s = 0.0
        self._stall_s = 0.0
        self._push_until_t = None

    def check(self, rx, ry, hdg, bx, by, enemies, skip=False):
        now = time.monotonic()
        own_speed, dt = None, None
        if self._last_pos is not None:
            px, py, pt = self._last_pos
            dt = now - pt
            if dt > 1e-3:
                own_speed = math.hypot(rx - px, ry - py) / dt
        self._last_pos = (rx, ry, now)

        rel = _wrap_deg(math.degrees(math.atan2(bx - rx, by - ry)) - hdg)

        if self._push_until_t is not None:
            if now < self._push_until_t:
                # Half toward the target bearing, half dead ahead, at full power.
                return _wrap_deg(rel * 0.5), 1.0
            self._push_until_t = None # dwell elapsed, fall through

        if own_speed is None or dt is None:
            return None
        # Displacement-stall trigger: the drive stack commands real motion (slew
        # level above the wall-safe floor) yet measured speed stays under
        # jam_speed_mm_s for a full window. Catches wedges with no tracked enemy
        # in contact (the teammate isn't tracked; corner pockets). Evidence decays
        # at 2x, a tick gap past stall_dt_gap_s resets it, and skip=True stands it
        # down. The scripted duel freeze (zero translation) and a goalie holding
        # its line (near-zero command) can never reach the window.
        if skip or dt > stall_dt_gap_s:
            self._stall_s = 0.0
        elif (_last_slew_speed >= stall_cmd_floor
                and own_speed < jam_speed_mm_s):
            self._stall_s = min(stall_window_s, self._stall_s + dt)
            if self._stall_s >= stall_window_s:
                self._stall_s = 0.0
                self._push_until_t = now + jam_hold_s
                return _wrap_deg(rel * 0.5), 1.0
        else:
            self._stall_s = max(0.0, self._stall_s - 2.0 * dt)
        jammed = False
        if not skip:
            for e in enemies:
                ed = math.hypot(e["x"] - rx, e["y"] - ry)
                if ed > jam_contact_mm:
                    continue
                eb = math.degrees(math.atan2(e["x"] - rx, e["y"] - ry))
                if abs(_wrap_deg(eb - hdg)) <= jam_ahead_deg:
                    jammed = True
                    break
        if jammed and own_speed < jam_speed_mm_s:
            self._jam_s = min(jam_enter_s, self._jam_s + dt)
        else:
            self._jam_s = max(0.0, self._jam_s - 2.0 * dt)
        if self._jam_s >= jam_enter_s:
            self._jam_s = 0.0
            self._push_until_t = now + jam_hold_s
            return _wrap_deg(rel * 0.5), 1.0
        return None

# Goalie box stand-off arbitration. Inside our own box the keeper's clear-charge used to
# hand its ball distance to _enemy_guard (which then skips avoidance, the contest bypass)
# while the jam check stood down for the same contact. So a keeper charging a ball with an
# enemy camped on it was neither steered around nor recovered: full speed into a braced
# body, and any shove it landed pushed the ball toward our net.
#
# The rule is evidence, not geometry: a shoulder-lock is sustained zero displacement while
# the drive is really commanding a clear. "Stand off whenever the enemy is closer to the
# ball" was rejected, since closeness doesn't mean the enemy is between us and the ball.
standoff_window_s = 0.5 # sustained zero-displacement while commanding a real clear; the same
                             # window the C2 displacement-stall trigger uses
standoff_release_mm = 120.0 # ball travel that ends a stand-off: the enemy's control broke (the
                             # ball rolled off their dribbler) or the picture changed entirely


class BoxStandoff:
    """goalie-only arbitration for a box stand-off.

    Call check() once per tick just before the keeper's clear-charge gate. True means don't
    charge this tick: the keeper falls through to its goal-side line, where _enemy_guard has
    no contest bypass, so it slides around the enemy and stays between ball and net.

    Arming: evidence accumulates only while the drive commands a clear (slew above the
    wall-safe floor, so a keeper holding station can't arm it), displacement stays under
    jam_speed_mm_s, the ball is in our box, and a tracked enemy is camped on it. No enemy
    means no arm; JamRecovery's stall trigger covers that case.

    Release: one-way, undone only by the ball leaving the box, no enemy holding it, or the
    ball moving standoff_release_mm. No timer: a timed release would just re-arm the shove
    on a schedule.

    Kept apart from JamRecovery on purpose: its answer to a stall is a full-power push,
    which in front of our own net can only push the ball goalward.
    """

    def __init__(self):
        """fresh latch state for a new goalie run."""
        self.reset()

    def reset(self):
        """forget every accumulator and the latch (new run / role switch)."""
        self._last_pos = None # (rx, ry, t) for the own-speed estimate
        self._stall_s = 0.0 # accumulated stand-off evidence (s)
        self._hold_ball = None # ball position latched when the stand-off was declared

    def holding(self):
        """True while an armed stand-off is still live (the caller must not charge)."""
        return self._hold_ball is not None

    def check(self, rx, ry, bx, by, enemies, in_box):
        """one per tick; True = stand off instead of charging. in_box is the caller's test that
        the ball is inside our goal box, where a shoulder-lock can become an own goal.
        """
        now = time.monotonic()
        own_speed, dt = None, None
        if self._last_pos is not None:
            px, py, pt = self._last_pos
            dt = now - pt
            if dt > 1e-3:
                own_speed = math.hypot(rx - px, ry - py) / dt
        self._last_pos = (rx, ry, now)

        if self._hold_ball is not None:
            hx, hy = self._hold_ball
            if (not in_box or not _enemy_holds_ball(bx, by, enemies)
                    or math.hypot(bx - hx, by - hy) > standoff_release_mm):
                self._hold_ball = None
                return False
            return True

        if own_speed is None or dt is None or dt > stall_dt_gap_s:
            # no usable displacement sample (first tick), or a tick gap that
            # must never count as stand-off time: evidence resets
            self._stall_s = 0.0
            return False
        if (in_box and _enemy_holds_ball(bx, by, enemies)
                and _last_slew_speed >= stall_cmd_floor
                and own_speed < jam_speed_mm_s):
            self._stall_s = min(standoff_window_s, self._stall_s + dt)
            if self._stall_s >= standoff_window_s:
                self._stall_s = 0.0
                self._hold_ball = (bx, by)
                return True
        else:
            self._stall_s = max(0.0, self._stall_s - 2.0 * dt)
        return False


# Rush: use rush_speed only when the straight path ahead is clear of enemies (reasoned,
# not field-validated). How far ahead an enemy still counts as blocking:
rush_lookahead_mm = 1200.0
# perpendicular half-width of the lane (about 2x robot_radius_mm plus margin)
rush_lane_half_width_mm = 300.0


def _lane_clear(rx, ry, hdg, bearing_rel, enemies):
    """True if no tracked enemy sits in the lane ahead along bearing_rel: the gate for
    rush_speed over base_speed.
    """
    if not enemies:
        return True
    bf = math.radians(bearing_rel + hdg)
    fwd_x, fwd_y = math.sin(bf), math.cos(bf) # unit vector, field frame
    for e in enemies:
        dx, dy = e["x"] - rx, e["y"] - ry
        ahead = dx * fwd_x + dy * fwd_y # projection onto the path
        lateral = dx * fwd_y - dy * fwd_x # perpendicular offset
        if 0.0 < ahead <= rush_lookahead_mm and abs(lateral) <= rush_lane_half_width_mm:
            return False
    return True
# Noise gate on the ball-velocity lead. A still ball's velocity estimate jitters around
# zero, so mixing in anything above 1e-6 mm/s flipped the lead on and off every tick and
# the bearing (and spin) snapped side to side. Below ball_vel_fade_min_mms it contributes
# nothing, above ball_vel_fade_full_mms fully, and it fades in linearly between.
ball_vel_fade_min_mms = 150.0
ball_vel_fade_full_mms = 450.0


def _add_ball_velocity(drive_deg, cmd, hdg, vbx, vby):
    """capture command mixing: turn the command (robot-frame bearing, command units) into a
    velocity, add the ball's field velocity rotated into the robot frame (through the fade
    gate above), and convert back.
    """
    v_cap = cmd / max_speed_cmd * vmax_full_cmd_mms
    d_rad = math.radians(drive_deg)
    vx = v_cap * math.sin(d_rad)
    vy = v_cap * math.cos(d_rad)

    vb = math.hypot(vbx, vby)
    if vb > ball_vel_fade_min_mms:
        w = min(1.0, (vb - ball_vel_fade_min_mms)
                / (ball_vel_fade_full_mms - ball_vel_fade_min_mms))
        a_rel = math.atan2(vbx, vby) - math.radians(hdg) # field -> robot
        vx += w * vb * math.sin(a_rel)
        vy += w * vb * math.cos(a_rel)

    v = math.hypot(vx, vy)
    if v < 1e-6:
        return drive_deg, 0.0
    bearing = math.degrees(math.atan2(vx, vy))
    new_cmd = v / vmax_full_cmd_mms * max_speed_cmd
    # Motor.drive would scale an over-budget command down anyway; cap it here so the
    # returned (and logged) number is one the chassis can hold on this bearing
    cap = _vmax_mms(bearing) / vmax_full_cmd_mms * max_speed_cmd
    return bearing, min(new_cmd, cap)
def _orbit_approach(bx, by, rx=None, ry=None, hdg=None):
    """direct pursuit to a ball at robot-frame (bx, by) (forward = +y, right = +x): point
    straight at it. The old tangent-circle orbit missed the ball outright on hardware.

    With our pose supplied, the bearing is first bent along any wall we're already close to
    (the same active_walls data _wall_guard uses), blended toward each blocking wall's
    tangent as clearance shrinks. _wall_guard can only react to the bearing it's handed, so
    a target pointing into a wall, or into a corner pocket, would fight it every tick; this
    curls the target along the wall early and leaves _wall_guard as the safety net.
    """
    if math.hypot(bx, by) < 1e-6:
        return 0.0, 0.0
    rel = math.degrees(math.atan2(bx, by))
    if not wall_keepout_enabled or rx is None:
        return rel, 1.0
    world = math.radians(_wrap_deg(rel + hdg))
    vx, vy = math.sin(world), math.cos(world)
    walls = FieldModel.active_walls(rx, ry, keep_min_mm + wall_slide_zone_mm)
    for dist, nx, ny, keep in walls:
        if nx == 0.0 and ny == 0.0:
            continue
        into = -(vx * nx + vy * ny) # positive: this target currently points into the wall
        if into <= 0.0:
            continue
        # 0 at keep + wall_slide_zone_mm out, rising to 1 at the keep line (the
        # slide-zone ramp)
        factor = max(0.0, min(1.0, (keep + wall_slide_zone_mm - dist) / wall_slide_zone_mm))
        if factor <= 0.0:
            continue
        tx, ty = -ny, nx # tangent along the wall, signed to keep progress toward the target
        if tx * vx + ty * vy < 0.0:
            tx, ty = -tx, -ty
        weight = factor * min(1.0, into)
        vx += weight * (tx - vx)
        vy += weight * (ty - vy)
        mag = math.hypot(vx, vy)
        if mag > 1e-6:
            vx, vy = vx / mag, vy / mag
    new_rel = _wrap_deg(math.degrees(math.atan2(vx, vy)) - hdg)
    return new_rel, 1.0
# sideline-route waypoint: this far off the enemy goal's centre x, out at the sideline, so
# the last stretch is a short cross-face drive outside the goal box
sideline_route_corner_x_mm = 330.0
# Terminal curve-in (see the curve_in constants): blend the ball bearing toward a point
# set back behind the ball on the goal line, by a fraction that ramps up as range closes.
# With no goal or pose it passes _orbit_approach's answer straight through.
def _curve_in_approach(bx, by, rx=None, ry=None, hdg=None, goal=None):
    """robot-frame bearing and speed fraction for the final approach: direct pursuit at range,
    curling onto the ball mouth-aligned as it closes.
    """
    bearing, frac = _orbit_approach(bx, by, rx, ry, hdg)
    if not curve_in_enabled or goal is None or rx is None:
        return bearing, frac
    dist_mm = math.hypot(bx, by)
    if dist_mm >= curve_in_near_mm:
        return bearing, frac # far out: pure direct pursuit, unchanged
    gx, gy = goal
    # the ball in the field frame, then the approach point behind it on the ball-goal
    # line
    world = math.radians(_wrap_deg(bearing + hdg))
    fx, fy = rx + dist_mm * math.sin(world), ry + dist_mm * math.cos(world)
    dgx, dgy = gx - fx, gy - fy
    dg = math.hypot(dgx, dgy) or 1.0
    # the offset scales with the ramp and the goal offset shrinks near the goal line,
    # so the curve eases out; past the goal line the straight bearing wins
    ramp = max(0.0, min(1.0, (curve_in_near_mm - dist_mm)
                        / max(1.0, curve_in_near_mm - curve_in_full_mm)))
    stand = curve_in_full_mm * (1.0 - 0.5 * min(1.0, dg / curve_in_full_mm))
    tx, ty = fx + dgx / dg * stand, fy + dgy / dg * stand
    tvx, tvy = tx - rx, ty - ry
    if math.hypot(tvx, tvy) < 1e-6:
        return bearing, frac
    target_rel = _wrap_deg(math.degrees(math.atan2(tvx, tvy)) - hdg)
    off = _wrap_deg(target_rel - bearing)
    if abs(off) > curve_in_max_off_deg:
        off = curve_in_max_off_deg if off > 0.0 else -curve_in_max_off_deg
    return _wrap_deg(bearing + ramp * off), frac


# Ball hiding along a sideline. Far from goal with an enemy upfield and nobody in front yet,
# gain depth with the ball tucked on the wall side of the body, so a defender has to come
# around us to reach it; the heading does the hiding. Armed on distance to the enemy goal,
# not on the lane being blocked (that's the sideline route). It starts at 900 and ends at
# 600 mm of goal depth (hysteresis), runs at 0.8 of cruise, and tucks 15 mm short of where
# the wall guard takes over: our guard holds the body centre keep_min_mm (150) off the wall,
# so the tuck line is keep_min_mm + ball_hiding_line_margin_mm.
ball_hiding_enabled = True
ball_hiding_start_dist_mm = 900.0 # goal depth at which hiding arms
ball_hiding_end_dist_mm = 600.0 # ...and releases below this (hysteresis)
ball_hiding_line_threshold_mm = 140.0 # tuck line from the wall, before our guard's offset
ball_hiding_line_margin_mm = 15.0 # slack between the tuck and the wall guard
ball_hiding_speed_ratio = 400.0 / 500.0 # hide speed as a fraction of the carry speed
# Hide only when an enemy is actually upfield of us.
ball_hiding_enemy_gate = True


def _ball_hide_step(hiding, tucked, rx, ry, hdg, enemy_goal, enemies, ball_xy=None):
    """one tick of the sideline ball hide.

    hiding and tucked are the caller's state. Returns (hiding, tucked, plan): plan is None
    while inactive, else (drive_rel, heading_rel, speed_frac_ratio, near_line), robot-frame.
    Until the line is reached the drive points at the wall and the heading faces it, so the
    ball sits on the wall side; after that the drive turns upfield while the heading keeps
    facing the wall.

    tucked is latched: re-deriving it every tick flipped
    between legs whenever the robot drifted a few mm out of the band, and against our
    guard's restoring push that limit-cycled (71 bearing flips in one 600-tick carry). The
    arrival is one-way, like the start/end band.

    It returns bearings and doesn't pre-empt the caller's steering. ball_xy, when given, is
    what the goal distance is measured to: the ball is a mouth-length ahead of us.
    """
    if not ball_hiding_enabled:
        return False, False, None # the flag also cancels a hide in flight, not just the arming
    gx, gy = enemy_goal
    # +1 when the enemy goal is the far one in y
    into = 1.0 if gy >= FieldModel.field_y / 2.0 else -1.0
    depth_y = ry if ball_xy is None else ball_xy[1]
    goal_depth_mm = into * (gy - depth_y) # how far the ball still has to travel to the goal
    # Arm only with an enemy upfield: hiding exists to hide from someone, and their
    # goalie camping the box counts. The flag reverts to open-field arming. Release stays
    # distance-only, since an enemy closing in once tucked is when hiding is worth
    # most.
    enemy_ahead = any((e["y"] - ry) * into > 0.0 for e in (enemies or ()))

    if not hiding and (not ball_hiding_enemy_gate or enemy_ahead) and goal_depth_mm >= ball_hiding_start_dist_mm:
        hiding = True
    elif hiding and goal_depth_mm < ball_hiding_end_dist_mm:
        hiding = False
    if not hiding:
        return False, False, None

    # Its side pick: whichever sideline we are nearer, and then everything else follows that wall.
    side = 1.0 if rx >= FieldModel.cx else -1.0
    wall_x = FieldModel.field_x if side > 0 else 0.0
    wall_dist_mm = abs(wall_x - rx)
    near_line = wall_dist_mm <= keep_min_mm + ball_hiding_line_margin_mm
    tucked = tucked or near_line

    # face the wall so the ball rides between us and it: +90 deg for the right wall,
    # -90 for the left (drive and heading coincide until the tuck ends)
    heading_rel = _wrap_deg(90.0 * side - hdg)
    drive_rel = (_wrap_deg(math.degrees(math.atan2(gx - rx, gy - ry)) - hdg)
                 if tucked else heading_rel)
    return True, tucked, (drive_rel, heading_rel, ball_hiding_speed_ratio, near_line)


# Sideline carry route: the flank waypoint, from where the carry began (ax, ay) and the
# enemy goal: straight out to the near sideline, clear of the goal box by the corner
# offset.
def _sideline_route_point(ax, ay, enemy_goal):
    """(wx, wy) of the route's mid waypoint, or None when the geometry doesn't call for it.
    Only carries starting in the central band and our own half get the route: one already
    out wide has no escape leg, one deep in attack should drive straight in, and one in our
    own goal box should leave the short way.
    """
    gx, gy = enemy_goal
    corner = sideline_route_corner_x_mm
    margin = Perception.robot_radius_mm + 40.0 # body clears the wall, not just the centre
    if abs(ay - gy) < FieldModel.field_y / 2.0:
        return None # already deep in the attacking half
    # nearer sideline, the same side-pick as the recovery bias
    side = 1.0 if ax >= FieldModel.cx else -1.0
    edge = FieldModel.bx1 if side > 0 else FieldModel.bx0
    own_gy = 0.0 if gy > FieldModel.field_y / 2 else FieldModel.field_y
    if _in_own_box(ax, ay, (FieldModel.cx, own_gy),
                   Perception.robot_radius_mm):
        return None # inside our own goal box: drive out the short way, don't flank
    if (ax - edge) * side >= corner + margin:
        return None # already out wide past the waypoint line, no leg to make
    return (edge + side * (corner + margin), ay)


recovery_standoff_mm = 350.0 # clear of the ball, so the next approach starts fresh
# lateral bias for the recovery target: without it the target sits on the ball-to-goal
# line a beaten robot is often already on, where _enemy_guard can only damp a dead-on
# approach and the drive stalls. Biasing to the side we already lean guarantees a swing
# round
recovery_lateral_mm = 300.0


def _ball_beaten(ry, by, own_goal):
    """True if the ball is deeper into our half than we are: an attacker has beaten us. Driving
    straight at it would hit their back while the shot lane stays open, so recover goal-side
    first.
    """
    own_gy = own_goal[1]
    into = 1.0 if own_gy < FieldModel.field_y / 2 else -1.0
    return into * (ry - own_gy) > into * (by - own_gy)


def _defend_recovery_target(rx, ry, bx, by, own_goal, standoff_mm, lateral_mm):
    """a point standoff_mm from the ball toward our own goal, offset lateral_mm to the side of
    the ball-goal line we're already on, so the next approach meets the attacker's front.
    """
    own_gx, own_gy = own_goal
    dxg, dyg = own_gx - bx, own_gy - by
    dg = math.hypot(dxg, dyg) or 1.0
    ux, uy = dxg / dg, dyg / dg # unit vector, ball -> own goal
    lx, ly = -uy, ux # perpendicular
    side = 1.0 if (rx - bx) * lx + (ry - by) * ly >= 0.0 else -1.0
    return (bx + ux * standoff_mm + lx * side * lateral_mm,
            by + uy * standoff_mm + ly * side * lateral_mm)


def _in_own_box(x, y, own_goal, margin):
    """True if a robot circle of radius `margin` at (x, y) touches our own goal box."""
    _gx, gy = own_goal
    into = 1.0 if gy < FieldModel.field_y / 2 else -1.0
    bx0 = FieldModel.bx0 - margin
    bx1 = FieldModel.bx1 + margin
    gd = FieldModel.goal_depth + margin
    if not (bx0 <= x <= bx1):
        return False
    return y <= gd if into > 0 else y >= FieldModel.field_y - gd


def _teammate_blocks_own_box(teammate_pos, own_goal):
    """True only when the teammate's fresh reported position is in or touching our defending
    box, so we hold back rather than also enter (RCJA 5.11). A stale link gives None, which
    allows entry.
    """
    if teammate_pos is None:
        return False
    tx, ty = teammate_pos
    return _in_own_box(tx, ty, own_goal, Perception.robot_radius_mm)


# real clearance past the box edge for a hold target, so a settling robot doesn't drift
# back across the line and re-trigger the guard every other tick
box_hold_clear_mm = 60.0


def _clamp_outside_own_box(x, y, own_goal, margin):
    """push (x, y) clear of our own goal box (plus box_hold_clear_mm) if it lands inside or
    touching it.
    """
    _gx, gy = own_goal
    into = 1.0 if gy < FieldModel.field_y / 2 else -1.0
    bx0 = FieldModel.bx0 - margin
    bx1 = FieldModel.bx1 + margin
    gd = FieldModel.goal_depth + margin
    if bx0 <= x <= bx1:
        if into > 0 and y <= gd:
            y = gd + box_hold_clear_mm
        elif into < 0 and y >= FieldModel.field_y - gd:
            y = FieldModel.field_y - gd - box_hold_clear_mm
    return x, y
