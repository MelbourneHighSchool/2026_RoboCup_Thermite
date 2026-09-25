"""Drive layer: speed model, keep-out guards (wall/enemy), jam/beaten recovery,
carry/shield possession creep, pass-lane race gate, and finishing-angle
selection. The largest, most cross-cutting module in bot/ - most functions
here take explicit args (rx, ry, hdg, enemies, ...) rather than reading
bot.state directly; none of the functions extracted into this module touch
_state/_lock (verified against the source range this was cut from - the only
_state/_lock use in that range, in _wheel_odom_delta_mm, is odometry, not
motion, and stayed in bot/odometry.py).

_add_ball_velocity lives here (not in bot/tracking.py, which explicitly left
it behind for this reason - see its own module docstring): it converts a
capture command into robot-frame mm/s using max_speed_cmd/vmax_full_cmd_mms/
_vmax_mms, the drive-speed model defined below, not anything from tracking.

Documented per-bot gaps (same pattern as bot1_config.py/bot2_config.py
elsewhere, e.g. bot/lidar.py, bot/vision.py): wall_slide_zone_mm, rush_speed,
base_speed and handle_exclusion_deg are per-bot (already in bot1_config.py/
bot2_config.py) and are referenced here only inside function bodies, left
undefined at module scope until the ROBOT_ID selection mechanism lands.
pass_eject_speed and flick_range_mm below are NOT per-bot gaps - they hold
the identical value in both mainrunbot1.py and mainrunbot2.py (0.05, 900.0)
- so they're copied here as plain module constants (this module's only use
of them is the drive-speed-model math in pass_ball_speed_mms/
carry_1v1_gap_mm); the rest of the pass/flick/tap-in logic they otherwise
feed stays unextracted, for a later possession-controller stage.

_last_slew_speed (module-level, used by _slew_drive) is this module's own
rebound global for per-call acceleration-slew state - code outside this
module must read/write it via `import bot.motion as motion;
motion._last_slew_speed`, never a from-import.
"""

import math
import time

from bot.field import FieldModel, wrap_deg as _wrap_deg
from bot.hardware import Motor
from bot.perception import Perception

# Per-bot gap: pass/flick constants (identical across bots, see module
# docstring) needed by this module's own drive-speed-model math
pass_eject_speed = 0.05
flick_range_mm   = 900.0

# Ball capture: direct-pursuit steering (see _orbit_approach), one system for StrikerController, GoalieController, and CamRunController alike (sec 4.1/4.9).
capture_cone_half_deg      = 20.0 # half-angle about the dwibbler direction
capture_cone_half_width_mm = 35.0 # lateral half-width (dwibbler mouth half-width)

# Drive speed model: normalised command <-> mm/s. One wheel at full scale
# is Motor.MAX_REV_PER_S, body speed is sqrt2 times rim speed. Re-check if
# the wheels are ever changed.
wheel_diameter_mm  = 50.0
max_speed_cmd      = 1.0 # full scale, in the normalised units
vmax_full_cmd_mms  = (math.sqrt(2.0) * math.pi * wheel_diameter_mm
                      * Motor.MAX_REV_PER_S) # about 7346 mm/s, uncapped

# Bearing matters too: achievable fraction of vmax is MOTOR_CAP_FRAC/(|c|+|s|), 0.90 on a cardinal bearing, 0.636 on a diagonal.

# Per-loop acceleration slew-rate cap:
# the commanded translation |speed| may only RISE this much per control loop
# (loop_dt, about 0.02s / 50Hz - see loop_dt below). Deceleration and
# direction reversals are never capped, only a rising magnitude is, matching
# Motor.drive's own "direction changes instant" design (see its ACCEL/DECEL
# comment in MotorFuncs_Proto1.py) - this only smooths a sudden jump straight
# to a big command, it doesn't re-introduce a general drive ramp. At 0.06 per
# loop / 50Hz, 0 -> rush_speed (0.5, see bot1_config.py/bot2_config.py) takes
# about 9 loops (~0.18s) and 0 -> full
# scale (1.0) about 17 loops (~0.33s): fast enough to stay responsive, slow
# enough that a sudden full-speed command doesn't ask real wheels for
# instant torque and skid/tip. overrideAcc bypasses the cap entirely for
# calls that need instant full response regardless (final ball-capture
# press, emergency wall/enemy pushback).
ACCEL_SLEW_PER_LOOP = 0.06
_last_slew_speed    = 0.0


def _slew_drive(bearing, speed, rot_r=0.0, rot_theta=0.0, rot_speed=0.0, overrideAcc=False):
    """Motor.drive wrapper that caps how fast commanded translation |speed| can rise per control loop (never caps a decrease, a direction reversal, or rot_speed - only a rising magnitude is ramped). overrideAcc=True sends the command straight through, uncapped."""
    global _last_slew_speed
    speed = float(speed)
    if not overrideAcc and speed > _last_slew_speed:
        speed = min(speed, _last_slew_speed + ACCEL_SLEW_PER_LOOP)
    _last_slew_speed = speed
    Motor.drive(bearing, speed, rot_r=rot_r, rot_theta=rot_theta, rot_speed=rot_speed)


def _vmax_mms(bearing_deg):
    """top speed (mm/s) the chassis can actually reach along a robot-frame bearing, i.e. full command less Motor.drive's per-wheel cap."""
    rad  = math.radians(bearing_deg)
    c, s = abs(math.cos(rad)), abs(math.sin(rad))
    peak = c + s # peak wheel command per unit of translation command
    return vmax_full_cmd_mms * Motor.MOTOR_CAP_FRAC / peak


def _brake_speed_frac(dist_mm):
    """max command fraction that can still brake to a stop in dist_mm, given a conservative achievable-deceleration estimate (Motor.DECEL_MAX_FRAC_PER_S * vmax_full_cmd_mms - no longer an active ramp, see COLLISION_ACCEL_G's own comment, kept only as this floor). A stopping distance scales with speed squared (real mass, F=ma, not an instant-turn point mass), so easing off linearly with distance - as every "hold this point" drive used to - lets a fast approach still be moving once it arrives, overshoot, and correct back past it. This is the same profile in reverse: the safe speed at a given distance, not an arbitrary ramp divisor."""
    a_max = Motor.DECEL_MAX_FRAC_PER_S * vmax_full_cmd_mms # mm/s^2
    return math.sqrt(max(0.0, 2.0 * a_max * dist_mm)) / vmax_full_cmd_mms
# Pass race-condition gate (sec 3.21): score a worst-case race, matching simulator.py's own model.
dwibble_rim_r_m    = 0.014 # roller radius (28 mm diameter incl. the PU wrap)
pass_ball_speed_mms = (pass_eject_speed * Motor.MAX_RPM / 60.0
                       * 2.0 * math.pi * dwibble_rim_r_m * 1000.0)
# pass_race_speed_mms (base_speed / max_speed_cmd * vmax_full_cmd_mms in the
# original) is deliberately NOT a module constant here: base_speed is the
# per-bot gap documented in this module's docstring, and a module-level
# expression referencing it would make importing this module fail outright
# (unlike a gap used only inside a function body, which merely raises if that
# function is ever actually called). It is computed inline in
# _pass_race_open below instead - same value, same per-call cost (two
# float divides), just deferred past import time.
# receiver/target must beat every tracked enemy to the pass by at least this long
pass_race_margin_s  = 0.15


def _pass_race_open(passer_xy, target_xy, enemies, enemy_vel=None):
    """race half of the pass-lane gate: True unless some tracked enemy could reach `target_xy` at pass_race_speed_mms at least pass_race_margin_s before the ball itself (travelling at pass_ball_speed_mms) would get there. enemy_vel is an optional {id: (vx, vy)} dict (EnemyVelocityTracker's own return shape, _state["enemy_vel"]): when given, an enemy's own position is led forward by the ball's travel time first - an enemy already closing on the target is a real threat sooner than its last static fix would suggest. Enemies with no id/velocity entry (or when enemy_vel is None) fall back to the old static-position race, unchanged."""
    pass_race_speed_mms = base_speed / max_speed_cmd * vmax_full_cmd_mms
    px, py = passer_xy
    tx, ty = target_xy
    dist_pt = math.hypot(tx - px, ty - py)
    if dist_pt < 1.0 or not enemies:
        return True
    t_ball = dist_pt / pass_ball_speed_mms
    for e in enemies:
        ex, ey = e["x"], e["y"]
        if enemy_vel is not None:
            vx, vy = enemy_vel.get(e.get("id"), (0.0, 0.0))
            ex, ey = ex + vx * t_ball, ey + vy * t_ball
        t_enemy = math.hypot(ex - tx, ey - ty) / pass_race_speed_mms
        if t_enemy < t_ball + pass_race_margin_s:
            return False
    return True
# Finishing-angle selection (README sec. 3.22-adjacent): when driving the
# ball into the enemy goal, instead of only ever aiming at the goal-line
# centre, evaluate multiple candidate finishing bearings - direct shots
# across the mouth plus a bank-shot off the far mouth wall - and drive at
# whichever one actually has a clear, scoring path. This file's own bearing
# convention is math.atan2(dx, dy) (0 = straight +y, positive toward +x) with
# y as the depth axis (goal lines at y=0/field_y) and x as the mouth-width
# axis (FieldModel.sx0/sx1); every function below works in that frame directly.
#
# IMPORTANT (see the "no candidate scores" investigation note further
# down, near where this is used in StrikerController): _goal_shot_aim can
# legitimately return aim_found=False - e.g. every candidate is blocked,
# or the ball is already past the goal line so back_y - ball_y is ~0 and
# the function bails out early. Callers MUST treat aim_found=False as
# "no opinion, use the existing fallback bearing", never index into a
# None aim_bearing.
finish_ball_radius_mm = 21.5 # ball_diameter_mm / 2
finish_post_clear_mm  = finish_ball_radius_mm + 5.0 # half a nominal goal-line/post width
finish_side_wall_clearance_deg = 5.0 # min angular clearance off a mouth wall for a bank shot


def _finish_robot_clear_mm():
    """Enemy-corridor clearance for a shot path: robot radius plus ball radius. A function (not a module constant) so it always reflects Perception.robot_radius_mm even though Perception is defined above this point in the file."""
    return Perception.robot_radius_mm + finish_ball_radius_mm


def _goal_depth_lines(goal_y_line):
    """(mouth_y, back_y): depth lines for the goal whose goal-line sits at
    goal_y_line (0 or FieldModel.field_y). mouth_y is the entrance to the
    box (FieldModel.goal_depth in from the line, the shot target's
    'reachable' edge); back_y is the narrower net's back wall
    (goal_depth - slot_depth in)."""
    direction = 1.0 if goal_y_line <= FieldModel.field_y / 2.0 else -1.0
    mouth_y = goal_y_line + direction * FieldModel.goal_depth
    back_y  = goal_y_line + direction * (FieldModel.goal_depth - FieldModel.slot_depth)
    return mouth_y, back_y


def _finish_mouth_sector(ball_x, ball_y, mouth_y):
    """Near/far mouth-wall bearings and the opposite inside wall's x, for a
    shot taken from (ball_x, ball_y) at depth mouth_y."""
    mouth_x_min = FieldModel.sx0 + finish_ball_radius_mm
    mouth_x_max = FieldModel.sx1 - finish_ball_radius_mm
    if ball_x >= (FieldModel.sx0 + FieldModel.sx1) / 2.0:
        near_x, far_x = mouth_x_max, mouth_x_min
        opposite_x = FieldModel.sx0 + finish_ball_radius_mm
    else:
        near_x, far_x = mouth_x_min, mouth_x_max
        opposite_x = FieldModel.sx1 - finish_ball_radius_mm
    near_bearing = math.degrees(math.atan2(near_x - ball_x, mouth_y - ball_y))
    far_bearing  = math.degrees(math.atan2(far_x - ball_x, mouth_y - ball_y))
    return near_bearing, far_bearing, opposite_x


def _finish_ray_hit(ball_x, ball_y, bearing_deg, wall_x=None, wall_y=None):
    """Intersection of a forward ray at bearing_deg (0=+y, atan2(dx,dy)
    convention) with x=wall_x or y=wall_y. Returns (x, y) or None."""
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


def _finish_scoring_path(ball_x, ball_y, bearing_deg, back_y, mouth_y, mouth_sector):
    """Direct/rebound ball-path segments when bearing_deg scores, else None."""
    near_bearing, far_bearing, opposite_x = mouth_sector
    span = _wrap_deg(far_bearing - near_bearing)
    from_near = _wrap_deg(bearing_deg - near_bearing)
    if abs(span) + 1e-9 < finish_side_wall_clearance_deg:
        return None
    if from_near * span <= 0:
        return None
    if abs(from_near) + 1e-9 < finish_side_wall_clearance_deg:
        return None
    if abs(from_near) > abs(span) + 1e-9:
        return None
    if not _finish_clears_posts(ball_x, ball_y, bearing_deg, mouth_y):
        return None

    start = (ball_x, ball_y)
    back_hit = _finish_ray_hit(ball_x, ball_y, bearing_deg, wall_y=back_y)
    side_hit = _finish_ray_hit(ball_x, ball_y, bearing_deg, wall_x=opposite_x)
    hits_side_first = (
        side_hit is not None
        and min(mouth_y, back_y) <= side_hit[1] <= max(mouth_y, back_y)
        and abs(side_hit[1] - ball_y) + 1e-9 < abs(back_y - ball_y)
    )
    if not hits_side_first:
        if back_hit is not None and FieldModel.sx0 <= back_hit[0] <= FieldModel.sx1:
            return [(start, back_hit)]
        return None

    # A wall bounce off the mouth's side wall reverses the x component of
    # the ball's direction, i.e. the rebound bearing is -bearing_deg.
    rebound_hit = _finish_ray_hit(side_hit[0], side_hit[1], -bearing_deg, wall_y=back_y)
    if rebound_hit is None or not FieldModel.sx0 <= rebound_hit[0] <= FieldModel.sx1:
        return None
    return [(start, side_hit), (side_hit, rebound_hit)]


def _finish_kick_scores(ball_x, ball_y, bearing_deg, back_y, mouth_y,
                         enemy_bot_positions=None, mouth_sector=None):
    """True when bearing_deg scores and its complete path clears every
    tracked enemy (each treated as a robot-radius-plus-ball-radius
    corridor around its centre)."""
    if mouth_sector is None:
        mouth_sector = _finish_mouth_sector(ball_x, ball_y, mouth_y)
    path = _finish_scoring_path(ball_x, ball_y, bearing_deg, back_y, mouth_y, mouth_sector)
    if path is None:
        return False
    clearance = _finish_robot_clear_mm()
    for e in enemy_bot_positions or ():
        bx, by = e["x"], e["y"]
        if any(_finish_point_to_segment_distance(bx, by, start, end) <= clearance
               for start, end in path):
            return False
    return True


def _goal_shot_aim(ball_x, ball_y, goal_y_line, enemy_bot_positions=None):
    """Return (aim_bearing_deg, True) for the best-scoring direct or
    bank-shot finishing angle into the goal whose line is at goal_y_line,
    or (None, False) if nothing evaluated clears every tracked enemy (the
    caller falls back to the plain goal-centre bearing)."""
    mouth_y, back_y = _goal_depth_lines(goal_y_line)
    dx_back = back_y - ball_y
    if abs(dx_back) < 1e-6:
        return None, False

    # Bail if the ball is already at or past the goal's own back wall (the
    # candidate math below assumes the ball still has to travel FORWARD
    # through the mouth to reach back_y; once it's already deeper than
    # back_y, back_y sits behind the ball rather than ahead of it, and the
    # formulas below produce a self-consistent but geometrically backward
    # bearing - e.g. pointing back out of the goal instead of further in.
    # This can legitimately happen against this test harness (no real
    # goal-scoring detection, see commit 020fa69's note) but is also just
    # generally the "already effectively in the net" regime where no
    # finishing-angle search is needed: fall back to the caller's own
    # plain direct bearing instead of trusting a candidate here.
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

    candidates = []
    if aim_x_min <= aim_x_max:
        # Prefer the centre, but try off-centre direct shots when a bot blocks it.
        for fraction in (0.5, 0.25, 0.75, 0.0, 1.0):
            aim_x = aim_x_min + (aim_x_max - aim_x_min) * fraction
            candidates.append(math.degrees(math.atan2(aim_x - ball_x, back_y - ball_y)))

    # Bank shot: opposite inside wall, as close to the back as possible, with near-wall clearance.
    mouth_sector = _finish_mouth_sector(ball_x, ball_y, mouth_y)
    near_bearing, far_bearing, opposite_x = mouth_sector
    span = _wrap_deg(far_bearing - near_bearing)
    if abs(span) + 1e-9 >= finish_side_wall_clearance_deg:
        ideal = math.degrees(math.atan2(opposite_x - ball_x, back_y - ball_y))
        clear = _wrap_deg(near_bearing + math.copysign(finish_side_wall_clearance_deg, span))
        from_near = _wrap_deg(ideal - near_bearing)
        if from_near * span > 0 and abs(from_near) + 1e-9 >= finish_side_wall_clearance_deg:
            candidates.append(ideal if abs(from_near) <= abs(span) + 1e-9 else far_bearing)
        else:
            candidates.append(clear)

    for aim in candidates:
        if _finish_kick_scores(ball_x, ball_y, aim, back_y, mouth_y,
                                enemy_bot_positions, mouth_sector=mouth_sector):
            return aim, True
    return None, False
# Carry/shield decision (sec 3.23): has_ball's drive-in fallback (below, once no
# tap-in/pass/flick fired) used to always push straight at goal regardless of
# whether the lane was actually winnable. Ported from an earlier simulator.py
# "carry" state (commit 934bb0f, formation/possession redesign part 2) plus a
# decoupled heading/translation drive technique: hold a shield heading fixed
# while translating independently of it, instead of a stationary hold-and-turn.
# Our own trigger (_carry_should_press, defender-geometry based) drives when
# to creep.
carry_1v1_gap_mm       = flick_range_mm * 1.2 # 2nd-nearest enemy must be at
                             # least this far off to call it a genuine 1v1,
                             # not a double-team
carry_level_y_mm       = Perception.robot_radius_mm * 2.5
carry_level_x_mm       = 500.0
carry_lane_margin_mm   = Perception.robot_radius_mm * 0.7
carry_engage_dist_mm   = 700.0 # shield-facing kicks in once the nearest enemy
                             # is this close
carry_shield_goal_bias = 0.25 # fraction of the shield heading blended back
                             # toward goal, so it isn't facing directly
                             # backward the whole time
# Creep (drive-while-shielding): while shielding, translate away from the
# blocking defender (working clear into open space) blended with a small
# forward-to-goal component, capped well below base_speed so this stays a
# deliberate creep, never a full-speed dash that would undermine the shield.
carry_creep_speed_frac  = 0.5  # fraction of base_speed used for the creep
carry_creep_away_weight = 0.65 # blend: away-from-defender vs toward-goal
                              # (1.0 = pure sideways-away, 0.0 = pure upfield)


def _path_blocked(sx, sy, gx, gy, enemies, margin):
    """True if a tracked enemy's body sits in the straight corridor between
    (sx, sy) and (gx, gy) - circle-vs-segment perpendicular-distance check,
    clamped to the segment."""
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


def _carry_should_press(rx, ry, gx, gy, enemies):
    """True if there's a genuine, winnable central lane to goal right now -
    False means hold up and shield (_shield_heading_deg/_carry_creep) instead
    of driving into or detouring around traffic. Only a genuine 1v1 (no
    second real threat), or a defender that's roughly level with us but too
    far horizontally to cut the lane off in time, make pressing worth it."""
    if not enemies:
        return True
    if _path_blocked(rx, ry, gx, gy, enemies, carry_lane_margin_mm):
        return False
    nearest = min(enemies, key=lambda e: math.hypot(e["x"] - rx, e["y"] - ry))
    others  = [e for e in enemies if e is not nearest]
    other_dist = min((math.hypot(e["x"] - rx, e["y"] - ry) for e in others),
                     default=float("inf"))
    genuine_1v1   = other_dist > carry_1v1_gap_mm
    level_and_far = (abs(nearest["y"] - ry) < carry_level_y_mm
                     and abs(nearest["x"] - rx) > carry_level_x_mm)
    return genuine_1v1 or level_and_far


def _shield_heading_deg(rx, ry, hdg, gx, gy, enemies,
                        engage_dist_mm=carry_engage_dist_mm):
    """Relative heading (deg, same convention as goal_rel elsewhere - feed
    straight into rot_speed via turn_gain) to turn by so the chassis shields
    the ball (carried at the bot's own front) from the nearest enemy: face
    away from the nearest enemy, biased carry_shield_goal_bias back toward
    goal so it isn't facing directly backward. Beyond engage_dist_mm there's
    no real pressure yet, so this just faces goal."""
    nearest = min(enemies, key=lambda e: math.hypot(e["x"] - rx, e["y"] - ry))
    de = math.hypot(nearest["x"] - rx, nearest["y"] - ry)
    goal_bearing = math.degrees(math.atan2(gx - rx, gy - ry))
    if de > engage_dist_mm:
        return _wrap_deg(goal_bearing - hdg)
    ang_away_abs = math.degrees(math.atan2(rx - nearest["x"], ry - nearest["y"]))
    target_abs = (ang_away_abs
                 + _wrap_deg(goal_bearing - ang_away_abs) * carry_shield_goal_bias)
    return _wrap_deg(target_abs - hdg)


def _carry_creep(rx, ry, hdg, gx, gy, enemies):
    """Bounded translation to creep toward open space while shielding,
    decoupled from the shield heading - translation moves independently of a
    fixed shield rotation, driven by our own defender-aware trigger. Blends a vector away from the nearest (blocking)
    defender with a small forward-to-goal component, so the robot works
    sideways clear of the defender and edges upfield, deliberately never
    straight back into them (the away component is never zero-weighted).
    Returns (drive_rel_deg, speed_cmd)."""
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
    cn = math.hypot(cx, cy)
    if cn < 1e-6:
        return 0.0, 0.0
    cx, cy = cx / cn, cy / cn
    drive_rel = _wrap_deg(math.degrees(math.atan2(cx, cy)) - hdg)
    return drive_rel, base_speed * carry_creep_speed_frac
# Wall keep-out: decompose the commanded velocity against each nearby wall's normal, damping only the into-wall part so the bot can slide along a wall at full speed.
wall_keepout_enabled = True
keep_min_mm          = 150.0 # centre-to-wall: never touch a wall
# goal_flank_keep_mm: enough clearance to clear the wheel-cutout wedging that started this, without needing the full keep_min_mm standoff so close to a goal.
goal_flank_keep_mm   = 50.0

# FieldModel.active_walls (bot/field.py) reads keep_min_mm/goal_flank_keep_mm
# as ITS OWN module-level globals (a plain name lookup inside that method, not
# something the caller can satisfy just by having them in scope) - bot/
# field.py's own docstring flags this as a forward gap left for "a later
# stage's shared constants module" to wire up. This is that stage: publish
# both onto the bot.field module now that they're defined.
import bot.field as _field_module
_field_module.keep_min_mm        = keep_min_mm
_field_module.goal_flank_keep_mm = goal_flank_keep_mm
# damp the into-wall component within this band outside the keep line. Must
# cover one pose-staleness interval (~100ms) of travel at rush_speed, or a
# fast, close approach can cross keep_min_mm before the guard ever sees it;
# 400 assumes rush_speed around 0.5, widen further if rush_speed goes up.
# wall_slide_zone_mm itself is a documented per-bot gap (see module
# docstring): it is already in bot1_config.py/bot2_config.py, not
# redefined here, and is referenced only inside function bodies below
# (_wall_guard, _orbit_approach).

# wall_safe_speed_cmd is the old base_speed (0.083), kept as its own constant since _wall_guard steers on an about 100ms-stale pose.
wall_safe_speed_cmd  = 0.083
wall_push_gain       = wall_safe_speed_cmd / 60.0 # outward command per mm of penetration
wall_push_max        = wall_safe_speed_cmd # cap on the restoring push


def _wall_guard(rx, ry, hdg, bearing_rel, cmd):
    """constrain a translation command so the robot keeps clear of the four field-boundary walls and each goal box's own flanks (the mouth/back wall stay exempt, see active_walls), each at its own keep-out distance. Fixes a real corner-stall bug: the old code resolved each nearby wall independently in a single pass, cancelling/damping that wall's own into-wall velocity component using the vector already reduced by the previous wall. At a
    true concave corner (e.g. a boundary wall and a goal-box flank meeting, with a
    ball tucked right in the pocket) this is not just imprecise, it is the
    mathematically exact clip of the constraint set: a commanded bearing pointing
    diagonally INTO the pocket has a negative component against BOTH walls' normals
    at once, so cancelling each one in turn drives the command to (0, 0) even though
    a real, reachable path along the corner exists. The hard-zone clip (never allow further penetration of a wall already inside `keep`) is now run to convergence over every such wall, idempotent, so a corner with several concurrently-violated walls settles correctly; the slide-zone damp (the soft band outside `keep`) still runs once, targeting a proportional "closing speed" that re-applying to its own output would keep eroding. When that clip still collapses the command well below what was asked for, bend the ORIGINAL command onto the SUM of every currently-blocking wall's own outward normal rather than just cancelling into them, then re-run the same clip on the bent result - bending onto a lone nearest wall's tangent alone can send the escape straight at a second wall just as tight and get re-crushed by that wall's own damp, while bending onto the sum of every blocking wall's away-normal is guaranteed non-negative against every one of them simultaneously (dot product identity: for unit "away" normals n1, n2, (n1+n2)*n1 = 1 + cos(theta) >= 0 for any real angle theta, and likewise for n2, generalising to any number of walls), the standard "escape away from the corner point" direction, not just one wall's tangent."""
    if not wall_keepout_enabled:
        return bearing_rel, cmd
    walls = FieldModel.active_walls(rx, ry, keep_min_mm + wall_slide_zone_mm)
    if not walls:
        return bearing_rel, cmd

    bf = math.radians(bearing_rel + hdg)
    vx0, vy0 = cmd * math.sin(bf), cmd * math.cos(bf) # original commanded field-frame velocity
    cmd_mag = cmd

    # Positional restoring push: summed over every currently-penetrated wall
    # (independent of the velocity clip below, which _resolve now handles
    # with proper multi-wall convergence).
    push_x = push_y = 0.0
    for dist, nx, ny, keep in walls:
        if nx == 0.0 and ny == 0.0:
            continue
        if dist <= keep:
            push = min(wall_push_max, wall_push_gain * (keep - dist))
            push_x += push * nx
            push_y += push * ny

    def _resolve(vx, vy):
        """hard-zone clip (run to convergence) plus one slide-zone damp pass; returns the clipped velocity and the (nx, ny) of every wall that actually constrained it."""
        blocking = []
        for _ in range(3): # converge the hard-zone clip over every concurrently-violated wall (a corner)
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
                factor  = (dist - keep) / wall_slide_zone_mm # 0 at keep -> 1 at edge
                # Surviving into-wall speed is additionally capped in absolute terms at wall_safe_speed_cmd.
                closing = max(factor * vn, -wall_safe_speed_cmd)
                vx += (closing - vn) * nx
                vy += (closing - vn) * ny
                blocking.append((nx, ny))
        return vx, vy, blocking

    vx, vy, blocking = _resolve(vx0, vy0)

    # De-dup blocking normals (the hard-zone loop can revisit the same wall
    # over its convergence passes) and keep only ones that are genuinely
    # distinct directions - a single wall's own normal, summed with itself,
    # is still just that wall's normal, and bending the ORIGINAL command onto
    # a lone wall's outward normal would blast the robot straight away from
    # it regardless of where it actually wants to go. That degenerate case is
    # not a corner; trust the clip above exactly as before. Only >= 2 walls
    # whose normals are not near-parallel form a real concave pocket worth
    # bending around.
    distinct = []
    for nx, ny in blocking:
        if not any(nx * dnx + ny * dny > 0.98 for dnx, dny in distinct):
            distinct.append((nx, ny))

    # Also require the ORIGINAL command to have been a real, meaningful drive
    # request (a chase/contest-speed command, well above wall_safe_speed_cmd -
    # matching the bug report's own real numbers, vx 0.218/vy 0.123) AND the
    # clipped result to have collapsed to near-nothing in absolute terms, not
    # just relative to the request. A slow creep near a wall (careful final
    # positioning, a goalie holding its line) naturally has a small commanded
    # magnitude on its own, with nothing to do with a corner; a relative
    # "collapsed by 75%" test alone still flags that, and then bending it onto
    # a full-cmd_mag escape direction wildly overshoots what was asked for in a
    # direction that was never requested. Caught by testing against the
    # goalie's own positioning near its goal-box corner in the test suite's
    # real_vs_real_match.py's repeated trials.
    if (len(distinct) >= 2 and cmd_mag > wall_safe_speed_cmd * 1.5
            and math.hypot(vx, vy) < 0.05):
        # The clip still collapsed the command near zero even though it had real
        # magnitude - the commanded bearing pointed into a concave pocket formed
        # by every wall in `distinct`. Bend the ORIGINAL command onto their
        # combined escape direction and re-run it through the same clip so the
        # bend can never re-violate a wall already enforced above.
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
        # Was `< 1.0`, comparing a real command against full-scale 1.0 always lost, zeroing the drive command near a wall instead of the damped/redirected velocity the slide-band logic computed.
        return bearing_rel, 0.0
    new_cmd  = min(new_cmd, max_speed_cmd)
    new_bear = _wrap_deg(math.degrees(math.atan2(vx, vy)) - hdg)
    return new_bear, new_cmd
# Enemy keep-out: same into-obstacle decomposition as _wall_guard, but against each tracked enemy's own position, so a chase steers around a blocking robot instead of driving straight through it. Off entirely once close enough to the ball to legitimately contest for it - this is meant to stop driving through an enemy on the way to a ball that's still out of reach, not to back off a real 50/50.
enemy_avoid_enabled       = True
enemy_avoid_keep_mm       = 260.0 # centre-to-centre distance before contact is imminent (~2x robot_radius_mm plus a small margin)
enemy_avoid_slide_zone_mm = 300.0 # damp the into-enemy component within this band outside the keep line, same idea as wall_slide_zone_mm
# This close to the ball, avoidance turns off entirely (sec 4.1). Was 300.0,
# only 40mm past enemy_avoid_keep_mm - when an enemy is standing ON the
# ball, our distance to the ball and our distance to that enemy are nearly
# the same number, so the "still avoiding" and "stop avoiding, go contest"
# zones nearly coincided and fought each other right at the boundary instead
# of handing off cleanly, leaving us stalled a few hundred mm short of a
# held ball. Widened to comfortably more than 2x robot_radius_mm past
# enemy_avoid_keep_mm so there's real room to accelerate through the gap
# once avoidance disengages - enough to have genuinely committed to the
# contest before contact, not so far out that a normal chase (ball not
# actually held) starts bulldozing through a merely-nearby enemy.
enemy_contest_dist_mm     = 550.0
# Committed approach speed once we're inside enemy_contest_dist_mm AND an
# enemy is close enough to the ball to be the one holding/hiding it (see
# _enemy_holds_ball): this deliberately bypasses _brake_speed_frac's
# gentle stopping-distance cap for that case specifically, since we're
# win a real shoving contest against similarly-built opponents - braking to a soft stop right
# at their body just glides up next to them and never dispossesses.
# Above base_speed (open-field cruise) so contact is real, below rush_speed
# (straight-line sprint) since this is still a final, steered approach.
contest_speed_cmd        = 0.25
# an enemy within this of the ball counts as holding/hiding it, for the
# contest-speed override above - generous enough to cover the enemy's own
# body radius plus the ball sitting tucked against it (robot_radius_mm plus
# a margin for the ball not being dead-centre on the enemy).
contest_hold_dist_mm     = Perception.robot_radius_mm + 180.0


def _enemy_holds_ball(bx, by, enemies):
    """True if some tracked enemy is close enough to (bx, by) to plausibly be the one holding/hiding the ball there, rather than the ball sitting free."""
    return any(math.hypot(e["x"] - bx, e["y"] - by) <= contest_hold_dist_mm
               for e in enemies)


def _approach_speed_cap(dist_mm, bx, by, enemies, fallback_cmd):
    """the speed cap to use on a ball approach: contest_speed_cmd (bypassing _brake_speed_frac's gentle stopping cap) once we're inside enemy_contest_dist_mm with an enemy plausibly holding the ball there, else fallback_cmd (normally _brake_speed_frac's own controlled-stop cap) unchanged."""
    if (dist_mm <= enemy_contest_dist_mm and enemies
            and _enemy_holds_ball(bx, by, enemies)):
        return max(fallback_cmd, contest_speed_cmd)
    return fallback_cmd


def _enemy_guard(rx, ry, hdg, bearing_rel, cmd, enemies, ball_dist_mm=None):
    """like _wall_guard but for tracked enemy robots (enemies, from _state["enemies"]): damps/deflects the into-enemy component of a translation command so the path slides around instead of driving straight through. ball_dist_mm is our own distance to the ball right now, not any per-enemy distance - pass it whenever this drive is actually chasing a known ball so a genuine close-range contest isn't steered away from; omit it (None) for drives with no ball to contest (positioning, yield-hold, lost-ball fallback), where avoidance always applies."""
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
            factor  = (dist - enemy_avoid_keep_mm) / enemy_avoid_slide_zone_mm
            closing = max(factor * vn, -wall_safe_speed_cmd)
            vx += (closing - vn) * nx
            vy += (closing - vn) * ny
    vx += push_x
    vy += push_y

    new_cmd = math.hypot(vx, vy)
    if new_cmd < 1e-3:
        return bearing_rel, 0.0
    new_cmd  = min(new_cmd, max_speed_cmd)
    new_bear = _wrap_deg(math.degrees(math.atan2(vx, vy)) - hdg)
    return new_bear, new_cmd
# General jam recovery: a body-contact stuck detector for what this codebase's other two
# recovery mechanisms leave uncovered. stuck_deep_* only bails out a striker carrying the
# ball deep under sustained contest, via a pass; _wall_guard's corner-escape plus
# _orbit_approach's wall-aware bend already cover a robot wedging against a WALL. What's
# left, and what this covers, is a robot jammed against ANOTHER ROBOT anywhere on the
# field: an enemy dead-ahead in contact range while our own measured displacement stalls,
# hysteresis-debounced, escapes with a fixed-dwell full-power push half toward the current
# target bearing, half dead-ahead. Kept deliberately separate from the
# enemy_contest_dist_mm / contest_speed_cmd machinery above: that's EXPECTED,
# low-relative-motion contact (a real 50/50 shoving match) already handled by the
# contest-speed bypass, so this detector stands down (skip=True, see the call sites in
# GoalieController.tick/StrikerController.tick) whenever that condition holds, rather than
# double-handling a legitimate sustained shove.
jam_contact_mm = 2.0 * Perception.robot_radius_mm + 20.0  # both bots the
                                                            # same radius, plus a
                                                            # small touching margin
jam_ahead_deg  = 35.0   # "opponent dead ahead" half-angle gate
jam_speed_mm_s = 90.0   # "going nowhere" own-speed floor
# Conservative on purpose: a goalie holding a
# tight defensive line, or a striker's deliberate slow final approach, both command
# real (if modest) motion and keep producing real (if modest) displacement - own_speed
# rarely stays pinned this far below jam_speed_mm_s for this long unless something is
# genuinely blocking progress, not just being careful.
jam_enter_s    = 1.2
jam_hold_s     = 0.5    # push-through dwell once declared


class JamRecovery:
    """hysteresis jam detector + push-through override, one instance per controller (Striker/Goalie) alongside its other per-tick state. Call check() once near the top of tick(), right after pose/enemies are in hand, with the ball-or-heading target bearing this tick would otherwise chase; a non-None result is the (drive_angle, speed_frac) to drive THIS tick instead of running the controller's normal steering at all. None means proceed as normal - this is strictly additive, a controller that never calls check() (or always gets None back) behaves exactly as before. skip=True suppresses new jam evidence this tick (an already-armed push-through still runs out its dwell) - pass it whenever the caller recognizes this contact as an expected, already-handled situation (see enemy_contest_dist_mm above) rather than a genuine jam."""

    def __init__(self):
        self._last_pos     = None   # (rx, ry, t) - own-speed estimate
        self._jam_s        = 0.0    # accumulated jam evidence (s)
        self._push_until_t = None   # set once declared; push-through lasts
                                      # until this monotonic time

    def reset(self):
        self._last_pos     = None
        self._jam_s        = 0.0
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
            self._push_until_t = None   # dwell elapsed, fall through

        if not enemies or own_speed is None or dt is None:
            return None
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
            self._jam_s        = 0.0
            self._push_until_t = now + jam_hold_s
            return _wrap_deg(rel * 0.5), 1.0
        return None
# Rush: whether the straight-line path ahead is clear enough of enemies to
# go full-send (rush_speed, sec 3.1). Reasoned, not field-validated.
# how far ahead along the drive direction an enemy still counts as blocking
rush_lookahead_mm       = 1200.0
# perpendicular half-width of the lane (about 2x robot_radius_mm plus margin)
rush_lane_half_width_mm = 300.0


def _lane_clear(rx, ry, hdg, bearing_rel, enemies):
    """True if no tracked enemy sits in the lane ahead of the straight-line path (bearing_rel, robot-frame) we're about to drive, the gate for using rush_speed instead of base_speed."""
    if not enemies:
        return True
    bf = math.radians(bearing_rel + hdg)
    fwd_x, fwd_y = math.sin(bf), math.cos(bf) # unit vector, field frame
    for e in enemies:
        dx, dy = e["x"] - rx, e["y"] - ry
        ahead   = dx * fwd_x + dy * fwd_y # projection onto the path
        lateral = dx * fwd_y - dy * fwd_x # perpendicular offset
        if 0.0 < ahead <= rush_lookahead_mm and abs(lateral) <= rush_lane_half_width_mm:
            return False
    return True
# Noise gate on _add_ball_velocity's ball-velocity contribution. A stationary
# ball's velocity estimate is not zero, it jitters around zero (camera fix
# quantisation + the estimator's own window), so the old "mix in anything above
# 1e-6 mm/s" test flipped the lead term in and out every tick a still ball's
# estimate crossed the threshold - the commanded bearing (and rot_speed, which
# is drive_angle * turn_gain downstream) snapped side to side with it. Below
# ball_vel_fade_min_mms the estimate is treated as pure noise and contributes
# nothing; above ball_vel_fade_full_mms it's a genuinely moving ball and the
# full velocity is mixed in; between the two the contribution fades in
# linearly, so a real (slow) roll ramps the lead in smoothly instead of
# snapping.
ball_vel_fade_min_mms  = 150.0
ball_vel_fade_full_mms = 450.0


def _add_ball_velocity(drive_deg, cmd, hdg, vbx, vby):
    """Method 6 command mixing: convert the capture command (robot-frame bearing + command units) to a physical robot-frame velocity, add the ball's field velocity rotated into the robot frame, and convert back. The ball-velocity term passes through the ball_vel_fade_* noise gate above (zero below min, full above, linear between) so a jittering near-stationary estimate can't oscillate the capture bearing."""
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
    # Motor.drive would scale an over-budget command down anyway, but only
    # after it has been asked for, cap it here so the number we hand back
    # (and log) is one the chassis can actually hold on this bearing.
    cap = _vmax_mms(bearing) / vmax_full_cmd_mms * max_speed_cmd
    return bearing, min(new_cmd, cap)
def _orbit_approach(bx, by, rx=None, ry=None, hdg=None):
    """direct-pursuit steer to a ball at robot-frame (bx, by), forward=+y right=+x: point straight at the ball's bearing, no tangent-circle/side offset. Real-hardware testing showed the old orbit (tangent-circle) approach missing the ball outright; this replaces it with plain direct pursuit. has_ball already re-aims toward the enemy goal once possession is confirmed (see the has_ball state), so no pre-alignment side selection is needed here. When rx/ry/hdg (our own field-frame pose) are supplied, the raw ball bearing above is bent away from any wall we're already close to before it's ever handed onward, using the same FieldModel.active_walls data _wall_guard itself keys off - this is upstream of _wall_guard on purpose: _wall_guard only gets to react to whatever bearing it's handed, so a target that already points straight into a wall (or, worse, into the pocket of two walls at once - a ball tucked into a corner) forces it to fight that bad request every tick right at the boundary. Here the raw straight-at-the-ball direction is instead blended toward each blocking wall's own tangent as clearance shrinks, applied in turn for any number of concurrently-blocking walls (same spirit as _wall_guard's own multi-wall handling); the result is a target bearing that already curls along the wall well before the hard keep-out line, so _wall_guard's own clip/bend rarely has anything left to correct - it stays the final safety net, not the routine corrector."""
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
        # 0 at (keep + wall_slide_zone_mm) out -> 1 right at the keep line, same ramp as _wall_guard's own slide-zone damp.
        factor = max(0.0, min(1.0, (keep + wall_slide_zone_mm - dist) / wall_slide_zone_mm))
        if factor <= 0.0:
            continue
        tx, ty = -ny, nx # tangent along the wall; pick the sign that keeps progress toward the target
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
recovery_standoff_mm = 350.0 # clear of the ball, so the next approach starts fresh instead of re-engaging mid-recovery
# Lateral bias for the recovery target, same order as enemy_avoid_keep_mm+
# _slide_zone_mm: without it, the target sits on the ball-to-own-goal line,
# which a beaten robot is often already roughly lined up on too - _enemy_guard
# can only cancel/damp a dead-on approach, not route around one with zero
# lateral room, so the recovery drive stalls short instead of ever reaching
# the correct side. Biasing to whichever side the robot's already leaning
# guarantees an actual swing around, not a stall.
recovery_lateral_mm  = 300.0


def _ball_beaten(ry, by, own_goal):
    """True if the ball's gotten deeper into our own half than we currently are - an attacker's beaten us to it. Driving straight at the ball from here means approaching from the enemy-goal side, contacting the attacker's rear/flank while their front (and the shot lane) stays completely clear. Recover goal-side first instead (_defend_recovery_target)."""
    own_gy = own_goal[1]
    into   = 1.0 if own_gy < FieldModel.field_y / 2 else -1.0
    return into * (ry - own_gy) > into * (by - own_gy)


def _defend_recovery_target(rx, ry, bx, by, own_goal, standoff_mm, lateral_mm):
    """a point standoff_mm from the ball toward our own goal, offset lateral_mm to whichever side of the ball-goal line we're already on: get back goal-side of a ball an attacker's beaten us to before re-engaging, so the next approach reaches their front (blocking the shot) instead of their rear. The lateral bias is what actually gets us there - see recovery_lateral_mm's own comment."""
    own_gx, own_gy = own_goal
    dxg, dyg = own_gx - bx, own_gy - by
    dg = math.hypot(dxg, dyg) or 1.0
    ux, uy = dxg / dg, dyg / dg   # unit vector, ball -> own goal
    lx, ly = -uy, ux              # perpendicular
    side = 1.0 if (rx - bx) * lx + (ry - by) * ly >= 0.0 else -1.0
    return (bx + ux * standoff_mm + lx * side * lateral_mm,
            by + uy * standoff_mm + ly * side * lateral_mm)


# Rule 5.11 (multiple defence): only one of us may be in/touching our own
# defending box at a time. The dynamic role-swap keeps this true INDIRECTLY
# most of the time (only one bot is ever assigned the defensive role), but
# that's never been a hard guarantee against a Bluetooth-latency race or a
# bot already mid-chase into the box when a swap lands. These are an
# explicit belt-and-braces check, applied right before any path that would
# actually drive a bot into its own box (goalie chase-and-clear, striker
# beaten-recovery).
def _in_own_box(x, y, own_goal, margin):
    """True if a robot circle of radius `margin` centred at (x, y) touches our own goal box - same corner-rect model (goal box widened by margin) as Perception._inside_playable's box test, just for one goal end."""
    _gx, gy = own_goal
    into = 1.0 if gy < FieldModel.field_y / 2 else -1.0
    bx0 = FieldModel.bx0 - margin
    bx1 = FieldModel.bx1 + margin
    gd  = FieldModel.goal_depth + margin
    if not (bx0 <= x <= bx1):
        return False
    return y <= gd if into > 0 else y >= FieldModel.field_y - gd


def _teammate_blocks_own_box(teammate_pos, own_goal):
    """True only when the teammate's freshly-reported (Bluetooth) position is already in/touching our own defending box, so we should hold back rather than also enter (Rule 5.11). teammate_pos is already None whenever the peer link is stale/lost (TeamState.pose_max_age_s / peer_timeout_s gating upstream, sec 3.2), so a dropped Bluetooth packet naturally falls through to "allow entry" here instead of ever blocking play on stale data."""
    if teammate_pos is None:
        return False
    tx, ty = teammate_pos
    return _in_own_box(tx, ty, own_goal, Perception.robot_radius_mm)


# Real clearance (not just past the touching threshold to 1mm) the "hold
# back" fallback pushes its target past our own box's edge by, so a
# braking/settling robot doesn't keep drifting back across the touching
# line and re-triggering the guard every other tick.
box_hold_clear_mm = 60.0


def _clamp_outside_own_box(x, y, own_goal, margin):
    """push (x, y) clear of our own goal box (along y, same box model as _in_own_box, plus box_hold_clear_mm of real margin) if it lands inside/touching it - used to keep a recovery/hold target from entering the box when the teammate's already there."""
    _gx, gy = own_goal
    into = 1.0 if gy < FieldModel.field_y / 2 else -1.0
    bx0 = FieldModel.bx0 - margin
    bx1 = FieldModel.bx1 + margin
    gd  = FieldModel.goal_depth + margin
    if bx0 <= x <= bx1:
        if into > 0 and y <= gd:
            y = gd + box_hold_clear_mm
        elif into < 0 and y >= FieldModel.field_y - gd:
            y = FieldModel.field_y - gd - box_hold_clear_mm
    return x, y
