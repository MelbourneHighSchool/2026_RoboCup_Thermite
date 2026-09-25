"""Role controllers: GoalieController, StrikerController, CamRunController -
the state machines that turn tracked ball/enemy/teammate state into drive
commands each control tick. The highest fan-in module in bot/, importing
from nearly every earlier-extracted module (bot.motion for the drive-speed
model, keep-out guards, jam recovery, carry/shield and finishing-angle
logic; bot.dwibbler for possession sensing and roller control; bot.tracking
for ball/teammate/enemy velocity+memory helpers; bot.vision for the front-
goal colour global; bot.field/bot.perception for field geometry and robot
size; bot.state for the shared _state dict/_lock; bot.network for the
role-swap/yield tunables already extracted there; bot.compass for fused
heading; bot.logs for the --capturelog session finisher).

Extracted verbatim from mainrunbot1.py's goalie/striker/camrun section
(originally lines 6067-7192: goalie_*/escort_* tunables, GoalieController,
loop_dt, support_* tunables, StrikerController, _camrun_forward_bearing,
CamRunController).

loop_dt (shared tick period, 0.02s / 50Hz) is defined here because
GoalieController.tick/StrikerController.tick/CamRunController.tick all use
it directly for their own capture-hold/pass-hold/stuck-hold hysteresis
accumulators - it is NOT purely controllers-specific: _play_loop (still in
the monolith, destined for bot/main.py) also sleeps this once per tick. A
later bot/main.py extraction should import loop_dt from here rather than
redefine it, to keep the play-loop sleep and every controller's hysteresis
math using the identical constant.

role_swap_margin_mm/yield_ball_near_mm/yield_hold_mm already live in
bot/network.py (stage 4 extracted them alongside _run_master's dynamic role
handoff, which reads them too) - imported from there, not redefined here.
support_ball_near_mm is set equal to yield_ball_near_mm, same as the
monolith did, since support and yield share the same "close enough to just
chase it" standoff.

Documented per-bot gaps (same pattern as every earlier stage): base_speed
and rush_speed are per-bot (already in bot1_config.py/bot2_config.py) and
are referenced here only inside GoalieController.tick/
StrikerController.tick/CamRunController.tick method bodies, left undefined
at module scope until the ROBOT_ID selection mechanism lands - `import
bot.controllers` still succeeds (Python does not evaluate a method body at
import time), and pyflakes flags both as undefined names, which is expected
and matches bot/motion.py's own documented gap for the same two names.

_calib (used by _goal_positions' goal-centre-x nudge) is a genuine
calibration-stage gap, not a per-bot one: `_calib`/`_load_calib` are the
monolith's calibration-persistence globals (around what will become
bot/calibration.py - a still-mechanical, not-yet-done later stage; the
`bot/calibration.py` file already on disk at HEAD is a stale leftover from
the abandoned historical refactor with different names, `calib`/
`load_calib`, not `_calib`/`_load_calib`, so it is NOT a reliable source to
import from here). Stubbed as an empty dict so `_goal_positions` still
behaves correctly (falls back to FieldModel.cx via `or`, exactly as the
monolith does with an unpopulated _calib) rather than a bare undefined name
- this keeps GoalieController/StrikerController.tick callable (and the
end-to-end tick() smoke test in this stage's verification meaningful)
ahead of that later calibration stage supplying the real thing.

flick_range_mm and pass_eject_speed are imported from bot.motion rather
than redefined here: bot/motion.py's own docstring already documents that
it copied these two (identical across bot1/bot2) as plain constants for its
own carry_1v1_gap_mm/pass_ball_speed_mms math, ahead of this stage. Reusing
that copy here (instead of a second independent copy) keeps a single source
of truth for both modules.
"""

import math
import time

import bot.state as state
import bot.vision as vision
from bot.compass import _fused_heading
from bot.dwibbler import (
    _possession, _set_dwibbler, _dwibble_carry_speed, capture_dist_mm,
    dwibble_contest_dist_mm, dwibble_on_dist_mm, dwibble_speed,
    possess_capture_window_s, possess_on_s, possess_spinup_s,
)
from bot.field import FieldModel, wrap_deg as _wrap_deg
from bot.hardware import Motor
from bot.logs import _finish_capture_log
import bot.motion as motion
from bot.motion import (
    JamRecovery, _add_ball_velocity, _approach_speed_cap, _ball_beaten,
    _brake_speed_frac, _carry_creep, _carry_should_press,
    _clamp_outside_own_box, _defend_recovery_target, _enemy_guard,
    _enemy_holds_ball, _goal_shot_aim, _lane_clear, _orbit_approach,
    _pass_race_open, _shield_heading_deg, _slew_drive,
    _wall_guard, capture_cone_half_deg,
    capture_cone_half_width_mm, enemy_contest_dist_mm, flick_range_mm,
    pass_eject_speed, recovery_lateral_mm, recovery_standoff_mm,
    wall_safe_speed_cmd,
)
# _teammate_blocks_own_box is read through bot.motion (motion.X), not
# from-imported: the test suite's scenario_dual_goalie_box.py monkeypatches it
# by direct reassignment on the mainrunbot*.py shim, which forwards the
# write onto bot.motion - a from-import here would freeze a stale bound
# copy that the monkeypatch could never reach, same reasoning as
# bot.network.bt_team_enabled / bot.odometry.wheel_odom_enabled in
# bot/main.py's own comment.
from bot.network import yield_ball_near_mm, yield_hold_mm
from bot.perception import Perception
from bot.tracking import (
    BallMemory, BallVelocityEstimator, KnownOcclusion,
    TeammateVelocityEstimator, ball_mem_min_conf,
)

# Stage 7 resolved the calibration-stage gap documented above: bot/calibration.py
# now holds the real `calib` dict and `goal_positions()`, extracted verbatim
# from the monolith. Import them directly rather than keeping a stub - their
# logic was byte-for-byte identical to the stub this replaced.
from bot.calibration import calib as _calib, goal_positions as _goal_positions


# State machine (behaviour-state-name string constants; bot/network.py's
# _run_master already documents referencing `has_ball`/`seek`/`passing`/
# `flick_shot` as a forward gap onto this module).
seek       = "seek"
has_ball   = "has_ball"
# carrying the ball but electing to release it to the teammate instead of driving it in
# ourselves
passing    = "pass"
flick_shot = "flick_shot" # spin-release shot around a blocker (sec. 3.22)
# shared non-blocking dwibbler-reverse countdown entered from both passing and
# flick_shot (replaces their old time.sleep-in-tick release, which stalled the
# whole 50Hz control loop for the dwell); also published in _state for logs.
ejecting   = "ejecting"

# Heading gain, pure P, no D.
turn_gain = 0.0012

# camera capture condition must hold this long, one glitch frame can't flip seek->has_ball
# (accumulate/decay has_ball pattern below)
capture_confirm_s = 0.15

capture_ball_vel = True # add ball velocity to the capture command

# Passing (sec 3.21).
pass_enabled          = True
# teammate must be at least this much closer to the enemy goal before we consider passing
pass_teammate_gain_mm = 400.0
# don't pass to a teammate this close, just dwibble together / drive in ourselves
pass_min_range_mm     = 300.0
# a reversed-dwibbler eject can't reliably cross more than this
pass_max_range_mm     = 1500.0
pass_lane_block_mm    = Perception.robot_radius_mm + 60.0 # corridor
                                # half-width for the enemy-in-lane check
# don't bother passing if we're already this close to goal ourselves
pass_min_own_dist_mm  = 500.0
# settle/advance this long after winning the ball before considering a pass, swept in
# simulator.py, flat from 1.6s to 4s, worse below
pass_settle_s         = 1.6
# lead the target by this many seconds of the teammate's estimated velocity (a through
# pass)
pass_lead_s           = 0.4
pass_align_deg        = 15.0 # turn to within this of the target before firing
pass_confirm_s        = 0.15 # hold the aim this long before releasing
pass_eject_s          = 0.25 # how long to reverse the dwibbler for
# how long a fired pass_target keeps publishing to the teammate afterwards
pass_target_max_age_s = 1.5

# Deep-stuck bail-out: a striker still carrying deep in its own half and contested by a
# nearby enemy for a sustained stretch bails out to the teammate via a real pass instead
# of grinding it out alone, even though the teammate (typically the goalie, planted near
# our own goal) isn't "closer to the enemy goal" the way the ordinary pass-teammate-gain
# check above requires. This only widens the trigger for the same pass mechanism below
# (still gated on pass_min_range_mm/pass_max_range_mm and the lane/race checks, unchanged)
# - it never abandons the ball mid-carry or flips roles directly. The BT dynamic role
# swap (sec 4.8) picks the receiving teammate up as striker once the ball's fused position
# shows up near them after the pass fires; role_swap_margin_mm/my_state == has_ball in
# _run_master still guarantee roles never flip while either bot is mid-carry.
# stuck_hold_s (accumulate/decay, matching capture_hold_s's own pattern) ramps up by
# loop_dt while deep+contested and drains by loop_dt the instant either lets up, capped at
# stuck_deep_contest_s + stuck_deep_arm_window_s - "armed" (>= stuck_deep_contest_s) stays
# true for up to stuck_deep_arm_window_s after the contesting enemy actually backs off.
stuck_deep_enabled       = True
stuck_deep_depth_mm      = 1400.0 # within this of our own goal (own_into-relative) is "deep"
stuck_deep_contest_s     = 2.5   # contested (dwibble_contest_dist_mm) this long arms the bail-out
stuck_deep_arm_window_s  = 3.0   # keep trying to pass this long after the contest eases

# Tap-in: a direct feed to a teammate already close enough to finish, unconditional of pass_settle_s, with its own clear-lane-to-goal check.
tap_in_enabled  = True
# teammate must already be this close to goal (= flick_range_mm) to count as a tap-in
tap_in_range_mm = 900.0

# Scoring (no solenoid fitted): has_ball drives the ball to the goal and cuts the roller once close, the body's own push does the work.
goal_eject_range_mm = 600.0

# Flick shot (sec. 3.22): reversed-dwibbler eject fired mid-spin instead of from a stop, no solenoid needed.
flick_enabled        = True
# Minimum continuous carry (from carry_start_t) before a blocked lane is
# allowed to trigger the eject-in-place above: without this, re-entering
# has_ball off a fresh contested capture (blocker still adjacent) evaluates
# "blocked" on the very next tick and reverses the dwibbler right back at
# them, looping capture/eject/recapture in place instead of ever driving.
flick_settle_s        = 0.5
flick_kick_speed      = dwibble_speed # max authority, more spin is better
flick_snap_s          = 0.22 # total snap duration


# Goalie role

# standoff distance from own goal centre while shadowing a known ball, a 2D radius now
# (see GoalieController's docstring), not a fixed blocking-line depth
goalie_line_mm       = 400.0
goalie_clear_dist_mm = 600.0 # ball closer than this -> charge and clear it
goalie_x_slack_mm    = 150.0 # lateral travel clamped to goal-box width +/- this

# Quadratic arc-radius shrink: when the ball sits near the
# goal-centre axis (straight in front of the net) the shadow radius shrinks
# to goalie_line_mm * goalie_radius_min_factor for tighter, closer coverage;
# as the ball swings wide toward either post the radius eases back up to the
# full goalie_line_mm via an angle^2 ramp, so a wide-angle ball still gets
# the full standoff distance (more time to react, less chance of being
# skinned near-post). International RCJ Soccer rules (not this codebase's
# prior "Aussie rules" box assumption) forbid a robot - including the
# defending goalie - from FULLY entering the penalty box; only some part of
# the robot's footprint needs to stay outside. So the floor isn't picked
# against the goal line, it's picked against FieldModel.goal_depth (300mm,
# the box's own front edge) plus Perception.robot_radius_mm (105mm): at
# min_factor*goalie_line_mm (260mm) the robot's own radius still pokes its
# near edge past the box's front edge by about 65mm of margin, so the
# tightest shadow point is legal even though its centre sits inside the box.
# (The box's corners are filleted/rounded under international rules, but
# that only shrinks the box's true interior near a corner relative to this
# square-corner model, i.e. it's the conservative direction for this check;
# this radius floor is reached with the target near the goal-centre axis,
# nowhere near a corner, so the fillet doesn't change the margin above.)
goalie_radius_min_factor = 0.65

# Ball-velocity lead: nudges
# the shadow target's source ball position along the ball's own predicted
# travel direction instead of only its current fix, cutting camera-to-drive
# reaction lag. Expressed here as a lead TIME (BallVelocityEstimator reports
# mm/s, not mm/tick), clamped so a velocity-estimator glitch can't fling the
# target off the arc.
goalie_vel_lead_s      = 0.15
goalie_vel_lead_max_mm = 200.0

# Escort/support is the closest real-robot equivalent of simulator.py's TacticalBrainMixin escort logic, with only two roles here.
escort_enabled       = True
# ball must be at least this far from our own goal, along the attack axis, before the
# goalie considers escorting, comfortably past midfield (FieldModel.field_y about = 2430)
escort_min_depth_mm  = 1600.0
# lateral offset from the teammate, toward whichever side of the pitch has more room
escort_lat_mm        = 300.0
# how far past the teammate (further into the attack) the support spot leads
escort_lead_mm       = 150.0

# Back-wall geometry, depths from own_goal's raw gy: goal_backwall_mm (226) is the true scoring wall, goal_line_mm (300) is the mouth.
goal_backwall_mm      = 226.0
goal_line_mm          = 300.0
goalie_backwall_tol_mm = 174.0


class GoalieController:
    """defensive role, three regimes: shadow (ball known, teammate not deep) at standoff radius goalie_line_mm on the goal-ball line, escort, or probability cover."""
    def __init__(self):
        """fresh BallMemory (and ball-velocity estimator, for the arc lead term) for a new goalie run."""
        self.ball_mem = BallMemory()
        self.ball_vel = BallVelocityEstimator()
        self.jam = JamRecovery() # general body-contact stuck detector, see its class docstring

    def tick(self):
        """run one control tick of the goalie behaviour, see the class docstring."""
        ball_mem = self.ball_mem
        ball_vel = self.ball_vel
        with state._lock:
            pose         = state._state["pose"]
            _pose_imu    = state._state["pose_imu"]
            _imu_now     = state._state["imu_heading"]
            ball         = state._state["ball"]
            enemies      = state._state["enemies"]
            remote_ball  = state._state["remote_ball"]
            run_mode     = state._state["run_mode"]
            teammate_pos = state._state["teammate_pos_bt"]
            peer_state   = state._state["peer_state"]
            state._state["my_state"] = "goalie"

        # Steer on a heading carried forward from the IMU, not one that is up
        # to a whole lidar revolution old (see _fused_heading).
        pose  = _fused_heading(pose, _pose_imu, _imu_now)
        now_t = time.monotonic()

        # Directly-known ball (our camera -> teammate's sighting), tracked
        # even while idle (known positions only); BallMemory fusion (seen/
        # infer/candidates below) only runs once we're in "run".
        ball_xy  = None
        ball_src = "cam"
        if ball is not None and pose is not None:
            rx, ry, hdg = pose
            angle_deg, dist_mm = ball
            b_rad = math.radians(angle_deg + hdg)
            ball_xy = (rx + dist_mm * math.sin(b_rad),
                       ry + dist_mm * math.cos(b_rad))
            if run_mode == "run":
                ball_mem.seen(ball_xy[0], ball_xy[1], now_t)
                ball_vel.update(now_t, ball_xy[0], ball_xy[1])
        elif remote_ball is not None:
            ball_xy = (remote_ball[0], remote_ball[1])
            # Same rule as the striker: only a peer packet carrying a real
            # sighting refreshes the memory (see state._state["remote_ball"]).
            if run_mode == "run" and remote_ball[3] == "cam":
                ball_mem.seen(ball_xy[0], ball_xy[1], now_t)
            ball_src = "remote"
        with state._lock:
            state._state["ball_est"] = ((ball_xy[0], ball_xy[1], 1.0, ball_src)
                                  if ball_xy is not None else None)

        if run_mode != "run":
            Motor.stopall()
            _set_dwibbler(False)
            return

        if pose is None:
            Motor.stopall()
            _set_dwibbler(False)
            return

        # General jam recovery (see JamRecovery's own docstring): a body-contact stuck
        # override, checked once per tick ahead of all normal steering below. skip=True
        # whenever we're already inside a recognized, expected near-ball contest
        # (enemy_contest_dist_mm) - that low-relative-motion shove is handled by the
        # contest-speed bypass further down, not by this general-purpose detector.
        rx_j, ry_j, hdg_j = pose
        if ball_xy is not None:
            bx_j, by_j = ball_xy
            jam_skip = (math.hypot(bx_j - rx_j, by_j - ry_j) <= enemy_contest_dist_mm
                        and _enemy_holds_ball(bx_j, by_j, enemies))
        else:
            bx_j = rx_j + math.sin(math.radians(hdg_j))
            by_j = ry_j + math.cos(math.radians(hdg_j))
            jam_skip = False
        jam_cmd = self.jam.check(rx_j, ry_j, hdg_j, bx_j, by_j, enemies, skip=jam_skip)
        if jam_cmd is not None:
            drive_rel, spd = jam_cmd
            _set_dwibbler(False)
            # overrideAcc=True: the push-through dwell is only jam_hold_s
            # (0.5s); the slew cap alone would spend most of it still ramping
            # toward full scale (~0.33s to reach 1.0), neutering the escape.
            # This is exactly the "emergency wall/enemy pushback" case the
            # overrideAcc bypass exists for (see motion.py's own comment).
            _slew_drive(drive_rel, spd, rot_speed=drive_rel * turn_gain,
                        overrideAcc=True)
            return

        # run_mode == "run" guarantees slot_goal is set (_apply_slot_state).
        with state._lock:
            goal = state._state["slot_goal"]
        own_goal, enemy_goal = _goal_positions(goal)
        gx, gy = own_goal
        into   = 1.0 if gy < FieldModel.field_y / 2 else -1.0
        line_y = gy + into * goalie_line_mm
        clamp_lo = gx - FieldModel.goal_width / 2 - goalie_x_slack_mm
        clamp_hi = gx + FieldModel.goal_width / 2 + goalie_x_slack_mm

        rx, ry, hdg = pose

        # known ball: enemy-sim line-keeper
        if ball_xy is not None:
            # Charge-and-clear: ball in our zone -> grab and carry it upfield.
            ball_depth = into * (ball_xy[1] - gy)
            if ball_depth < goal_line_mm:
                chase_y = ball_xy[1] # crossed, go all in
            else:
                safe_y  = gy + into * (goal_backwall_mm + goalie_backwall_tol_mm)
                chase_y = (max(ball_xy[1], safe_y) if into > 0
                          else min(ball_xy[1], safe_y))
            ball_robot_dist = math.hypot(ball_xy[0] - rx, ball_xy[1] - ry)
            # goal_side: are we still between our own goal and the ball? If an
            # attacker has gotten the ball behind us (closer to our own net
            # than we are), charging straight at its current spot drives
            # through the attacker from the wrong side, shoving them (and the
            # ball) further toward our own net. Skip the charge and
            # fall through to shadow/probability-cover below, which already
            # repositions onto the ball-to-goal line, to get goal-side again
            # first. Still always charges once the ball's actually crossed
            # into the goal box (ball_depth < goal_line_mm) - too dangerous
            # to wait out regardless of side.
            goal_side = (into * (ry - gy) <= ball_depth
                        or ball_depth < goal_line_mm)
            # Rule 5.11 guard: if the teammate's last-known (fresh) position
            # is already in/touching our own box, don't also charge in -
            # fall through to shadow/probability-cover below instead, same
            # as the goal_side False case above.
            teammate_in_box = motion._teammate_blocks_own_box(teammate_pos, own_goal)
            if (ball_robot_dist <= goalie_clear_dist_mm and goal_side
                    and not teammate_in_box):
                # Same direct-pursuit approach as StrikerController (sec 4.1).
                dx_, dy_ = ball_xy[0] - rx, chase_y - ry
                chase_dist = math.hypot(dx_, dy_)
                rel = _wrap_deg(math.degrees(math.atan2(dx_, dy_)) - hdg)
                drive_angle, frac = _orbit_approach(
                    chase_dist * math.sin(math.radians(rel)),
                    chase_dist * math.cos(math.radians(rel)),
                    rx, ry, hdg)
                turn = drive_angle * turn_gain
                # goalie_clear_dist_mm (600) decides whether to charge at all, a driving/positioning call, made early on purpose so the keeper is already moving.
                if ball_robot_dist <= dwibble_on_dist_mm:
                    _set_dwibbler(True, speed=_dwibble_carry_speed(
                        rx, ry, enemies, turn))
                else:
                    _set_dwibbler(False)
                # Carrying the clear upfield keeps keep_min_mm off the four perimeter walls (goal boxes excluded); the robot itself is free to cross the white line.
                rush = _lane_clear(rx, ry, hdg, drive_angle, enemies)
                cmd = min((rush_speed if rush else base_speed) * frac,
                          _brake_speed_frac(chase_dist))
                drive_angle, cmd = _enemy_guard(rx, ry, hdg, drive_angle, cmd,
                                                enemies, ball_robot_dist)
                drive_angle, cmd = _wall_guard(rx, ry, hdg, drive_angle, cmd)
                _slew_drive(drive_angle, cmd, rot_speed=turn)
                return

            # Escort/support: the teammate genuinely has the ball and is deep enough that guarding an unthreatened net costs more than it's worth.
            ball_depth_now = into * (ball_xy[1] - gy)
            if (escort_enabled and teammate_pos is not None
                    and peer_state in (has_ball, passing)
                    and ball_depth_now >= escort_min_depth_mm):
                tx, ty = teammate_pos
                side = 1.0 if tx <= FieldModel.field_x / 2 else -1.0
                target_x = max(clamp_lo, min(clamp_hi, tx + side * escort_lat_mm))
                target_y = ty + into * escort_lead_mm
                face_x, face_y = ball_xy
            else:
                # Shadow: standoff radius goalie_line_mm from goal centre, on the goal-centre -> ball line, narrows the angle the same way simulator.py's DefenderBot._shadow_pos does, rather than sliding along a fixed-depth line.
                #
                # Two adjustments to that base geometry:
                #  1. velocity lead: the goal-centre -> ball
                #     line is built from where the ball is HEADING, not just
                #     its current fix, using ball_vel (updated above whenever
                #     a fresh direct camera fix comes in).
                #  2. quadratic radius shrink: the
                #     standoff radius itself shrinks toward
                #     goalie_radius_min_factor as the (lead-adjusted) ball
                #     nears the goal-centre axis, and eases back to the full
                #     goalie_line_mm as it swings wide toward either post.
                vbx, vby = ball_vel.velocity()
                lead_x, lead_y = vbx * goalie_vel_lead_s, vby * goalie_vel_lead_s
                lead_mm = math.hypot(lead_x, lead_y)
                if lead_mm > goalie_vel_lead_max_mm:
                    scale = goalie_vel_lead_max_mm / lead_mm
                    lead_x *= scale
                    lead_y *= scale
                lead_bx, lead_by = ball_xy[0] + lead_x, ball_xy[1] + lead_y
                dxg, dyg = lead_bx - gx, lead_by - gy
                dg = math.hypot(dxg, dyg) or 1.0
                # Angle between the goal-centre -> (lead) ball line and the
                # goal-centre -> field-centre axis; 0 = ball dead ahead, 0.5*pi = ball level with
                # the goal line off to a post.
                axis_dot   = dyg * into
                axis_cross = dxg * into
                angle_off_axis = math.atan2(abs(axis_cross), axis_dot)
                axis_frac = max(0.0, min(1.0, angle_off_axis / (0.5 * math.pi)))
                axis_frac = max(0.0, min(1.0, 4.0 * axis_frac ** 2))
                radius_factor = (goalie_radius_min_factor
                                 + (1.0 - goalie_radius_min_factor) * axis_frac)
                radius = goalie_line_mm * radius_factor
                target_x = max(clamp_lo, min(clamp_hi,
                               gx + dxg * radius / dg))
                target_y = gy + dyg * radius / dg
                if teammate_in_box:
                    # Rule 5.11 guard, continued: shadow's own standoff
                    # radius can shrink (goalie_radius_min_factor) to less
                    # than the box depth when the ball's near the centre
                    # axis, so the normal single-goalie shadow point can
                    # itself land inside the box - fine for the sole
                    # legitimate defender, not fine as our "hold back"
                    # fallback when the teammate's already in there. Push
                    # it just clear instead.
                    target_x, target_y = _clamp_outside_own_box(
                        target_x, target_y, own_goal, Perception.robot_radius_mm)
                face_x, face_y = ball_xy
        else:
            # unknown ball: probability cover
            cands = ball_mem.candidates(now_t, enemies)
            with state._lock:
                state._state["ball_est"] = ((cands[0][0], cands[0][1], cands[0][2],
                                       "mem") if cands else None)
            target_y = line_y
            if cands:
                # Confidence-weighted shot-on-goal intercept on our line.
                sx = wsum = 0.0
                for cx_, cy_, p in cands:
                    denom = gy - cy_
                    bx = (cx_ + (line_y - cy_) / denom * (gx - cx_)
                          if abs(denom) > 1e-6 else cx_)
                    sx   += p * bx
                    wsum += p
                target_x = max(clamp_lo, min(clamp_hi, sx / wsum if wsum else gx))
                face_x, face_y, _p = max(cands, key=lambda c: c[2])
            else:
                target_x = gx # nothing known: centre
                face_x, face_y = gx, FieldModel.field_y / 2

        # Not clearing -> no dwibbler; slide along the line to target_x.
        _set_dwibbler(False)
        dx_, dy_  = target_x - rx, target_y - ry
        dist_t    = math.hypot(dx_, dy_)
        drive_rel = _wrap_deg(math.degrees(math.atan2(dx_, dy_)) - hdg)
        speed     = min(base_speed, _brake_speed_frac(dist_t))
        turn = _wrap_deg(math.degrees(math.atan2(face_x - rx, face_y - ry)) - hdg) * turn_gain
        # goalie_line_mm (400) already keeps the shadow clear of its own perimeter wall, so the keep-out only guards the side walls here.
        drive_rel, speed = _enemy_guard(rx, ry, hdg, drive_rel, speed, enemies)
        drive_rel, speed = _wall_guard(rx, ry, hdg, drive_rel, speed)
        _slew_drive(drive_rel, speed, rot_speed=turn)


# shared tick period, _play_loop sleeps this once per tick; StrikerController uses the
# same constant for its capture-hold hysteresis timer
loop_dt = 0.02

# Support/receiving position: the striker's mirror of GoalieController's escort logic
# above - when the teammate genuinely has the ball (peer_state, same as escort's own
# check) and we don't have a closer, more direct chase of our own, go to an open spot
# for a give-and-go instead of running a redundant, independent chase toward a ball the
# teammate already controls. Unlike the goalie's escort spot - which has no defensive-
# box constraint since it's already right next to its own goal - this target must also
# (a) stay off the carrier so it isn't just recreating the crowding problem escorting
# solves, and (b) stay out of our own defensive box. It's deliberately sized to land
# inside the existing pass logic's own acceptance window (pass_min_range_mm..
# pass_max_range_mm, sec 3.21) so the spot we move to is a plausible pass_target
# candidate for KnownOcclusion/_pass_race_open to actually pick, not just open space.
support_enabled       = True
# a directly reachable ball this close overrides support - a real, close chase beats
# standing off; same standoff distance the yield logic above already uses
support_ball_near_mm  = yield_ball_near_mm
# lateral offset from the teammate, toward whichever side of the pitch has more room
# (same "roomier side" pick as escort_lat_mm). Sized together with support_lead_mm so
# hypot(lat, lead) sits comfortably inside pass_min_range_mm..pass_max_range_mm rather
# than right at an edge case's mercy.
support_lat_mm        = 550.0
# how far further into the attack (past the teammate, toward the enemy goal) the
# support spot leads - mirrors escort_lead_mm but larger: a striker going open for a
# give-and-go wants a real head start up the pitch, not the goalie's token few cm
support_lead_mm       = 400.0
# never let the support spot land inside, or too near, our own defensive box, no matter
# how far back the teammate currently is - goal_line_mm (300, the box mouth) plus a
# clear margin so the striker isn't just camped on the goal line
support_min_own_depth_mm = goal_line_mm + 400.0


class StrikerController:
    """attacking role, a 3-state machine (seek / has_ball / passing) at about 50 Hz."""
    def __init__(self):
        """fresh FSM state (starts in seek) and fresh velocity/memory helpers for a new striker run."""
        self.state           = seek
        self.ball_vel        = BallVelocityEstimator()
        self.ball_mem        = BallMemory()
        self.teammate_vel    = TeammateVelocityEstimator()
        self.last_ball_fix   = None
        self.last_teammate_pos = None
        self.capture_hold_s  = 0.0 # camera capture-condition hysteresis accumulator
        self.possession_seen = False # dwibbler detector confirmed the current carry
        self.carry_start_t   = 0.0 # monotonic time this carry began (pass settle)
        # monotonic time the camera last had the ball inside dwibble_on_dist_mm
        self.close_ball_t    = 0.0
        self.pass_target     = None # (x, y) chosen when has_ball -> passing
        self.pass_hold_s     = 0.0 # passing alignment hysteresis accumulator
        self.pass_expiry_t   = 0.0 # monotonic time pass_target stops publishing
        # last has_ball tick was inside goal_eject_range_mm (roller deliberately cut, see
        # the stall-loss check's own guard below)
        self.near_goal_eject = False
        # accumulate/decay hold for the deep-stuck bail-out (see stuck_deep_enabled),
        # same pattern as capture_hold_s: ramps up while deep+contested, drains
        # otherwise, so a brief let-up doesn't instantly disarm it
        self.stuck_hold_s = 0.0
        # non-blocking dwibbler-eject countdown (state == ejecting): monotonic
        # deadline and which kind of eject (pass release vs flick abort) is
        # running, so the reversal can outlive the tick that started it
        # without ever time.sleep()ing inside the 50Hz control loop.
        self.eject_until_t = 0.0
        self.eject_is_pass = True
        self.jam = JamRecovery() # general body-contact stuck detector, see its class docstring

    def tick(self):
        """run one control tick of the striker behaviour, see the class docstring."""
        ball_vel, ball_mem = self.ball_vel, self.ball_mem
        teammate_vel = self.teammate_vel

        # pass_target keeps publishing for pass_target_max_age_s after we
        # fire (sec. 3.21), so the teammate has time to see and act on it,
        # then expires, unless we're still actively lining one up.
        if (self.state != passing and self.pass_target is not None
                and time.monotonic() >= self.pass_expiry_t):
            self.pass_target = None

        with state._lock:
            pose            = state._state["pose"]
            _pose_imu       = state._state["pose_imu"]
            _imu_now        = state._state["imu_heading"]
            ball            = state._state["ball"]
            enemies         = state._state["enemies"]
            enemy_vel_est   = state._state["enemy_vel"]
            remote_ball     = state._state["remote_ball"]
            yield_now       = state._state["yield_striker"]
            run_mode        = state._state["run_mode"]
            teammate_pos    = state._state["teammate_pos_bt"]
            peer_pass_target = state._state["peer_pass_target"]
            peer_state      = state._state["peer_state"]
            state._state["my_state"]    = self.state
            state._state["pass_target"] = self.pass_target
            # Reset every tick, not just when a new one is computed, or a stale zone from a previous tick would keep reading out for --motionlog.
            state._state["capture_zone"] = None

        # Steer on a heading carried forward from the IMU, not one that is up
        # to a whole lidar revolution old (see _fused_heading).
        pose = _fused_heading(pose, _pose_imu, _imu_now)

        if teammate_pos is not None and teammate_pos != self.last_teammate_pos:
            teammate_vel.update(time.monotonic(), teammate_pos[0], teammate_pos[1])
            self.last_teammate_pos = teammate_pos
        elif teammate_pos is None:
            teammate_vel.reset()
            self.last_teammate_pos = None

        # fused ball estimate (field frame), shared over the team link and used by seek when our own camera is blind.
        ball_est = None
        if pose is not None:
            if ball is not None:
                _b = math.radians(ball[0] + pose[2])
                ball_est = (pose[0] + ball[1] * math.sin(_b),
                            pose[1] + ball[1] * math.cos(_b),
                            1.0, "cam")
            elif remote_ball is not None:
                ball_est = (remote_ball[0], remote_ball[1],
                            min(1.0, remote_ball[2]), "remote")
                # Only a real sighting counts as one.
                if run_mode == "run" and remote_ball[3] == "cam":
                    ball_mem.seen(remote_ball[0], remote_ball[1],
                                  time.monotonic())
            elif run_mode == "run":
                _inf = ball_mem.infer(time.monotonic(), enemies)
                if _inf is not None:
                    ball_est = (_inf[0], _inf[1], _inf[2], "mem")
                elif peer_pass_target is not None:
                    ball_est = (peer_pass_target[0], peer_pass_target[1],
                               0.5, "pass")
        with state._lock:
            state._state["ball_est"] = ball_est

        if run_mode != "run":
            Motor.stopall()
            _set_dwibbler(False)
            # Idle abandons an in-progress pass rather than leave it broadcasting a target that's never going to fire.
            if self.state == passing:
                self.state = seek
            self.pass_target = None
            with state._lock:
                state._state["pass_target"] = None
            return

        # General jam recovery (see JamRecovery's own docstring): a body-contact stuck
        # override, checked once per tick ahead of all normal steering below. skip=True
        # whenever we're already inside a recognized, expected near-ball contest
        # (enemy_contest_dist_mm) - that low-relative-motion shove is handled by the
        # contest-speed bypass (_approach_speed_cap) further down, not by this
        # general-purpose detector.
        if pose is not None:
            rx_j, ry_j, hdg_j = pose
            if ball_est is not None:
                bx_j, by_j = ball_est[0], ball_est[1]
                jam_skip = (math.hypot(bx_j - rx_j, by_j - ry_j) <= enemy_contest_dist_mm
                            and _enemy_holds_ball(bx_j, by_j, enemies))
            else:
                bx_j = rx_j + math.sin(math.radians(hdg_j))
                by_j = ry_j + math.cos(math.radians(hdg_j))
                jam_skip = False
            jam_cmd = self.jam.check(rx_j, ry_j, hdg_j, bx_j, by_j, enemies, skip=jam_skip)
            if jam_cmd is not None:
                drive_rel, spd = jam_cmd
                _set_dwibbler(False)
                # overrideAcc=True, same as the goalie's own jam push above:
                # the 0.5s dwell can't afford the ~0.33s slew ramp to full
                # scale (see that site's comment).
                _slew_drive(drive_rel, spd, rot_speed=drive_rel * turn_gain,
                            overrideAcc=True)
                return

        # run_mode == "run" guarantees slot_goal is set (_apply_slot_state).
        with state._lock:
            goal = state._state["slot_goal"]
        own_goal, enemy_goal = _goal_positions(goal)

        # seek: direct-pursuit capture (sec 4.1)
        if self.state == seek:
            # Roller already loaded -> we have it, whatever the camera says, the dwibbler-stall detector is the only sensor that sees a ball flush in the mouth (self-occluded from the camera).
            if (_possession.has_ball
                    and time.monotonic() - self.close_ball_t
                        <= possess_capture_window_s):
                _set_dwibbler(True)
                self.state           = has_ball
                self.possession_seen = True
                self.capture_hold_s  = 0.0
                self.carry_start_t   = time.monotonic()
                self.stuck_hold_s    = 0.0
                print("[main] ball captured -> attacking (dwibbler)",
                      flush=True)
                if state._capture_log is not None:
                    state._capture_log.event("captured", "dwibbler-stall route")
                return

            # Yield to the teammate (hold a covering point).
            ball_near = ball is not None and ball[1] <= yield_ball_near_mm
            if yield_now and not ball_near and pose is not None:
                rx, ry, hdg = pose
                tx, ty = own_goal
                if ball_est is not None:
                    bx_, by_ = ball_est[0], ball_est[1]
                else:
                    bx_, by_ = FieldModel.cx, FieldModel.field_y / 2
                ang = math.atan2(bx_ - tx, by_ - ty)
                hx  = tx + yield_hold_mm * math.sin(ang)
                hy  = ty + yield_hold_mm * math.cos(ang)
                dx_, dy_  = hx - rx, hy - ry
                dist_t    = math.hypot(dx_, dy_)
                drive_rel = _wrap_deg(math.degrees(math.atan2(dx_, dy_))
                                      - hdg)
                face = _wrap_deg(math.degrees(math.atan2(bx_ - rx,
                                                         by_ - ry)) - hdg)
                _set_dwibbler(False)
                ball_vel.reset()
                self.last_ball_fix = None
                spd = min(base_speed, _brake_speed_frac(dist_t))
                drive_rel, spd = _enemy_guard(rx, ry, hdg, drive_rel, spd, enemies)
                drive_rel, spd = _wall_guard(rx, ry, hdg, drive_rel, spd)
                _slew_drive(drive_rel, spd, rot_speed=face * turn_gain)
                return

            # Support/receiving position: the teammate already has the ball
            # (peer_state) and we don't have a closer, more direct chase of
            # our own - a genuinely close/reachable ball still wins (same
            # standoff support_ball_near_mm uses as the yield check above),
            # otherwise go open for a give-and-go instead of running an
            # independent chase toward a ball someone else already controls
            # (mirrors GoalieController's escort/support logic, sec above).
            close_ball = ball is not None and ball[1] <= support_ball_near_mm
            if (support_enabled and not close_ball and teammate_pos is not None
                    and peer_state in (has_ball, passing) and pose is not None):
                rx, ry, hdg = pose
                ogx, ogy = own_goal
                own_into = 1.0 if ogy < FieldModel.field_y / 2 else -1.0
                tx, ty = teammate_pos
                side = 1.0 if tx <= FieldModel.field_x / 2 else -1.0
                target_x = max(0.0, min(FieldModel.field_x,
                                        tx + side * support_lat_mm))
                target_y = ty + own_into * support_lead_mm
                depth = own_into * (target_y - ogy)
                if depth < support_min_own_depth_mm:
                    target_y = ogy + own_into * support_min_own_depth_mm
                if ball_est is not None:
                    face_x, face_y = ball_est[0], ball_est[1]
                else:
                    face_x, face_y = tx, ty
                dx_, dy_  = target_x - rx, target_y - ry
                dist_t    = math.hypot(dx_, dy_)
                drive_rel = _wrap_deg(math.degrees(math.atan2(dx_, dy_))
                                      - hdg)
                face = _wrap_deg(math.degrees(math.atan2(face_x - rx,
                                                         face_y - ry)) - hdg)
                _set_dwibbler(False)
                ball_vel.reset()
                self.last_ball_fix = None
                spd = min(base_speed, _brake_speed_frac(dist_t))
                drive_rel, spd = _enemy_guard(rx, ry, hdg, drive_rel, spd, enemies)
                drive_rel, spd = _wall_guard(rx, ry, hdg, drive_rel, spd)
                # _slew_drive, not Motor.drive: every other drive path here is
                # slew-capped, and an uncapped escort command jumps 0 ->
                # base_speed in one tick straight out of a stop (idle/lost
                # pose), a jerk every other path is protected from.
                _slew_drive(drive_rel, spd, rot_speed=face * turn_gain)
                return

            if ball is not None:
                angle_deg, dist_mm = ball
                if pose is not None:
                    # Absolute field frame (origin bottom-left corner): project the ball into field coords for velocity/memory tracking, and to pick which tangent side to circle from so we end up facing goal (sec 4.1).
                    rx, ry, hdg = pose
                    b_rad = math.radians(angle_deg + hdg)
                    bx = rx + dist_mm * math.sin(b_rad)
                    by = ry + dist_mm * math.cos(b_rad)
                    if ball is not self.last_ball_fix:
                        now_t = time.monotonic()
                        ball_vel.update(now_t, bx, by)
                        ball_mem.seen(bx, by, now_t)
                        self.last_ball_fix = ball
                    rel_rad = math.radians(angle_deg)
                    lat_mm  = dist_mm * math.sin(rel_rad)
                    vbx, vby = ball_vel.velocity()
                    # Computed unconditionally (not just under capture_ball_vel)
                    # so --capturelog still sees the ball's velocity even with
                    # Method 6 switched off.
                    own_into = 1.0 if own_goal[1] < FieldModel.field_y / 2 else -1.0
                    ball_depth_now = own_into * (by - own_goal[1])
                    if _ball_beaten(ry, by, own_goal) and ball_depth_now >= goal_line_mm:
                        # Beaten: an attacker's got the ball deeper into our own
                        # half than we currently are. A direct approach from here
                        # reaches only their rear/flank - their front, and the
                        # shot lane, stay clear.
                        # Get goal-side of the ball first instead (sec 4.1).
                        tx_, ty_ = _defend_recovery_target(
                            rx, ry, bx, by, own_goal,
                            recovery_standoff_mm, recovery_lateral_mm)
                        if motion._teammate_blocks_own_box(teammate_pos, own_goal):
                            # Rule 5.11 guard: teammate's already in/touching
                            # our own box (presumably the goalie) - hold
                            # goal-side of the ball but don't also enter.
                            tx_, ty_ = _clamp_outside_own_box(
                                tx_, ty_, own_goal, Perception.robot_radius_mm)
                        dxr, dyr = tx_ - rx, ty_ - ry
                        drive_angle = _wrap_deg(
                            math.degrees(math.atan2(dxr, dyr)) - hdg)
                        turn      = drive_angle * turn_gain
                        in_cone   = False
                        rush      = False
                        drive_cmd = min(base_speed,
                                        _brake_speed_frac(math.hypot(dxr, dyr)))
                    else:
                        drive_angle, speed_frac = _orbit_approach(
                            dist_mm * math.sin(rel_rad),
                            dist_mm * math.cos(rel_rad),
                            rx, ry, hdg)
                        turn = drive_angle * turn_gain
                        in_cone = (abs(angle_deg) <= capture_cone_half_deg
                                  and abs(lat_mm) <= capture_cone_half_width_mm)
                        # Clear lane ahead (sec 3.1/_lane_clear) -> rush_speed
                        # for the open chase, never once centred: that's the
                        # final approach (sec 4.1), not a straight-line sprint.
                        rush = not in_cone and _lane_clear(
                            rx, ry, hdg, drive_angle, enemies)
                        drive_cmd = (rush_speed if rush else base_speed) * speed_frac
                        if capture_ball_vel:
                            # Method 6: x_o = x_b + v_o
                            drive_angle, drive_cmd = _add_ball_velocity(
                                drive_angle, drive_cmd, hdg, vbx, vby)
                    # Real mass behind this (F=ma): a linear/flat approach speed can still be moving too fast to actually stop at the ball, overshooting the capture instead of a controlled trap. Bypassed via _approach_speed_cap once an enemy is plausibly holding the ball this close in - see contest_speed_cmd's own comment, this is the "shove through, don't brake to a gentle stop at their body" case.
                    drive_cmd = _approach_speed_cap(
                        dist_mm, bx, by, enemies, min(drive_cmd, _brake_speed_frac(dist_mm)))
                else:
                    # No lidar fix yet: no field frame, so no y=0 tie-break and no ball-velocity compensation either.
                    ball_vel.reset()
                    self.last_ball_fix = None
                    b_rad  = math.radians(angle_deg)
                    lat_mm = dist_mm * math.sin(b_rad)
                    drive_angle, speed_frac = _orbit_approach(
                        dist_mm * math.sin(b_rad),
                        dist_mm * math.cos(b_rad))
                    turn = drive_angle * turn_gain
                    in_cone = (abs(angle_deg) <= capture_cone_half_deg
                              and abs(lat_mm) <= capture_cone_half_width_mm)
                    drive_cmd = min(base_speed * speed_frac,
                                    _brake_speed_frac(dist_mm))
                    rush = False # no pose -> no field-frame lane to check
                with state._lock:
                    state._state["capture_zone"] = "cone" if in_cone else None
                # Draw the ball in once it's centred / close.
                _set_dwibbler(in_cone or dist_mm < dwibble_on_dist_mm)
                if dist_mm < dwibble_on_dist_mm:
                    # Arms the dwibbler-only capture above for possess_capture_window_s, this is the last moment the camera can see a ball that is about to disappear into the mouth.
                    self.close_ball_t = time.monotonic()
                if pose is not None:
                    # keep_min_mm applies here same as always, we're
                    # still chasing, not carrying (we can't dig a ball
                    # flush against a wall, by design).
                    drive_angle, drive_cmd = _enemy_guard(
                        rx, ry, hdg, drive_angle, drive_cmd, enemies, dist_mm)
                    drive_angle, drive_cmd = _wall_guard(
                        rx, ry, hdg, drive_angle, drive_cmd)
                _slew_drive(drive_angle, drive_cmd, rot_speed=turn)
                if state._capture_log is not None and pose is not None:
                    state._capture_log.tick(
                        self.state, state._state["capture_zone"], rx, ry, hdg,
                        bx, by, (vbx, vby), drive_angle, drive_cmd)
                    if state._capture_log.is_done():
                        _finish_capture_log()
                # Camera capture = close and centred, sustained for capture_confirm_s.
                if dist_mm < capture_dist_mm and in_cone:
                    self.capture_hold_s += loop_dt
                else:
                    self.capture_hold_s = max(0.0, self.capture_hold_s - loop_dt)
                if self.capture_hold_s >= capture_confirm_s:
                    self.state = has_ball
                    self.possession_seen = False
                    self.capture_hold_s  = 0.0
                    self.carry_start_t   = time.monotonic()
                    self.stuck_hold_s    = 0.0
                    print("[main] ball captured -> attacking", flush=True)
                    if state._capture_log is not None:
                        state._capture_log.event("captured", "camera route")
            else:
                ball_vel.reset()
                self.last_ball_fix = None
                # Camera lost the ball, chase the fused estimate (the subordinate's sighting or the remembered/attributed position) before blind-spinning.
                _set_dwibbler(False)
                if (ball_est is not None
                        and ball_est[2] >= ball_mem_min_conf):
                    mbx, mby = ball_est[0], ball_est[1]
                    rx, ry, hdg = pose
                    own_into = 1.0 if own_goal[1] < FieldModel.field_y / 2 else -1.0
                    ball_depth_now = own_into * (mby - own_goal[1])
                    with state._lock:
                        state._state["capture_zone"] = None
                    if _ball_beaten(ry, mby, own_goal) and ball_depth_now >= goal_line_mm:
                        # Beaten, same as the seek/ball-seen branch above (sec 4.1):
                        # get goal-side of the memory estimate before chasing it.
                        tx_, ty_ = _defend_recovery_target(
                            rx, ry, mbx, mby, own_goal,
                            recovery_standoff_mm, recovery_lateral_mm)
                        if motion._teammate_blocks_own_box(teammate_pos, own_goal):
                            # Rule 5.11 guard, same as the seek/ball-seen branch above.
                            tx_, ty_ = _clamp_outside_own_box(
                                tx_, ty_, own_goal, Perception.robot_radius_mm)
                        dxr, dyr = tx_ - rx, ty_ - ry
                        dist_r = math.hypot(dxr, dyr)
                        drive_angle = _wrap_deg(
                            math.degrees(math.atan2(dxr, dyr)) - hdg)
                        turn = drive_angle * turn_gain
                        cmd  = min(base_speed, _brake_speed_frac(dist_r))
                        drive_angle, cmd = _enemy_guard(
                            rx, ry, hdg, drive_angle, cmd, enemies)
                    else:
                        dx_, dy_ = mbx - rx, mby - ry
                        dist_est = math.hypot(dx_, dy_)
                        rel = _wrap_deg(math.degrees(math.atan2(dx_, dy_)) - hdg)
                        drive_angle, speed_frac = _orbit_approach(
                            dist_est * math.sin(math.radians(rel)),
                            dist_est * math.cos(math.radians(rel)),
                            rx, ry, hdg)
                        turn = drive_angle * turn_gain
                        drive_angle, cmd = _enemy_guard(
                            rx, ry, hdg, drive_angle,
                            _approach_speed_cap(
                                dist_est, mbx, mby, enemies,
                                min(base_speed * speed_frac, _brake_speed_frac(dist_est))),
                            enemies, dist_est)
                    drive_angle, cmd = _wall_guard(
                        rx, ry, hdg, drive_angle, cmd)
                    _slew_drive(drive_angle, cmd, rot_speed=turn)
                else:
                    with state._lock:
                        state._state["capture_zone"] = "search"
                    # Completely lost: fall back toward one bot radius below field centre (own-goal side) instead of holding open field. No in-place spin while searching: the camera's own fisheye ring is a full 360-degree scan regardless of heading (sec 3.6), so turning the chassis doesn't see anything new.
                    if pose is not None:
                        rx, ry, hdg = pose
                        fall_sign = (1.0 if own_goal[1] > FieldModel.field_y / 2
                                     else -1.0)
                        fx = FieldModel.cx
                        fy = (FieldModel.field_y / 2
                              + fall_sign * Perception.robot_radius_mm)
                        dx_, dy_  = fx - rx, fy - ry
                        dist_f    = math.hypot(dx_, dy_)
                        drive_rel = _wrap_deg(math.degrees(math.atan2(dx_, dy_))
                                              - hdg)
                        spd = min(base_speed, _brake_speed_frac(dist_f))
                        drive_rel, spd = _enemy_guard(rx, ry, hdg, drive_rel, spd, enemies)
                        drive_rel, spd = _wall_guard(rx, ry, hdg, drive_rel, spd)
                        _slew_drive(drive_rel, spd)
                    else:
                        # No pose yet -> no fallback point to drive to, hold still.
                        _slew_drive(0, 0)

        # has_ball: carry the ball into the enemy goal
        elif self.state == has_ball:
            # Full grip as the default hold, overridden below once pose is known, either graduated (contested/turning/straight, see _dwibble_carry_speed) or reversed to eject near the goal.
            _set_dwibbler(True)
            if _possession.has_ball:
                self.possession_seen = True
            # If ball reappears far away we dropped it, go back to seek
            if ball is not None:
                _, dist_mm = ball
                if dist_mm > capture_dist_mm * 2:
                    self.state = seek
                    self.possession_seen = False
                    _possession.reset()
                    print("[main] lost ball -> seeking", flush=True)
                    return
            # Dwibbler-stall loss check: the roller free-spinning again means the mouth is empty even if the camera sees nothing. Also bails if possession was never confirmed at all within a fair grace period (the camera-only capture route can enter has_ball without ever having grabbed anything), not just after a confirmed carry goes empty.
            grace_elapsed = (time.monotonic() - self.carry_start_t
                            >= possess_spinup_s + possess_on_s)
            if ((self.possession_seen or grace_elapsed)
                    and _possession.available
                    and not _possession.has_ball
                    and not self.near_goal_eject):
                never_confirmed = not self.possession_seen
                self.state = seek
                self.possession_seen = False
                print("[main] dwibbler empty -> seeking"
                      + (" (never confirmed)" if never_confirmed else ""),
                      flush=True)
                return

            if pose is None:
                # No LiDAR fix yet: just drive forward and hope.
                _slew_drive(0, wall_safe_speed_cmd)
            else:
                rx, ry, hdg = pose
                gx, gy = enemy_goal
                d_us = math.hypot(gx - rx, gy - ry)

                # Tap-in, see tap_in_enabled's own comment for why this
                # runs unconditionally, ahead of and separate from the
                # ordinary settle_s-gated pass below.
                if (pass_enabled and tap_in_enabled and teammate_pos is not None
                        and math.hypot(gx - teammate_pos[0],
                                       gy - teammate_pos[1]) <= tap_in_range_mm):
                    tx, ty = teammate_pos
                    tvx, tvy = teammate_vel.velocity()
                    tap_target = (min(max(tx + tvx * pass_lead_s, 0.0),
                                      FieldModel.field_x),
                                 min(max(ty + tvy * pass_lead_s, 0.0),
                                     FieldModel.field_y))
                    d_tap_us = math.hypot(tap_target[0] - rx, tap_target[1] - ry)
                    tap_blockers = [(e["x"], e["y"], pass_lane_block_mm)
                                    for e in enemies]
                    if (d_tap_us <= pass_max_range_mm
                            and KnownOcclusion(tap_target, tap_blockers
                                              ).observed(gx, gy)
                            and KnownOcclusion((rx, ry), tap_blockers
                                              ).observed(*tap_target)
                            and _pass_race_open((rx, ry), tap_target, enemies, enemy_vel_est)):
                        self.state       = passing
                        self.pass_target = tap_target
                        self.pass_hold_s = 0.0
                        print(f"[main] tap-in -> passing to "
                             f"{tap_target[0]:.0f},{tap_target[1]:.0f}",
                             flush=True)
                        return

                # Pass instead of driving in ourselves, if the teammate is meaningfully closer to goal and within eject range (README sec. 3.21).
                settled = (time.monotonic() - self.carry_start_t
                           >= pass_settle_s)

                # Deep-stuck bail-out: still carrying deep in our own half and
                # contested by a nearby enemy for a sustained stretch - widen
                # the pass trigger below so we hand off to the teammate
                # (typically the goalie) even though they aren't closer to the
                # enemy goal, rather than keep grinding it out under pressure
                # right in front of our own net. stuck_hold_s stays armed for
                # stuck_deep_arm_window_s after the contest eases (see the
                # constants' own comment) - required for this to ever fire at
                # all, since _pass_race_open below can't pass while the
                # contesting enemy is still literally on top of us.
                own_into = 1.0 if own_goal[1] < FieldModel.field_y / 2 else -1.0
                deep_now = (own_into * (ry - own_goal[1])) <= stuck_deep_depth_mm
                contested_now = any(
                    math.hypot(e["x"] - rx, e["y"] - ry) <= dwibble_contest_dist_mm
                    for e in enemies)
                if stuck_deep_enabled and deep_now:
                    cap = stuck_deep_contest_s + stuck_deep_arm_window_s
                    if contested_now:
                        self.stuck_hold_s = min(cap, self.stuck_hold_s + loop_dt)
                    else:
                        self.stuck_hold_s = max(0.0, self.stuck_hold_s - loop_dt)
                else:
                    self.stuck_hold_s = 0.0
                stuck_deep = (stuck_deep_enabled
                             and self.stuck_hold_s >= stuck_deep_contest_s)

                if (pass_enabled and (settled or stuck_deep)
                        and teammate_pos is not None
                        and (d_us > pass_min_own_dist_mm or stuck_deep)):
                    tx, ty  = teammate_pos
                    d_tm    = math.hypot(gx - tx, gy - ty)
                    d_tm_us = math.hypot(tx - rx, ty - ry)
                    if ((d_tm + pass_teammate_gain_mm < d_us or stuck_deep)
                            and pass_min_range_mm <= d_tm_us <= pass_max_range_mm):
                        # Lead the target by the teammate's own estimated
                        # velocity, a through pass if they're moving,
                        # a fixed-spot pass if they're standing still.
                        tvx, tvy = teammate_vel.velocity()
                        target = (min(max(tx + tvx * pass_lead_s, 0.0),
                                     FieldModel.field_x),
                                 min(max(ty + tvy * pass_lead_s, 0.0),
                                     FieldModel.field_y))
                        # Don't release into a covered lane, a tracked enemy between us and the target (or behind a goal box) means the eject just hands it over.
                        blockers = [(e["x"], e["y"], pass_lane_block_mm)
                                   for e in enemies]
                        if (KnownOcclusion((rx, ry), blockers).observed(*target)
                                and _pass_race_open((rx, ry), target, enemies, enemy_vel_est)):
                            self.state       = passing
                            self.pass_target = target
                            self.pass_hold_s = 0.0
                            print(f"[main] passing to "
                                 f"{target[0]:.0f},{target[1]:.0f}", flush=True)
                            return

                # Drive the ball into the enemy goal. Default to the
                # straight goal-centre bearing, but evaluate finishing-angle
                # candidates (direct off-centre shots + a bank shot off the
                # far mouth wall) and steer at whichever one actually has a
                # clear, scoring path when a tracked enemy is in the way of
                # the straight line (see _goal_shot_aim above).
                goal_bearing = math.degrees(math.atan2(gx - rx, gy - ry))
                aim_bearing, aim_found = _goal_shot_aim(rx, ry, gy, enemies)
                if aim_found:
                    goal_bearing = aim_bearing
                goal_rel     = _wrap_deg(goal_bearing - hdg)

                # Not attempting an actual shot yet: the wind-up/snap turn could get stuck rotating in place with the dwibbler spinning and the chassis going nowhere, so a blocked lane just reverses the dwibbler in place instead.
                # Gated on flick_settle_s of continuous carry first: right
                # after a contested capture, the blocker is still standing
                # right where we just took the ball from, so evaluating
                # "blocked" immediately just reverses the dwibbler straight
                # back into their control - a self-dispossession loop
                # (capture -> instantly re-eject -> they/we recapture ->
                # repeat) that never lets the drive-to-goal code below even
                # run a single tick. Waiting lets that straight-line drive
                # try to open the lane (or clear the goal_eject_range_mm
                # band entirely) before we give up and eject. Also skipped
                # when the finishing-angle search above already found a
                # clear direct or bank-shot line - goal_bearing is already
                # steering at it, no need to abort.
                settled_for_flick = (time.monotonic() - self.carry_start_t
                                      >= flick_settle_s)
                if (flick_enabled and settled_for_flick
                        and goal_eject_range_mm <= d_us < flick_range_mm
                        and enemies and not aim_found):
                    blockers = [(e["x"], e["y"], pass_lane_block_mm)
                               for e in enemies]
                    lane_open = KnownOcclusion((rx, ry),
                                               blockers).observed(gx, gy)
                    if not lane_open:
                        # Straight into the shared non-blocking eject countdown
                        # (flick mode); the old flick_shot state just slept
                        # flick_snap_s in-tick, stalling the control loop.
                        _set_dwibbler(True, speed=-flick_kick_speed)
                        self.eject_until_t = time.monotonic() + flick_snap_s
                        self.eject_is_pass = False
                        self.state         = ejecting
                        print("[main] blocked -> reversing dwibbler "
                             "(no shot attempt yet)", flush=True)
                        return

                # Carry/shield decision (sec 3.23): no tap-in, no pass, no open
                # shot (aim_found False - the finishing-angle search above
                # found nothing that scores) and not yet inside
                # flick_range_mm - press for goal only if the lane is
                # genuinely winnable right now (_carry_should_press), else
                # hold the shield heading and creep clear instead of forcing a
                # drive into/around a defender that can actually contest it.
                # The pass/tap-in checks above already recheck fresh every
                # tick ahead of this, so the moment the defender commits and a
                # window opens, the very next tick's pass takes it - this
                # only covers the frames before that.
                if (not aim_found and enemies and d_us >= flick_range_mm
                        and not _carry_should_press(rx, ry, gx, gy, enemies)):
                    shield_rel = _shield_heading_deg(rx, ry, hdg, gx, gy, enemies)
                    creep_rel, creep_cmd = _carry_creep(rx, ry, hdg, gx, gy, enemies)
                    _set_dwibbler(True, speed=_dwibble_carry_speed(
                        rx, ry, enemies, shield_rel * turn_gain))
                    _slew_drive(creep_rel, creep_cmd,
                               rot_speed=shield_rel * turn_gain)
                    return

                # No solenoid: temporary scoring strategy is drive-in, not a kick.
                self.near_goal_eject = d_us < goal_eject_range_mm
                if self.near_goal_eject:
                    _set_dwibbler(False)
                else:
                    # Not yet close enough to eject: graduated grip for the straight-line push, full if an enemy is closing in or we're turning hard to line up, eased off otherwise (see _dwibble_carry_speed).
                    _set_dwibbler(True, speed=_dwibble_carry_speed(
                        rx, ry, enemies, goal_rel * turn_gain))
                # _wall_guard is bypassed here for now: the goal mouth's own flanking boundary segments sit close enough to read as a wall.
                _slew_drive(goal_rel, base_speed, rot_speed=goal_rel * turn_gain)

        # passing: line up on the pass target and release the ball off the
        # dwibbler (reversed briefly), no solenoid needed (sec. 3.21).
        elif self.state == passing:
            # Same loss checks as has_ball above, not a stricter visible-right-now test, since the ball sitting in the mouth is often invisible to the camera.
            _set_dwibbler(True)
            if _possession.has_ball:
                self.possession_seen = True
            if ball is not None:
                _, dist_mm = ball
                if dist_mm > capture_dist_mm * 2:
                    self.state       = seek
                    self.pass_target = None
                    with state._lock:
                        state._state["pass_target"] = None
                    self.possession_seen = False
                    _possession.reset()
                    print("[main] lost ball mid-pass -> seeking", flush=True)
                    return
            if ((self.possession_seen
                 or time.monotonic() - self.carry_start_t
                    >= possess_spinup_s + possess_on_s)
                    and _possession.available and not _possession.has_ball):
                self.state       = seek
                self.pass_target = None
                with state._lock:
                    state._state["pass_target"] = None
                self.possession_seen = False
                print("[main] dwibbler empty mid-pass -> seeking", flush=True)
                return

            if pose is None:
                _slew_drive(0, 0)
                return
            rx, ry, hdg = pose
            tx, ty = self.pass_target
            rel = _wrap_deg(math.degrees(math.atan2(tx - rx, ty - ry)) - hdg)
            if abs(rel) > pass_align_deg:
                self.pass_hold_s = 0.0
                _slew_drive(0, 0, rot_speed=rel * turn_gain)
                return
            self.pass_hold_s += loop_dt
            if self.pass_hold_s < pass_confirm_s:
                _slew_drive(0, 0, rot_speed=rel * turn_gain)
                return
            # Aligned and settled, release: reverse the dwibbler briefly.
            # time.sleep would stall the whole 50Hz control loop for
            # pass_eject_s (0.25s, 12+ dead ticks holding a spin-in-place),
            # so instead stage the eject as a countdown: the dwibbler stays
            # reversed until the deadline, then the next tick cleans up and
            # hands back to seek. Same shape as any other per-tick state.
            _set_dwibbler(True, speed=-pass_eject_speed)
            self.eject_until_t    = time.monotonic() + pass_eject_s
            self.eject_is_pass    = True
            self.pass_hold_s      = 0.0
            self.pass_expiry_t    = time.monotonic() + pass_target_max_age_s
            _possession.reset()
            self.state            = ejecting
            print("[main] pass released", flush=True)

        # ejecting: pass/flick dwibbler-reverse countdown, one shared state
        # (see the passing/flick_shot release sites above/below) - runs the
        # reversal for its full dwell without ever blocking the control loop,
        # then transitions back to seek.
        elif self.state == ejecting:
            if time.monotonic() < self.eject_until_t:
                _set_dwibbler(True,
                              speed=-(pass_eject_speed if self.eject_is_pass
                                      else flick_kick_speed))
                # Hold still-ish while ejecting; a spin-in-place would fight
                # the ball leaving the mouth.
                _slew_drive(0, 0)
                return
            _set_dwibbler(False)
            if not self.eject_is_pass:
                _possession.reset()
                self.possession_seen = False
                print("[main] dwibbler reversed (no shot attempt) -> seeking",
                      flush=True)
            self.state = seek

        # shoot: align to the enemy goal and kick (solenoid)
        # elif self.state == shoot:             # <- enable with solenoid
        #     self.state = _shoot_tick(pose, enemy_goal,
        #                              time.monotonic() - t_shoot0)


def _camrun_forward_bearing(front_goal, imu_now, forward0):
    """current best estimate of the robot-frame bearing (0 = dead ahead right now) to "forward", the direction camrun pushes the ball and circles its approach around (sec 4.9)."""
    if (front_goal is not None and vision._enemy_goal_colour is not None
            and front_goal[0] == vision._enemy_goal_colour):
        return front_goal[1]
    if imu_now is not None and forward0 is not None:
        return -_wrap_deg(imu_now - forward0)
    return 0.0


class CamRunController:
    """run_mode "camrun" (sec 4.9): camera and dwibbler-stall possession only, permanent no-lidar mode, never reads state._state["pose"]."""
    def __init__(self):
        """fresh FSM state (starts in seek)."""
        self.state           = seek
        self.capture_hold_s  = 0.0
        self.possession_seen = False
        self.close_ball_t    = 0.0
        self.carry_start_t   = 0.0 # monotonic time this carry began

    def tick(self):
        """run one control tick of the camera-only behaviour, see the class docstring."""
        with state._lock:
            ball        = state._state["ball"]
            front_goal  = state._state["front_goal"]
            run_mode    = state._state["run_mode"]
            imu_now     = state._state["imu_heading"]
            forward0    = state._state["camrun_forward_heading"]
            state._state["my_state"] = self.state
            # Reset every tick, same reasoning as StrikerController's own top-of-tick reset, otherwise a stale zone would keep reading out.
            state._state["capture_zone"] = None

        if run_mode != "camrun":
            Motor.stopall()
            _set_dwibbler(False)
            self.state = seek
            return

        fwd_bearing = _camrun_forward_bearing(front_goal, imu_now, forward0)

        # seek: direct-pursuit approach the ball (sec 4.1/_orbit_approach).
        if self.state == seek:
            if (_possession.has_ball
                    and time.monotonic() - self.close_ball_t
                        <= possess_capture_window_s):
                _set_dwibbler(True)
                self.state           = has_ball
                self.possession_seen = True
                self.capture_hold_s  = 0.0
                self.carry_start_t   = time.monotonic()
                print("[main] ball captured -> attacking (camrun, dwibbler)",
                      flush=True)
                return

            if ball is not None:
                angle_deg, dist_mm = ball
                b_rad  = math.radians(angle_deg)
                bx, by = dist_mm * math.sin(b_rad), dist_mm * math.cos(b_rad)
                drive_angle, speed_frac = _orbit_approach(bx, by)
                drive_cmd = min(base_speed * speed_frac, _brake_speed_frac(dist_mm))
                # Nose tracks the drive target, which is now just the ball's own bearing (direct pursuit).
                turn = drive_angle * turn_gain

                lat_mm  = dist_mm * math.sin(math.radians(angle_deg))
                in_cone = (abs(angle_deg) <= capture_cone_half_deg
                           and abs(lat_mm) <= capture_cone_half_width_mm)
                with state._lock:
                    state._state["capture_zone"] = "cone" if in_cone else None
                _set_dwibbler(in_cone or dist_mm < dwibble_on_dist_mm)
                if dist_mm < dwibble_on_dist_mm:
                    self.close_ball_t = time.monotonic()
                # No pose -> _wall_guard can't run (needs rx, ry), same as
                # every no-fix branch elsewhere in this file.
                _slew_drive(drive_angle, drive_cmd, rot_speed=turn)
                if dist_mm < capture_dist_mm and in_cone:
                    self.capture_hold_s += loop_dt
                else:
                    self.capture_hold_s = max(0.0, self.capture_hold_s - loop_dt)
                if self.capture_hold_s >= capture_confirm_s:
                    self.state           = has_ball
                    self.possession_seen = False
                    self.capture_hold_s  = 0.0
                    self.carry_start_t   = time.monotonic()
                    print("[main] ball captured -> attacking (camrun)",
                          flush=True)
            else:
                _set_dwibbler(False)
                with state._lock:
                    state._state["capture_zone"] = "search"
                # No in-place spin: the camera's fisheye ring already scans a full 360 degrees regardless of heading (sec 3.6), and camrun has no pose to safely translate toward.
                _slew_drive(0, 0)

        # has_ball: push the ball toward fwd_bearing.
        elif self.state == has_ball:
            _set_dwibbler(True)
            if _possession.has_ball:
                self.possession_seen = True
            if ball is not None and ball[1] > capture_dist_mm * 2:
                self.state = seek
                self.possession_seen = False
                _possession.reset()
                print("[main] lost ball -> seeking (camrun)", flush=True)
                return
            if ((self.possession_seen
                 or time.monotonic() - self.carry_start_t
                    >= possess_spinup_s + possess_on_s)
                    and _possession.available and not _possession.has_ball):
                self.state = seek
                self.possession_seen = False
                print("[main] dwibbler empty -> seeking (camrun)", flush=True)
                return

            # No pose -> no _wall_guard, same reasoning as StrikerController's own no-fix has_ball branch: wall_safe_speed_cmd, not base_speed, is the one translation command in this class with zero wall protection.
            _slew_drive(0, wall_safe_speed_cmd, rot_speed=fwd_bearing * turn_gain)
