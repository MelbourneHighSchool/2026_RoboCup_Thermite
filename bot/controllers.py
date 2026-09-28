"""Role controllers: GoalieController, StrikerController and CamRunController, the state
machines that turn ball, enemy and teammate state into drive commands each control tick.
"""

import math
import time

import bot.state as state
import bot.vision as vision
from bot.compass import _fused_heading
from bot.drive_config import base_speed, rush_speed
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
    BoxStandoff, JamRecovery, ShotCommitment, _add_ball_velocity, _approach_speed_cap, _ball_beaten,
    _brake_speed_frac, _carry_creep, _carry_should_press,
    _clamp_outside_own_box, _curve_in_approach, _defend_recovery_target,
    _drive_speed_bias,
    _heading_spin,
    _ball_hide_step, _enemy_guard, _enemy_holds_ball, _goal_shot_aim, _goalie_speed_frac, _lane_clear,
    _orbit_approach, _pass_race_open, _shield_heading_deg, _shot_release_scores,
    _sideline_route_point,
    _slew_drive, _teammate_guard, _turn_about_ball, _wall_guard, capture_cone_half_deg,
    capture_cone_half_width_mm, enemy_contest_dist_mm, flick_range_mm, goalie_max_frac,
    pass_eject_frac, pass_race_margin_s, pass_reach_mm,
    recovery_lateral_mm, recovery_standoff_mm,
    sideline_route_slow_mm,
    wall_safe_speed_cmd,
)
# read through bot.motion (motion.X), not from-imported, so a test can substitute the rule
# itself (sim-tests/test_goalie_box_rule.py does)
from bot.perception import Perception
from bot.tracking import (
    BallKalman, BallMemory, BallVelocityEstimator, KnownOcclusion,
    TeammateVelocityEstimator, ball_mem_min_conf,
)

# field calibration: the measured goal centres, if a calibration was saved
from bot.calibration import calib as _calib, goal_positions as _goal_positions


# Behaviour states (bot/network.py's master also reads these names).
seek = "seek"
has_ball = "has_ball"
# carrying the ball but electing to release it to the teammate instead of driving it in ourselves.
passing = "pass"
flick_shot = "flick_shot" # spin-release shot around a blocker
# shared non-blocking dwibbler-reverse countdown for pass releases and flick aborts (a
# sleep in the tick used to stall the whole 50 Hz loop)
ejecting = "ejecting"

# Heading control is motion._heading_spin (3 deg deadband, error/60 to saturation).

# the camera capture condition must hold this long, so one glitch frame can't flip seek ->
# has_ball
capture_confirm_s = 0.15

capture_ball_vel = True # add ball velocity to the capture command

# Kickoff hold (RCJA pre-match positions). maybe_start (bot/main.py) sets "kicking" or
# "receiving" at play start; it clears after kickoff_hold_s or once the ball moves toward
# us. During the hold each role drives to its kickoff spot and neither chases into illegal
# territory or engages an enemy. It runs after the ball-estimate block (so the estimate
# stays published) and before the jam check (a legally placed opponent at centre isn't a
# jam).
kickoff_hold_s = 4.0 # placement window after play starts
# kicking: striker stands off short of the centre ball (the real chase starts when the
# hold lifts)
kickoff_striker_standoff_mm = 120.0
# receiving: striker straddles the ball line this far short of the box's front edge
kickoff_striker_line_mm = 60.0 # clearance behind the box edge -> body straddles it
# kicking: goalie depth from the goal line, fully inside the box (centre + radius <=
# goal_depth needs <= 195)
kickoff_goalie_front_mm = 150.0
# receiving: goalie depth from the goal line, fully in the box behind the striker's line
kickoff_goalie_box_depth_mm = 150.0

# Passing is disabled: the stack is sound in simulation, but its failure modes (releases
# overshooting ~40% past the receiver, face-first teammate collisions, tap-ins refused by
# the race gate) were never tunable on hardware, and a mis-tuned pass is worse than none:
# it hands the ball over and the reversed roller drops our possession. pass_enabled
# False stops both the tap-in and the ordinary pass at their first gate, so the carrier
# plays the shot/carry/shield ladder. Every gate, constant and test is intact: set this
# and tap_in_enabled True to restore it.
pass_enabled = False
# teammate must be at least this much closer to the enemy goal before we pass (both robots
# attacking)
pass_teammate_gain_mm = 400.0
# Relief pass (the gate's second arm): with a keeper as the only teammate, "closer to the
# enemy goal" can never hold. This arm passes when our lane to goal is blocked while
# the teammate's is open, and the teammate is no more than pass_relief_slack_mm further
# from goal.
pass_relief_enabled = True
pass_relief_slack_mm = 600.0
# don't pass to a teammate this close, just dwibble together / drive in ourselves
pass_min_range_mm = 300.0
# a reversed-roller eject can't reach further than the pass model's ceiling
# (motion.pass_reach_mm)
pass_max_range_mm = pass_reach_mm
# corridor half-width for the enemy-in-lane check
pass_lane_block_mm = Perception.robot_radius_mm + 60.0
pass_min_own_dist_mm = 500.0 # don't pass if we're already this close to goal ourselves
# settle/advance this long after winning the ball before passing (swept in the simulator:
# flat from 1.6 to 4 s, worse below)
pass_settle_s = 1.6
# lead the target by this many seconds of the teammate's estimated velocity (a through pass).
pass_lead_s = 0.4
pass_align_deg = 15.0 # turn to within this of the target before firing
pass_confirm_s = 0.15 # hold the aim this long before releasing
pass_eject_s = 0.25 # how long to reverse the dwibbler for
# how long a fired pass_target keeps publishing to the teammate afterwards
pass_target_max_age_s = 1.5

# Deep-stuck bail-out: a striker still carrying deep in its own half under sustained
# contest passes to the teammate (usually the keeper) even though the teammate isn't
# closer to the enemy goal. It only widens the pass trigger (range, lane and race checks
# unchanged), never abandons the ball or flips roles itself. stuck_hold_s accumulates
# while deep and contested, drains otherwise, and stays armed for stuck_deep_arm_window_s
# after the enemy backs off (the race gate can't pass while they're still on top of us).
stuck_deep_enabled = True
stuck_deep_depth_mm = 1400.0 # within this of our own goal (own_into-relative) is "deep"
stuck_deep_contest_s = 2.5 # contested (dwibble_contest_dist_mm) this long arms the bail-out
stuck_deep_arm_window_s = 3.0 # keep trying to pass this long after the contest eases

# Tap-in: a direct feed to a teammate already close enough to finish, regardless of
# pass_settle_s, with its clear-lane check. Disabled with passing (its race gate
# refused most tap-in shapes); re-enable together with pass_enabled.
tap_in_enabled = False
# teammate must already be this close to goal (= flick_range_mm) to count as a tap-in
tap_in_range_mm = 900.0

# Scoring (no solenoid fitted): has_ball drives the ball at goal and cuts the roller once
# close; the body's push is the shot.
goal_eject_range_mm = 600.0

# Shot release gate. With no solenoid the ball leaves in the direction the body faces, so
# the question is "would a shot along our current facing go in?".
# motion._shot_release_scores answers it with the finishing predicate itself: the facing
# ray must reach the back wall between the posts, clear of the slot walls and every
# tracked enemy.
#
# It replaced a 15 deg tolerance around the planned aim, which refused releases that
# clearly score and allowed ones that don't. shot_align_enabled False removes the gate
# entirely; shot_release_ray_enabled False restores the old tolerance (the only reader of
# shot_align_tol_deg).
shot_align_enabled = True
shot_release_ray_enabled = True
shot_align_tol_deg = 15.0 # the old tolerance: read only when shot_release_ray_enabled is False

# Flick shot: a reversed-roller eject fired mid-spin instead of from a stop.
flick_enabled = True
# minimum continuous carry before a blocked lane may trigger the eject: without it a fresh
# contested capture reads "blocked" next tick and reverses the ball straight back to the
# blocker
flick_settle_s = 0.5
flick_kick_speed = dwibble_speed # max authority, more spin is better
flick_snap_s = 0.22 # total snap duration


# Duel escape: an enemy parked in front of the mouth is a face-off JamRecovery
# deliberately skips (the near-ball contest), so the ball sits between two bodies going
# nowhere. A scripted escape: freeze, strafe around the blocker's flank (on the side it
# sits off our bearing), then push through. Backs off if jam recovery fires mid-escape
# (the flank isn't open).
duel_escape_enabled = True
duel_enemy_range_mm = 130.0 # an enemy this close in front of the mouth counts as parked on us
duel_time_s = 0.5 # parked this long before the escape fires
duel_freeze_s = 0.1 # step 1: stop (dwibbler keeps gripping)
duel_strafe_s = 0.5 # step 2: strafe around the flank
# strafe bearing off our ball bearing; 90 is what has flown, 45 is worth an A/B
duel_strafe_off_deg = 90.0
# fraction of base_speed, leaving the guards room to work
duel_strafe_speed_frac = 0.8
duel_cooldown_s = 1.5 # before the escape can re-arm once ended

# Kalman ball tracking: the controllers steer on the filter's position and velocity, which
# also dead-reckons through short occlusions with decaying confidence before BallMemory
# takes over. False falls back to the least-squares velocity estimator; every tier below
# runs either way.
ball_kalman_enabled = True
# the Kalman estimate only replaces a direct source at or above this confidence (else the
# memory tier wins)
ball_kalman_takeover_conf = 0.25


# Goalie role

# standoff from own goal centre while shadowing a known ball (a radius, see
# GoalieController)
goalie_line_mm = 400.0
goalie_clear_dist_mm = 600.0 # ball closer than this -> charge and clear it
# Goalie box stand-off: when a clear inside our box has measurably become a shoulder-lock
# (no displacement for standoff_window_s while commanding a real clear, an enemy camped on
# the ball), the keeper stands down to its goal-side line instead of shoving the ball
# toward our net. motion.BoxStandoff owns detection and release; False restores
# always-charge.
goalie_box_standoff_enabled = True

# Roller pre-arm for the keeper: roller on whenever the ball is within 1000 mm. The keeper
# used to spin the roller only at contact inside its charge, and command it off everywhere
# else, so a ball arriving at a keeper holding its line met a dead roller and could bounce
# anywhere, including into our net. A live roller drags it into the mouth, toward the body
# and away from our goal.
#
# It only covers the known-ball steering region, not the kickoff hold or the jam escape.
# Speed is our graduated carry grip rather than a flat "on". dwibble_on_dist_mm (150)
# restores contact-range-only behaviour.
goalie_dwibble_prearm_mm = 1000.0
goalie_x_slack_mm = 150.0 # lateral travel clamped to goal-box width +/- this

# Quadratic radius shrink: with the ball near the goal-centre axis the shadow radius
# shrinks to goalie_line_mm * goalie_radius_min_factor for tighter cover, easing back to
# the full radius as it swings wide (more time to react near-post). At the minimum (260
# mm) the robot's radius still takes part of it past the box's 300 mm front edge (about 65
# mm), so the tightest shadow point stays legal (RCJA 5.8.2: part of the goalie must be
# able to leave the box).
goalie_radius_min_factor = 0.65

# Ball-velocity lead: shift the shadow's ball position along its travel direction, as a
# lead time, clamped so a velocity glitch can't fling the target off the arc
goalie_vel_lead_s = 0.15
goalie_vel_lead_max_mm = 200.0

# Sector clamp: the widest off-axis angle the shadow target may take. 55 deg covers about
# 330 mm either side of centre at the full radius, keeping the goalie central enough for a
# switch.
goalie_sector_max_rad = math.radians(55.0)

# Escort/support: the goalie's two-role version of the simulator's escort logic.
escort_enabled = True
# the ball must be at least this far from our goal along the attack axis before the goalie
# escorts (past midfield)
escort_min_depth_mm = 1600.0
# lateral offset from the teammate, toward whichever side of the pitch has more room
escort_lat_mm = 300.0
# how far past the teammate (further into the attack) the support spot leads
escort_lead_mm = 150.0

# back-wall geometry from own_goal's y: goal_backwall_mm (226) is the scoring wall,
# goal_line_mm (300) the mouth
goal_backwall_mm = 226.0
goal_line_mm = 300.0
goalie_backwall_tol_mm = 174.0


def _goalie_dwibble_radius_mm():
    """the keeper's roller-on radius: contact range widened to the pre-arm radius; read by both
    of its dribbler sites.
    """
    return max(dwibble_on_dist_mm, goalie_dwibble_prearm_mm)


class GoalieController:
    """defensive role, three regimes: shadow (ball known, teammate not deep) on the goal-ball
    line, escort, or probability cover.
    """
    def __init__(self):
        """fresh BallMemory, velocity estimator and Kalman tracker for a new goalie run."""
        self.ball_mem = BallMemory()
        self.ball_vel = BallVelocityEstimator()
        self.ball_kal = BallKalman()
        self.jam = JamRecovery() # general body-contact stuck detector, see its class docstring
        self.box_standoff = BoxStandoff() # box stand-off arbitration (motion.BoxStandoff)

    def tick(self):
        """run one control tick of the goalie behaviour, see the class docstring."""
        ball_mem = self.ball_mem
        ball_vel = self.ball_vel
        with state._lock:
            pose = state._state["pose"]
            _pose_imu = state._state["pose_imu"]
            _imu_now = state._state["imu_heading"]
            ball = state._state["ball"]
            enemies = state._state["enemies"]
            remote_ball = state._state["remote_ball"]
            run_mode = state._state["run_mode"]
            teammate_pos = state._state["teammate_pos_bt"]
            peer_state = state._state["peer_state"]
            # a keeper whose dribbler holds the ball reports it, so the role
            # swap hands it the striker role
            state._state["my_state"] = has_ball if _possession.has_ball else "goalie"

        # steer on the heading carried forward with the IMU, not one a revolution
        # old
        pose = _fused_heading(pose, _pose_imu, _imu_now)
        now_t = time.monotonic()

        # Directly known ball (our camera, else the teammate's), tracked even
        # while idle; BallMemory fusion only runs once in "run".
        ball_xy = None
        ball_src = "cam"
        ball_conf = 1.0
        if ball is not None and pose is not None:
            rx, ry, hdg = pose
            angle_deg, dist_mm = ball
            b_rad = math.radians(angle_deg + hdg)
            ball_xy = (rx + dist_mm * math.sin(b_rad),
                       ry + dist_mm * math.cos(b_rad))
            if run_mode == "run":
                ball_mem.seen(ball_xy[0], ball_xy[1], now_t)
                ball_vel.update(now_t, ball_xy[0], ball_xy[1])
                if ball_kalman_enabled:
                    self.ball_kal.update(now_t, ball_xy[0], ball_xy[1])
        elif remote_ball is not None:
            ball_xy = (remote_ball[0], remote_ball[1])
            # only a peer packet with a real sighting refreshes the memory
            if run_mode == "run" and remote_ball[3] == "cam":
                ball_mem.seen(ball_xy[0], ball_xy[1], now_t)
                # a teammate's sighting is in the same field frame: feed
                # the filter while our camera is blind, so the coast
                # covers the whole both-blind gap
                if ball_kalman_enabled:
                    self.ball_kal.update(now_t, ball_xy[0], ball_xy[1])
            ball_src = "remote"
        # Kalman tier: our camera is blind and no fresh teammate sighting came in,
        # so coast the filter's prediction with decaying confidence; below the
        # takeover confidence (or past the coast cap) the memory tier decides.
        if (ball_kalman_enabled and run_mode == "run" and ball_xy is None):
            _kal = self.ball_kal.est(now_t)
            if _kal is not None and _kal[4] >= ball_kalman_takeover_conf:
                ball_xy = (_kal[0], _kal[1])
                ball_src = "kal"
                ball_conf = _kal[4]
        with state._lock:
            state._state["ball_est"] = ((ball_xy[0], ball_xy[1], ball_conf,
                                         ball_src)
                                  if ball_xy is not None else None)

        if run_mode != "run":
            Motor.stopall()
            _set_dwibbler(False)
            return

        if pose is None:
            Motor.stopall()
            _set_dwibbler(False)
            return

        # kickoff hold: legal pre-match placement, ahead of the jam check
        _own_g, _ = _goal_positions(state._state["slot_goal"])
        _ko = _kickoff_spot("goalie", _own_g,
                            (ball_xy[0], ball_xy[1]) if ball_xy is not None
                            else None)
        if _ko is not None:
            rx_g, ry_g, hdg_g = pose
            tx_g, ty_g, face_g = _ko
            dx_g, dy_g = tx_g - rx_g, ty_g - ry_g
            dist_g = math.hypot(dx_g, dy_g)
            drive_rel = _wrap_deg(math.degrees(math.atan2(dx_g, dy_g)) - hdg_g)
            face = _wrap_deg(face_g - hdg_g)
            # the keeper's positioning uses the goalie braking law
            # (motion._goalie_speed_frac): overshooting the block line is the
            # one failure a keeper can't have
            spd = _goalie_speed_frac(dist_g)
            _set_dwibbler(False)
            drive_rel, spd = _wall_guard(rx_g, ry_g, hdg_g, drive_rel, spd)
            _slew_drive(drive_rel, spd, rot_speed=_heading_spin(face))
            return

        # Jam recovery, checked ahead of all normal steering. skip=True inside a
        # recognised near-ball contest (enemy_contest_dist_mm), which the contest
        # speed handles.
        rx_j, ry_j, hdg_j = pose
        if ball_xy is not None:
            bx_j, by_j = ball_xy
            dist_to_ball = math.hypot(bx_j - rx_j, by_j - ry_j)
            jam_skip = (dist_to_ball <= enemy_contest_dist_mm
                        and _enemy_holds_ball(bx_j, by_j, enemies))
        else:
            bx_j = rx_j + math.sin(math.radians(hdg_j))
            by_j = ry_j + math.cos(math.radians(hdg_j))
            jam_skip = False
        jam_cmd = self.jam.check(rx_j, ry_j, hdg_j, bx_j, by_j, enemies, skip=jam_skip)
        if jam_cmd is not None:
            drive_rel, spd = jam_cmd
            _set_dwibbler(False)
            # overrideAcc: the 0.5 s push-through can't spend ~0.33 s of it
            # ramping up under the slew cap
            _slew_drive(drive_rel, spd, rot_speed=_heading_spin(drive_rel),
                        overrideAcc=True)
            return

        # run_mode == "run" guarantees slot_goal is set (_apply_slot_state).
        with state._lock:
            goal = state._state["slot_goal"]
        own_goal, enemy_goal = _goal_positions(goal)
        gx, gy = own_goal
        into = 1.0 if gy < FieldModel.field_y / 2 else -1.0
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
                safe_y = gy + into * (goal_backwall_mm + goalie_backwall_tol_mm)
                chase_y = (max(ball_xy[1], safe_y) if into > 0
                          else min(ball_xy[1], safe_y))
            ball_robot_dist = math.hypot(ball_xy[0] - rx, ball_xy[1] - ry)
            # goal_side: are we still between our goal and the ball? If an
            # attacker has the ball behind us, charging its spot drives
            # through them from the wrong side and shoves the ball toward our
            # net, so fall through to shadow/cover, which gets us goal-side
            # first. A ball inside the box is always charged (subject to the
            # stand-off arbiter below).
            goal_side = (into * (ry - gy) <= ball_depth
                        or ball_depth < goal_line_mm)
            # RCJA 5.11: if the teammate is already in or touching our box,
            # don't also charge in
            teammate_in_box = motion._teammate_blocks_own_box(teammate_pos, own_goal)
            # Box stand-off (motion.BoxStandoff): a clear that has measurably
            # become a shoulder-lock stands down to the goal-side line, whose
            # _enemy_guard has no contest bypass and slides around the enemy.
            # This gates the whole charge path (roller, speed, guards).
            stand_off = (goalie_box_standoff_enabled and self.box_standoff.check(
                rx, ry, ball_xy[0], ball_xy[1], enemies, ball_depth < goal_line_mm))
            if (ball_robot_dist <= goalie_clear_dist_mm and goal_side
                    and not teammate_in_box and not stand_off):
                # same direct-pursuit approach as StrikerController
                dx_, dy_ = ball_xy[0] - rx, chase_y - ry
                chase_dist = math.hypot(dx_, dy_)
                rel = _wrap_deg(math.degrees(math.atan2(dx_, dy_)) - hdg)
                drive_angle, frac = _orbit_approach(
                    chase_dist * math.sin(math.radians(rel)),
                    chase_dist * math.cos(math.radians(rel)),
                    rx, ry, hdg)
                turn = _heading_spin(drive_angle)
                # goalie_clear_dist_mm decides whether to charge at all,
                # early so the keeper is already moving. The pre-arm radius
                # is never below that gate, so a charge always runs with the
                # roller live.
                if ball_robot_dist <= _goalie_dwibble_radius_mm():
                    _set_dwibbler(True, speed=_dwibble_carry_speed(
                        rx, ry, enemies, turn))
                else:
                    _set_dwibbler(False)
                # carrying the clear upfield keeps keep_min_mm off the
                # perimeter walls (goal boxes excluded)
                rush = _lane_clear(rx, ry, hdg, drive_angle, enemies)
                cmd = min((rush_speed if rush else base_speed) * frac,
                          _brake_speed_frac(chase_dist))
                drive_angle, cmd = _enemy_guard(rx, ry, hdg, drive_angle, cmd,
                                                enemies, ball_robot_dist)
                drive_angle, cmd = _teammate_guard(rx, ry, hdg, drive_angle, cmd, teammate_pos)
                drive_angle, cmd = _wall_guard(rx, ry, hdg, drive_angle, cmd)
                _slew_drive(drive_angle, cmd, rot_speed=turn)
                return

            # Escort/support: the teammate has the ball and is deep enough
            # that guarding an unthreatened net costs more than it's worth.
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
                # Shadow: stand goalie_line_mm from goal centre on the
                # goal-ball line (the simulator's
                # DefenderBot._shadow_pos), with two adjustments: the line
                # is built from where the ball is heading (velocity lead,
                # from the Kalman state when enabled), and the radius
                # shrinks quadratically as the ball nears the goal-centre
                # axis.
                _kal = self.ball_kal.est(now_t) if ball_kalman_enabled else None
                if _kal is not None:
                    vbx, vby = _kal[2], _kal[3]
                else:
                    vbx, vby = ball_vel.velocity()
                lead_x, lead_y = vbx * goalie_vel_lead_s, vby * goalie_vel_lead_s
                lead_mm = math.hypot(lead_x, lead_y)
                if lead_mm > goalie_vel_lead_max_mm:
                    scale = goalie_vel_lead_max_mm / lead_mm
                    lead_x *= scale
                    lead_y *= scale
                lead_bx, lead_by = ball_xy[0] + lead_x, ball_xy[1] + lead_y
                dxg, dyg = lead_bx - gx, lead_by - gy
                # angle between the goal->ball line and the
                # goal->field-centre axis: 0 dead ahead, pi/2 level with
                # the goal line
                axis_dot = dyg * into
                axis_cross = dxg * into
                angle_off_axis = math.atan2(abs(axis_cross), axis_dot)
                # Sector clamp: with the ball far out on a wing the shadow
                # would slide all the way to the post, over-committing
                # just as a square pass switches the attack. Capping the
                # angle at goalie_sector_max_rad keeps cover where shots
                # come from, and a switch costs a short slide instead of
                # an arc sweep. The radius stays the shrunk one.
                clamped_angle = min(angle_off_axis, goalie_sector_max_rad)
                side_sign = 1.0 if axis_cross >= 0.0 else -1.0
                axis_frac = max(0.0, min(1.0, clamped_angle / (0.5 * math.pi)))
                axis_frac = max(0.0, min(1.0, 4.0 * axis_frac ** 2))
                radius_factor = (goalie_radius_min_factor
                                 + (1.0 - goalie_radius_min_factor) * axis_frac)
                radius = goalie_line_mm * radius_factor
                target_x = max(clamp_lo, min(clamp_hi,
                               gx + side_sign * math.sin(clamped_angle) * radius))
                target_y = gy + into * math.cos(clamped_angle) * radius
                if teammate_in_box:
                    # RCJA 5.11, continued: the shrunk shadow point
                    # can land inside the box, fine for the sole
                    # defender but not as a hold-back point with the
                    # teammate already in there: push it clear
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
                    sx += p * bx
                    wsum += p
                target_x = max(clamp_lo, min(clamp_hi, sx / wsum if wsum else gx))
                face_x, face_y, _p = max(cands, key=lambda c: c[2])
            else:
                target_x = gx # nothing known: centre
                face_x, face_y = gx, FieldModel.field_y / 2

        # Not clearing: slide along the line to target_x. The roller is set
        # further down once this tick's facing turn is known.
        dx_, dy_ = target_x - rx, target_y - ry
        dist_t = math.hypot(dx_, dy_)
        drive_rel = _wrap_deg(math.degrees(math.atan2(dx_, dy_)) - hdg)
        # the keeper's cap, not base_speed: it repositions at up to 3000 mm/s
        speed = min(goalie_max_frac, _brake_speed_frac(dist_t))
        turn = _heading_spin(_wrap_deg(math.degrees(math.atan2(face_x - rx, face_y - ry)) - hdg))
        # Roller pre-arm: a known ball inside the keeper's radius keeps the roller
        # spinning while we hold the line. Unknown ball (probability cover) or one
        # outside the radius: off.
        if (ball_xy is not None
                and math.hypot(ball_xy[0] - rx, ball_xy[1] - ry)
                    <= _goalie_dwibble_radius_mm()):
            _set_dwibbler(True, speed=_dwibble_carry_speed(rx, ry, enemies, turn))
        else:
            _set_dwibbler(False)
        # goalie_line_mm (400) already keeps the shadow off its own end wall, so
        # this guards the side walls
        drive_rel, speed = _enemy_guard(rx, ry, hdg, drive_rel, speed, enemies)
        # no teammate veto on purpose: the escort target is computed from
        # teammate_pos, so vetoing on that same report would aim the keeper at a
        # ghost (a stale link gives None anyway)
        drive_rel, speed = _teammate_guard(rx, ry, hdg, drive_rel, speed, None)
        drive_rel, speed = _wall_guard(rx, ry, hdg, drive_rel, speed)
        _slew_drive(drive_rel, speed, rot_speed=turn)


# shared tick period: the play loop sleeps this once per tick
loop_dt = 0.02

# Support position: the striker's mirror of the goalie's escort. When the teammate has the
# ball and we have no closer chase of our own, go to an open spot for a give-and-go
# instead of a redundant chase. It must stay off the carrier and out of our own box, and
# is sized to sit inside the pass range window so it's a plausible pass target.
support_enabled = True
# a directly reachable ball this close overrides support
support_ball_near_mm = 600.0
# lateral offset from the teammate toward the roomier side; with support_lead_mm,
# hypot(lat, lead) sits well inside the pass range
support_lat_mm = 550.0
# how far past the teammate toward the enemy goal the support spot leads
support_lead_mm = 400.0
# keep the support spot clear of our own box: goal_line_mm (300) plus margin
support_min_own_depth_mm = goal_line_mm + 400.0


# Seek arbitration: each seek option scores itself on one scale and the best runs, so
# option strength is visible (_state["seek_util"], the debug page) and tunable in one
# place. The roller-stall capture stays an unconditional override above it. Support scores
# 0.5 while the teammate is releasing the ball to us (beating a chase past ~624 mm) and
# 0.35 while it's still carrying (so a chase inside ~945 mm wins: two bots parking in
# formation is worse than keep-away). There is no switch back to the old fixed ladder; it no longer exists.
#
# The chase score decays with distance; this is its 1/e length (at 900 mm a chase is worth
# ~0.37).
seek_util_chase_mm = 900.0
# support while the teammate is releasing the ball (passing): the receiving slot is where
# we want to be
seek_util_support = 0.5
# demoted support while the teammate is still carrying it (has_ball)
seek_util_support_carry = 0.35

def _seek_util_chase(dist_mm):
    """chase option's score (see seek_util_chase_mm above)."""
    return math.exp(-dist_mm / seek_util_chase_mm)

def _seek_arbiter(support_ok, peer_state, ball_robot_dist, util_log):
    """pick this tick's seek behaviour (see the block comment above)."""
    # no usable ball estimate: nothing to chase, score 0 and let support take the tick
    opts = [("chase", (_seek_util_chase(ball_robot_dist)
                      if ball_robot_dist is not None else 0.0))]
    if support_ok:
        score = (seek_util_support if peer_state == passing
                 else seek_util_support_carry)
        opts.append(("support", score))
    opts.sort(key=lambda o: o[1], reverse=True)
    util_log.extend(opts)
    return opts[0][0]


def _kickoff_spot(role, own_goal, ball_est):
    """this role's kickoff target (field mm) and face bearing, or None if the hold isn't active."""
    with state._lock:
        kind = state._state["kickoff_role"]
        until = state._state["kickoff_until_t"]
    if kind is None or until is None or time.monotonic() >= until:
        if kind is not None:
            # expired: clear it once so the debug page stops showing a hold
            with state._lock:
                state._state["kickoff_role"] = None
                state._state["kickoff_until_t"] = None
        return None

    gx, gy = own_goal
    into = 1.0 if gy < FieldModel.field_y / 2 else -1.0 # own goal -> field direction

    # the ball starts at centre on kick-off; use that while the estimate is blind
    bx, by = (ball_est[0], ball_est[1]) if ball_est is not None \
        else (FieldModel.cx, FieldModel.field_y / 2)

    if kind == "kicking" and role == "striker":
        # standoff short of the centre ball, on the own-goal -> ball line
        dx, dy = bx - gx, by - gy
        d = math.hypot(dx, dy) or 1.0
        tx = bx - dx / d * kickoff_striker_standoff_mm
        ty = by - dy / d * kickoff_striker_standoff_mm
    elif kind == "receiving" and role == "striker":
        # just short of the box's front edge on the goal->ball line, the body
        # straddling the line
        tx = gx + (bx - gx) * 0.0 # straight out from goal centre
        tx = gx
        ty = gy + into * (FieldModel.goal_depth - Perception.robot_radius_mm
                          + kickoff_striker_line_mm)
    elif kind == "kicking" and role == "goalie":
        tx = gx
        ty = gy + into * kickoff_goalie_front_mm
    else: # receiving goalie: fully in the box, behind the striker's line
        tx = gx
        ty = gy + into * kickoff_goalie_box_depth_mm

    # early relief for the kicking striker: the ball moving toward our half (or off
    # centre) means the opponent struck it
    if kind == "kicking" and role == "striker" and ball_est is not None:
        moved = math.hypot(bx - FieldModel.cx, by - FieldModel.field_y / 2)
        toward_us = into * (by - FieldModel.field_y / 2) < -50.0
        if moved > 150.0 or toward_us:
            with state._lock:
                state._state["kickoff_role"] = None
                state._state["kickoff_until_t"] = None
            return None

    face = (math.degrees(math.atan2(bx - tx, by - ty))
            if ball_est is not None else 0.0)
    return tx, ty, face


class StrikerController:
    """attacking role, a 3-state machine (seek / has_ball / passing) at about 50 Hz."""
    def __init__(self):
        """fresh FSM state (starts in seek) and fresh velocity/memory helpers for a new striker
        run.
        """
        self.state = seek
        self.ball_vel = BallVelocityEstimator()
        self.ball_kal = BallKalman()
        self.ball_mem = BallMemory()
        self.teammate_vel = TeammateVelocityEstimator()
        self.last_ball_fix = None
        self.last_teammate_pos = None
        self.capture_hold_s = 0.0 # camera capture-condition hysteresis accumulator
        self.facing_away_ticks = 0 # consecutive seek ticks with the ball well off our heading
        self.possession_seen = False # dwibbler detector confirmed the current carry
        self.carry_start_t = 0.0 # monotonic time this carry began (pass settle)
        # monotonic time the camera last had the ball inside dwibble_on_dist_mm
        self.close_ball_t = 0.0
        self.pass_target = None # (x, y) chosen when has_ball -> passing
        self.pass_hold_s = 0.0 # passing alignment hysteresis accumulator
        self.pass_expiry_t = 0.0 # monotonic time pass_target stops publishing
        # last has_ball tick was inside goal_eject_range_mm (roller deliberately
        # cut)
        self.near_goal_eject = False
        # accumulate/decay hold for the deep-stuck bail-out
        self.stuck_hold_s = 0.0
        # (x, y) where this carry began: the anchor for the shield creep's retreat
        # bound
        self.carry_anchor = None
        # sideline route waypoint armed at capture time, None when not armed (or
        # retired)
        self.sideline_route = None
        # non-blocking eject countdown (state == ejecting): deadline and kind
        # (pass or flick)
        self.eject_until_t = 0.0
        self.eject_is_pass = True
        # command magnitude of a pass release in flight, set at the release from
        # the distance to cross. The default (half of pass_reach_mm) is a
        # fail-safe, so an eject entered without a release still reverses the
        # roller
        self.eject_speed = pass_eject_frac(pass_reach_mm / 2.0)
        self.jam = JamRecovery() # general body-contact stuck detector, see its class docstring
        # finishing-lane commitment across ticks (motion.ShotCommitment), reset at
        # each new carry
        self.shot = ShotCommitment()
        # ball-hiding state across ticks (motion._ball_hide_step), reset at each
        # new carry
        self.hiding = False
        self.hide_tucked = False
        # duel escape: armed while an enemy sits parked in front of the mouth;
        # None = idle
        self.duel_t = 0.0 # parked-enemy hysteresis accumulator
        self.duel_phase = None # None / "freeze" / "strafe"
        self.duel_phase_t = 0.0 # monotonic time the current phase started
        self.duel_side = 1.0 # strafe side: +1 right, -1 left
        self.duel_end_t = 0.0 # monotonic time the last escape ended (cooldown)

    def tick(self):
        """run one control tick of the striker behaviour, see the class docstring."""
        ball_vel, ball_mem = self.ball_vel, self.ball_mem
        teammate_vel = self.teammate_vel

        # pass_target keeps publishing for pass_target_max_age_s after a release
        # so the teammate can act on it
        if (self.state != passing and self.pass_target is not None
                and time.monotonic() >= self.pass_expiry_t):
            self.pass_target = None

        with state._lock:
            pose = state._state["pose"]
            _pose_imu = state._state["pose_imu"]
            _imu_now = state._state["imu_heading"]
            ball = state._state["ball"]
            enemies = state._state["enemies"]
            enemy_vel_est = state._state["enemy_vel"]
            remote_ball = state._state["remote_ball"]
            run_mode = state._state["run_mode"]
            teammate_pos = state._state["teammate_pos_bt"]
            peer_pass_target = state._state["peer_pass_target"]
            peer_state = state._state["peer_state"]
            state._state["my_state"] = self.state
            state._state["pass_target"] = self.pass_target
            # reset every tick, so a stale zone can't keep reading out for
            # --motionlog
            state._state["capture_zone"] = None

        # steer on the heading carried forward with the IMU, not one a revolution
        # old
        pose = _fused_heading(pose, _pose_imu, _imu_now)

        if teammate_pos is not None and teammate_pos != self.last_teammate_pos:
            teammate_vel.update(time.monotonic(), teammate_pos[0], teammate_pos[1])
            self.last_teammate_pos = teammate_pos
        elif teammate_pos is None:
            teammate_vel.reset()
            self.last_teammate_pos = None

        # fused ball estimate (field frame), shared over the team link and used by
        # seek when our own camera is blind.
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
                    # feed the filter with the teammate's sighting
                    # while our camera is blind
                    if ball_kalman_enabled:
                        self.ball_kal.update(time.monotonic(),
                                             remote_ball[0], remote_ball[1])
            elif run_mode == "run":
                # Kalman tier: camera blind and no remote sighting, so
                # coast the filter's prediction with decaying confidence
                # before BallMemory's attribution (the long-term story)
                # takes over.
                _kal = self.ball_kal.est(time.monotonic()) \
                    if ball_kalman_enabled else None
                if _kal is not None and _kal[4] >= ball_kalman_takeover_conf:
                    ball_est = (_kal[0], _kal[1], _kal[4], "kal")
                else:
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
            # Idle abandons an in-progress pass rather than leave it
            # broadcasting a target that's never going to fire.
            if self.state == passing:
                self.state = seek
            self.pass_target = None
            with state._lock:
                state._state["pass_target"] = None
            return

        # Kickoff hold: legal placement before the first chase; after the estimate
        # block (the spot helper reads ball_est) and before the jam check.
        _ko = None
        if pose is not None:
            _rx, _ry, _rh = pose
            _own_g, _ = _goal_positions(state._state["slot_goal"])
            _ko = _kickoff_spot("striker", _own_g, ball_est)
        if _ko is not None:
            tx_k, ty_k, face_k = _ko
            dx_k, dy_k = tx_k - _rx, ty_k - _ry
            dist_k = math.hypot(dx_k, dy_k)
            drive_rel = _wrap_deg(math.degrees(math.atan2(dx_k, dy_k)) - _rh)
            face = _wrap_deg(face_k - _rh)
            spd = min(base_speed, _brake_speed_frac(dist_k))
            _set_dwibbler(False)
            drive_rel, spd = _enemy_guard(_rx, _ry, _rh, drive_rel, spd, enemies)
            drive_rel, spd = _teammate_guard(_rx, _ry, _rh, drive_rel, spd, teammate_pos)
            drive_rel, spd = _wall_guard(_rx, _ry, _rh, drive_rel, spd)
            _slew_drive(drive_rel, spd, rot_speed=_heading_spin(face))
            return

        # Duel escape, after the kickoff hold and before the jam check
        # (JamRecovery skips the near-ball contest, so an enemy parked on the
        # mouth would be a permanent face-off). Seek only; in has_ball the
        # shield/creep/flick ladder handles enemies. It drives through the guards,
        # so a flank that isn't open just bends the strafe until the timer
        # expires.
        if duel_escape_enabled and pose is not None and self.state == seek:
            rx_d, ry_d, hdg_d = pose
            now_d = time.monotonic()
            if self.duel_phase is not None and ball_est is None:
                # ball lost mid-script: not a face-off any more, stand down
                self.duel_phase = None
                self.duel_t = 0.0
                self.duel_end_t = now_d
            elif self.duel_phase is None:
                # arm: an enemy within duel_enemy_range_mm of a point that
                # far ahead of us, with the ball close enough to make it a
                # contest
                fx_d = rx_d + math.sin(math.radians(hdg_d)) * duel_enemy_range_mm
                fy_d = ry_d + math.cos(math.radians(hdg_d)) * duel_enemy_range_mm
                parker = None
                for e in enemies:
                    if math.hypot(e["x"] - fx_d, e["y"] - fy_d) <= duel_enemy_range_mm:
                        parker = e
                        break
                # no ball read at all: nothing to contest, so no duel
                # evidence this tick (stand down rather than read a
                # missing estimate)
                ball_duel = (math.hypot(ball_est[0] - rx_d, ball_est[1] - ry_d)
                             if ball_est is not None else None)
                if (parker is not None and ball_duel is not None
                        and ball_duel <= enemy_contest_dist_mm
                        and now_d - self.duel_end_t >= duel_cooldown_s):
                    self.duel_t += loop_dt
                    if self.duel_t >= duel_time_s:
                        # strafe around whichever flank the enemy isn't on
                        rel_e = _wrap_deg(math.degrees(math.atan2(
                            parker["x"] - rx_d, parker["y"] - ry_d)) - hdg_d)
                        self.duel_side = 1.0 if rel_e > 0 else -1.0
                        self.duel_phase = "freeze"
                        self.duel_phase_t = now_d
                        print("[main] duel escape: freeze", flush=True)
                else:
                    self.duel_t = max(0.0, self.duel_t - 2.0 * loop_dt)
            elif self.duel_phase == "freeze":
                # step 1: stop dead, roller gripping; the parked enemy
                # often overruns the ball
                if now_d - self.duel_phase_t >= duel_freeze_s:
                    self.duel_phase = "strafe"
                    self.duel_phase_t = now_d
                    print("[main] duel escape: strafe", flush=True)
                else:
                    _set_dwibbler(True)
                    _slew_drive(0.0, 0.0, rot_speed=0.0)
                    return
            else: # "strafe"
                if now_d - self.duel_phase_t >= duel_strafe_s:
                    self.duel_phase = None
                    self.duel_t = 0.0
                    self.duel_end_t = now_d
                else:
                    # step 2: sideways around the flank, still facing the ball
                    strafe_rel = -self.duel_side * duel_strafe_off_deg
                    face_d = _wrap_deg(math.degrees(math.atan2(
                        ball_est[0] - rx_d, ball_est[1] - ry_d)) - hdg_d)
                    _set_dwibbler(True)
                    strafe_rel, spd_d = _enemy_guard(
                        rx_d, ry_d, hdg_d, strafe_rel,
                        duel_strafe_speed_frac * base_speed, enemies)
                    strafe_rel, spd_d = _teammate_guard(rx_d, ry_d, hdg_d, strafe_rel, spd_d, teammate_pos)
                    strafe_rel, spd_d = _wall_guard(rx_d, ry_d, hdg_d,
                                                    strafe_rel, spd_d)
                    _slew_drive(strafe_rel, spd_d, rot_speed=_heading_spin(face_d))
                    return

        # Jam recovery, checked ahead of all normal steering. skip=True inside a
        # recognised near-ball contest, which _approach_speed_cap's contest speed
        # handles.
        if pose is not None:
            rx_j, ry_j, hdg_j = pose
            if ball_est is not None:
                bx_j, by_j = ball_est[0], ball_est[1]
                dist_to_ball = math.hypot(bx_j - rx_j, by_j - ry_j)
                jam_skip = (dist_to_ball <= enemy_contest_dist_mm
                            and _enemy_holds_ball(bx_j, by_j, enemies))
            else:
                bx_j = rx_j + math.sin(math.radians(hdg_j))
                by_j = ry_j + math.cos(math.radians(hdg_j))
                jam_skip = False
            jam_cmd = self.jam.check(rx_j, ry_j, hdg_j, bx_j, by_j, enemies, skip=jam_skip)
            if jam_cmd is not None:
                drive_rel, spd = jam_cmd
                _set_dwibbler(False)
                # overrideAcc, as the goalie's jam push: the 0.5 s dwell
                # can't spend ~0.33 s ramping up
                _slew_drive(drive_rel, spd, rot_speed=_heading_spin(drive_rel),
                            overrideAcc=True)
                return

        # run_mode == "run" guarantees slot_goal is set (_apply_slot_state).
        with state._lock:
            goal = state._state["slot_goal"]
        own_goal, enemy_goal = _goal_positions(goal)

        # seek: direct-pursuit capture
        if self.state == seek:
            # roller loaded -> we have it, whatever the camera says (it can't
            # see a ball flush in the mouth)
            if (_possession.has_ball
                    and time.monotonic() - self.close_ball_t
                        <= possess_capture_window_s):
                _set_dwibbler(True)
                self.state = has_ball
                self.possession_seen = True
                self.capture_hold_s = 0.0
                self.facing_away_ticks = 0
                self.carry_start_t = time.monotonic()
                self.stuck_hold_s = 0.0
                self.shot.reset()
                self.hiding = False
                self.hide_tucked = False
                self.carry_anchor = (pose[0], pose[1]) if pose is not None else None
                print("[main] ball captured -> attacking (dwibbler)",
                      flush=True)
                if state._capture_log is not None:
                    state._capture_log.event("captured", "dwibbler-stall route")
                return

            # Seek arbitration (_seek_arbiter): every option scores itself and
            # the best runs this tick. close_ball reads the fused estimate, not
            # just the camera: with the camera blind and the estimate at our
            # feet, support would otherwise park us beside the ball.
            ball_robot_dist = (math.hypot(ball_est[0] - pose[0],
                                          ball_est[1] - pose[1])
                               if (ball_est is not None and pose is not None)
                               else None)
            close_ball = ((ball is not None and ball[1] <= support_ball_near_mm)
                           or (ball_robot_dist is not None
                               and ball_robot_dist <= support_ball_near_mm))
            support_ok = (support_enabled and not close_ball
                           and teammate_pos is not None
                           and peer_state in (has_ball, passing)
                           and pose is not None)
            util_opts = []
            util_choice = _seek_arbiter(support_ok, peer_state, ball_robot_dist,
                                        util_opts)
            with state._lock:
                state._state["seek_util"] = (util_choice, tuple(util_opts))
            # support: run to the give-and-go slot when the arbiter picks it
            if util_choice == "support":
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
                dx_, dy_ = target_x - rx, target_y - ry
                dist_t = math.hypot(dx_, dy_)
                drive_rel = _wrap_deg(math.degrees(math.atan2(dx_, dy_))
                                      - hdg)
                face = _wrap_deg(math.degrees(math.atan2(face_x - rx,
                                                         face_y - ry)) - hdg)
                _set_dwibbler(False)
                ball_vel.reset()
                self.last_ball_fix = None
                spd = min(base_speed, _brake_speed_frac(dist_t))
                drive_rel, spd = _enemy_guard(rx, ry, hdg, drive_rel, spd, enemies)
                drive_rel, spd = _teammate_guard(rx, ry, hdg, drive_rel, spd, teammate_pos)
                drive_rel, spd = _wall_guard(rx, ry, hdg, drive_rel, spd)
                # _slew_drive, not Motor.drive: an uncapped escort command
                # would jump to base_speed in one tick from a stop
                _slew_drive(drive_rel, spd, rot_speed=_heading_spin(face))
                return

            if ball is not None:
                angle_deg, dist_mm = ball
                if pose is not None:
                    # field frame (origin bottom-left): the ball's
                    # field position for velocity/memory tracking
                    rx, ry, hdg = pose
                    b_rad = math.radians(angle_deg + hdg)
                    bx = rx + dist_mm * math.sin(b_rad)
                    by = ry + dist_mm * math.cos(b_rad)
                    if ball is not self.last_ball_fix:
                        now_t = time.monotonic()
                        ball_vel.update(now_t, bx, by)
                        ball_mem.seen(bx, by, now_t)
                        if ball_kalman_enabled:
                            self.ball_kal.update(now_t, bx, by)
                        self.last_ball_fix = ball
                    rel_rad = math.radians(angle_deg)
                    lat_mm = dist_mm * math.sin(rel_rad)
                    # live velocity: the Kalman state while enabled,
                    # else the least-squares estimator
                    if ball_kalman_enabled:
                        _kal = self.ball_kal.est(time.monotonic())
                        vbx, vby = ((_kal[2], _kal[3]) if _kal is not None
                                    else ball_vel.velocity())
                    else:
                        vbx, vby = ball_vel.velocity()
                    # computed every tick so --capturelog sees the
                    # velocity even with the lead switched off
                    own_into = 1.0 if own_goal[1] < FieldModel.field_y / 2 else -1.0
                    ball_depth_now = own_into * (by - own_goal[1])
                    if _ball_beaten(ry, by, own_goal) and ball_depth_now >= goal_line_mm:
                        # beaten: the ball is deeper in our half
                        # than we are; a direct approach reaches
                        # only the attacker's back, so get
                        # goal-side first
                        tx_, ty_ = _defend_recovery_target(
                            rx, ry, bx, by, own_goal,
                            recovery_standoff_mm, recovery_lateral_mm)
                        if motion._teammate_blocks_own_box(teammate_pos, own_goal):
                            # RCJA 5.11: the teammate is in
                            # our box, so hold goal-side of
                            # the ball without entering
                            tx_, ty_ = _clamp_outside_own_box(
                                tx_, ty_, own_goal, Perception.robot_radius_mm)
                        dxr, dyr = tx_ - rx, ty_ - ry
                        drive_angle = _wrap_deg(
                            math.degrees(math.atan2(dxr, dyr)) - hdg)
                        turn = _heading_spin(drive_angle)
                        in_cone = False
                        rush = False
                        drive_cmd = min(base_speed,
                                        _brake_speed_frac(math.hypot(dxr, dyr)))
                    else:
                        drive_angle, speed_frac = _curve_in_approach(
                            dist_mm * math.sin(rel_rad),
                            dist_mm * math.cos(rel_rad),
                            rx, ry, hdg, goal=enemy_goal)
                        turn = _heading_spin(drive_angle)
                        in_cone = (abs(angle_deg) <= capture_cone_half_deg
                                  and abs(lat_mm) <= capture_cone_half_width_mm)
                        # clear lane ahead -> rush_speed for the
                        # open chase, but never once centred on
                        # the final approach
                        rush = not in_cone and _lane_clear(
                            rx, ry, hdg, drive_angle, enemies)
                        drive_cmd = (rush_speed if rush else base_speed) * speed_frac
                        if capture_ball_vel:
                            # Method 6: x_o = x_b + v_o
                            drive_angle, drive_cmd = _add_ball_velocity(
                                drive_angle, drive_cmd, hdg, vbx, vby)
                    # a flat approach speed can arrive too fast to
                    # stop at the ball; _approach_speed_cap swaps in
                    # the contest speed when an enemy is holding the
                    # ball this close
                    drive_cmd = _approach_speed_cap(
                        dist_mm, bx, by, enemies, min(drive_cmd, _brake_speed_frac(dist_mm)))
                else:
                    # No lidar fix yet: no field frame, so no y=0
                    # tie-break and no ball-velocity compensation
                    # either.
                    ball_vel.reset()
                    self.last_ball_fix = None
                    b_rad = math.radians(angle_deg)
                    lat_mm = dist_mm * math.sin(b_rad)
                    drive_angle, speed_frac = _orbit_approach(
                        dist_mm * math.sin(b_rad),
                        dist_mm * math.cos(b_rad))
                    turn = _heading_spin(drive_angle)
                    in_cone = (abs(angle_deg) <= capture_cone_half_deg
                              and abs(lat_mm) <= capture_cone_half_width_mm)
                    drive_cmd = min(base_speed * speed_frac,
                                    _brake_speed_frac(dist_mm))
                    rush = False # no pose -> no field-frame lane to check
                # Approach speed shaping: ease off the last stretch when
                # the drive is off-axis or the ball is close. A cap only,
                # faded out by speed_bias_far_mm.
                drive_cmd *= _drive_speed_bias(drive_angle, dist_mm)
                # Facing-away recovery: the ball has sat well off our
                # heading for a long run of ticks, so come at it from the
                # far side of the field. Needs a pose, so the no-fix path
                # skips it.
                if motion._facing_away(angle_deg):
                    self.facing_away_ticks += 1
                else:
                    self.facing_away_ticks = 0
                if (motion.facing_away_recovery_enabled and rx is not None
                        and self.facing_away_ticks >= motion.facing_away_recovery_ticks):
                    drive_angle = motion._facing_away_mirror_angle(drive_angle, rx)
                    drive_cmd *= motion.facing_away_recovery_scale
                    turn = _heading_spin(drive_angle)
                with state._lock:
                    state._state["capture_zone"] = "cone" if in_cone else None
                # Draw the ball in once it's centred / close.
                _set_dwibbler(in_cone or dist_mm < dwibble_on_dist_mm)
                if dist_mm < dwibble_on_dist_mm:
                    # arms the roller-only capture for
                    # possess_capture_window_s: the last moment the
                    # camera sees a ball about to disappear into the
                    # mouth
                    self.close_ball_t = time.monotonic()
                if pose is not None:
                    # keep_min_mm applies while chasing (we don't dig
                    # a ball flush against a wall)
                    drive_angle, drive_cmd = _enemy_guard(
                        rx, ry, hdg, drive_angle, drive_cmd, enemies, dist_mm)
                    drive_angle, drive_cmd = _teammate_guard(rx, ry, hdg, drive_angle, drive_cmd, teammate_pos)
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
                    self.capture_hold_s = 0.0
                    self.facing_away_ticks = 0
                    self.carry_start_t = time.monotonic()
                    self.stuck_hold_s = 0.0
                    self.shot.reset()
                    self.hiding = False
                    self.hide_tucked = False
                    self.carry_anchor = (rx, ry)
                    # sideline route: the waypoint is computed once,
                    # at capture, from where the carry started
                    self.sideline_route = _sideline_route_point(
                        rx, ry, enemy_goal)
                    print("[main] ball captured -> attacking", flush=True)
                    if state._capture_log is not None:
                        state._capture_log.event("captured", "camera route")
            else:
                ball_vel.reset()
                self.last_ball_fix = None
                # Camera lost the ball: chase the fused estimate (teammate
                # sighting or memory) before blind-spinning.
                _set_dwibbler(False)
                # drain the facing-away count while the camera is blind (1
                # per tick, mirroring the accumulate), so its evidence
                # goes stale instead of freezing
                self.facing_away_ticks = max(0, self.facing_away_ticks - 1)
                if (ball_est is not None
                        and ball_est[2] >= ball_mem_min_conf):
                    mbx, mby = ball_est[0], ball_est[1]
                    rx, ry, hdg = pose
                    own_into = 1.0 if own_goal[1] < FieldModel.field_y / 2 else -1.0
                    ball_depth_now = own_into * (mby - own_goal[1])
                    with state._lock:
                        state._state["capture_zone"] = None
                    if _ball_beaten(ry, mby, own_goal) and ball_depth_now >= goal_line_mm:
                        # beaten, as above: get goal-side of the
                        # memory estimate before chasing it
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
                        turn = _heading_spin(drive_angle)
                        cmd = min(base_speed, _brake_speed_frac(dist_r))
                        drive_angle, cmd = _enemy_guard(
                            rx, ry, hdg, drive_angle, cmd, enemies)
                        drive_angle, cmd = _teammate_guard(rx, ry, hdg, drive_angle, cmd, teammate_pos)
                    else:
                        dx_, dy_ = mbx - rx, mby - ry
                        dist_est = math.hypot(dx_, dy_)
                        rel = _wrap_deg(math.degrees(math.atan2(dx_, dy_)) - hdg)
                        drive_angle, speed_frac = _curve_in_approach(
                            dist_est * math.sin(math.radians(rel)),
                            dist_est * math.cos(math.radians(rel)),
                            rx, ry, hdg, goal=enemy_goal)
                        turn = _heading_spin(drive_angle)
                        drive_angle, cmd = _enemy_guard(
                            rx, ry, hdg, drive_angle,
                            _approach_speed_cap(
                                dist_est, mbx, mby, enemies,
                                min(base_speed * speed_frac, _brake_speed_frac(dist_est))),
                            enemies, dist_est)
                        drive_angle, cmd = _teammate_guard(rx, ry, hdg, drive_angle, cmd, teammate_pos)
                    drive_angle, cmd = _wall_guard(
                        rx, ry, hdg, drive_angle, cmd)
                    _slew_drive(drive_angle, cmd, rot_speed=turn)
                else:
                    with state._lock:
                        state._state["capture_zone"] = "search"
                    # completely lost: fall back toward one radius
                    # below field centre (own side). No spin: the
                    # fisheye ring already sees 360 degrees
                    if pose is not None:
                        rx, ry, hdg = pose
                        fall_sign = (1.0 if own_goal[1] > FieldModel.field_y / 2
                                     else -1.0)
                        fx = FieldModel.cx
                        fy = (FieldModel.field_y / 2
                              + fall_sign * Perception.robot_radius_mm)
                        dx_, dy_ = fx - rx, fy - ry
                        dist_f = math.hypot(dx_, dy_)
                        drive_rel = _wrap_deg(math.degrees(math.atan2(dx_, dy_))
                                              - hdg)
                        spd = min(base_speed, _brake_speed_frac(dist_f))
                        drive_rel, spd = _enemy_guard(rx, ry, hdg, drive_rel, spd, enemies)
                        drive_rel, spd = _teammate_guard(rx, ry, hdg, drive_rel, spd, teammate_pos)
                        drive_rel, spd = _wall_guard(rx, ry, hdg, drive_rel, spd)
                        _slew_drive(drive_rel, spd)
                    else:
                        # No pose yet -> no fallback point to drive to, hold still.
                        _slew_drive(0, 0)

        # has_ball: carry the ball into the enemy goal
        elif self.state == has_ball:
            # full grip by default; below, once the pose is known, graduated
            # or reversed to eject
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
            # Roller loss check: a free-spinning roller means the mouth is
            # empty. Also bail if possession was never confirmed within a
            # grace period (the camera-only capture can enter has_ball without
            # holding anything).
            grace_elapsed = (time.monotonic() - self.carry_start_t
                            >= possess_spinup_s + possess_on_s)
            if ((self.possession_seen or grace_elapsed)
                    and _possession.available
                    and not _possession.has_ball
                    and not self.near_goal_eject):
                never_confirmed = not self.possession_seen
                self.state = seek
                self.possession_seen = False
                self.sideline_route = None # carry lost, the flank run died with it
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

                # tap-in: runs ahead of, and separate from, the
                # settle-gated pass below
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
                        self.state = passing
                        self.pass_target = tap_target
                        self.pass_hold_s = 0.0
                        # handing the ball off, the route is not
                        # ours to finish
                        self.sideline_route = None
                        print(f"[main] tap-in -> passing to "
                             f"{tap_target[0]:.0f},{tap_target[1]:.0f}",
                             flush=True)
                        return

                # pass instead of driving in, if the teammate is
                # meaningfully closer to goal and in eject range
                settled = (time.monotonic() - self.carry_start_t
                           >= pass_settle_s)

                # Deep-stuck bail-out: widen the pass trigger so a striker
                # contested deep in its own half hands off to the teammate
                # even though they aren't closer to the enemy goal.
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
                    tx, ty = teammate_pos
                    d_tm = math.hypot(gx - tx, gy - ty)
                    d_tm_us = math.hypot(tx - rx, ty - ry)
                    blockers = [(e["x"], e["y"], pass_lane_block_mm)
                               for e in enemies]
                    # teammate suitability, two arms: more advanced
                    # than us, or the relief case (our lane is shut,
                    # theirs isn't, and they're close enough that
                    # recycling to them doesn't just give ground)
                    advanced = (d_tm + pass_teammate_gain_mm < d_us)
                    relief = (pass_relief_enabled
                                and d_tm - d_us <= pass_relief_slack_mm
                                and not KnownOcclusion((rx, ry), blockers
                                                       ).observed(gx, gy)
                                and KnownOcclusion((tx, ty), blockers
                                                   ).observed(gx, gy))
                    if ((advanced or relief or stuck_deep)
                            and pass_min_range_mm <= d_tm_us <= pass_max_range_mm):
                        # lead the target by the teammate's
                        # velocity: a through pass if they're
                        # moving
                        tvx, tvy = teammate_vel.velocity()
                        target = (min(max(tx + tvx * pass_lead_s, 0.0),
                                     FieldModel.field_x),
                                 min(max(ty + tvy * pass_lead_s, 0.0),
                                     FieldModel.field_y))
                        # don't release into a covered lane: an
                        # enemy between us and the target means
                        # the eject just hands it over
                        if (KnownOcclusion((rx, ry), blockers).observed(*target)
                                and _pass_race_open((rx, ry), target, enemies, enemy_vel_est)):
                            self.state = passing
                            self.pass_target = target
                            self.pass_hold_s = 0.0
                            # handing the ball off, the route
                            # is not ours to finish
                            self.sideline_route = None
                            print(f"[main] passing to "
                                 f"{target[0]:.0f},{target[1]:.0f}", flush=True)
                            return

                # Ball hiding along a sideline (motion's ball_hiding_*
                # block): far from goal with an enemy upfield, keep the
                # ball on the wall side and gain depth along the line.
                # Arming and release are held inside
                # motion._ball_hide_step. It outranks the shot/carry
                # ladder, but not the tap-in and pass checks above, and
                # not an armed sideline route.
                hide_deferred = (motion.sideline_route_enabled
                                 and self.sideline_route is not None)
                if hide_deferred:
                    self.hiding = False
                    self.hide_tucked = False
                if ball is not None and not hide_deferred:
                    _bd_rel, _bd_d = ball
                    _b_rad = math.radians(_bd_rel + hdg)
                    hide_ball = (rx + _bd_d * math.sin(_b_rad),
                                 ry + _bd_d * math.cos(_b_rad))
                else:
                    hide_ball = None
                self.hiding, self.hide_tucked, hide_plan = _ball_hide_step(
                    self.hiding, self.hide_tucked, rx, ry, hdg, enemy_goal, enemies,
                    ball_xy=hide_ball)
                # while deferred the step still re-arms on its inputs,
                # so the gate sits on the drive: the route branch below
                # owns this tick
                if hide_plan is not None and not hide_deferred:
                    hide_rel, hide_head_rel, hide_ratio, hide_near = hide_plan
                    hide_rel, hide_cmd = _wall_guard(rx, ry, hdg, hide_rel,
                                                     base_speed * hide_ratio)
                    _set_dwibbler(True, speed=_dwibble_carry_speed(rx, ry, enemies,
                                                                   hide_head_rel))
                    _slew_drive(hide_rel, hide_cmd,
                                rot_speed=_heading_spin(hide_head_rel))
                    return

                # Drive the ball into the enemy goal: default to the
                # goal-centre bearing, but when an enemy is in the way
                # evaluate direct finishing angles across the mouth and
                # steer at one that scores (_goal_shot_aim; the ball must
                # strike the back wall, RCJA 5.5.1).
                aim_bearing, aim_found = _goal_shot_aim(rx, ry, gy, enemies,
                                                        preferred_bearing=self.shot.aim)
                # Commit to a lane (motion.ShotCommitment): aim_found
                # stays this tick's raw answer and drives the abort/flick
                # gates, but we steer on the committed bearing, held while
                # a lane is merely blocked. steer_bearing is None once the
                # commitment is dropped, and the goal-centre fallback
                # takes over.
                steer_bearing, _shot_ready = self.shot.select(
                    aim_bearing if aim_found else None, time.monotonic())
                if steer_bearing is not None:
                    goal_bearing = steer_bearing
                else:
                    goal_bearing = math.degrees(math.atan2(gx - rx, gy - ry))
                goal_rel = _wrap_deg(goal_bearing - hdg)

                # No shot attempt yet: a blocked lane reverses the roller
                # in place (a wind-up turn could get stuck spinning). Only
                # after flick_settle_s of carry, or a fresh contested
                # capture reverses the ball straight back to the blocker
                # in a loop. Skipped when the finishing search already
                # found a clear line.
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
                        # straight into the shared non-blocking
                        # eject countdown (flick mode)
                        _set_dwibbler(True, speed=-flick_kick_speed)
                        self.eject_until_t = time.monotonic() + flick_snap_s
                        self.eject_is_pass = False
                        self.state = ejecting
                        print("[main] blocked -> reversing dwibbler "
                             "(no shot attempt yet)", flush=True)
                        return

                # Carry/shield: no tap-in, pass or open shot, and not yet
                # inside flick_range_mm. Press for goal only if the lane
                # is winnable now (_carry_should_press), else hold the
                # shield heading and creep clear. The pass checks above
                # re-run every tick, so an opening is taken next tick.
                if (not aim_found and enemies and d_us >= flick_range_mm
                        and not _carry_should_press(rx, ry, gx, gy, enemies)):
                    # sideline route: inside the hold gate it outranks
                    # the shield/creep. Retired on arrival or when an
                    # enemy camps the waypoint; an opened lane goes to
                    # the shot logic first.
                    if (motion.sideline_route_enabled
                            and self.sideline_route is not None):
                        wpx, wpy = self.sideline_route
                        if any(math.hypot(e["x"] - wpx, e["y"] - wpy)
                               < enemy_contest_dist_mm for e in enemies):
                            self.sideline_route = None
                        else:
                            dxw, dyw = wpx - rx, wpy - ry
                            dist_w = math.hypot(dxw, dyw)
                            if dist_w < sideline_route_slow_mm:
                                self.sideline_route = None
                                print("[main] sideline route reached -> carry",
                                      flush=True)
                            else:
                                route_rel = _wrap_deg(
                                    math.degrees(math.atan2(dxw, dyw)) - hdg)
                                # same shield facing as
                                # the creep: keep the
                                # chassis between the ball
                                # and the nearest enemy
                                shield_rel = _shield_heading_deg(
                                    rx, ry, hdg, gx, gy, enemies)
                                _set_dwibbler(True, speed=_dwibble_carry_speed(
                                    rx, ry, enemies, shield_rel))
                                _slew_drive(route_rel, base_speed,
                                            rot_speed=_heading_spin(shield_rel))
                                return
                    shield_rel = _shield_heading_deg(rx, ry, hdg, gx, gy, enemies)
                    creep_rel, creep_cmd = _carry_creep(rx, ry, hdg, gx, gy, enemies,
                                                        carry_anchor=self.carry_anchor)
                    _set_dwibbler(True, speed=_dwibble_carry_speed(
                        rx, ry, enemies, shield_rel))
                    _slew_drive(creep_rel, creep_cmd,
                               rot_speed=_heading_spin(shield_rel))
                    return

                # No solenoid: scoring is drive-in. The near-goal eject
                # cuts the roller and the body's momentum is the shot
                # (RCJA 4.6.6: a dribbler must release to score), so the
                # release gate asks whether the facing we already hold
                # scores (motion._shot_release_scores). A lane that has
                # gone blocked holds the ball instead of firing into the
                # blocker.
                if not shot_align_enabled:
                    aligned_now = True
                elif shot_release_ray_enabled:
                    aligned_now = _shot_release_scores(rx, ry, hdg, gy, enemies)
                else:
                    # pre-substitution proxy, kept as the revert arm of the A/B
                    aligned_now = (abs(_wrap_deg(goal_bearing - hdg))
                                   <= shot_align_tol_deg)

                self.near_goal_eject = (d_us < goal_eject_range_mm
                                        and aligned_now)
                if self.near_goal_eject:
                    _set_dwibbler(False)
                else:
                    # not yet close enough or lined up: graduated grip
                    # for the push (full when contested or turning
                    # hard)
                    _set_dwibbler(True, speed=_dwibble_carry_speed(
                        rx, ry, enemies, _heading_spin(goal_rel)))
                # no _wall_guard here: the goal's flanking segments sit
                # close enough to read as a wall (the white-line gate in
                # _slew_drive still applies)
                _slew_drive(goal_rel, base_speed, rot_speed=_heading_spin(goal_rel))

        # passing: line up on the pass target and release off the reversed roller
        elif self.state == passing:
            # same loss checks as has_ball: a ball in the mouth is often
            # invisible to the camera
            _set_dwibbler(True)
            if _possession.has_ball:
                self.possession_seen = True
            if ball is not None:
                _, dist_mm = ball
                if dist_mm > capture_dist_mm * 2:
                    self.state = seek
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
                self.state = seek
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
            # Turn about the ball, not on the spot (motion._turn_about_ball),
            # so the mouth stays on it for the whole turn. ball[0] is the
            # camera bearing; with no fix the held ball is dead ahead.
            ball_rel = ball[0] if ball is not None else 0.0
            if abs(rel) > pass_align_deg:
                self.pass_hold_s = 0.0
                _slew_drive(*_turn_about_ball(ball_rel, _heading_spin(rel)))
                return
            self.pass_hold_s += loop_dt
            if self.pass_hold_s < pass_confirm_s:
                _slew_drive(*_turn_about_ball(ball_rel, _heading_spin(rel)))
                return
            # Aligned and settled: release. Reverse the roller at the speed
            # this pass's length needs (motion.pass_eject_frac, the number the
            # race gate scored), as a countdown rather than a sleep that would
            # stall the loop for 0.25 s.
            self.eject_speed = pass_eject_frac(math.hypot(tx - rx, ty - ry))
            _set_dwibbler(True, speed=-self.eject_speed)
            self.eject_until_t = time.monotonic() + pass_eject_s
            self.eject_is_pass = True
            self.pass_hold_s = 0.0
            self.pass_expiry_t = time.monotonic() + pass_target_max_age_s
            _possession.reset()
            self.state = ejecting
            print("[main] pass released", flush=True)

        # ejecting: the shared pass/flick roller-reverse countdown; runs its full
        # dwell without blocking, then back to seek
        elif self.state == ejecting:
            if time.monotonic() < self.eject_until_t:
                _set_dwibbler(True,
                              speed=-(self.eject_speed if self.eject_is_pass
                                      else flick_kick_speed))
                # Hold still-ish while ejecting; a spin-in-place would
                # fight the ball leaving the mouth.
                _slew_drive(0, 0)
                return
            _set_dwibbler(False)
            if not self.eject_is_pass:
                _possession.reset()
                self.possession_seen = False
                print("[main] dwibbler reversed (no shot attempt) -> seeking",
                      flush=True)
            self.state = seek

        # shoot hooks: align to the enemy goal and kick (enable with a solenoid)
        # elif self.state == shoot:
        #     self.state = _shoot_tick(pose, enemy_goal, time.monotonic() - t_shoot0)


def _camrun_forward_bearing(front_goal, imu_now, forward0):
    """best estimate of the robot-frame bearing (0 = dead ahead) to "forward", the direction
    camrun pushes the ball.
    """
    if (front_goal is not None and vision._enemy_goal_colour is not None
            and front_goal[0] == vision._enemy_goal_colour):
        return front_goal[1]
    if imu_now is not None and forward0 is not None:
        return -_wrap_deg(imu_now - forward0)
    return 0.0


class CamRunController:
    """run_mode "camrun": camera and roller-stall possession only, no lidar; never reads the
    pose.
    """
    def __init__(self):
        """fresh FSM state (starts in seek)."""
        self.state = seek
        self.capture_hold_s = 0.0
        self.possession_seen = False
        self.close_ball_t = 0.0
        self.carry_start_t = 0.0 # monotonic time this carry began

    def tick(self):
        """run one control tick of the camera-only behaviour, see the class docstring."""
        with state._lock:
            ball = state._state["ball"]
            front_goal = state._state["front_goal"]
            run_mode = state._state["run_mode"]
            imu_now = state._state["imu_heading"]
            forward0 = state._state["camrun_forward_heading"]
            state._state["my_state"] = self.state
            # reset every tick, as the striker does
            state._state["capture_zone"] = None

        if run_mode != "camrun":
            Motor.stopall()
            _set_dwibbler(False)
            self.state = seek
            return

        fwd_bearing = _camrun_forward_bearing(front_goal, imu_now, forward0)

        # seek: direct-pursuit approach to the ball (_orbit_approach)
        if self.state == seek:
            if (_possession.has_ball
                    and time.monotonic() - self.close_ball_t
                        <= possess_capture_window_s):
                _set_dwibbler(True)
                self.state = has_ball
                self.possession_seen = True
                self.capture_hold_s = 0.0
                self.carry_start_t = time.monotonic()
                print("[main] ball captured -> attacking (camrun, dwibbler)",
                      flush=True)
                return

            if ball is not None:
                angle_deg, dist_mm = ball
                b_rad = math.radians(angle_deg)
                bx, by = dist_mm * math.sin(b_rad), dist_mm * math.cos(b_rad)
                drive_angle, speed_frac = _orbit_approach(bx, by)
                drive_cmd = min(base_speed * speed_frac, _brake_speed_frac(dist_mm))
                # Nose tracks the drive target, which is now just the
                # ball's bearing (direct pursuit).
                turn = _heading_spin(drive_angle)

                lat_mm = dist_mm * math.sin(math.radians(angle_deg))
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
                    self.state = has_ball
                    self.possession_seen = False
                    self.capture_hold_s = 0.0
                    self.carry_start_t = time.monotonic()
                    print("[main] ball captured -> attacking (camrun)",
                          flush=True)
            else:
                _set_dwibbler(False)
                with state._lock:
                    state._state["capture_zone"] = "search"
                # no spin: the fisheye ring already sees 360 degrees, and
                # camrun has no pose to translate toward
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

            # no pose -> no _wall_guard, so this is kept to
            # wall_safe_speed_cmd
            _slew_drive(0, wall_safe_speed_cmd, rot_speed=_heading_spin(fwd_bearing))
