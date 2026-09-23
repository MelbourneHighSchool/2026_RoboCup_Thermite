"""Bench-logging session finishers, log-header const dicts, and the
--motionlog poll thread. Extracted verbatim from mainrunbot1.py's logging
section (around what was lines 4921-5065 before this extraction).

Several *_consts()/  _motion_log_thread() functions here reference globals
that belong to LATER refactor stages (vision.py, motion.py, controllers.py)
that do not exist yet as bot.* modules - e.g. Perception, LidarCoords,
FieldModel's field_x/field_y, base_speed, turn_gain, dwibble_speed,
capture_cone_half_deg and friends, ball_vel_* tunables, handle_exclusion_deg,
and the module-level _possession (BallPossession) singleton. Those names are
left as plain (undefined-for-now) globals in this module's function bodies,
exactly as they were plain globals in the monolith - Python does not
evaluate a function body until it's called, so `import bot.logs` succeeds
today even though calling _capture_log_consts()/_lidar_log_consts()/
_motion_log_consts()/_motion_log_thread() will NameError until a later
stage's module supplies those names (e.g. via `from bot.vision import *`
equivalent wiring in bot/main.py, or explicit imports added when those
modules are extracted). This mirrors the existing land-mine pattern for
bot.odometry._wheel_odom: don't paper over a not-yet-extracted dependency
with a guess, just extract this module's own code faithfully and leave the
gap documented for the stage that fills it.

What IS already extracted and is properly imported here: imu_fusion_enabled
and COLLISION_ACCEL_G from bot.compass, wheel_odom_enabled from
bot.odometry, and the shared _lock/_state/_lidar_log/_capture_log/
_motion_log from bot.state (accessed as state.<name> per bot/state.py's own
docstring, since _lidar_log/_capture_log/_motion_log are rebound globals).
"""

import math
import time

import bot.state as state
from bot.compass import imu_fusion_enabled, COLLISION_ACCEL_G
from bot.hardware import Motor
from bot.odometry import wheel_odom_enabled


_lidar_bad_rms     = 55.0 # mm, frames above this are "bad"
_lidar_min_inliers = 30 # frames below this are "bad"
_lidar_recover_n   = 3 # consecutive bad frames before re-searching


def _finish_lidar_log():
    """close the --lidarlog session and print where the .txt landed."""
    log = state._lidar_log
    if log is None:
        return
    state._lidar_log = None
    path = log.close()
    print(f"[lidarlog] wrote {path}, commit and push it, or read the "
          "SUMMARY block at the end", flush=True)


def _finish_capture_log():
    """close the --capturelog session and print where the .txt landed."""
    log = state._capture_log
    if log is None:
        return
    state._capture_log = None
    path = log.close()
    print(f"[capturelog] wrote {path}, commit and push it, or read the "
          "SUMMARY block at the end", flush=True)


def _finish_motion_log():
    """close the --motionlog session and print where the .txt landed."""
    log = state._motion_log
    if log is None:
        return
    state._motion_log = None
    path = log.close()
    print(f"[motionlog] wrote {path}, commit and push it, or read the "
          "SUMMARY block at the end", flush=True)


def _capture_log_consts():
    """every constant needed to interpret a capture log, as a flat dict."""
    return {
        "capture_cone_half_deg":      capture_cone_half_deg,
        "capture_cone_half_width_mm": capture_cone_half_width_mm,
        "capture_ball_vel":           capture_ball_vel,
        "ball_vel_window_s":          ball_vel_window_s,
        "ball_vel_jump_mm":           ball_vel_jump_mm,
        "ball_vel_min_span_s":        ball_vel_min_span_s,
        "ball_vel_stationary_std_mm": ball_vel_stationary_std_mm,
        "base_speed":                 base_speed,
        "turn_gain":                  turn_gain,
    }


def _lidar_log_consts():
    """every constant needed to interpret a lidar log, as a flat dict."""
    return {
        "inlier_threshold_mm":  Perception.inlier_threshold_mm,
        "max_iters":            Perception.max_iters,
        "converge_mm":          Perception.converge_mm,
        "robot_radius_mm":      Perception.robot_radius_mm,
        "scan_step_deg":        round(math.degrees(Perception.scan_step_rad), 3),
        "lidar_bad_rms":        _lidar_bad_rms,
        "lidar_min_inliers":    _lidar_min_inliers,
        "lidar_recover_n":      _lidar_recover_n,
        "min_range_mm":         LidarCoords.min_range_mm,
        "max_range_mm":         LidarCoords.max_range_mm,
        "min_intensity":        LidarCoords.min_intensity,
        "angle_offset_deg":     LidarCoords.angle_offset_deg,
        "angle_sign":           LidarCoords.angle_sign,
        "handle_exclusion_deg": handle_exclusion_deg,
        "imu_fusion_enabled":   imu_fusion_enabled,
        "field_x":              FieldModel.field_x,
        "field_y":              FieldModel.field_y,
    }


# --motionlog poll rate (sec. 1).
motionlog_hz = 20.0
_MOTION_LOG_MOTORS = ("nw", "ne", "sw", "se", "dwibble")


def _motion_log_consts():
    """every constant needed to interpret a --motionlog log, as a flat dict."""
    return {
        "motionlog_hz":       motionlog_hz,
        "base_speed":         base_speed,
        "turn_gain":          turn_gain,
        "dwibble_speed":      dwibble_speed,
        "MOTOR_CAP_FRAC":     Motor.MOTOR_CAP_FRAC,
        "imu_fusion_enabled":     imu_fusion_enabled,
        "MOTOR_MAX_RAW":          Motor.MOTOR_MAX_RAW,
        "wheel_odom_enabled": wheel_odom_enabled,
        "COLLISION_ACCEL_G":  COLLISION_ACCEL_G,
    }


def _motion_log_thread():
    """poll Motor._last_sent/pose/IMU/QDR/collision at motionlog_hz and feed _motion_log until the session's duration is up, then close it."""
    period = 1.0 / motionlog_hz
    while state._motion_log is not None:
        with state._lock:
            pose      = state._state["pose"]
            imu       = state._state["imu_heading"]
            my_state  = state._state["my_state"]
            run_mode  = state._state["run_mode"]
            slot_role = state._state["slot_role"]
            ball_seen = state._state["ball"] is not None
            ball_est  = state._state["ball_est"]
            zone      = state._state["capture_zone"]
        cmds = {m: Motor._last_sent.get(m) for m in _MOTION_LOG_MOTORS}
        qdr = {}
        for m in _MOTION_LOG_MOTORS:
            motor = Motor.motors.get(m)
            reading = motor.read_qdr() if motor is not None else None
            qdr[m] = reading[1] / Motor.MOTOR_MAX_RAW if reading is not None else None
        with state._lock:
            collided = (state._state["collision_t"] is not None
                       and time.monotonic() - state._state["collision_t"] < period)
        # drib/drib_avail read _possession directly, same as the play loops
        # do, thread-safe by GIL-atomicity (BallPossession's own docstring),
        # not _state, since it was never published there.
        fsm = {"state": my_state, "run_mode": run_mode, "role": slot_role,
               "zone": zone, "ball_seen": ball_seen,
               "ball_est_src": ball_est[3] if ball_est is not None else None,
               "ball_est_conf": ball_est[2] if ball_est is not None else None,
               "drib": _possession.has_ball, "drib_avail": _possession.available}
        state._motion_log.tick(cmds, pose, imu, qdr, fsm, collision=collided)
        if state._motion_log.is_done():
            break
        time.sleep(period)
    _finish_motion_log()
