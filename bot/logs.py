"""Bench-log session finishers, the constants each log header records, and the --motionlog poll
thread.
"""

import math
import time

import bot.motion as _motion
import bot.state as state
import bot.compass as _compass
import bot.odometry as _odometry
import bot.drive_config as _drive_config
from bot.hardware import Motor


_lidar_bad_rms = 55.0 # mm, frames above this are "bad"
_lidar_min_inliers = 30 # frames below this are "bad"
_lidar_recover_n = 3 # consecutive bad frames before re-searching


def _finish_lidar_log():
    """close the --lidarlog session and print where the .txt landed."""
    log = state._lidar_log
    if log is None:
        return
    state._lidar_log = None
    path = log.close()
    print(f"[lidarlog] wrote {path}, commit and push it, or read the "
          "summary at the end", flush=True)


def _finish_capture_log():
    """close the --capturelog session and print where the .txt landed."""
    log = state._capture_log
    if log is None:
        return
    state._capture_log = None
    path = log.close()
    print(f"[capturelog] wrote {path}, commit and push it, or read the "
          "summary at the end", flush=True)


def _finish_motion_log():
    """close the --motionlog session and print where the .txt landed."""
    log = state._motion_log
    if log is None:
        return
    state._motion_log = None
    path = log.close()
    print(f"[motionlog] wrote {path}, commit and push it, or read the "
          "summary at the end", flush=True)


def _capture_log_consts():
    """every constant needed to interpret a capture log, as a flat dict."""
    # lazy: controllers and tracking import this module, so a top-level import would cycle
    import bot.controllers as _controllers
    from bot.tracking import (ball_vel_jump_mm, ball_vel_min_span_s,
                              ball_vel_stationary_std_mm, ball_vel_window_s)
    return {
        "capture_cone_half_deg": _motion.capture_cone_half_deg,
        "capture_cone_half_width_mm": _motion.capture_cone_half_width_mm,
        "capture_ball_vel": _controllers.capture_ball_vel,
        "ball_vel_window_s": ball_vel_window_s,
        "ball_vel_jump_mm": ball_vel_jump_mm,
        "ball_vel_min_span_s": ball_vel_min_span_s,
        "ball_vel_stationary_std_mm": ball_vel_stationary_std_mm,
        "base_speed": _drive_config.base_speed,
        "heading_deadband_deg": _motion.heading_deadband_deg,
        "heading_saturation_deg": _motion.heading_saturation_deg,
    }


def _lidar_log_consts():
    """every constant needed to interpret a lidar log, as a flat dict."""
    # lazy: lidar imports this module, so a top-level import would cycle
    import bot.vision as _vision
    from bot.field import FieldModel
    from bot.lidar import LidarCoords
    from bot.perception import Perception
    return {
        "inlier_threshold_mm": Perception.inlier_threshold_mm,
        "max_iters": Perception.max_iters,
        "converge_mm": Perception.converge_mm,
        "robot_radius_mm": Perception.robot_radius_mm,
        "scan_step_deg": round(math.degrees(Perception.scan_step_rad), 3),
        "lidar_bad_rms": _lidar_bad_rms,
        "lidar_min_inliers": _lidar_min_inliers,
        "lidar_recover_n": _lidar_recover_n,
        "min_range_mm": LidarCoords.min_range_mm,
        "max_range_mm": LidarCoords.max_range_mm,
        "min_intensity": LidarCoords.min_intensity,
        "angle_offset_deg": LidarCoords.angle_offset_deg,
        "angle_sign": LidarCoords.angle_sign,
        "handle_exclusion_deg": getattr(_vision, "handle_exclusion_deg", None),
        "imu_fusion_enabled": _compass.imu_fusion_enabled,
        "field_x": FieldModel.field_x,
        "field_y": FieldModel.field_y,
    }


# --motionlog poll rate
motionlog_hz = 20.0
_MOTION_LOG_MOTORS = ("nw", "ne", "sw", "se", "dwibble")


def _motion_log_consts():
    """every constant needed to interpret a --motionlog log, as a flat dict."""
    from bot.dwibbler import dwibble_speed
    return {
        "motionlog_hz": motionlog_hz,
        "base_speed": _drive_config.base_speed,
        "heading_deadband_deg": _motion.heading_deadband_deg,
        "heading_saturation_deg": _motion.heading_saturation_deg,
        "dwibble_speed": dwibble_speed,
        "MOTOR_CAP_FRAC": Motor.MOTOR_CAP_FRAC,
        "imu_fusion_enabled": _compass.imu_fusion_enabled,
        "MOTOR_MAX_RAW": Motor.MOTOR_MAX_RAW,
        "wheel_odom_enabled": _odometry.wheel_odom_enabled,
        "COLLISION_ACCEL_G": _compass.COLLISION_ACCEL_G,
    }


def _motion_log_thread():
    """poll motor commands, pose, IMU, QDR and collisions at motionlog_hz into _motion_log
    until its duration is up, then close it.
    """
    from bot.dwibbler import _possession
    period = 1.0 / motionlog_hz
    while state._motion_log is not None:
        with state._lock:
            pose = state._state["pose"]
            imu = state._state["imu_heading"]
            my_state = state._state["my_state"]
            run_mode = state._state["run_mode"]
            slot_role = state._state["slot_role"]
            ball_seen = state._state["ball"] is not None
            ball_est = state._state["ball_est"]
            zone = state._state["capture_zone"]
        cmds = {m: Motor._last_sent.get(m) for m in _MOTION_LOG_MOTORS}
        qdr = {}
        for m in _MOTION_LOG_MOTORS:
            motor = Motor.motors.get(m)
            reading = motor.read_qdr() if motor is not None else None
            qdr[m] = reading[1] / Motor.MOTOR_MAX_RAW if reading is not None else None
        with state._lock:
            collided = (state._state["collision_t"] is not None
                       and time.monotonic() - state._state["collision_t"] < period)
        # _possession is read directly, as the play loop does (plain attribute
        # reads are atomic under the GIL); it was never published into _state
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
