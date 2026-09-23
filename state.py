"""Shared runtime state: the _state dict/_lock and the handful of rebindable
module-level globals that back it, extracted verbatim from mainrunbot1.py's
"Shared state" section.

This is a leaf module - no imports from any other bot.* module - meant to
be the dependency root every other bot.* module imports from. Several of
these globals (mode, _status_led, _lidar_log, _capture_log, _motion_log,
_jpeg_bytes) get reassigned via `global` in the current monolith; other
modules should access them as attributes on this module (state.mode,
state._status_led, ...), never via `from bot.state import mode`, so that a
later reassignment here is visible to every importer.
"""

import threading


# Shared state
_lock  = threading.Lock()
_state = {
    "pose":  None, # (x, y, heading_deg)
    "ball":  None, # (angle_deg, dist_mm) or None
    # (w, h, row, subpixel_shift) of the detected blob in unwrap cells, diagnostic only,
    # see _subpixel_centre
    "ball_blob": None,
    "frame": None, # BGR frame at capture res
    "mask":  None, # orange mask (uint8)
    "cx_px": None, # ball centroid x in capture pixels
    "cy_px": None, # ball centroid y in capture pixels
    "radius": None, # detected circle radius in capture pixels (or None)
    # live-tunable HSV (used by --hsv mode)
    "hsv_lower": None,
    "hsv_upper": None,
    # two-bot collaboration
    "teammate_pos": None, # (x, y) via UDP fallback, or None
    "teammate_pos_bt": None, # (x, y) via bluetooth team link (takes priority)
    # (x, y, conf, src) ball fix from the peer, from the subordinate this is always its
    # own camera sighting; from the master it is the fused estimate, so src says whether
    # it is a sighting at all before it can count as one
    "remote_ball": None,
    "ball_est":    None, # (x, y, conf, src) our fused ball estimate
    "enemies":      [], # list of detection dicts from detect_robots
    "enemy_vel":    {}, # {track_id: (vx, vy)} field mm/s, from EnemyVelocityTracker
    "cam_fps":      0.0, # camera capture fps (for debug overlay)
    "lidar_hz":     0.0, # lidar spin speed in Hz (from packet speed field)
    "lidar_pts":    [], # latest raw robot-frame points [(x,y), ...]
    # Wheel-slip cross-check (diagnostic only, see WheelSlipMonitor's own block
    # comment): 1.0 = wheel-claimed and lidar-observed speed agree, sliding to
    # 0.0 as they diverge (a wheel probably slipping).
    "wheel_slip_trust": 1.0,
    "wheel_speed_mms":  0.0, # EMA wheel-odometry speed this revolution
    "lidar_speed_mms":  0.0, # EMA lidar/ICP-observed speed this revolution
    # Most recent wobble Finding dict (see detect_wobble's own block comment),
    # or None between reports - pure telemetry, never read by anything that
    # steers the robot.
    "wobble": None,
    # new sensing / team-play state
    "drib_has_ball": False, # dwibbler-stall possession (BallPossession)
    # camera saw the dwibble_mark colour in the mouth notch this frame,
    # QDR-stall's second vote, off (dwibble_mark_enabled=False) until the
    # real sticker exists
    "dwibble_mark_seen": False,
    # _dwibble_camera_thread's own direct sighting (dwibble_cam_enabled), and the raw orange-pixel fraction behind it, for debug
    "dwibble_cam_seen": False,
    "dwibble_cam_frac": 0.0,
    "imu_heading":   None, # BNO08x yaw (relative, deg cw+) or None
    # monotonic time of the last COLLISION_ACCEL_G spike, or None, WheelOdometry's contact
    # gate
    "collision_t":   None,
    # True once _compass_thread has seen no good BNO08x reading for
    # imu_fault_hold_s straight (a sustained fault, not one transient bus
    # error - that's already absorbed inside _compass_thread's own
    # try/except). Auto-clears the instant a good reading resumes.
    "imu_fault":         False,
    # Sticky IMU-fault forced-stop latch (see imu_fault_hold_s): set the
    # instant imu_fault first goes True, and stays set - forcing _play_loop
    # to stop driving and run_mode back to "idle" - until the operator
    # explicitly re-picks colour/role (_maybe_start) with the IMU healthy
    # again.
    "imu_pause_latched": False,
    # subsystem name -> status string, last published by _health_thread /
    # _compass_thread's own fault report (_report_health); read by the
    # debug HTTP page's /health endpoint.
    "health":            {},
    # imu_heading as it was when "pose" was published, so consumers can carry the heading
    # forward at IMU rate between revolutions (_fused_heading)
    "pose_imu":      None,
    "peer_state":    None, # subordinate's reported state ("seek"/"has_ball"/...)
    "yield_striker": False, # bt thread: teammate is taking the ball, hold back
    "my_state":      None, # our own behaviour state, published over the link
    # (x, y) field-frame target while we're passing (sec. 3.21), published to the teammate
    "pass_target":     None,
    # (x, y) the teammate's pass_target, read over the link, lowest-priority ball_est
    # source
    "peer_pass_target": None,
    # "cone" (centred/close, sec 4.1) | "search" (blind-spinning) | None
    # (chasing but not yet centred, or not chasing at all this tick), for
    # --motionlog (sec. 1) to read outside the play loop.
    "capture_zone":  None,
    "solo":          True, # no live teammate -> full one-robot game (default)
    # None (undecided), True (RFCOMM server), or False (RFCOMM client), decided once by _negotiate_bt_role for whichever robot wins the role race.
    "bt_is_master":  None,
    # True if we attack the y=0 goal, set from default_slot_goal the instant both buttons are pressed (_maybe_start); the robot always starts low with the goal colour already correct, so this is never re-derived from the camera.
    "attack_low":    None,
    "open_goal":     None, # {bearing_deg, open_deg, blocked_frac} aim target
    # (colour, bearing_deg) within cam_front_half_deg of dead ahead, or None, see
    # detect_front_goal. CamRunController's own forward-bearing fix (sec 4.9), no lidar involved.
    "front_goal":    None,
    # imu_heading when camrun (sec 4.9) was entered, locks the attacking
    # direction for the session. Set by _select_camrun; None until camrun
    # starts or if the IMU is down.
    "camrun_forward_heading": None,
    # slot_role is one of two independent picks (main()'s buttons); the enemy goal colour lives in the module-global _enemy_goal_colour, not here, since the camera-vote fallback also writes it.
    "slot_goal":     None, # None | "low" | "high"
    # None | "striker" | "goalie", role actually Played, may be overwritten by the dynamic
    # handoff (sec 4.8)
    "slot_role":     None,
    # role this bot's own button picked, untouched by the handoff.
    "own_slot_role": None,
    # "idle"/"run"/"calib"/"camrun": run starts once both sets are picked (or forced by Calib); camrun is forced the same way, by the keyboard-only toggle.
    "run_mode":      "idle",
    # set on a real side change (a fresh colour pick, or the goal-lock correcting the
    # provisional guess) so _lidar_thread re-searches instead of coasting on a pose fitted
    # against the old guess
    "force_relocalise": False,
}

mode  = "run" # "run" | "hsv"

_jpeg_cond  = threading.Condition()
_jpeg_bytes = None # latest rendered MJPEG frame
# notified by _camera_thread each time it publishes a new _state["frame"],
# lets _render_thread re-render on arrival instead of free-spinning
# re-encoding the same stale frame
_frame_cond = threading.Condition()
# gpiozero LED, set in main(), also toggled by _calib_routine when calibration finishes on
# its own
_status_led = None
# lidar_debug.LidarLogger while --lidarlog is running, None otherwise. Every hook is
# behind an "is not None" check, so a normal run pays one pointer comparison per scan and
# nothing else.
_lidar_log  = None
# capture_debug.CaptureLogger while --capturelog is running, None otherwise, same "is not
# None" pattern, checked once per StrikerController tick.
_capture_log = None
# motion_debug.MotionLogger while --motionlog is running, None otherwise, fed by its own
# polling thread (_motion_log_thread), not the play loop.
_motion_log  = None


def _apply_slot_state():
    """call under _lock after mutating slot_role from the Bluetooth dynamic role handoff (sec 4.8), main()'s own buttons manage run_mode themselves and don't call this."""
    if _state["run_mode"] != "run":
        _state["run_mode"] = "idle"
