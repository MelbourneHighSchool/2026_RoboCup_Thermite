"""Shared runtime state: the _state dict, its lock, and the rebindable module globals."""

import threading


_lock = threading.Lock()
_state = {
    "pose": None, # (x, y, heading_deg)
    # monotonic time "pose" was published (the scan's timestamp), so readers can
    # judge its age
    "pose_t": None,
    # (x, y, heading, t): the last lidar fit carried forward with wheel odometry and
    # the fused IMU heading at about 50 Hz (PosePropagator, bot/odometry.py). Its only
    # writer never writes "pose", so it can't feed itself. None before the first fit
    # or with pose_propagate_enabled off.
    "pose_live": None,
    "ball": None, # (angle_deg, dist_mm) or None
    # (w, h, row, subpixel_shift) of the ball blob in unwrap cells, diagnostic only
    "ball_blob": None,
    "frame": None, # BGR frame at capture res
    "mask": None, # orange mask (uint8)
    "cx_px": None, # ball centroid x in capture pixels
    "cy_px": None, # ball centroid y in capture pixels
    "radius": None, # detected circle radius in capture pixels (or None)
    # live-tunable HSV (used by --hsv mode)
    "hsv_lower": None,
    "hsv_upper": None,
    # two-bot collaboration
    "teammate_pos": None, # (x, y) via UDP fallback, or None
    "teammate_pos_bt": None, # (x, y) via bluetooth team link (takes priority)
    # (x, y, conf, src) ball fix from the peer: the subordinate sends its own camera
    # sighting, the master sends its fused estimate, so check src before counting it
    # as a sighting
    "remote_ball": None,
    "ball_est": None, # (x, y, conf, src) our fused ball estimate
    # Kickoff hold, set by maybe_start at play start and read by both controllers.
    # None = normal play (and what it decays to). "kicking": our kick-off, the striker
    # drives to the centre ball and the goalie holds in front of goal. "receiving":
    # theirs, the striker holds just short of the box edge and the goalie sits in the
    # box behind it.
    "kickoff_role": None,
    "kickoff_until_t": None, # monotonic time the hold expires
    "enemies": [], # list of detection dicts from detect_robots
    "enemy_vel": {}, # {track_id: (vx, vy)} field mm/s, from EnemyVelocityTracker
    "cam_fps": 0.0, # camera capture fps (for debug overlay)
    "lidar_hz": 0.0, # lidar spin speed in Hz (from packet speed field)
    "lidar_pts": [], # latest raw robot-frame points [(x,y), ...]
    # wheel-slip cross-check (diagnostic only): 1.0 when wheel and lidar speeds agree,
    # falling to 0.0 as they diverge
    "wheel_slip_trust": 1.0,
    "wheel_speed_mms": 0.0, # EMA wheel-odometry speed this revolution
    "lidar_speed_mms": 0.0, # EMA lidar/ICP-observed speed this revolution
    # latest wobble finding dict (detect_wobble), or None; telemetry only
    "wobble": None,
    "drib_has_ball": False, # dwibbler-stall possession (BallPossession)
    # camera saw the dwibble_mark colour in the mouth notch this frame: the QDR
    # stall's second vote (off until the real sticker exists)
    "dwibble_mark_seen": False,
    # second camera's direct sighting and the orange-pixel fraction behind it, for
    # debug
    "dwibble_cam_seen": False,
    "dwibble_cam_frac": 0.0,
    "imu_heading": None, # BNO08x yaw (relative, deg cw+) or None
    # monotonic time of the last COLLISION_ACCEL_G spike, WheelOdometry's contact gate
    "collision_t": None,
    # True once there has been no good BNO08x reading for imu_fault_hold_s straight;
    # clears the moment a good reading resumes
    "imu_fault": False,
    # sticky forced stop: set when imu_fault goes True, and held (play loop stopped,
    # run_mode forced to idle) until the operator re-picks colour/role with the IMU
    # healthy again
    "imu_pause_latched": False,
    # subsystem -> status string, from _health_thread and the compass thread; served
    # at /health
    "health": {},
    # imu_heading at the moment "pose" was published, so the heading can be carried
    # forward at IMU rate between revolutions (_fused_heading)
    "pose_imu": None,
    "peer_state": None, # subordinate's reported state ("seek"/"has_ball"/...)
    "my_state": None, # our behaviour state, published over the link
    # (x, y) field-frame target while passing, published to the teammate
    "pass_target": None,
    # the teammate's pass_target, the lowest-priority ball_est source
    "peer_pass_target": None,
    # "cone" (ball centred and close), "search" (blind spin) or None, for --motionlog
    "capture_zone": None,
    # (choice, ((name, score), ...)) from the striker's seek arbiter, best first;
    # telemetry only
    "seek_util": None,
    "solo": True, # no live teammate -> full one-robot game (default)
    # None (undecided), True (RFCOMM server) or False (client), settled once by
    # _negotiate_bt_role
    "bt_is_master": None,
    # True if we attack the y=0 goal. Fixed when play starts and never re-derived from
    # the camera.
    "attack_low": None,
    "open_goal": None, # {bearing_deg, open_deg, blocked_frac} aim target
    # (colour, bearing_deg) of a goal near dead ahead, or None (detect_front_goal);
    # camrun's forward fix, no lidar involved
    "front_goal": None,
    # imu_heading when camrun started, locking the attacking direction; None before
    # camrun or with no IMU
    "camrun_forward_heading": None,
    # The two button picks. The enemy goal colour lives in vision._enemy_goal_colour
    # instead, because the camera-vote fallback writes it too.
    "slot_goal": None, # None | "low" | "high"
    # None | "striker" | "goalie": the role actually played, which the dynamic handoff
    # may change
    "slot_role": None,
    # the role this bot's button picked, untouched by the handoff
    "own_slot_role": None,
    # "idle" | "run" | "calib" | "camrun"
    "run_mode": "idle",
    # set on a real side change so the lidar thread re-searches instead of coasting on
    # a pose fitted against the old side
    "force_relocalise": False,
    # mirror of match_recorder.record_match for the debug page; the module flag is the
    # source of truth
    "record_match": False,
}

mode = "run" # "run" | "hsv"

_jpeg_cond = threading.Condition()
jpeg_bytes = None # latest rendered MJPEG frame
# notified each time the camera thread publishes a frame, so the render thread re-renders
# on arrival instead of spinning on a stale one
_frame_cond = threading.Condition()
# the running logger for --lidarlog / --capturelog / --motionlog, None otherwise. Every
# hook checks "is not None", so a normal run pays one comparison.
_lidar_log = None
_capture_log = None
_motion_log = None


def _apply_slot_state():
    """call under _lock after the Bluetooth role handoff changes slot_role; the buttons in
    main() manage run_mode themselves
    """
    if _state["run_mode"] != "run":
        _state["run_mode"] = "idle"
