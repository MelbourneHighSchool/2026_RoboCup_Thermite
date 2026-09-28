"""Ball-retention roller (dwibbler) control and possession sensing: roller stall, marker, and
mouth camera.
"""

import math
import time

import cv2
import numpy as np

# The Pi camera stack only exists on the Pi.
try:
    from picamera2 import Picamera2
except ImportError: # not on the Pi
    Picamera2 = None

import bot.diagnostics as diagnostics
import bot.state as state
from bot.diagnostics import _mark_health_t
from bot.dwibble_cam_calib import get_calib as _get_calib
from bot.hardware import Motor

# Ball capture: camera detection distance shared with the possession FSM.
capture_dist_mm = 200 # ball this close and centred (in_cone) = captured

# Dwibbler (ball-retention roller): spins to draw the ball in and hold it.
#
# It runs in speed mode (command mode 12, see Motor.__init__) because possession detection
# below compares measured roller speed against the command. Torque mode (constant grip,
# the better roller mode in principle) has no commanded speed to sag against, so switch
# only together with a replacement possession signal (a break-beam, or the mouth camera).
dwibble_speed = 1

# rim speed at 1800 rpm is about 2.64 m/s, so capture takes well under one control tick
# once the ball touches the roller: there is no spin-up window to buy lead time for
dwibble_on_dist_mm = 150.0 # spin up when the ball is this close

# Graduated grip while carrying: softer once held, full grip only when contested or
# turning.
dwibble_contest_dist_mm = 350.0 # enemy this close counts as contested
dwibble_turn_frac_full = 0.071 # |rot_speed| past this = "turning"
# grip while carrying uncontested and straight, as a fraction of the full hold. The old
# 0.15 let a braking or turning carry at the new base speed roll the ball off.
dwibble_straight_frac = 0.6
# [elecangle, sincos] for the dwibbler motor (motorcalib.py); motorcheck.py imports it
dwibble_calib = [1489797888, 1236]

# Possession from roller load (QDR): the roller is speed-controlled, so a seated ball
# drags measured speed below command. Senses a ball flush in the mouth, where the camera
# can't see it.
possess_speed_frac = 0.55 # measured < this x commanded = roller is loaded
possess_free_frac = 0.80 # measured >= this x commanded proves the reading means anything
possess_spinup_s = 0.30 # ignore readings this long after the roller turns on
possess_on_s = 0.12 # sustained sag needed to declare possession
possess_off_s = 0.40 # sustained free-spin needed to declare it lost
# how long a roller-only capture is believed after the camera last had the ball close
# (confirms, never announces from nothing)
possess_capture_window_s = 2.0
# EMA on the measured/commanded ratio: real captures chatter loaded/free every 1-2 ticks
possess_smooth_tau_s = 0.06

# The sag test needs a command floor too: below half the hold a real ball's sag is too
# small for the QDR to resolve. The straight-carry grip (0.6) clears it, so possession is
# sensed through a straight carry as well as a contested one.
possess_min_cmd_frac = 0.5

# Roller-arm marker: a sticker visible only once the arm pivots up under the ball. Off
# until the real sticker is fitted and tuned.
dwibble_mark_enabled = False
dwibble_mark_lower = np.array([140, 120, 80], dtype=np.uint8) # provisional
# magenta placeholder, distinct from the orange ball and the cyan/yellow goals
dwibble_mark_upper = np.array([160, 255, 255], dtype=np.uint8)
# marker pixels in the notch before believing the arm is raised
dwibble_mark_min_px = 3

# Second camera (Camera Module 3) aimed into the dwibbler mouth: a direct visual
# possession check. Unlike the marker it can assert possession on its own, since it is a
# direct sighting; it only ever pushes toward "loaded", never toward "empty". Enabled by
# diagnostics.dwibble_cam_enabled.
#
# The camera rides the dwibbler, so it swings back and tilts up once the ball touches the
# roller: pixel position isn't a fixed function of distance. Until the bench regression
# exists (dwibble_cam_calib) it is a possession indicator only, never a range or bearing.
dwibble_cam_index = 1 # Picamera2 camera_num; the Pi 5's second CSI port
dwibble_cam_res = (320, 240) # small and fast, no bearing/range math needed
# Starting point copied from the main camera's orange; this camera sees an enclosed mouth
# under its lighting, so retune against the real ball.
dwibble_cam_lower = np.array([6, 171, 13], dtype=np.uint8)
dwibble_cam_upper = np.array([20, 255, 255], dtype=np.uint8)
dwibble_cam_min_frac = 0.35 # fraction of the frame that must read ball-orange to call it seen
dwibble_cam_confirm_s = 0.08 # sustained sighting before trusting it (debounces a stray reflection)


class BallPossession:
    """hysteresis possession flag fed by the roller monitor and the mouth camera."""

    def __init__(self):
        """start assuming no ball and no working QDR reads yet."""
        self.has_ball = False
        self.available = False # QDR reads are working
        self._load_s = 0.0 # accumulated loaded evidence (s)
        self._free_s = 0.0 # accumulated free-spin evidence (s)
        self._last_t = None

    def update(self, t, loaded):
        """feed one sample: loaded = roller commanded on and speed sagging."""
        # two threads feed this, so a sample can arrive stamped before the last one
        dt = 0.0 if self._last_t is None else max(0.0, min(0.1, t - self._last_t))
        self._last_t = max(t, self._last_t or t)
        if loaded:
            self._load_s = min(possess_on_s, self._load_s + dt)
            self._free_s = max(0.0, self._free_s - dt)
        else:
            self._free_s = min(possess_off_s, self._free_s + dt)
            self._load_s = max(0.0, self._load_s - dt)
        if self._load_s >= possess_on_s:
            self.has_ball = True
        elif self._free_s >= possess_off_s:
            self.has_ball = False

    def reset(self):
        """clear the flag and both accumulators."""
        self.has_ball = False
        self._load_s = 0.0
        self._free_s = 0.0


_possession = BallPossession()

# The last command _set_dwibbler sent, which the monitor thread compares the QDR speed
# against.
_dwibble_cmd = 0
_dwibble_on_t = 0.0 # monotonic time the roller last switched on


def _set_dwibbler(on, speed=None):
    """spin the dwibbler at dwibble_speed (or an explicit `speed`, e.g. reversed to release) or
    stop it; every control loop goes through here.
    """
    global _dwibble_cmd, _dwibble_on_t
    if not on:
        cmd = 0
    else:
        cmd = speed if speed is not None else dwibble_speed
    # restart the spin-up window on off -> on or a reversal: both cross zero speed,
    # which reads exactly like a stall until the roller settles
    was_off_or_reversed = _dwibble_cmd == 0 or (cmd > 0) != (_dwibble_cmd > 0)
    if cmd != 0 and was_off_or_reversed:
        _dwibble_on_t = time.monotonic()
    _dwibble_cmd = cmd
    Motor.dwibble(cmd)


def _dwibble_carry_speed(rx, ry, enemies, rot_speed_cmd):
    """dwibble_speed for a carrying tick: full grip when turning hard or contested, else the
    straight-carry fraction.
    """
    if abs(rot_speed_cmd) >= dwibble_turn_frac_full:
        return dwibble_speed
    for e in (enemies or []):
        if math.hypot(e["x"] - rx, e["y"] - ry) <= dwibble_contest_dist_mm:
            return dwibble_speed
    return dwibble_speed * dwibble_straight_frac


def _dwibbler_monitor_thread():
    """poll the roller's measured speed at about 50 Hz and keep the possession flag current."""
    m = Motor.motors.get("dwibble")
    if m is None:
        return
    # latches on the first healthy free-spin and never clears: "can this roller reach
    # its command", not "is it loaded now"
    roller_ok = False
    # EMA state, cleared whenever the roller is off so a new command doesn't inherit
    # it
    smoothed_speed = None
    prev_sample_t = None
    while True:
        t = time.monotonic()
        cmd = _dwibble_cmd
        qdr = m.read_qdr()
        if qdr is None:
            _possession.available = False
        else:
            _, speed, _err1, _err2 = qdr
            # normalise the raw QDR speed to the -1..1 command scale
            speed = abs(speed) / Motor.MOTOR_MAX_RAW
            if cmd == 0:
                smoothed_speed = None
                prev_sample_t = None
            elif smoothed_speed is None:
                smoothed_speed = speed
                prev_sample_t = t
            else:
                dt = max(1e-3, t - prev_sample_t)
                alpha = 1.0 - math.exp(-dt / possess_smooth_tau_s)
                smoothed_speed = alpha * speed + (1.0 - alpha) * smoothed_speed
                prev_sample_t = t
            speed = smoothed_speed if smoothed_speed is not None else speed
            settled = cmd != 0 and t - _dwibble_on_t >= possess_spinup_s
            # trusted: settled, and commanded high enough for a sag to be
            # resolvable
            trusted = settled and abs(cmd) >= possess_min_cmd_frac * dwibble_speed
            if trusted and abs(speed) >= possess_free_frac * abs(cmd):
                if not roller_ok:
                    print("[dwibble] roller reaches its commanded speed, "
                          "stall possession detection is live", flush=True)
                roller_ok = True
            # an unproven roller reports nothing rather than a ball;
            # camera-only fallback
            _possession.available = roller_ok
            if trusted:
                # a ball loads the roller and sags its speed below command
                loaded = (roller_ok
                          and abs(speed) < possess_speed_frac * abs(cmd))
                # second vote from the roller-arm marker, when fitted
                if dwibble_mark_enabled:
                    with state._lock:
                        mark_seen = state._state["dwibble_mark_seen"]
                    loaded = loaded and mark_seen
                _possession.update(t, loaded)
            # else: untrusted sample (light grip, spinning up, or off): freeze
            # the flag rather than feed it a reading that can't tell loaded
            # from empty
        with state._lock:
            state._state["drib_has_ball"] = _possession.has_ball
        time.sleep(0.02)


def _dwibble_camera_thread():
    """run the second camera, aimed into the dwibbler mouth: a direct visual possession check
    published to _state["dwibble_cam_seen"].
    """
    if not diagnostics.dwibble_cam_enabled:
        return
    if Picamera2 is None:
        print("[dwibblecam] picamera2 not installed, disabled", flush=True)
        return
    try:
        picam2 = Picamera2(camera_num=dwibble_cam_index)
        config = picam2.create_video_configuration(
            main={"size": dwibble_cam_res, "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
    except Exception as e: # noqa: BLE001
        print(f"[dwibblecam] could not start (camera_num={dwibble_cam_index}): "
              f"{e}, possession stays QDR/marker-only", flush=True)
        return
    print(f"[dwibblecam] live at {dwibble_cam_res[0]}x{dwibble_cam_res[1]}, "
          f"camera_num={dwibble_cam_index}", flush=True)

    seen_s = 0.0
    last_t = None
    calib = _get_calib()
    while True:
        frame = picam2.capture_array("main")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, dwibble_cam_lower, dwibble_cam_upper)
        frac = float(mask.mean()) / 255.0

        # ball blob geometry, for the debug overlay and (once calibrated) distance
        h, w = mask.shape[:2]
        ball_dx = ball_dy = ball_r = None
        if frac >= 0.005:
            mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                     np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                (bx, by), br = cv2.minEnclosingCircle(c)
                ball_dx = bx - w / 2.0
                ball_dy = by - h / 2.0
                ball_r = br

        # calibrated distance: None until calib_points.json exists
        dist_mm = None
        if ball_dy is not None:
            est = calib.distance_mm(ball_dy, ball_r)
            if est is not None:
                dist_mm = est[0]

        t = time.monotonic()
        dt = 0.0 if last_t is None else min(0.1, t - last_t)
        last_t = t
        if frac >= dwibble_cam_min_frac:
            seen_s = min(dwibble_cam_confirm_s, seen_s + dt)
        else:
            seen_s = 0.0
        seen = seen_s >= dwibble_cam_confirm_s
        if seen:
            _possession.update(t, True)
        with state._lock:
            state._state["dwibble_cam_seen"] = seen
            state._state["dwibble_cam_frac"] = frac
            state._state["dwibble_cam_ball"] = (
                None if ball_dx is None else
                (ball_dx, ball_dy, ball_r, dist_mm))
            # raw frame + mask for the debug page's second-camera panel
            state._state["dwibble_cam_frame"] = frame
            state._state["dwibble_cam_mask"] = mask
        _mark_health_t("dwibble_cam")
        time.sleep(0.02)
