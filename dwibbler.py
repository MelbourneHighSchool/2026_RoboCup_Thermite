"""Ball-retention roller (dwibbler, i2c 29) control and possession sensing.

BallPossession is a hysteresis flag fed by two independent evidence sources:
the dwibbler-motor QDR sag monitor (_dwibbler_monitor_thread, the primary
signal, roller load implies a ball seated against it) and an optional second
camera aimed into the mouth (_dwibble_camera_thread, off by default until
fitted, can only push possession toward True, never toward False - see its
own docstring).

Land mine (same shape as bot.odometry's _wheel_odom): _dwibble_cmd/
_dwibble_on_t are module-level globals written by _set_dwibbler (the single
chokepoint every control loop goes through) and read by
_dwibbler_monitor_thread from a different thread. Anything importing this
module for those must not rebind them at import time; use the module
attribute (e.g. `bot.dwibbler._dwibble_cmd`), not a frozen `from` import.
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

import bot.state as state
from bot.diagnostics import _mark_health_t
from bot.hardware import Motor

# Ball capture: camera detection radius/timing shared with the possession FSM.
capture_dist_mm   = 200 # ball this close and centred (in_cone) = captured

# Dwibbler (ball-retention roller, i2c 29): spins to draw the ball in and hold it.
dwibble_speed      = 1

# dwibble_on_dist_mm: rim speed at 1800 rpm is about 2.64 m/s, fast enough that capture happens well under one control tick once the ball touches the roller, no "spin-up and settle" window to buy lead time for.
dwibble_on_dist_mm = 150.0 # spin up when the ball is this close

# Graduated grip while carrying: the roller runs softer once held, full grip only when contested or turning hard.
dwibble_contest_dist_mm = 350.0 # enemy this close counts as contested
dwibble_turn_frac_full  = 0.071 # |rot_speed| past this = "turning"
# grip fraction while carrying uncontested straight, loose on purpose, ball can still be
# stolen
dwibble_straight_frac   = 0.15
# [elecangle, sincos] for the dwibbler, motorcalib.py on i2c 29.
# motorcheck.py uses the same pair for the same motor/address; update both
# together if either changes.
dwibble_calib      = [1489797888, 1236]

# Ball possession via dwibbler-motor load (rotom QDR): roller is speed-controlled, so a ball seated against it drags measured speed below command, senses possession where the camera can't see a ball flush in the mouth.
possess_speed_frac = 0.55 # measured < this x commanded = roller is loaded
possess_free_frac  = 0.80 # measured >= this x commanded proves the reading means anything
possess_spinup_s   = 0.30 # ignore readings this long after the roller turns on
possess_on_s       = 0.12 # sustained sag needed to declare possession
possess_off_s      = 0.40 # sustained free-spin needed to declare it lost
possess_capture_window_s = 2.0 # how long a dwibbler-only capture is
                            # believed after the camera last had the ball
                            # close, confirms, never announces from nothing
# EMA on the measured/commanded ratio before thresholding: real captures chatter loaded/free every 1-2 ticks, damped before the on/off timers see it.
possess_smooth_tau_s = 0.06

# Sag test also needs a command floor: dwibble_speed*straight_frac (0.045) never sags even with a real ball, too small for QDR to resolve.
possess_min_cmd_frac = 0.5

# Dwibbler-raised visual marker (sec 3.16): a sticker on the roller arm, visible only once the arm pivots up under a ball's weight, off by default until fitted and tuned.
dwibble_mark_enabled = False
dwibble_mark_lower  = np.array([140, 120, 80],  dtype=np.uint8) # Provisional
# a magenta/pink placeholder distinct from the orange ball and the cyan/yellow goal markers, picked before the real sticker exists.
dwibble_mark_upper  = np.array([160, 255, 255], dtype=np.uint8)
# marker-coloured pixels in the notch before believing the arm is actually raised, same
# idiom as min_orange_px
dwibble_mark_min_px = 3

# Second camera (Camera Module 3) aimed straight into the dwibbler mouth: a
# direct visual possession check, off by default until it's actually mounted
# and this section bench-retuned. Unlike dwibble_mark_enabled's role as a
# QDR-sag confirmatory AND, this one can assert possession on its own, since
# it's a direct sighting rather than an inferred proxy; see
# _dwibble_camera_thread's own comment for why it only ever pushes toward
# "loaded", never toward "empty".
dwibble_cam_enabled   = False
dwibble_cam_index     = 1 # Picamera2 camera_num; the Pi 5's second CSI port
dwibble_cam_res       = (320, 240) # small and fast, no bearing/range math needed here
# Starting point only, copied from orange_lower/upper: this camera has its own
# exposure/lighting (an enclosed mouth cavity, likely needs its own LED), re-tune
# against the real ball before trusting it.
dwibble_cam_lower     = np.array([6, 171, 13],   dtype=np.uint8)
dwibble_cam_upper     = np.array([20, 255, 255], dtype=np.uint8)
dwibble_cam_min_frac  = 0.35 # fraction of the frame that must read ball-orange to call it "seen"
dwibble_cam_confirm_s = 0.08 # sustained sighting needed before trusting it, debounces a stray reflection


class BallPossession:
    """hysteresis possession flag fed by the dwibbler monitor thread (accumulate/decay has_ball pattern)."""

    def __init__(self):
        """start assuming no ball and no working QDR reads yet."""
        self.has_ball  = False
        self.available = False # QDR reads are actually working
        self._load_s   = 0.0 # accumulated loaded evidence (s)
        self._free_s   = 0.0 # accumulated free-spin evidence (s)
        self._last_t   = None

    def update(self, t, loaded):
        """feed one sample: loaded = roller commanded on and speed sagging."""
        dt = 0.0 if self._last_t is None else min(0.1, t - self._last_t)
        self._last_t = t
        if loaded:
            self._load_s = min(possess_on_s,  self._load_s + dt)
            self._free_s = max(0.0, self._free_s - dt)
        else:
            self._free_s = min(possess_off_s, self._free_s + dt)
            self._load_s = max(0.0, self._load_s - dt)
        if self._load_s >= possess_on_s:
            self.has_ball = True
        elif self._free_s >= possess_off_s:
            self.has_ball = False

    def reset(self):
        """clear the flag and both accumulators (e.g. on losing/re-seeking the ball)."""
        self.has_ball = False
        self._load_s = 0.0
        self._free_s = 0.0


_possession = BallPossession()

# Written by _set_dwibbler (the single dwibbler chokepoint), read by the
# monitor thread to know what command it should compare the QDR speed against.
_dwibble_cmd    = 0
_dwibble_on_t   = 0.0 # monotonic time the roller was last switched on


def _set_dwibbler(on, speed=None):
    """spin the dwibbler at dwibble_speed (or an explicit `speed`, e.g. reversed to release a pass) or stop it, the single chokepoint every control loop goes through."""
    global _dwibble_cmd, _dwibble_on_t
    if not on:
        cmd = 0
    else:
        cmd = speed if speed is not None else dwibble_speed
    # Restart the spin-up window on an off -> on transition, or a direction reversal (e.g. the pass eject), both have to cross zero speed and settle again, which reads exactly like a stall until they have.
    was_off_or_reversed = _dwibble_cmd == 0 or (cmd > 0) != (_dwibble_cmd > 0)
    if cmd != 0 and was_off_or_reversed:
        _dwibble_on_t = time.monotonic()
    _dwibble_cmd = cmd
    Motor.dwibble(cmd)


def _dwibble_carry_speed(rx, ry, enemies, rot_speed_cmd):
    """dwibble_speed fraction for a carrying tick (has_ball / passing / the goalie's charge), not the capture approach, which stays at full dwibble_speed regardless (see that constant's own comment)."""
    if abs(rot_speed_cmd) >= dwibble_turn_frac_full:
        return dwibble_speed
    for e in (enemies or []):
        if math.hypot(e["x"] - rx, e["y"] - ry) <= dwibble_contest_dist_mm:
            return dwibble_speed
    return dwibble_speed * dwibble_straight_frac


def _dwibbler_monitor_thread():
    """poll the dwibbler's measured speed at about 50 Hz and keep the possession score current."""
    m = Motor.motors.get("dwibble")
    if m is None:
        return
    # Latches on the first healthy free-spin and never clears: it answers "is
    # this roller capable of reaching its command", not "is it loaded now".
    roller_ok = False
    # EMA state (possess_smooth_tau_s), cleared whenever the roller is off so a fresh
    # command doesn't inherit a stale value
    smoothed_speed = None
    prev_sample_t  = None
    while True:
        t   = time.monotonic()
        cmd = _dwibble_cmd
        qdr = m.read_qdr()
        if qdr is None:
            _possession.available = False
        else:
            _, speed, _err1, _err2 = qdr
            # read_qdr's speed is raw driver units; cmd is the -1..1 fraction
            # _set_dwibbler last asked for.  Normalise the measurement so the
            # two are comparable.
            speed = abs(speed) / Motor.MOTOR_MAX_RAW
            if cmd == 0:
                smoothed_speed = None
                prev_sample_t  = None
            elif smoothed_speed is None:
                smoothed_speed = speed
                prev_sample_t  = t
            else:
                dt    = max(1e-3, t - prev_sample_t)
                alpha = 1.0 - math.exp(-dt / possess_smooth_tau_s)
                smoothed_speed = alpha * speed + (1.0 - alpha) * smoothed_speed
                prev_sample_t  = t
            speed   = smoothed_speed if smoothed_speed is not None else speed
            settled = cmd != 0 and t - _dwibble_on_t >= possess_spinup_s
            # trusted: settled and commanded near enough to full dwibble_speed
            # for a sag to actually be resolvable, see possess_min_cmd_frac's
            # own comment (real field data: the eased/straight grip level
            # never sags at all, proving nothing either way).
            trusted = settled and abs(cmd) >= possess_min_cmd_frac * dwibble_speed
            if trusted and abs(speed) >= possess_free_frac * abs(cmd):
                if not roller_ok:
                    print("[dwibble] roller reaches its commanded speed, "
                          "stall possession detection is live", flush=True)
                roller_ok = True
            # An unproven roller reports nothing rather than reporting a ball:
            # available False is the documented camera-only fallback.
            _possession.available = roller_ok
            if trusted:
                # A ball loads the roller and sags its measured speed below
                # command (possess_speed_frac's own comment: "< this x
                # commanded = loaded"). This used to compare with >=, the
                # opposite sense, which reported "has ball" while the roller
                # was spinning freely and "no ball" while it was actually
                # loaded - backwards possession detection on real hardware.
                loaded = (roller_ok
                          and abs(speed) < possess_speed_frac * abs(cmd))
                # Second vote from the dwibble-mark camera sighting (double authentication), published by _camera_thread each frame from the same mouth-notch wedge the ball-in-mouth detection uses (dwibble_mark_enabled's own comment).
                if dwibble_mark_enabled:
                    with state._lock:
                        mark_seen = state._state["dwibble_mark_seen"]
                    loaded = loaded and mark_seen
                _possession.update(t, loaded)
            # else: untrusted sample (eased grip, not yet settled, or off),
            # freeze has_ball/the accumulators rather than feed them a
            # reading that can't distinguish loaded from empty.
        with state._lock:
            state._state["drib_has_ball"] = _possession.has_ball
        time.sleep(0.02)


def _dwibble_camera_thread():
    """own the second Picamera2 (dwibble_cam_index) aimed straight into the dwibbler mouth: a direct visual possession check, published to _state["dwibble_cam_seen"].

    Only ever pushes _possession toward "loaded", never toward "empty": a
    confirmed sighting calls _possession.update(t, True) directly, same
    accumulator the QDR monitor thread feeds, but a non-sighting doesn't
    call update() at all. The QDR thread stays the sole source of decay, so
    the ball briefly rotating out of this camera's narrow view mid-carry
    can never falsely clear a real QDR-confirmed carry; this thread can
    only ever help confirm possession sooner; not un-confirm it.
    """
    if not dwibble_cam_enabled:
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
    while True:
        frame = picam2.capture_array("main")
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(hsv, dwibble_cam_lower, dwibble_cam_upper)
        frac  = float(mask.mean()) / 255.0

        t  = time.monotonic()
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
        _mark_health_t("dwibble_cam")
        time.sleep(0.02)
