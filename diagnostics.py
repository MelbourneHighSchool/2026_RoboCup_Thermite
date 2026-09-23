"""Centralized health/fault tracking and the wobble diagnostic.

Every safety/localisation-relevant thread stamps a
"last good reading" monotonic timestamp via _mark_health_t; _health_thread
(started from main()) turns staleness into a single console-visible
per-subsystem fault line. The IMU gets one extra piece: a sustained (not
transient) fault latches a forced stop that survives until the operator
re-picks colour/role with the IMU healthy again (see bot.compass's
_compass_thread fault latch and _play_loop's gate).

The wobble diagnostic is a softer, purely-telemetry cousin of the sustained
IMU fault above: that one is for a genuinely DEAD IMU (no readings at all).
This is for a heading/pose estimate that IS producing readings but is
buzzing - confined to about one robot-body's extent, not really going
anywhere, yet still showing a real position/accel spike with sign flips -
distinguishing a genuine wobble (noisy heading fusion, a wheel chattering
against a wall, an unsteady gait) from a clean stop, which also sits still
but has no spike left. Diagnostic only: nothing here steers the robot, it
only names the pathology so a bench run's console/debug page says WHY the
pose looked jittery instead of leaving it to be guessed off a video. Fed
from _lidar_thread with each newly-published pose (see wobble_observe
below), not from _play_loop's own faster tick, so duplicate same-revolution
samples never masquerade as spikes.

Both bot.diagnostics and its NEW-this-session content (this whole module)
did not exist in the stale historical bot/ package; this module is new.

Land mine: `_wobble_last_t` is a module-level global rebound via `global`
inside wobble_reset()/wobble_observe(). Any OTHER module reading it must do
`import bot.diagnostics as diagnostics; diagnostics._wobble_last_t`, never
`from bot.diagnostics import _wobble_last_t` - the latter freezes a stale
reference (None, at import time) forever and will never see later updates.
"""

import collections
import math
import time

import bot.state as state
from bot.hardware import Motor


# how long a subsystem may go without a fresh reading before it's "stale"
health_stale_s   = 1.0
# consecutive seconds of IMU staleness before the forced-stop latches; a lone
# transient bus-read exception (already handled in _compass_thread, one
# missed ~10ms sample) recovers well under this and never latches anything.
imu_fault_hold_s = 1.0

_health   = {} # subsystem name -> last-reported status string, console-diffed by _report_health
_health_t = {
    "lidar":  None, # monotonic time of the last published (good) pose
    "camera": None, # monotonic time of the last published camera frame
    "dwibble_cam": None, # monotonic time of the last dwibble-cam frame (only while dwibble_cam_enabled)
    "imu":    None, # monotonic time of the last good BNO08x sample
}

# Not yet extracted into any bot.* module (belongs to the camera/vision
# config, a later stage) - kept here as the one place _health_thread needs
# it. A later vision-stage extraction should replace this with an import
# from wherever it ends up living, and this line should be deleted then.
dwibble_cam_enabled = False


# Wobble diagnostic
wobble_window_n     = 5    # (t, x, y) samples kept; the detector uses the last 4/5
wobble_settled_mms  = 120.0 # "not going anywhere" speed, every sample in the window
wobble_box_mm       = 120.0 # ... confined inside this bounding box, about one robot body
wobble_accel_mms2   = 900.0 # ... yet still spiking this hard
wobble_min_flips    = 2    # accel/velocity sign changes needed: buzzing, not one clean stop
wobble_report_gap_s = 1.0  # rate limit so a persistent wobble doesn't spam a line per tick

_wobble_hist   = collections.deque(maxlen=wobble_window_n) # (t, x, y)
_wobble_last_t = None # last time a wobble Finding was reported


def _wobble_sign_flips(vals, dead):
    """how many times a sequence changes sign, ignoring |v| <= dead."""
    flips, last = 0, 0
    for v in vals:
        s = 0 if abs(v) <= dead else (1 if v > 0 else -1)
        if s and last and s != last:
            flips += 1
        if s:
            last = s
    return flips


def detect_wobble(hist):
    """buzzing in place: slow, spatially confined, but still accelerating and
    reversing - distinct from a clean stop, which has no accel left. Pure:
    takes a short (t, x, y) history, returns a findings dict or None."""
    h = list(hist)[-wobble_window_n:]
    if len(h) < 4:
        return None
    xs = [s[1] for s in h]
    ys = [s[2] for s in h]
    box = max(max(xs) - min(xs), max(ys) - min(ys))
    if box > wobble_box_mm:
        return None

    vels = [] # (t_mid, vx, vy)
    for i in range(len(h) - 1):
        dt = h[i + 1][0] - h[i][0]
        if dt <= 1e-6:
            return None # duplicate timestamps, nothing to differentiate
        vels.append((0.5 * (h[i][0] + h[i + 1][0]),
                     (h[i + 1][1] - h[i][1]) / dt,
                     (h[i + 1][2] - h[i][2]) / dt))
    if any(math.hypot(v[1], v[2]) > wobble_settled_mms for v in vels):
        return None # still travelling; that is motion, not a wobble

    accels = []
    for i in range(len(vels) - 1):
        dt = vels[i + 1][0] - vels[i][0]
        if dt <= 1e-6:
            continue
        accels.append(((vels[i + 1][1] - vels[i][1]) / dt,
                       (vels[i + 1][2] - vels[i][2]) / dt))
    if not accels:
        return None
    peak = max(math.hypot(ax, ay) for ax, ay in accels)
    if peak < wobble_accel_mms2:
        return None # confined and slow with no accel left = it simply stopped

    # Reversals on whichever axis is doing the buzzing; with wobble_window_n=5
    # there are only 2 accel samples, so also count the velocity's own
    # reversals rather than demanding more accel samples than the window
    # can hold.
    fx  = _wobble_sign_flips([a[0] for a in accels], 0.25 * wobble_accel_mms2)
    fy  = _wobble_sign_flips([a[1] for a in accels], 0.25 * wobble_accel_mms2)
    fvx = _wobble_sign_flips([v[1] for v in vels], 0.2 * wobble_settled_mms)
    fvy = _wobble_sign_flips([v[2] for v in vels], 0.2 * wobble_settled_mms)
    flips = max(fx, fy, fvx, fvy)
    if flips < wobble_min_flips:
        return None
    return {"kind": "wobble", "severity": min(1.0, peak / (2.0 * wobble_accel_mms2)),
            "box_mm": box, "peak_accel_mms2": peak, "flips": flips}


def wobble_reset():
    """forget the wobble history and rate limiter (e.g. on a lidar re-localise - a fresh global-search fix is a discontinuous jump, not real motion, and would otherwise misread as a wobble/spike)."""
    global _wobble_last_t
    _wobble_hist.clear()
    _wobble_last_t = None


def wobble_observe(pose, t):
    """feed one newly-published pose ((x, y, hdg)) at monotonic time t;
    publishes a finding to _state["wobble"] (rate-limited to one report per
    wobble_report_gap_s) and clears it once the wobble stops. Safe to call
    every published pose."""
    global _wobble_last_t
    if pose is None:
        return
    _wobble_hist.append((float(t), float(pose[0]), float(pose[1])))
    finding = detect_wobble(_wobble_hist)
    if finding is None:
        return
    if _wobble_last_t is not None and t - _wobble_last_t < wobble_report_gap_s:
        return
    _wobble_last_t = t
    with state._lock:
        state._state["wobble"] = finding
    print(f"[diag] wobble sev={finding['severity']:.2f}"
         f" box={finding['box_mm']:.0f}mm accel={finding['peak_accel_mms2']:.0f}mm/s2",
         flush=True)


def _mark_health_t(name, t=None):
    """stamp `name` (a key of _health_t) as having produced a good reading at `t` (default: now)."""
    with state._lock:
        _health_t[name] = t if t is not None else time.monotonic()


def _report_health(name, status):
    """print only on change, and publish into _state["health"] for the debug page's /health endpoint."""
    with state._lock:
        changed = _health.get(name) != status
        _health[name] = status
        state._state["health"] = dict(_health)
    if changed:
        print(f"[health] {name}: {status}", flush=True)


def _health_thread():
    """poll every subsystem's staleness (lidar/camera/dwibble-cam/BT/motors) at ~5 Hz and report; the IMU reports its own fault directly from _compass_thread since it needs sub-poll-period latching."""
    while True:
        now = time.monotonic()
        with state._lock:
            t = dict(_health_t)
            solo = state._state["solo"]
            imu_fault = state._state["imu_fault"]

        lidar_t = t["lidar"]
        if lidar_t is None:
            _report_health("lidar", "no fix yet")
        elif now - lidar_t > health_stale_s:
            _report_health("lidar", f"STALE ({now - lidar_t:.1f}s since last fix)")
        else:
            _report_health("lidar", "ok")

        cam_t = t["camera"]
        if cam_t is None:
            _report_health("camera", "no frames yet")
        elif now - cam_t > health_stale_s:
            _report_health("camera", f"STALE ({now - cam_t:.1f}s since last frame)")
        else:
            _report_health("camera", "ok")

        if not dwibble_cam_enabled:
            _report_health("dwibble_cam", "disabled")
        else:
            dc_t = t["dwibble_cam"]
            if dc_t is None:
                _report_health("dwibble_cam", "no frames yet")
            elif now - dc_t > health_stale_s:
                _report_health("dwibble_cam", f"STALE ({now - dc_t:.1f}s since last frame)")
            else:
                _report_health("dwibble_cam", "ok")

        # BT team link: "solo" is a normal, expected mode when only one robot
        # is on/powered, not a fault - informational only, never escalated.
        _report_health("team_link", "solo (no teammate)" if solo else "ok")

        _report_health("imu", "FAULT (no reading)" if imu_fault else "ok")

        try:
            motor_ok = any(m is not None and m.read_qdr() is not None
                          for m in Motor.motors.values())
            _report_health("motors", "ok" if motor_ok else "NO QDR FEEDBACK")
        except Exception as e: # noqa: BLE001, health polling itself must never crash the robot
            _report_health("motors", f"health check error: {e}")

        time.sleep(0.2)
