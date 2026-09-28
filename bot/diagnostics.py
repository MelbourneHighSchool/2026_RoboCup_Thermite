"""Centralized health/fault tracking and the wobble diagnostic."""

import collections
import math
import time

import bot.state as state
from bot.hardware import Motor


# how long a subsystem may go without a fresh reading before it's "stale"
health_stale_s = 1.0
# seconds of IMU staleness before the forced stop latches; one missed ~10 ms sample
# recovers well inside this
imu_fault_hold_s = 1.0

_health = {} # subsystem -> last reported status, diffed so the console only prints changes
_health_t = {
    "lidar": None, # monotonic time of the last published (good) pose
    "camera": None, # monotonic time of the last published camera frame
    # monotonic time of the last dwibble-cam frame (only while dwibble_cam_enabled)
    "dwibble_cam": None,
    "imu": None, # monotonic time of the last good BNO08x sample
}

# Second camera (Camera Module 3 in the dwibbler mouth) fitted. Lives here because the
# health thread is its only reader outside the camera thread.
dwibble_cam_enabled = True


# Wobble diagnostic: buzzing in place, telemetry only.
wobble_window_n = 5 # (t, x, y) samples kept; the detector uses the last 4/5
wobble_settled_mms = 120.0 # "not going anywhere" speed, every sample in the window
wobble_box_mm = 120.0 # ... confined inside this bounding box, about one robot body
wobble_accel_mms2 = 900.0 # ... yet still spiking this hard
wobble_min_flips = 2 # accel/velocity sign changes needed: buzzing, not one clean stop
wobble_report_gap_s = 1.0 # rate limit so a persistent wobble doesn't spam a line per tick

_wobble_hist = collections.deque(maxlen=wobble_window_n) # (t, x, y)
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
    """buzzing in place: slow and confined, yet still accelerating and reversing (a clean stop
    has no accel left). Pure: takes a short (t, x, y) history, returns a finding dict or
    None.
    """
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

    # Count reversals on whichever axis is buzzing. A 5-sample window only yields 2
    # accel samples, so velocity reversals count too.
    fx = _wobble_sign_flips([a[0] for a in accels], 0.25 * wobble_accel_mms2)
    fy = _wobble_sign_flips([a[1] for a in accels], 0.25 * wobble_accel_mms2)
    fvx = _wobble_sign_flips([v[1] for v in vels], 0.2 * wobble_settled_mms)
    fvy = _wobble_sign_flips([v[2] for v in vels], 0.2 * wobble_settled_mms)
    flips = max(fx, fy, fvx, fvy)
    if flips < wobble_min_flips:
        return None
    return {"kind": "wobble", "severity": min(1.0, peak / (2.0 * wobble_accel_mms2)),
            "box_mm": box, "peak_accel_mms2": peak, "flips": flips}


def wobble_reset():
    """forget the history and rate limiter, e.g. after a re-localise (a fresh global fix is a
    jump, not motion).
    """
    global _wobble_last_t
    _wobble_hist.clear()
    _wobble_last_t = None


def wobble_observe(pose, t):
    """feed one published pose (x, y, hdg) at monotonic t; publishes a finding to
    _state["wobble"], at most one per wobble_report_gap_s.
    """
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
    """stamp `name` (a _health_t key) as having produced a good reading at `t` (default: now)."""
    with state._lock:
        _health_t[name] = t if t is not None else time.monotonic()


def _report_health(name, status):
    """print only on change, and publish into _state["health"] for the debug page."""
    with state._lock:
        changed = _health.get(name) != status
        _health[name] = status
        state._state["health"] = dict(_health)
    if changed:
        print(f"[health] {name}: {status}", flush=True)


def _health_thread():
    """poll every subsystem's staleness at about 5 Hz and report it. The IMU fault latch itself
    lives in the compass thread, which needs finer timing than this poll.
    """
    while True:
        now = time.monotonic()
        with state._lock:
            t = dict(_health_t)
            solo = state._state["solo"]
            imu_fault = state._state["imu_fault"]
            # the compass thread reports "disabled" once when no BNO08x answers; keep that
            imu_absent = _health.get("imu", "").startswith("disabled")

        lidar_t = t["lidar"]
        if lidar_t is None:
            _report_health("lidar", "no fix yet")
        elif now - lidar_t > health_stale_s:
            _report_health("lidar", f"stale ({now - lidar_t:.1f}s since last fix)")
        else:
            _report_health("lidar", "ok")

        cam_t = t["camera"]
        if cam_t is None:
            _report_health("camera", "no frames yet")
        elif now - cam_t > health_stale_s:
            _report_health("camera", f"stale ({now - cam_t:.1f}s since last frame)")
        else:
            _report_health("camera", "ok")

        if not dwibble_cam_enabled:
            _report_health("dwibble_cam", "disabled")
        else:
            dc_t = t["dwibble_cam"]
            if dc_t is None:
                _report_health("dwibble_cam", "no frames yet")
            elif now - dc_t > health_stale_s:
                _report_health("dwibble_cam", f"stale ({now - dc_t:.1f}s since last frame)")
            else:
                _report_health("dwibble_cam", "ok")

        # solo is a normal mode with one robot on, not a fault: informational only
        _report_health("team_link", "solo (no teammate)" if solo else "ok")

        if not imu_absent:
            _report_health("imu", "fault (no reading)" if imu_fault else "ok")

        try:
            motor_ok = any(m is not None and m.read_qdr() is not None
                          for m in Motor.motors.values())
            _report_health("motors", "ok" if motor_ok else "no QDR feedback")
        except Exception as e: # noqa: BLE001 (health polling must never crash the robot)
            _report_health("motors", f"health check error: {e}")

        time.sleep(0.2)
