"""BNO08x IMU: heading-delta assist + linear-accel collision gate (sec
3.18/3.18a). Lidar ICP stays the source of truth for (x, y, heading); the
IMU fills the gap when lidar is weakest (shoved, fast spins, re-search).
On-chip game rotation vector for heading only (no magnetometer, pure
noise next to 4 BLDCs). No hardware -> lidar-only, collision gate never
trips. imu.py (repo root) is an unrelated standalone bench script, not
used here.

Extracted verbatim from mainrunbot1.py's Compass class + IMU-fusion
helpers + _compass_thread (which lives much further down the monolith,
near the field-calibration section, but is gathered here since it is the
thread that owns this module's state).
"""

import collections
import math
import time

from bot.field import wrap_deg as _wrap_deg
import bot.state as state
from bot.diagnostics import imu_fault_hold_s, _mark_health_t, _report_health


class Compass:
    """wraps a BNO08x IMU for the heading-delta assist and collision-accel gate described above."""

    def __init__(self, address=0x4A):
        """connect to the BNO08x at `address` and enable the game rotation vector and linear acceleration reports."""
        import board
        from adafruit_bno08x.i2c import BNO08X_I2C
        from adafruit_bno08x import (BNO_REPORT_GAME_ROTATION_VECTOR,
                                     BNO_REPORT_LINEAR_ACCELERATION)

        self.address = address
        self.bno = BNO08X_I2C(board.I2C(), address=address)
        time.sleep(0.1)
        self.bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)
        self.bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
        time.sleep(0.1)

    @classmethod
    def try_init(cls, address=0x4A):
        """Compass, or None if the library or the sensor isn't there."""
        try:
            c = cls(address)
            c.read() # prove a real sample comes back
            print(f"[imu] BNO08x online at {hex(address)}", flush=True)
            return c
        except Exception as e: # noqa: BLE001, any failure = no IMU
            print(f"[imu] no BNO08x at {hex(address)}: {e}", flush=True)
            return None

    def read(self):
        """current yaw in degrees, clockwise-positive to match the robot's compass convention (pose heading: 0 deg = +y, cw+)."""
        x, y, z, w = self.bno.game_quaternion
        yaw = math.degrees(math.atan2(2.0 * (w * z + x * y),
                                      1.0 - 2.0 * (y * y + z * z)))
        return -yaw # BNO yaw is ccw+; robot frame is cw+

    def read_accel(self):
        """linear acceleration magnitude, m/s^2, gravity already removed by the chip's own fusion, see COLLISION_ACCEL_G."""
        ax, ay, az = self.bno.linear_acceleration
        return math.sqrt(ax * ax + ay * ay + az * az)

# Fold the IMU's heading delta into the lidar ICP prior each scan.  The
# BNO08x is confirmed working on the robot, so fusion is live; the compass
# thread still publishes _state["imu_heading"] for telemetry either way.
imu_fusion_enabled = True

# Timestamped heading history, written by _compass_thread at about 100 Hz and read by the lidar thread to deskew a scan (see _deskew).
imu_hist_s = 1.0
_imu_hist  = collections.deque() # (monotonic_t, heading_deg), under _lock


def _imu_heading_at(t):
    """IMU heading (deg) at monotonic time `t`, linearly interpolated between samples, or None if the history does not span it."""
    if len(_imu_hist) < 2 or not (_imu_hist[0][0] <= t <= _imu_hist[-1][0]):
        return None
    prev_t, prev_h = _imu_hist[0]
    for cur_t, cur_h in _imu_hist:
        if cur_t >= t:
            span = cur_t - prev_t
            if span <= 1e-9:
                return cur_h
            f = (t - prev_t) / span
            # Interpolate along the short way round, so a sample pair either
            # side of the +-180 wrap doesn't read as a near-full turn.
            return prev_h + f * _wrap_deg(cur_h - prev_h)
        prev_t, prev_h = cur_t, cur_h
    return prev_h


# Carrying the heading between lidar revolutions: pose updates at about 10 Hz but the control loop runs at 50 Hz, and that staleness caps turn_gain (sec 3.1).
heading_imu_carry = True # off -> controllers steer on the raw lidar pose
# a bigger delta than this means a stale reference, not a turn; fall back to the pose
heading_carry_max_deg = 90.0


def _fused_heading(pose, pose_imu, imu_now):
    """`pose` with its heading advanced by the IMU delta since the revolution that produced it."""
    if (not heading_imu_carry or pose is None
            or pose_imu is None or imu_now is None):
        return pose
    d = _wrap_deg(imu_now - pose_imu)
    if abs(d) > heading_carry_max_deg:
        return pose
    return (pose[0], pose[1], pose[2] + d)


def _imu_turn_between(t0, t1):
    """how far the robot actually turned (deg, cw+) between two monotonic times, from the IMU history, or None if it doesn't cover the window."""
    with state._lock:
        h0 = _imu_heading_at(t0)
        h1 = _imu_heading_at(t1)
    if h0 is None or h1 is None:
        return None
    return _wrap_deg(h1 - h0)


# BNO08x linear-accel magnitude past this is a shove/wall hit, not commanded driving. Motor.drive
# has no software accel/decel ramp any more (removed so direction changes are instant, not eased),
# so a hard commanded cut can genuinely swing real IMU-measured accel much higher than the old
# ramp's ~4.5g worst case - this is raised well above that with margin as an unverified guess,
# NEEDS bench/field retuning against real IMU logs of a deliberate hard direction change before
# trusting it not to false-trigger on ordinary play.
COLLISION_ACCEL_G  = 14.0


def _compass_thread():
    """poll the BNO08x at about 100 Hz into _state["imu_heading"] (relative yaw, deg cw+) and _state["collision_t"] (see COLLISION_ACCEL_G). Also owns the IMU health/fault latch (see imu_fault_hold_s / "imu_fault" / "imu_pause_latched" in _state): a lone transient bus-read exception (the try/except right below) just holds the last good value as always, but if NO good reading comes back for imu_fault_hold_s straight - a sustained fault, e.g. the sensor genuinely dropping off the bus mid-match - that's escalated into a forced-stop latch _play_loop enforces, not silently held forever. A robot that never had a BNO08x at all (c is None) is a normal, already-supported lidar-only configuration, not a fault: it never enters this loop, so it can never latch a pause."""
    c = Compass.try_init(0x4A) or Compass.try_init(0x4B) # SA0 low or high
    if c is None:
        _report_health("imu", "disabled (no BNO08x)")
        return
    fail_since = None # monotonic time the current run of bad reads started, or None
    while True:
        try:
            h = c.read()
            accel = c.read_accel()
        except Exception: # transient bus error, hold last value
            h = accel = None
        now = time.monotonic()
        if h is not None:
            with state._lock:
                state._state["imu_heading"] = h
                # Timestamped history, for deskewing a lidar revolution against the rotation that actually happened during it (see _imu_turn_between).
                _imu_hist.append((now, h))
                while _imu_hist and now - _imu_hist[0][0] > imu_hist_s:
                    _imu_hist.popleft()
            _mark_health_t("imu", now)
            fail_since = None
        elif fail_since is None:
            fail_since = now
        if accel is not None and accel > COLLISION_ACCEL_G * 9.80665:
            with state._lock:
                state._state["collision_t"] = now

        faulted = fail_since is not None and (now - fail_since) >= imu_fault_hold_s
        with state._lock:
            state._state["imu_fault"] = faulted
            if faulted:
                state._state["imu_pause_latched"] = True
        _report_health("imu", "FAULT (no reading)" if faulted else "ok")
        time.sleep(0.01)
