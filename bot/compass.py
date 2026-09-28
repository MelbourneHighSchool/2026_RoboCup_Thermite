"""BNO08x IMU: heading-delta assist and linear-accel collision gate.

Lidar stays the source of truth for (x, y, heading); the IMU fills the gaps where lidar is
weakest (shoves, fast spins, re-search). Game rotation vector only, no magnetometer (pure
noise next to four BLDCs). No IMU fitted -> lidar-only, and the collision gate never trips.
"""

import collections
import math
import time

from bot.field import wrap_deg as _wrap_deg
import bot.state as state
from bot.diagnostics import imu_fault_hold_s, _mark_health_t, _report_health


class Compass:
    """wraps a BNO08x for the heading assist and the collision gate."""

    def __init__(self, address=0x4A):
        """connect to the BNO08x at `address` and enable game rotation + linear acceleration
        reports.
        """
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
        """current yaw in degrees, clockwise-positive to match pose heading (0 deg = +y, cw+)."""
        x, y, z, w = self.bno.game_quaternion
        yaw = math.degrees(math.atan2(2.0 * (w * z + x * y),
                                      1.0 - 2.0 * (y * y + z * z)))
        return -yaw # BNO yaw is ccw+; robot frame is cw+

    def read_accel(self):
        """linear acceleration magnitude, m/s^2, gravity already removed by the chip."""
        ax, ay, az = self.bno.linear_acceleration
        return math.sqrt(ax * ax + ay * ay + az * az)

# Fold the IMU heading delta into each scan's localisation prior. The compass thread
# publishes imu_heading for telemetry either way.
imu_fusion_enabled = True

# Timestamped heading history (about 100 Hz), read by the lidar thread to deskew a scan.
imu_hist_s = 1.0
_imu_hist = collections.deque() # (monotonic_t, heading_deg), under _lock
# A query up to this far past the newest sample reads that sample. The lidar thread asks
# for the turn up to a scan's timestamp, usually within a millisecond of it and before
# the next 100 Hz sample exists; without the hold the deskew almost always fell back to
# the carried estimate instead of the IMU.
imu_hist_hold_s = 0.02


def _imu_heading_at(t):
    """IMU heading (deg) at monotonic time `t`, interpolated, or None if the history doesn't
    span it (allowing imu_hist_hold_s past the newest sample).
    """
    if len(_imu_hist) < 2 or t < _imu_hist[0][0]:
        return None
    newest_t, newest_h = _imu_hist[-1]
    if t > newest_t:
        return newest_h if t - newest_t <= imu_hist_hold_s else None
    prev_t, prev_h = _imu_hist[0]
    for cur_t, cur_h in _imu_hist:
        if cur_t >= t:
            span = cur_t - prev_t
            if span <= 1e-9:
                return cur_h
            f = (t - prev_t) / span
            # the short way round, so a pair either side of +-180 doesn't read
            # as a near-full turn
            return prev_h + f * _wrap_deg(cur_h - prev_h)
        prev_t, prev_h = cur_t, cur_h
    return prev_h


# Pose updates at about 10 Hz but the control loop runs at 50 Hz, so controllers carry the
# heading forward with the IMU between revolutions.
heading_imu_carry = True # off -> controllers steer on the raw lidar pose
# a bigger delta than this means a stale baseline, not a turn: fall back to the pose
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
    """how far the robot turned (deg, cw+) between two monotonic times, or None if not covered."""
    with state._lock:
        h0 = _imu_heading_at(t0)
        h1 = _imu_heading_at(t1)
    if h0 is None or h1 is None:
        return None
    return _wrap_deg(h1 - h0)


# Linear accel past this is a shove or wall hit, not driving. Motor.drive has no accel
# ramp, so a hard commanded direction change can spike well past the old ~4.5 g; 14 g is
# an unverified guess with margin. Retune against IMU logs of a deliberate hard reversal
# before trusting it.
COLLISION_ACCEL_G = 14.0


def _compass_thread():
    """poll the BNO08x at about 100 Hz into imu_heading and collision_t, and own the IMU fault
    latch.

    A lone bad read just holds the last value. No good reading for imu_fault_hold_s straight
    (the sensor dropping off the bus mid-match) sets imu_fault and the imu_pause_latched
    forced stop. A robot with no BNO08x at all is a normal lidar-only setup: it never enters
    the loop, so it never latches.
    """
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
                # history for deskewing a revolution against the rotation
                # that actually happened during it
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
        _report_health("imu", "fault (no reading)" if faulted else "ok")
        time.sleep(0.01)
