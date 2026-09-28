"""Wheel odometry and the wheel-vs-lidar slip diagnostic."""

import math
import time

import bot.compass as _compass
from bot.compass import heading_carry_max_deg
from bot.field import wrap_deg as _wrap_deg
import bot.state as state
from bot.hardware import Motor

# Drive wheel size, needed here for the QDR-to-mm conversion (motion.py carries its own
# copy for the speed model; change both if the wheels change).
wheel_diameter_mm = 50.0

# WheelOdometry: dead-reckoned translation between lidar revolutions from the four drive
# wheels' own QDR position readout.
POS_RAW_PER_OUTPUT_REV = Motor.RPM_TO_RAW * 60 # raw position units per output wheel revolution
wheel_odom_enabled = True


def _qdr_step(new, old):
    """change in a 32-bit QDR position counter, taken modulo 2^32 so a wrap (or a wheel
    reversing past zero) reads as the small step it is, not a 4-billion-unit jump.
    """
    return (new - old + 2 ** 31) % 2 ** 32 - 2 ** 31


class WheelOdometry:
    """dead-reckoned (forward_mm, right_mm) since the last poll(), from the drive wheels' QDR
    positions.
    """
    NAMES = ("nw", "ne", "sw", "se")

    def __init__(self):
        # name -> last raw QDR position, in drive()'s sign convention (see poll())
        self._raw = None
        self.t = None # monotonic time of that reading

    def poll(self, now):
        """(forward_mm, right_mm) since the last call, or None on a read failure or the first
        call.
        """
        raw = {}
        for n in self.NAMES:
            m = Motor.motors.get(n)
            qdr = m.read_qdr() if m is not None else None
            if qdr is None:
                return None
            # undo Motor.drive's send-time polarity flip so this matches
            # drive()'s own convention
            raw[n] = -qdr[0] if n in ("nw", "ne") else qdr[0]
        prev, self._raw, self.t = self._raw, raw, now
        if prev is None:
            return None
        mm = {n: _qdr_step(raw[n], prev[n]) / POS_RAW_PER_OUTPUT_REV
                 * math.pi * wheel_diameter_mm
              for n in self.NAMES}
        # sqrt2 because vmax_full_cmd_mms sums the four wheels' rim speed the same
        # way
        fwd = math.sqrt(2.0) * (mm["nw"] - mm["ne"] + mm["se"] - mm["sw"]) / 4.0
        right = math.sqrt(2.0) * (mm["nw"] + mm["ne"] + mm["sw"] + mm["se"]) / 4.0
        return fwd, right


_wheel_odom = None # WheelOdometry(), set up in main()

def _wheel_odom_delta_mm(pose, now):
    """this revolution's measured displacement in field-frame mm, rotated by pose's heading,
    for the lidar thread's deskew; None on a read failure or a collision in the interval.
    """
    if _wheel_odom is None:
        return None
    since = _wheel_odom.t
    fwd_right = _wheel_odom.poll(now)
    if fwd_right is None:
        return None
    with state._lock:
        collided = (since is not None and state._state["collision_t"] is not None
                   and since < state._state["collision_t"] <= now)
    if collided:
        return None
    fwd, right = fwd_right
    h = math.radians(pose[2])
    ch, sh = math.cos(h), math.sin(h)
    return ch * right + sh * fwd, ch * fwd - sh * right


# Wheel-slip cross-check: compare what the wheels claim against what the lidar saw, per
# revolution. Diagnostic only.
wheel_slip_min_speed_mms = 50.0 # below this, noise dominates the ratio: always trust
wheel_slip_zero_ratio = 0.5 # a mismatch this large (as a fraction of wheel speed) -> trust 0.0
wheel_slip_smooth = 0.3 # EMA smoothing on both speeds before comparing


class WheelSlipMonitor:
    """EMA-smoothed wheel-vs-lidar speed cross-check, fed once per lidar revolution."""

    def __init__(self):
        self.wheel_speed_mms = 0.0
        self.lidar_speed_mms = 0.0

    def reset(self):
        """forget the running speed estimate (e.g. on a re-localise)."""
        self.wheel_speed_mms = 0.0
        self.lidar_speed_mms = 0.0

    def update(self, wheel_disp_mm, lidar_disp_mm, dt):
        """feed one revolution's (wheel_disp_mm, lidar_disp_mm, dt); returns trust()."""
        if dt > 1e-6:
            a = wheel_slip_smooth
            self.wheel_speed_mms = (a * (wheel_disp_mm / dt)
                                    + (1.0 - a) * self.wheel_speed_mms)
            self.lidar_speed_mms = (a * (lidar_disp_mm / dt)
                                    + (1.0 - a) * self.lidar_speed_mms)
        return self.trust()

    def trust(self):
        """1.0 (trusted) down to 0.0: how well the wheels' speed matches the lidar's. Always
        1.0 below wheel_slip_min_speed_mms, where the ratio is noise.
        """
        if self.wheel_speed_mms < wheel_slip_min_speed_mms:
            return 1.0
        ratio = abs(self.wheel_speed_mms - self.lidar_speed_mms) / self.wheel_speed_mms
        if ratio >= wheel_slip_zero_ratio:
            return 0.0
        return 1.0 - ratio / wheel_slip_zero_ratio


# PosePropagator: carry the last lidar fit forward between revolutions.
#
# Heading was already carried with the IMU (_fused_heading); position was not, so "pose"
# sat up to a whole revolution (about 100 ms) old between fits. This extends the last fit
# with wheel odometry and the fused IMU heading and publishes "pose_live" = (x, y,
# heading, t) at about 50 Hz. It is not a second estimator: it never writes "pose", so the
# fit stays the single authority and nothing downstream can feed through it. Consumers opt
# in (today: the white-line gate).
#
# It owns its WheelOdometry instance: two pollers sharing one would reset each other's QDR
# baseline. The trust floor below only stops new wheel evidence being integrated; it
# doesn't steer anything, and the next fit re-bases regardless.

pose_propagate_enabled = True
pose_propagate_period_s = 0.02 # about 50 Hz, matched to the play loop
pose_propagate_max_age_s = 0.5 # older than this -> "pose_live" goes None
pose_propagate_max_jump_mm = 100.0 # one tick moving further than this is a QDR glitch
pose_propagate_trust_floor = 0.25 # slip trust below this -> ignore new wheel evidence
pose_propagate_coast_max_s = 0.3 # no wheel evidence this long -> fully back on the fit


class PosePropagator:
    """extend the last lidar fit forward with wheel odometry and the fused IMU heading."""

    def __init__(self):
        self.odom = WheelOdometry() # own instance, see the block comment above
        self._x0 = self._y0 = None # extension baseline = the last fit
        self._h0 = None
        self._t0 = None # monotonic time of that fit (its "pose_t")
        self._no_fix_since = None # first tick of the current run with no wheel evidence

    def tick(self, now):
        """one propagation step; publishes _state["pose_live"]."""
        with state._lock:
            pose = state._state["pose"]
            pt = state._state["pose_t"]
            pimu = state._state["pose_imu"]
            inow = state._state["imu_heading"]
            trust = state._state["wheel_slip_trust"]

        if pose is None or pt is None:
            with state._lock:
                state._state["pose_live"] = None
            self._t0 = None
            self._no_fix_since = None
            return

        # A fresh fit re-bases the extension, and that tick's wheel delta is
        # dropped: the fit already covers the same stretch of time, so integrating
        # both would double-count it.
        rebased = pt != self._t0
        if rebased:
            self._x0, self._y0, self._h0, self._t0 = pose[0], pose[1], pose[2], pt

        # advance the fit's heading by the IMU delta since its revolution, as
        # _fused_heading does
        h = pose[2]
        if _compass.imu_fusion_enabled and pimu is not None and inow is not None:
            d = _wrap_deg(inow - pimu)
            if abs(d) <= heading_carry_max_deg:
                h = h + d

        since_t = self.odom.t
        since = self.odom.poll(now) # poll even on a re-base tick, to keep the QDR baseline moving
        usable = False # did this tick produce wheel evidence we integrated?
        if since is not None and not rebased:
            with state._lock:
                collided = (since_t is not None
                            and state._state["collision_t"] is not None
                            and since_t < state._state["collision_t"] <= now)
            df, dr = since
            if (collided or trust < pose_propagate_trust_floor
                    or abs(df) > pose_propagate_max_jump_mm
                    or abs(dr) > pose_propagate_max_jump_mm):
                # a shove, a slipping wheel or a QDR glitch: drop it and
                # count the tick as evidence-less so the decay below
                # engages
                pass
            else:
                usable = True
                rad = math.radians(h)
                ch, sh = math.cos(rad), math.sin(rad)
                # this interval's robot-frame (forward, right), rotated
                # into the field frame
                self._x0 += ch * dr + sh * df
                self._y0 += ch * df - sh * dr
        if usable:
            self._no_fix_since = None
        elif self._no_fix_since is None:
            self._no_fix_since = now

        # With no wheel evidence (a motor bus stall), decay back onto the raw fit
        # instead of coasting on the last delta, so "pose_live" fails toward
        # "pose" rather than toward a frozen lie.
        if self._no_fix_since is not None:
            a = min(1.0, (now - self._no_fix_since) / pose_propagate_coast_max_s)
            self._x0 += (pose[0] - self._x0) * a
            self._y0 += (pose[1] - self._y0) * a

        with state._lock:
            if now - pt <= pose_propagate_max_age_s:
                state._state["pose_live"] = (self._x0, self._y0, h, now)
            else:
                state._state["pose_live"] = None


def _pose_propagator_thread():
    """daemon thread running PosePropagator at pose_propagate_period_s (started by bot.main)."""
    prop = PosePropagator()
    period = pose_propagate_period_s
    while True:
        now = time.monotonic()
        try:
            prop.tick(now)
        except Exception:
            # never let the extension take the process down; degrade to None
            with state._lock:
                state._state["pose_live"] = None
        time.sleep(max(0.001, period - (time.monotonic() - now)))
