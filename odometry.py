"""Wheel odometry and the wheel-vs-lidar slip diagnostic.

WheelOdometry: dead-reckoned translation between lidar revolutions, from the
four drive wheels' own QDR position readout, no new hardware, replaces a
12-sensor outward ToF ring that was never mounted.

WheelSlipMonitor (NEW this session): a mismatch between the wheels' own
claimed displacement and what the lidar/ICP fit concluded we actually moved
over the same revolution means the wheels are probably slipping (pushed
against another robot, a wheel skating off the carpet), not that the lidar
fit is wrong. DIAGNOSTIC ONLY: the lidar/ICP fit this compares against is
itself partly built from wheel odometry (the deskew step in _lidar_thread),
so feeding this trust score back into anything that steers off wheel
odometry would be checking the wheels against a witness that partly heard
its testimony from the wheels - not a genuinely independent check.
Published to _state["wheel_slip_trust"]/"wheel_speed_mms"/"lidar_speed_mms"
for the debug page/logs instead, a diagnostic, not fed back into control.

Land mine: `_wheel_odom` is a module-level global, None until main()
(bot.main, stage 3+) constructs a WheelOdometry() and assigns it, then read
from a different thread. Any OTHER module reading it must do
`import bot.odometry as odometry; odometry._wheel_odom`, never
`from bot.odometry import _wheel_odom` - the latter freezes a stale None
reference forever and will never see main()'s later assignment.
"""

import math

import bot.state as state
from bot.hardware import Motor

# Drive speed model: wheel_diameter_mm is not yet extracted into any bot.*
# module (it belongs to the drive-speed-model section, a later
# motion/config stage) - kept here as a leaf constant since
# POS_RAW_PER_OUTPUT_REV and WheelOdometry.poll() both need it directly. A
# later stage should replace this with an import from wherever the
# drive-speed model ends up living, and delete this local copy then.
wheel_diameter_mm = 50.0

# WheelOdometry: dead-reckoned translation between lidar revolutions, from the four drive wheels' own QDR position readout, no new hardware, replaces a 12-sensor outward ToF ring that was never mounted.
POS_RAW_PER_OUTPUT_REV = Motor.RPM_TO_RAW * 60 # raw ticks per output wheel revolution
wheel_odom_enabled = True


class WheelOdometry:
    """dead-reckoned (forward_mm, right_mm) since the last poll(), from the four drive wheels' own QDR position."""
    NAMES = ("nw", "ne", "sw", "se")

    def __init__(self):
        # name -> last raw QDR position, drive()'s own trans={} sign convention (see
        # poll())
        self._raw = None
        self.t = None # monotonic time of that reading

    def poll(self, now):
        """(forward_mm, right_mm) since the last call, or None on a motor read failure or the first call (nothing to diff against yet)."""
        raw = {}
        for n in self.NAMES:
            m = Motor.motors.get(n)
            qdr = m.read_qdr() if m is not None else None
            if qdr is None:
                return None
            # undo Motor.drive's own send-time polarity flip (se/sw wired
            # reversed) so this lands back in drive()'s trans={} convention
            raw[n] = -qdr[0] if n in ("nw", "ne") else qdr[0]
        prev, self._raw, self.t = self._raw, raw, now
        if prev is None:
            return None
        mm = {n: (raw[n] - prev[n]) / POS_RAW_PER_OUTPUT_REV
                 * math.pi * wheel_diameter_mm
              for n in self.NAMES}
        # sqrt2 because vmax_full_cmd_mms sums the same four wheels' rim speed the same way its own derivation does.
        fwd   = math.sqrt(2.0) * (mm["nw"] - mm["ne"] + mm["se"] - mm["sw"]) / 4.0
        right = math.sqrt(2.0) * (mm["nw"] + mm["ne"] + mm["sw"] + mm["se"]) / 4.0
        return fwd, right


_wheel_odom = None # WheelOdometry(), set up in main()

def _wheel_odom_delta_mm(pose, now):
    """call from _lidar_thread in place of the `last_delta` carry (see the comment there): a measured displacement instead of an extrapolated one, rotated from robot- into field-frame using pose's heading."""
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


# Wheel-slip cross-check
wheel_slip_min_speed_mms = 50.0 # below this, noise dominates the ratio - always trust
wheel_slip_zero_ratio    = 0.5  # a mismatch this large (or more) of the wheels' own speed -> trust 0.0
wheel_slip_smooth        = 0.3  # EMA smoothing on both speeds before comparing them


class WheelSlipMonitor:
    """EMA-smoothed per-revolution wheel-vs-lidar speed cross-check (see the
    block comment above). Fed once per lidar revolution from _lidar_thread
    with that revolution's wheel-measured and ICP-measured displacement."""

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
        """1.0 (fully trusted) down to 0.0: how well the wheels' own claimed
        speed matches the lidar-derived speed. Below wheel_slip_min_speed_mms
        the ratio is noise, not signal, so it's always trusted."""
        if self.wheel_speed_mms < wheel_slip_min_speed_mms:
            return 1.0
        ratio = abs(self.wheel_speed_mms - self.lidar_speed_mms) / self.wheel_speed_mms
        if ratio >= wheel_slip_zero_ratio:
            return 0.0
        return 1.0 - ratio / wheel_slip_zero_ratio
