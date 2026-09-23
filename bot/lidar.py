"""LiDAR ingestion (LidarReader/LidarCoords), the opposing-wall-sum ICP
validity gate, motion deskew, and _lidar_thread - the localisation +
robot-detection main loop.

lidar_native: optional C++ acceleration of LidarReader._crc8/_parse_packet
(~3-6x faster, see the sibling native/lidar_native.cpp), falls back to the
pure-Python implementation below if the compiled extension isn't built for
this platform. Land mine: this module lives one directory deeper than the
repo root (bot/lidar.py), so os.path.dirname(os.path.abspath(__file__))
resolves to bot/, not the root - the native/ directory hasn't moved, so
the sys.path insertion below goes one level up (.., "native") to still find it.
This import must never become a hard dependency: any ImportError (extension
not built for this platform) falls back to lidar_native = None and the pure
-Python path.

_lidar_thread is the highest-risk single function in this extraction: it
touches bot.state (_state/_lock), bot.odometry (_wheel_odom via the module
attribute, WheelSlipMonitor, _wheel_odom_delta_mm), bot.compass
(_imu_turn_between, imu_fusion_enabled), bot.diagnostics (wobble_reset/
wobble_observe, _mark_health_t), bot.field (FieldModel), bot.perception
(Perception), bot.tracking (RobotTracker, TeammateID, EnemyVelocityTracker,
KnownOcclusion), and bot.logs (the lidar-quality thresholds and
state._lidar_log, whose .skew/.drop/.event/.scan methods are called by
attribute, needing no LidarLogger import here). wall_sum_trusted is defined
in THIS module (not bot.tracking) even though it originally sat inside the
same source block as EnemyProfile/RobotTracker: it is wired directly into
_lidar_thread, right before Perception.localise, and has no other caller.
"""

import glob
import math
import os
import queue
import struct
import sys
import threading
import time

import numpy as np

import bot.state as state
from bot.compass import _imu_turn_between, imu_fusion_enabled
from bot.diagnostics import _mark_health_t, wobble_observe, wobble_reset
from bot.field import FieldModel, wrap_deg as _wrap_deg
from bot.localisation import MCL
from bot.logs import _lidar_bad_rms, _lidar_min_inliers, _lidar_recover_n
from bot.odometry import WheelSlipMonitor, _wheel_odom_delta_mm
from bot.perception import Perception
from bot.tracking import EnemyVelocityTracker, KnownOcclusion, RobotTracker, TeammateID

# MCL (bot/localisation.py) is the sole pose source: a persistent particle
# filter replaces Perception.localise/global_localise's point-to-line ICP
# entirely (not a fusion - see bot/localisation.py's own module docstring
# and this session's explicit direction). One instance, since a particle
# filter's whole value is carrying its belief forward between calls, not
# refitting from scratch every tick; bot/main.py reaches this same instance
# (`from bot.lidar import mcl`) to seed its startup IMU-yaw prior. detect_robots
# still goes through Perception (a separate, still-ICP-shaped geometry helper
# - see bot/perception.py's own docstring for why it stays there), fed
# straight off MCL's own inliers/outliers split (result_for's shape matches
# Perception.localise's, so this is a drop-in for _lidar_thread's callers).
mcl = MCL()


class LidarPoint:
    """one raw return from the STL-19P/LD19 lidar: angle_deg, distance_mm, intensity."""
    __slots__ = ("angle_deg", "distance_mm", "intensity")

    def __init__(self, angle_deg, distance_mm, intensity):
        """store one (angle, distance, intensity) return."""
        self.angle_deg   = angle_deg
        self.distance_mm = distance_mm
        self.intensity   = intensity

    def __repr__(self):
        return (f"LidarPoint(angle={self.angle_deg:.2f}deg, "
                f"dist={self.distance_mm}mm, intensity={self.intensity})")


# native/lidar_native.{so,pyd}: optional C++ port of LidarReader._crc8/
# _parse_packet (~3-6x faster measured, see the sibling native/lidar_native.cpp),
# the hottest, smallest loop in the pipeline (every raw serial packet, not
# just once per revolution). Falls back to the pure-Python implementation
# below (byte-for-byte verified against it, see
# native/test_lidar_native.py) if the compiled extension isn't built
# for this platform - this import must never be a hard dependency. See this
# module's own docstring for why the path now goes one level up from here.
try:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "native"))
    import lidar_native
except ImportError:
    lidar_native = None


class LidarReader:
    """reads the STL-19P / LD19 LiDAR over a serial UART and yields full scan revolutions as lists of LidarPoint."""

    packet_len     = 47
    points_per_pkt = 12
    baud_rate      = 230400

    _crc_table = (
        0x00, 0x4d, 0x9a, 0xd7, 0x79, 0x34, 0xe3, 0xae,
        0xf2, 0xbf, 0x68, 0x25, 0x8b, 0xc6, 0x11, 0x5c,
        0xa9, 0xe4, 0x33, 0x7e, 0xd0, 0x9d, 0x4a, 0x07,
        0x5b, 0x16, 0xc1, 0x8c, 0x22, 0x6f, 0xb8, 0xf5,
        0x1f, 0x52, 0x85, 0xc8, 0x66, 0x2b, 0xfc, 0xb1,
        0xed, 0xa0, 0x77, 0x3a, 0x94, 0xd9, 0x0e, 0x43,
        0xb6, 0xfb, 0x2c, 0x61, 0xcf, 0x82, 0x55, 0x18,
        0x44, 0x09, 0xde, 0x93, 0x3d, 0x70, 0xa7, 0xea,
        0x3e, 0x73, 0xa4, 0xe9, 0x47, 0x0a, 0xdd, 0x90,
        0xcc, 0x81, 0x56, 0x1b, 0xb5, 0xf8, 0x2f, 0x62,
        0x97, 0xda, 0x0d, 0x40, 0xee, 0xa3, 0x74, 0x39,
        0x65, 0x28, 0xff, 0xb2, 0x1c, 0x51, 0x86, 0xcb,
        0x21, 0x6c, 0xbb, 0xf6, 0x58, 0x15, 0xc2, 0x8f,
        0xd3, 0x9e, 0x49, 0x04, 0xaa, 0xe7, 0x30, 0x7d,
        0x88, 0xc5, 0x12, 0x5f, 0xf1, 0xbc, 0x6b, 0x26,
        0x7a, 0x37, 0xe0, 0xad, 0x03, 0x4e, 0x99, 0xd4,
        0x7c, 0x31, 0xe6, 0xab, 0x05, 0x48, 0x9f, 0xd2,
        0x8e, 0xc3, 0x14, 0x59, 0xf7, 0xba, 0x6d, 0x20,
        0xd5, 0x98, 0x4f, 0x02, 0xac, 0xe1, 0x36, 0x7b,
        0x27, 0x6a, 0xbd, 0xf0, 0x5e, 0x13, 0xc4, 0x89,
        0x63, 0x2e, 0xf9, 0xb4, 0x1a, 0x57, 0x80, 0xcd,
        0x91, 0xdc, 0x0b, 0x46, 0xe8, 0xa5, 0x72, 0x3f,
        0xca, 0x87, 0x50, 0x1d, 0xb3, 0xfe, 0x29, 0x64,
        0x38, 0x75, 0xa2, 0xef, 0x41, 0x0c, 0xdb, 0x96,
        0x42, 0x0f, 0xd8, 0x95, 0x3b, 0x76, 0xa1, 0xec,
        0xb0, 0xfd, 0x2a, 0x67, 0xc9, 0x84, 0x53, 0x1e,
        0xeb, 0xa6, 0x71, 0x3c, 0x92, 0xdf, 0x08, 0x45,
        0x19, 0x54, 0x83, 0xce, 0x60, 0x2d, 0xfa, 0xb7,
        0x5d, 0x10, 0xc7, 0x8a, 0x24, 0x69, 0xbe, 0xf3,
        0xaf, 0xe2, 0x35, 0x78, 0xd6, 0x9b, 0x4c, 0x01,
        0xf4, 0xb9, 0x6e, 0x23, 0x8d, 0xc0, 0x17, 0x5a,
        0x06, 0x4b, 0x9c, 0xd1, 0x7f, 0x32, 0xe5, 0xa8,
    )

    def __init__(self, port=None, baudrate=None):
        """port : serial port path (e.g. '/dev/ttyUSB0'), auto-detected via find_port() if None; baudrate overrides the default 230400."""
        self.port     = port if port is not None else self.find_port()
        self.baudrate = baudrate if baudrate is not None else self.baud_rate

    # helpers
    @staticmethod
    def find_port():
        """first plausible serial port path, prefers /dev/serial/by-id/ (stable across reboots) over /dev/ttyUSB*."""
        by_id = glob.glob("/dev/serial/by-id/*")
        if by_id:
            return by_id[0]
        usb = sorted(glob.glob("/dev/ttyUSB*"))
        if usb:
            return usb[0]
        raise RuntimeError(
            "No serial device found.  Check `ls /dev/serial/by-id/` and "
            "`dmesg | tail` after plugging the LiDAR in, and make sure your "
            "user is in the dialout group:\n"
            "    sudo usermod -a -G dialout $user  (then log out/in)")

    @classmethod
    def _crc8(cls, data):
        """packet checksum, per the LDROBOT DTOF protocol's CRC8 table."""
        crc = 0
        for b in data:
            crc = cls._crc_table[(crc ^ b) & 0xFF]
        return crc

    @classmethod
    def _parse_packet(cls, buf):
        """parse one 47-byte packet into (points, speed_dps): 12 LidarPoints and the motor spin speed in degrees/second (bytes 2-3, divide by 360 for Hz), or (None, None) on a bad header/CRC. Uses lidar_native (~3-6x faster) when built, byte-for-byte verified against this pure-Python fallback."""
        if lidar_native is not None:
            tuples, speed_dps = lidar_native.parse_packet(bytes(buf))
            if tuples is None:
                return None, None
            return [LidarPoint(a, d, i) for a, d, i in tuples], speed_dps

        if buf[0] != 0x54 or buf[1] != 0x2C:
            return None, None
        if cls._crc8(buf[:-1]) != buf[-1]:
            return None, None

        speed_dps   = struct.unpack_from("<H", buf, 2)[0]
        start_angle = struct.unpack_from("<H", buf, 4)[0]
        end_angle   = struct.unpack_from("<H", buf, 42)[0]
        span = end_angle - start_angle
        if span < 0:
            span += 36000 # wrapped through 360deg
        step = span / (cls.points_per_pkt - 1)

        points = []
        for i in range(cls.points_per_pkt):
            dist, inten = struct.unpack_from("<HB", buf, 6 + i * 3)
            angle = (start_angle + step * i) % 36000
            points.append(LidarPoint(angle / 100.0, dist, inten))
        return points, speed_dps

    # main interface
    def read_scans(self):
        """generator, yields one full revolution (list[LidarPoint]) at a time."""
        import serial
        ser = serial.Serial(self.port, self.baudrate, timeout=1)
        ser.reset_input_buffer()
        buf        = bytearray()
        revolution = []
        last_angle = None
        self.speed_dps = 0 # updated every packet; divide by 360 for Hz

        try:
            while True:
                chunk = ser.read(256)
                if not chunk:
                    continue
                buf.extend(chunk)

                while len(buf) >= self.packet_len:
                    if buf[0] != 0x54 or buf[1] != 0x2C:
                        del buf[0] # not aligned, resync byte by byte
                        continue
                    packet = bytes(buf[:self.packet_len])
                    points, speed_dps = self._parse_packet(packet)
                    if points is None:
                        del buf[0] # header matched but CRC failed
                        continue
                    self.speed_dps = speed_dps
                    del buf[:self.packet_len]
                    for p in points:
                        if last_angle is not None and p.angle_deg < last_angle - 180:
                            yield revolution
                            revolution = []
                        revolution.append(p)
                        last_angle = p.angle_deg
        finally:
            ser.close()


class LidarCoords:
    """polar-to-Cartesian conversion for STL-19P scans."""

    angle_sign         = 1 # flip to -1 if cloud is mirrored
    angle_offset_deg   = 0 # mounting correction (deg)
    max_range_mm       = 3500 # field diagonal about 3055 mm; filter beyond
    min_range_mm       = 95 # filter inside the robot's own footprint
    min_intensity      = 0 # raise to reject weak/noisy returns
    handle_centres_deg = (90.0, 270.0) # carry-handle azimuths (robot frame)
    handle_half_deg    = 5.0 # half-width of each blocked wedge

    class SimplePoint:
        """lightweight stand-in for LidarPoint."""
        __slots__ = ("angle_deg", "distance_mm", "intensity")
        def __init__(self, angle_deg, distance_mm, intensity):
            """store one (angle, distance, intensity) reading."""
            self.angle_deg   = angle_deg
            self.distance_mm = distance_mm
            self.intensity   = intensity

    # helpers
    @classmethod
    def _angle_diff(cls, a, b):
        """smallest absolute difference between two bearings (deg)."""
        return abs(((a - b + 180.0) % 360.0) - 180.0)

    @classmethod
    def is_handle_blocked(cls, bearing_deg):
        """True if a robot-frame bearing falls inside a carry-handle wedge."""
        return any(cls._angle_diff(bearing_deg, c) <= cls.handle_half_deg
                   for c in cls.handle_centres_deg)

    @classmethod
    def polar_to_xy(cls, point):
        """convert one LidarPoint (or SimplePoint) to robot-frame (x_mm, y_mm)."""
        bearing = cls.angle_sign * point.angle_deg + cls.angle_offset_deg
        rad     = math.radians(bearing)
        return (point.distance_mm * math.sin(rad),
                point.distance_mm * math.cos(rad))

    @classmethod
    def scan_to_xy(cls, scan):
        """convert a full revolution to parallel (xs, ys) mm arrays in robot frame."""
        xs, ys, _ = cls.scan_to_xy_frac(scan)
        return xs, ys

    @classmethod
    def scan_to_xy_frac(cls, scan):
        """scan_to_xy plus, per surviving point, how far through the revolution it was sampled (0.0 first return, 1.0 last)."""
        n = len(scan)
        if n == 0:
            return np.empty(0), np.empty(0), np.empty(0)
        angle = np.fromiter((p.angle_deg for p in scan), dtype=np.float64, count=n)
        dist  = np.fromiter((p.distance_mm for p in scan), dtype=np.float64, count=n)
        inten = np.fromiter((p.intensity for p in scan), dtype=np.float64, count=n)
        fracs = np.arange(n, dtype=np.float64) / (float(n - 1) if n > 1 else 1.0)

        bearing = cls.angle_sign * angle + cls.angle_offset_deg
        blocked = np.zeros(n, dtype=bool)
        for centre in cls.handle_centres_deg:
            diff = np.abs(((bearing - centre + 180.0) % 360.0) - 180.0)
            blocked |= diff <= cls.handle_half_deg

        mask = ((dist >= cls.min_range_mm) & (dist <= cls.max_range_mm)
                & (inten >= cls.min_intensity) & ~blocked)

        rad = np.radians(bearing[mask])
        d   = dist[mask]
        return d * np.sin(rad), d * np.cos(rad), fracs[mask]


# Opposing-wall-sum ICP validity gate
# With the field's true dimensions known, two lidar rays pointing opposite
# ways along one field axis must have perpendicular wall-distances summing to
# that dimension - if they both actually reached the boundary. A short sum
# means one of them stopped on something nearer, almost always an enemy body
# occluding it, so it is not "the wall is here" evidence. A long sum means one
# ray went past the nominal wall line (through a goal mouth), equally not
# boundary evidence. Wired into _lidar_thread, right before Perception.localise.
lidar_axis_tol_deg    = 12.0 # how far off a field axis a ray may still count
lidar_wall_sum_tol_mm = 100.0 # slack on the opposing-pair sum

# Absolute bearing of each field-axis direction, in this codebase's
# atan2(dx, dy) convention (0 = +y, 90 = +x), with the axis each belongs to.
_wall_sum_axis_dirs = (("y", +1, 0.0), ("x", +1, 90.0),
                       ("y", -1, 180.0), ("x", -1, -90.0))


def _wall_sum_reading_polar(r):
    """(robot-frame bearing_deg, distance_mm) out of either a plain (bearing, distance) pair or a LidarPoint-like object."""
    if hasattr(r, "distance_mm"):
        return float(r.angle_deg), float(r.distance_mm)
    return float(r[0]), float(r[1])


def wall_sum_trusted(pose, readings, axis_tol_deg=None, tol_mm=None):
    """per-reading "this ray really did terminate on a field boundary wall" flags, parallel to `readings` (True = trusted, including every reading the test simply has nothing to say about, so a caller can AND it into whatever gating it already does). pose is (rx, ry, hdg); only hdg is used (the opposing-pair sum does not depend on where in the field we stand) but the full pose is taken so a caller can pass _state["pose"] straight through. `readings` are robot-frame bearings, i.e. post-mounting-correction, either as (bearing_deg, distance_mm) pairs or LidarPoint-likes. Each axis direction is tested against the LONGEST opposing reading, since occlusion only ever shortens a ray - the longest one is the least likely to be the occluded half of the pair."""
    tol = lidar_wall_sum_tol_mm if tol_mm is None else tol_mm
    atol = lidar_axis_tol_deg if axis_tol_deg is None else axis_tol_deg
    hdg = pose[2]
    out = [True] * len(readings)
    # (axis, sign) -> [(index, perpendicular wall distance), ...]
    groups = {}
    for i, r in enumerate(readings):
        bearing, dist = _wall_sum_reading_polar(r)
        if dist <= 0.0:
            continue
        absb = _wrap_deg(hdg + bearing)
        for axis, sign, ref in _wall_sum_axis_dirs:
            off = _wrap_deg(absb - ref)
            if abs(off) <= atol:
                # Project onto the axis: an off-axis ray reaches the wall at
                # d, but the wall is d*cos(off) away along the normal.
                groups.setdefault((axis, sign), []).append(
                    (i, dist * math.cos(math.radians(off))))
                break

    for (axis, sign), items in groups.items():
        opp = groups.get((axis, -sign))
        if not opp:
            continue # no opposing ray, nothing to check against
        ref = max(p for nm, p in opp)
        expect = FieldModel.field_x if axis == "x" else FieldModel.field_y
        for i, perp in items:
            if abs(perp + ref - expect) > tol:
                out[i] = False
    return out


# Motion deskew: a revolution takes about 100ms, so ICP fits one rigid transform to points from different poses, badly enough to displace returns metres off at a 90 deg heading error.
deskew_enabled     = True
deskew_min_deg     = 0.15 # below this much rotation, not worth the work
deskew_min_mm      = 5.0 # ... or this much translation
deskew_max_deg     = 90.0 # sanity gate: a bigger "delta" than this in one
deskew_max_mm      = 400.0 # revolution is a relocalise jump, not motion


def _deskew(pts, fracs, dx, dy, dth, hdg):
    """undo the robot's own motion during one lidar revolution."""
    if not deskew_enabled or len(pts) < 2 or len(pts) != len(fracs):
        return pts
    if abs(dth) < deskew_min_deg and math.hypot(dx, dy) < deskew_min_mm:
        return pts # sitting still, nothing to undo
    if abs(dth) > deskew_max_deg or math.hypot(dx, dy) > deskew_max_mm:
        return pts # not motion; don't corrupt the scan

    P = np.asarray(pts, dtype=np.float64)
    r = 1.0 - np.asarray(fracs, dtype=np.float64) # residual of the sweep

    # Rotate each point by the heading change still to come after it was taken.
    a  = np.radians(r * dth)
    ca, sa = np.cos(a), np.sin(a)
    x = P[:, 0] * ca - P[:, 1] * sa
    y = P[:, 0] * sa + P[:, 1] * ca

    # ...then subtract the translation still to come, rotated into the robot
    # frame (this file's convention: field = R(H) . robot, with
    # R(H) = [[cos H, sin H], [-sin H, cos H]]).
    h = math.radians(hdg)
    ch, sh = math.cos(h), math.sin(h)
    tx = r * (dx * ch - dy * sh)
    ty = r * (dx * sh + dy * ch)
    return list(zip(x - tx, y - ty))


def _lidar_reader_thread(reader, out_queue):
    """owns the LidarReader generator and the Cartesian conversion (LidarCoords.scan_to_xy_frac), decoupled from _lidar_thread's own ICP/detection/team-link work."""
    for scan in reader.read_scans():
        t_scan = time.monotonic() # end of this revolution
        # scan itself rides along too (not just its xs/ys/fracs), the
        # --lidarlog raw-dump feature (LidarLogger.scan's scan_points)
        # wants the original polar points, not just what survived filtering.
        item = (t_scan, *LidarCoords.scan_to_xy_frac(scan), scan)
        try:
            out_queue.put_nowait(item)
        except queue.Full:
            try:
                out_queue.get_nowait() # drop the stale revolution
            except queue.Empty:
                pass
            out_queue.put_nowait(item)


def _lidar_thread():
    """LiDAR localisation + robot detection."""
    reader       = LidarReader()
    lidar_queue  = queue.Queue(maxsize=1)
    threading.Thread(target=_lidar_reader_thread, args=(reader, lidar_queue),
                     daemon=True).start()
    pose         = None
    bad_streak   = 0
    tracker      = RobotTracker()
    mate_id      = TeammateID()
    enemy_vel    = EnemyVelocityTracker()
    wheel_slip   = WheelSlipMonitor()
    imu_prev     = None
    prev_pub     = None # last published pose, for the deskew velocity
    prev_t       = None # ... and when it was published
    last_delta   = None # (dx, dy, dth) of motion per revolution, measured
    # real, tracked heading from the last good fix, only source of a search heading prior.
    # Never a guess (see "if pose is None" below for why).
    last_heading = None

    while True:
        t_scan, xs, ys, fracs, scan = lidar_queue.get()
        pts = list(zip(xs, ys))
        if not pts:
            continue

        # --lidarlog: read the logger once per scan, and track where this
        # scan's ICP prior came from as the sources below get their turn.
        log       = state._lidar_log
        prior_src = "carry"

        # A goal-side change invalidates the heading guess a previous global search was biased on, forcing a fresh one.
        with state._lock:
            force = state._state["force_relocalise"]
            if force:
                state._state["force_relocalise"] = False
        if force:
            if pose is not None:
                last_heading = pose[2]
            pose = None

        # IMU heading delta since the last scan, folded into the ICP prior, covers fast spins/shoves between revolutions ICP alone can't track.
        if imu_fusion_enabled:
            with state._lock:
                imu = state._state["imu_heading"]
            if imu is not None and imu_prev is not None:
                imu_d = _wrap_deg(imu - imu_prev)
                if pose is not None:
                    pose = (pose[0], pose[1], pose[2] + imu_d)
                    if log is not None and imu_d:
                        prior_src += "+imu"
                elif last_heading is not None:
                    last_heading = _wrap_deg(last_heading + imu_d)
            imu_prev = imu

        # initial / recovery global search
        if pose is None:
            with state._lock:
                slot_goal = state._state["slot_goal"]
            # rough_region is only a convenience tiebreak between two comparably-good ICP fits, never how the goal side is decided (slot_goal is fixed the instant both buttons are pressed, sec 3.19).
            rough = None
            if slot_goal is not None:
                rough = (FieldModel.cx,
                         FieldModel.field_y * 0.25 if slot_goal == "low"
                         else FieldModel.field_y * 0.75)
            # Heading must never derive from slot_goal, a robot placed facing the "wrong" way for a hard heading filter would never be found.
            if last_heading is not None:
                print("[lidar] searching for start pose"
                      f" (near {last_heading:.0f}deg)...", flush=True)
                g = mcl.global_localise(pts, rough_region=rough,
                                        known_heading=last_heading,
                                        heading_tolerance=45.0)
            else:
                print("[lidar] searching for start pose"
                      " (full heading sweep)...", flush=True)
                g = mcl.global_localise(pts, rough_region=rough)
            if g is None:
                if log is not None:
                    log.event("global-fail", f"no fit from {len(pts)} pts")
                continue
            pose       = g["pose"]
            prior_src  = "global"
            bad_streak = 0
            wobble_reset() # a re-localise jump is not real motion, don't misread it as one
            print(f"[lidar] pose ({pose[0]:.0f}, {pose[1]:.0f},"
                  f" {pose[2]:.1f}deg)  rms {g['rms_mm']:.1f} mm"
                  f"  inliers {g['inlier_count']}", flush=True)
            if log is not None:
                log.event("global",
                          f"x={pose[0]:.1f} y={pose[1]:.1f} h={pose[2]:.2f} "
                          f"rms={g['rms_mm']:.1f} inliers={g['inlier_count']} "
                          f"ambiguous={bool(g.get('ambiguous'))} "
                          f"resolved={bool(g.get('resolved'))}")

        # Motion deskew: undo the robot's own movement across the revolution before fitting, so ICP sees one rigid scan instead of a smeared one.
        period = 0.1
        if reader.speed_dps:
            period = min(0.5, max(0.02, 360.0 / reader.speed_dps))
        wheel_dx = wheel_dy = 0.0
        wheel_src = "none" # for the wheel-slip cross-check below
        if pose is not None:
            # Rotation and translation come from independent fallback chains, wheel odometry never solves for rotation.
            dth     = _imu_turn_between(t_scan - period, t_scan)
            dth_src = "imu"
            if dth is None and last_delta is not None:
                dth = last_delta[2]
                dth_src = "carry"

            # Translation has no IMU equivalent: the wheels' own measured displacement where it has a fix this revolution (see WheelOdometry.poll), else the last revolution's measured displacement carried forward at constant velocity.
            dx = dy = 0.0
            dxy_src = "none"
            odom_delta = _wheel_odom_delta_mm(pose, t_scan)
            if odom_delta is not None:
                dx, dy = odom_delta
                dxy_src = "odom"
            elif last_delta is not None:
                dx, dy = last_delta[0], last_delta[1]
                dxy_src = "carry"
            wheel_dx, wheel_dy, wheel_src = dx, dy, dxy_src

            if dth is not None:
                pts = _deskew(pts, fracs, dx, dy, dth, pose[2])
                if log is not None:
                    skew_src = dth_src if dth_src == dxy_src else f"{dth_src}+{dxy_src}"
                    log.skew(dx, dy, dth, skew_src, period)

        with state._lock:
            state._state["lidar_pts"] = pts # deskewed robot-frame points, for calib

        # Opposing-wall-sum validity gate: drop points whose bearing points
        # along a field axis but whose paired opposing-direction range
        # doesn't sum to the known field width/length - almost always an
        # enemy occluding what should be a wall return. Uses this revolution's
        # PRIOR pose as the reference frame (the corrected pose is exactly
        # what we're about to solve for), so it's skipped right after a fresh
        # global search (pose was just set from the search fit itself, close
        # enough not to need this) has no earlier prior to gate against -
        # in practice `pose` is always set by this point, so this only ever
        # runs with a real (if one-revolution-stale) heading.
        if pose is not None:
            readings = [(_wrap_deg(math.degrees(math.atan2(x, y))), math.hypot(x, y))
                       for x, y in pts]
            trust = wall_sum_trusted(pose, readings)
            filtered = [p for p, ok in zip(pts, trust) if ok]
            if filtered: # never hand localise an empty scan over this
                pts = filtered

        # MCL tick: motion-update the particles then weight them against this
        # (wall_sum_trusted-filtered) scan - see bot/localisation.py's own
        # localise() docstring for why max_iters is accepted/ignored here.
        prior  = pose
        result = mcl.localise(pts, pose)
        pose   = result["pose"]
        rms    = result["rms_mm"]
        n_in   = result["inlier_count"]

        # quality watchdog -> re-search if fit goes bad
        if rms > _lidar_bad_rms or n_in < _lidar_min_inliers:
            bad_streak += 1
            if log is not None:
                log.drop(rms, n_in, bad_streak)
            if bad_streak >= _lidar_recover_n:
                print(f"[lidar] lost fix (rms={rms:.1f} mm, inliers={n_in})"
                      ", re-searching...", flush=True)
                if log is not None:
                    log.event("lost-fix", f"rms={rms:.1f} inliers={n_in}")
                last_heading = pose[2]
                pose = None
                bad_streak = 0
            continue # don't publish a bad pose
        else:
            bad_streak = 0

        # Wheel-slip cross-check (diagnostic only, see WheelSlipMonitor's own
        # block comment): compare the wheels' own measured displacement this
        # revolution (wheel_dx/wheel_dy, already computed above for deskewing)
        # against what the ICP fit itself concluded we actually moved
        # (pose - prior). A real mismatch, wheels claiming motion the lidar
        # fit doesn't back up, means a wheel is probably slipping.
        if prior is not None and wheel_src == "odom":
            lidar_disp = math.hypot(pose[0] - prior[0], pose[1] - prior[1])
            wheel_disp = math.hypot(wheel_dx, wheel_dy)
            trust = wheel_slip.update(wheel_disp, lidar_disp, period)
            with state._lock:
                state._state["wheel_slip_trust"] = trust
                state._state["wheel_speed_mms"]  = wheel_slip.wheel_speed_mms
                state._state["lidar_speed_mms"]  = wheel_slip.lidar_speed_mms

        # Robot detection (geometry-gated, tracked across scans): occlusion is filled in from what we know blocks the view, walls, goal boxes, this scan's detected bots, and the teammate's own reported pose.
        dets = Perception.detect_robots(result["outliers"],
                                        (pose[0], pose[1]))

        with state._lock:
            # Bluetooth team-link pose beats the UDP fallback
            tm_pos = state._state["teammate_pos_bt"] or state._state["teammate_pos"]

        blockers = [(d["x"], d["y"], Perception.robot_radius_mm)
                    for d in dets]
        if tm_pos is not None:
            blockers.append((tm_pos[0], tm_pos[1],
                             Perception.robot_radius_mm))
        occ = KnownOcclusion((pose[0], pose[1]), blockers)
        tracked = tracker.update(dets, occ)

        # One friendly at most: every track is an enemy until positively
        # ID'ed against the teammate's broadcast; the label then sticks to
        # the track id, not to whoever happens to be nearest.
        _, enemies = mate_id.classify(tracked, tm_pos)

        # Field-frame enemy velocity (sec: EnemyVelocityTracker), fed straight
        # off the same confirmed enemy list - see _pass_race_open's own
        # enemy_vel_est parameter for the one live consumer.
        enemy_vel_est = enemy_vel.update(t_scan, enemies)

        with state._lock:
            state._state["pose"]     = pose
            # The IMU reading that goes with this pose.  Consumers advance the
            # heading from here at IMU rate rather than waiting a whole
            # revolution for the next one, see _fused_heading.
            state._state["pose_imu"] = state._state["imu_heading"]
            state._state["enemies"]  = enemies
            state._state["enemy_vel"] = enemy_vel_est
            state._state["lidar_hz"] = reader.speed_dps / 360.0
        _mark_health_t("lidar", t_scan) # a good fit was just published, see the quality watchdog above for what "good" means
        wobble_observe(pose, t_scan) # pure telemetry, see its own block comment
        # Per-revolution motion, measured between published poses.  Divided by
        # however many revolutions actually elapsed, so a dropped frame or two
        # scales the estimate instead of inflating it.
        if prev_pub is not None and prev_t is not None:
            n_rev = max(1.0, round((t_scan - prev_t) / period))
            last_delta = ((pose[0] - prev_pub[0]) / n_rev,
                          (pose[1] - prev_pub[1]) / n_rev,
                          _wrap_deg(pose[2] - prev_pub[2]) / n_rev)
        prev_pub, prev_t = pose, t_scan

        if log is not None:
            log.scan(n_raw=len(scan), n_used=len(pts),
                     hz=reader.speed_dps / 360.0,
                     prior=prior, prior_src=prior_src, post=pose,
                     rms=rms, n_in=n_in, n_out=len(result["outliers"]),
                     scan_points=scan)
            if log.is_done():
                from bot.logs import _finish_lidar_log
                _finish_lidar_log()
