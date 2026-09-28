"""Lidar ingestion (LidarReader, LidarCoords), the opposing-wall-sum gate, motion deskew, and
_lidar_thread: localisation and robot detection.
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
import bot.compass as _compass
from bot.compass import _imu_turn_between
from bot.diagnostics import _mark_health_t, wobble_observe, wobble_reset
from bot.field import FieldModel, wrap_deg as _wrap_deg
from bot.localisation import MCL
from bot.logs import _lidar_bad_rms, _lidar_min_inliers, _lidar_recover_n
from bot.odometry import WheelSlipMonitor, _wheel_odom_delta_mm
from bot.perception import Perception
from bot.tracking import EnemyVelocityTracker, KnownOcclusion, RobotTracker, TeammateID

# MCL (bot/localisation.py) is the only pose source. One persistent instance, since a
# particle filter's value is carrying its belief between calls; bot/main.py seeds its
# startup IMU prior through it. Robot detection still goes through Perception, fed off
# MCL's inlier/outlier split (result_for matches Perception.localise's return shape).
mcl = MCL()


class LidarPoint:
    """one raw return from the STL-19P/LD19 lidar: angle_deg, distance_mm, intensity."""
    __slots__ = ("angle_deg", "distance_mm", "intensity")

    def __init__(self, angle_deg, distance_mm, intensity):
        """store one (angle, distance, intensity) return."""
        self.angle_deg = angle_deg
        self.distance_mm = distance_mm
        self.intensity = intensity

    def __repr__(self):
        return (f"LidarPoint(angle={self.angle_deg:.2f}deg, "
                f"dist={self.distance_mm}mm, intensity={self.intensity})")


# Optional compiled core for _crc8/_parse_packet (about 6x faster), the hottest loop in
# the pipeline: every raw serial packet. Falls back to the Python below if not built;
# never a hard dependency.
try:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "native"))
    import lidar_native
except ImportError:
    lidar_native = None


class LidarReader:
    """reads the STL-19P / LD19 lidar over UART and yields full revolutions as lists of
    LidarPoint.
    """

    packet_len = 47
    points_per_pkt = 12
    baud_rate = 230400

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
        """port: serial port path, auto-detected by find_port() if None; baudrate overrides
        230400.
        """
        self.port = port if port is not None else self.find_port()
        self.baudrate = baudrate if baudrate is not None else self.baud_rate

    # helpers
    @staticmethod
    def find_port():
        """first plausible serial port, preferring /dev/serial/by-id/ (stable across reboots)."""
        by_id = glob.glob("/dev/serial/by-id/*")
        if by_id:
            return by_id[0]
        usb = sorted(glob.glob("/dev/ttyUSB*"))
        if usb:
            return usb[0]
        raise RuntimeError(
            "No serial device found. Check `ls /dev/serial/by-id/` and "
            "`dmesg | tail` after plugging the LiDAR in, and make sure your "
            "user is in the dialout group:\n"
            "    sudo usermod -a -G dialout $user (then log out/in)")

    @classmethod
    def _crc8(cls, data):
        """packet checksum, per the LDROBOT DTOF protocol's CRC8 table."""
        crc = 0
        for b in data:
            crc = cls._crc_table[(crc ^ b) & 0xFF]
        return crc

    @classmethod
    def _parse_packet(cls, buf):
        """parse one 47-byte packet into (12 LidarPoints, spin speed in deg/s), or (None, None)
        on a bad header or CRC. Uses lidar_native when built.
        """
        if lidar_native is not None:
            tuples, speed_dps = lidar_native.parse_packet(bytes(buf))
            if tuples is None:
                return None, None
            return [LidarPoint(a, d, i) for a, d, i in tuples], speed_dps

        if buf[0] != 0x54 or buf[1] != 0x2C:
            return None, None
        if cls._crc8(buf[:-1]) != buf[-1]:
            return None, None

        speed_dps = struct.unpack_from("<H", buf, 2)[0]
        start_angle = struct.unpack_from("<H", buf, 4)[0]
        end_angle = struct.unpack_from("<H", buf, 42)[0]
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
        buf = bytearray()
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

    angle_sign = 1 # flip to -1 if cloud is mirrored
    angle_offset_deg = 0 # mounting correction (deg)
    max_range_mm = 3500 # field diagonal about 3055 mm; filter beyond
    min_range_mm = 95 # filter inside the robot's footprint
    min_intensity = 0 # raise to reject weak/noisy returns
    handle_centres_deg = (90.0, 270.0) # carry-handle azimuths (robot frame)
    handle_half_deg = 5.0 # half-width of each blocked wedge

    class SimplePoint:
        """lightweight stand-in for LidarPoint."""
        __slots__ = ("angle_deg", "distance_mm", "intensity")
        def __init__(self, angle_deg, distance_mm, intensity):
            """store one (angle, distance, intensity) reading."""
            self.angle_deg = angle_deg
            self.distance_mm = distance_mm
            self.intensity = intensity

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
        rad = math.radians(bearing)
        return (point.distance_mm * math.sin(rad),
                point.distance_mm * math.cos(rad))

    @classmethod
    def scan_to_xy(cls, scan):
        """convert a full revolution to parallel (xs, ys) mm arrays in robot frame."""
        xs, ys, _ = cls.scan_to_xy_frac(scan)
        return xs, ys

    @classmethod
    def scan_to_xy_frac(cls, scan):
        """scan_to_xy plus, per surviving point, how far through the revolution it was taken
        (0.0 first, 1.0 last).
        """
        n = len(scan)
        if n == 0:
            return np.empty(0), np.empty(0), np.empty(0)
        angle = np.fromiter((p.angle_deg for p in scan), dtype=np.float64, count=n)
        dist = np.fromiter((p.distance_mm for p in scan), dtype=np.float64, count=n)
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
        d = dist[mask]
        return d * np.sin(rad), d * np.cos(rad), fracs[mask]


# Opposing-wall-sum gate: two rays pointing opposite ways along a field axis must have
# wall distances summing to that dimension if both really reached the boundary. A short
# sum means one stopped on something nearer (almost always an enemy); a long one went
# through a goal mouth. Either way it isn't "the wall is here" evidence.
lidar_axis_tol_deg = 12.0 # how far off a field axis a ray may still count
lidar_wall_sum_tol_mm = 100.0 # slack on the opposing-pair sum

# Bearing of each field-axis direction (atan2(dx, dy): 0 = +y, 90 = +x) and its axis.
_wall_sum_axis_dirs = (("y", +1, 0.0), ("x", +1, 90.0),
                       ("y", -1, 180.0), ("x", -1, -90.0))


def _wall_sum_reading_polar(r):
    """(robot-frame bearing_deg, distance_mm) from a (bearing, distance) pair or a
    LidarPoint-like.
    """
    if hasattr(r, "distance_mm"):
        return float(r.angle_deg), float(r.distance_mm)
    return float(r[0]), float(r[1])


def wall_sum_trusted(pose, readings, axis_tol_deg=None, tol_mm=None):
    """per-reading "this ray really ended on a boundary wall" flags, parallel to `readings`
    (True also where the test has nothing to say, so callers can and it in).

    Only pose's heading is used; the pair sum doesn't depend on where we stand. Readings are
    robot-frame bearings, as (bearing_deg, distance_mm) pairs or LidarPoint-likes. Each
    direction is tested against the longest opposing reading, since occlusion only ever
    shortens a ray.
    """
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
                # project onto the axis: an off-axis ray meets the wall at
                # d, which is d*cos(off) along the normal
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


# Motion deskew: a revolution takes about 100 ms, so without it the fit gets one rigid
# transform for points taken from different poses.
deskew_enabled = True
deskew_min_deg = 0.15 # below this much rotation, not worth the work
deskew_min_mm = 5.0 # ... or this much translation
deskew_max_deg = 90.0 # sanity gate: a bigger "delta" than this in one
deskew_max_mm = 400.0 # revolution is a relocalise jump, not motion


def _deskew(pts, fracs, dx, dy, dth, hdg):
    """undo the robot's motion during one lidar revolution."""
    if not deskew_enabled or len(pts) < 2 or len(pts) != len(fracs):
        return pts
    if abs(dth) < deskew_min_deg and math.hypot(dx, dy) < deskew_min_mm:
        return pts # sitting still, nothing to undo
    if abs(dth) > deskew_max_deg or math.hypot(dx, dy) > deskew_max_mm:
        return pts # not motion; don't corrupt the scan

    P = np.asarray(pts, dtype=np.float64)
    r = 1.0 - np.asarray(fracs, dtype=np.float64) # residual of the sweep

    # Rotate each point by the heading change still to come after it was taken.
    a = np.radians(r * dth)
    ca, sa = np.cos(a), np.sin(a)
    x = P[:, 0] * ca - P[:, 1] * sa
    y = P[:, 0] * sa + P[:, 1] * ca

    # ...then subtract the translation still to come, rotated into the robot frame
    # (field = R(H) . robot, R(H) = [[cos H, sin H], [-sin H, cos H]])
    h = math.radians(hdg)
    ch, sh = math.cos(h), math.sin(h)
    tx = r * (dx * ch - dy * sh)
    ty = r * (dx * sh + dy * ch)
    return list(zip(x - tx, y - ty))


def _lidar_reader_thread(reader, out_queue):
    """own the LidarReader and the Cartesian conversion, decoupled from _lidar_thread's heavier
    work.
    """
    for scan in reader.read_scans():
        t_scan = time.monotonic() # end of this revolution
        # the polar scan rides along for --lidarlog's raw dump
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
    reader = LidarReader()
    lidar_queue = queue.Queue(maxsize=1)
    threading.Thread(target=_lidar_reader_thread, args=(reader, lidar_queue),
                     daemon=True).start()
    pose = None
    bad_streak = 0
    tracker = RobotTracker()
    mate_id = TeammateID()
    enemy_vel = EnemyVelocityTracker()
    wheel_slip = WheelSlipMonitor()
    imu_prev = None
    prev_pub = None # last published pose, for the deskew velocity
    prev_t = None # ... and when it was published
    last_delta = None # (dx, dy, dth) of motion per revolution, measured
    # real heading from the last good fix, the only source of a search heading prior
    # (never a guess)
    last_heading = None

    while True:
        t_scan, xs, ys, fracs, scan = lidar_queue.get()
        pts = list(zip(xs, ys))
        if not pts:
            continue

        # --lidarlog: read the logger once per scan, and note where this scan's
        # prior came from
        log = state._lidar_log
        prior_src = "carry"

        # a goal-side change invalidates the heading a previous global search was
        # biased on
        with state._lock:
            force = state._state["force_relocalise"]
            if force:
                state._state["force_relocalise"] = False
        if force:
            if pose is not None:
                last_heading = pose[2]
            pose = None

        # Keep the prior's heading current with the IMU delta since the last scan:
        # the deskew and the wall-sum gate below run in its frame. (MCL takes the
        # turn in its motion update.)
        if _compass.imu_fusion_enabled:
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
            # rough_region is only a tiebreak between two comparable fits,
            # never how the goal side is decided (slot_goal is fixed when both
            # buttons are pressed)
            rough = None
            if slot_goal is not None:
                rough = (FieldModel.cx,
                         FieldModel.field_y * 0.25 if slot_goal == "low"
                         else FieldModel.field_y * 0.75)
            # the heading must never come from slot_goal: a robot placed
            # facing the "wrong" way would never be found under a hard heading
            # filter
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
            pose = g["pose"]
            prior_src = "global"
            bad_streak = 0
            wobble_reset() # a re-localise jump is not real motion, don't misread it as one
            print(f"[lidar] pose ({pose[0]:.0f}, {pose[1]:.0f},"
                  f" {pose[2]:.1f}deg) rms {g['rms_mm']:.1f} mm"
                  f"  inliers {g['inlier_count']}", flush=True)
            if log is not None:
                log.event("global",
                          f"x={pose[0]:.1f} y={pose[1]:.1f} h={pose[2]:.2f} "
                          f"rms={g['rms_mm']:.1f} inliers={g['inlier_count']} "
                          f"ambiguous={bool(g.get('ambiguous'))} "
                          f"resolved={bool(g.get('resolved'))}")

        # Motion deskew: undo the robot's movement across the revolution
        # before fitting.
        period = 0.1
        if reader.speed_dps:
            period = min(0.5, max(0.02, 360.0 / reader.speed_dps))
        wheel_dx = wheel_dy = 0.0
        wheel_src = "none" # for the wheel-slip cross-check below
        if pose is not None:
            # rotation and translation have independent fallback chains; the
            # wheels never solve rotation
            dth = _imu_turn_between(t_scan - period, t_scan)
            dth_src = "imu"
            if dth is None and last_delta is not None:
                dth = last_delta[2]
                dth_src = "carry"

            # translation: the wheels' measured displacement this revolution,
            # else the last revolution's carried forward
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

        # Opposing-wall-sum gate in the prior's frame (the corrected pose is what
        # we are about to solve for). Drops returns that point along a field axis
        # but don't sum to the field's size.
        if pose is not None:
            readings = [(_wrap_deg(math.degrees(math.atan2(x, y))), math.hypot(x, y))
                       for x, y in pts]
            trust = wall_sum_trusted(pose, readings)
            filtered = [p for p, ok in zip(pts, trust) if ok]
            if filtered: # never hand localise an empty scan over this
                pts = filtered

        # MCL tick: motion-update the particles, then weight them against this
        # filtered scan
        prior = pose
        result = mcl.localise(pts, pose)
        pose = result["pose"]
        rms = result["rms_mm"]
        n_in = result["inlier_count"]

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

        # Wheel-slip cross-check (diagnostic only): the wheels' displacement this
        # revolution against what the fit says we moved. Wheels claiming motion
        # the lidar doesn't back up are probably slipping.
        if prior is not None and wheel_src == "odom":
            lidar_disp = math.hypot(pose[0] - prior[0], pose[1] - prior[1])
            wheel_disp = math.hypot(wheel_dx, wheel_dy)
            trust = wheel_slip.update(wheel_disp, lidar_disp, period)
            with state._lock:
                state._state["wheel_slip_trust"] = trust
                state._state["wheel_speed_mms"] = wheel_slip.wheel_speed_mms
                state._state["lidar_speed_mms"] = wheel_slip.lidar_speed_mms

        # Robot detection, tracked across scans. Occlusion comes from what we know
        # blocks the view: walls, goal structures, this scan's bots, and the
        # teammate's reported pose.
        with state._lock:
            # Bluetooth team-link pose beats the UDP fallback
            tm_pos = state._state["teammate_pos_bt"] or state._state["teammate_pos"]
        dets = Perception.detect_robots(result["outliers"],
                                        (pose[0], pose[1]),
                                        teammate_pos=tm_pos)

        blockers = [(d["x"], d["y"], Perception.robot_radius_mm)
                    for d in dets]
        if tm_pos is not None:
            blockers.append((tm_pos[0], tm_pos[1],
                             Perception.robot_radius_mm))
        occ = KnownOcclusion((pose[0], pose[1]), blockers)
        tracked = tracker.update(dets, occ)

        # one friendly at most: every track is an enemy until ID'ed against the
        # teammate's broadcast, and the label sticks to the track id
        _, enemies = mate_id.classify(tracked, tm_pos)

        # field-frame enemy velocity from the same confirmed enemy list
        enemy_vel_est = enemy_vel.update(t_scan, enemies)

        with state._lock:
            state._state["pose"] = pose
            # the scan's timestamp, so consumers can judge the fit's age
            state._state["pose_t"] = t_scan
            # the IMU reading that goes with this pose, for carrying the
            # heading forward (_fused_heading)
            state._state["pose_imu"] = state._state["imu_heading"]
            state._state["enemies"] = enemies
            state._state["enemy_vel"] = enemy_vel_est
            state._state["lidar_hz"] = reader.speed_dps / 360.0
        # a good fit was just published, see the quality watchdog above for what
        # "good" means
        _mark_health_t("lidar", t_scan)
        wobble_observe(pose, t_scan) # pure telemetry, see its block comment
        # per-revolution motion between published poses, divided by the
        # revolutions that elapsed so a dropped frame scales the estimate instead
        # of inflating it
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
