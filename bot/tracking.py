"""Track persistence over Perception's per-scan detections, and the velocity, memory and Kalman
estimators built on the tracks.
"""

import collections
import math

import numpy as np

from bot.field import FieldModel
from bot.perception import Perception


def fit_circle_algebraic(points):
    """variable-radius circle fit (algebraic / Kasa): closed-form least squares for centre and
    radius over x^2 + y^2 + A x + B y + C = 0.
    """
    n = len(points)
    if n < 3:
        return None
    P = np.asarray(points, dtype=np.float64)
    x, y = P[:, 0], P[:, 1]
    z = x * x + y * y
    A = np.column_stack((x, y, np.ones(n)))
    try:
        sol, *_ = np.linalg.lstsq(A, -z, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, b, c = sol
    cx, cy = -a / 2.0, -b / 2.0
    r2 = cx * cx + cy * cy - c
    if r2 <= 0.0:
        return None
    return float(cx), float(cy), float(math.sqrt(r2))


# Per-enemy shape profiling: as we move around an enemy, the lidar sees its near-side arc
# from changing angles.
enemy_profile_enabled = True


class EnemyProfile:
    """accumulate an enemy's near-side boundary points into angular bins across scans and
    re-fit a variable-radius circle.
    """
    n_bins = 36 # 10 deg angular bins around the enemy
    min_bins = 12 # circumference coverage before the fit is trusted
    bin_smooth = 0.4 # EMA on each bin's boundary offset
    centre_gain = 0.35 # fraction of the fit's centre correction applied/scan
    r_min = 70.0 # sane enemy-radius clamp (mm)
    r_max = 175.0

    def __init__(self):
        """start with no bins filled in, coverage() is 0, radius unknown."""
        self._off = [None] * self.n_bins # (ox, oy) boundary offset per bin
        self.radius = None # learned enemy radius, or None
        self.centred = False # enough coverage to trust the fit

    def coverage(self):
        """how many of the n_bins angular bins have a boundary point yet."""
        return sum(1 for o in self._off if o is not None)

    def observe(self, points, cx, cy):
        """bin the boundary points (offset from the current centre), then refit."""
        for (px, py) in points:
            ox, oy = px - cx, py - cy
            b = int((math.atan2(oy, ox) + math.pi)
                    / (2.0 * math.pi) * self.n_bins) % self.n_bins
            if self._off[b] is None:
                self._off[b] = (ox, oy)
            else:
                o0x, o0y = self._off[b]
                self._off[b] = (o0x + self.bin_smooth * (ox - o0x),
                                o0y + self.bin_smooth * (oy - o0y))

        pts = [o for o in self._off if o is not None]
        if len(pts) < self.min_bins:
            self.centred = False
            return None
        fit = fit_circle_algebraic(pts)
        if fit is None:
            return None
        fx, fy, r = fit # circle centre in offset space
        if not (self.r_min <= r <= self.r_max):
            return None
        self.centred = True
        self.radius = r
        # (fx, fy) != 0 means the binning centre (cx, cy) was off by that much
        dx, dy = self.centre_gain * fx, self.centre_gain * fy
        for i, o in enumerate(self._off):
            if o is not None:
                self._off[i] = (o[0] - dx, o[1] - dy)
        return cx + dx, cy + dy


class KnownOcclusion:
    """model-based visibility: everything that can block the lidar's view is already known
    (walls, goal structures, this scan's detected bots).
    """

    clearance_mm = 50.0 # ray must clear a blocker/wall by this much

    def __init__(self, sensor_xy, blockers):
        """sensor_xy: the observing lidar's (x, y). blockers: [(x, y, radius), ...] of known
        bots (None entries dropped).
        """
        self.sx, self.sy = sensor_xy
        self.blockers = [b for b in blockers if b is not None]

    def observed(self, x, y):
        """True if a bot at (x, y) would be in clear view of the sensor."""
        dx, dy = x - self.sx, y - self.sy
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return True
        ux, uy = dx / d, dy / d

        # walls and goal structures block everything behind them
        d_wall = FieldModel.raycast(self.sx, self.sy, ux, uy)
        if d_wall is not None and d_wall < d - self.clearance_mm:
            return False

        # known bots block a corridor of their radius around their centre
        for (bx, by, r) in self.blockers:
            if math.hypot(bx - x, by - y) <= r + 1.0:
                continue # that is the spot being asked about
            fx, fy = bx - self.sx, by - self.sy
            t = fx * ux + fy * uy # blocker's position along the ray
            if t <= 1e-6 or t >= d:
                continue # behind the sensor / beyond the spot
            perp = abs(fx * uy - fy * ux)
            if perp <= r + self.clearance_mm:
                return False
        return True


class RobotTracker:
    """frame-to-frame persistence filter over Perception.detect_robots output."""

    confirm_hits = 2 # sightings before a track is reported
    max_misses = 4 # scans a track may coast while in clear view
    max_occluded = 80 # scans a track may hide inside a shadow sector
    gate_mm = 300.0 # association gate detection <-> track
    smooth = 0.5 # EMA weight of the new detection
    max_tracks = Perception.max_robots_on_field

    def __init__(self):
        """start with no tracks, the next confirmed track gets id 1."""
        self._tracks = []
        self._next_id = 1

    def reset(self):
        """drop every track (e.g. on a lost lidar fix / re-localise)."""
        self._tracks = []

    def update(self, detections, visibility=None):
        """feed one scan's detections (plus optional KnownOcclusion visibility); returns the
        confirmed tracks.
        """
        dets = list(detections)
        # greedy nearest-first association
        pairs = sorted(
            ((math.hypot(d["x"] - t["x"], d["y"] - t["y"]), i, j)
             for i, d in enumerate(dets)
             for j, t in enumerate(self._tracks)),
            key=lambda p: p[0])
        used_d, used_t = set(), set()
        for dist, i, j in pairs:
            if dist > self.gate_mm:
                break
            if i in used_d or j in used_t:
                continue
            used_d.add(i)
            used_t.add(j)
            t, d = self._tracks[j], dets[i]
            a = self.smooth
            t["x"] = t["x"] + a * (d["x"] - t["x"])
            t["y"] = t["y"] + a * (d["y"] - t["y"])
            for k in ("points", "span_mm", "resid_mm", "conf"):
                if k in d:
                    t[k] = d[k]
            t["hits"] += 1
            t["misses"] = 0
            t["occluded"] = 0
            # Per-enemy profile: accumulate this scan's boundary points and,
            # once enough of the circumference is mapped, pull the centre
            # toward the de-biased fit and publish the radius.
            if enemy_profile_enabled and d.get("points_xy"):
                corr = t["profile"].observe(d["points_xy"], t["x"], t["y"])
                if corr is not None:
                    t["x"], t["y"] = corr
                t["radius"] = t["profile"].radius or Perception.robot_radius_mm
                t["mapped"] = t["profile"].centred
                t["coverage"] = t["profile"].coverage()

        # Unmatched tracks: in clear view they burn misses; behind a known blocker
        # they coast without penalty (up to max_occluded).
        for j, t in enumerate(self._tracks):
            if j in used_t:
                continue
            if (visibility is not None
                    and not visibility.observed(t["x"], t["y"])):
                t["occluded"] += 1
                if t["occluded"] > self.max_occluded:
                    t["misses"] += 1 # stale ghost, let it die
            else:
                t["misses"] += 1
        self._tracks = [t for t in self._tracks if t["misses"] <= self.max_misses]

        # unmatched detections seed new tracks (up to the field cap)
        for i, d in enumerate(dets):
            if i in used_d or len(self._tracks) >= self.max_tracks:
                continue
            t = dict(d)
            t["id"] = self._next_id
            t["hits"] = 1
            t["misses"] = 0
            t["occluded"] = 0
            t["profile"] = EnemyProfile()
            t["radius"] = Perception.robot_radius_mm
            t["mapped"] = False
            t["coverage"] = 0
            if enemy_profile_enabled and d.get("points_xy"):
                t["profile"].observe(d["points_xy"], t["x"], t["y"])
            self._next_id += 1
            self._tracks.append(t)

        # Publish confirmed tracks without the internal keys (the live
        # EnemyProfile and raw boundary points), so shared state and the Bluetooth
        # serialiser stay clean.
        hide = ("profile", "points_xy")
        return [{**{k: v for k, v in t.items() if k not in hide},
                 "occluded": t["occluded"] > 0}
                for t in self._tracks if t["hits"] >= self.confirm_hits]


class TeammateID:
    """positive identification of the single friendly bot among tracks."""

    match_mm = 300.0 # candidate must sit this close to the broadcast pose
    confirm = 5 # consecutive matching scans to earn the ID
    revoke_mm = 600.0 # ID'ed track this far from a live broadcast ...
    revoke_n = 8 # ... for this many consecutive scans -> revoke

    def __init__(self):
        """start with nobody identified, everyone is an enemy until classify() earns an ID."""
        self._id = None # confirmed friendly track id
        self._cand = None # (track_id, consecutive_matches)
        self._diverge = 0

    def reset(self):
        """forget the identified friendly and any in-progress candidate."""
        self.__init__()

    def classify(self, tracks, teammate_pos):
        """tracks: confirmed RobotTracker output (dicts with 'id'); teammate_pos: the live
        broadcast (x, y), or None if silent.
        """
        # maintain an existing ID
        if self._id is not None:
            trk = next((t for t in tracks if t["id"] == self._id), None)
            if trk is None:
                print(f"[mate] track {self._id} died, friendly ID revoked",
                      flush=True)
                self._id, self._diverge = None, 0
            else:
                if teammate_pos is not None:
                    d = math.hypot(trk["x"] - teammate_pos[0],
                                   trk["y"] - teammate_pos[1])
                    if d > self.revoke_mm:
                        self._diverge += 1
                        if self._diverge >= self.revoke_n:
                            print(f"[mate] track {self._id} diverged "
                                  f"{d:.0f} mm from broadcast, ID revoked",
                                  flush=True)
                            self._id, self._diverge = None, 0
                    else:
                        self._diverge = 0
                if self._id is not None:
                    return trk, [t for t in tracks if t["id"] != self._id]

        # no ID: everyone is an enemy; earn one from a live broadcast
        if teammate_pos is not None and tracks:
            near = min(tracks,
                       key=lambda t: math.hypot(t["x"] - teammate_pos[0],
                                                t["y"] - teammate_pos[1]))
            d = math.hypot(near["x"] - teammate_pos[0],
                           near["y"] - teammate_pos[1])
            if d <= self.match_mm:
                if self._cand is not None and self._cand[0] == near["id"]:
                    self._cand = (near["id"], self._cand[1] + 1)
                else:
                    self._cand = (near["id"], 1)
                if self._cand[1] >= self.confirm:
                    self._id, self._cand = near["id"], None
                    print(f"[mate] track {self._id} identified as the "
                          f"friendly", flush=True)
                    return near, [t for t in tracks if t["id"] != self._id]
            else:
                self._cand = None
        else:
            self._cand = None
        return None, list(tracks)


# Enemy velocity: field-frame position differencing over a short window per track.
# RobotTracker already publishes absolute field positions, so no ego-motion subtraction is
# needed. Same jump-reset idiom as the ball and teammate estimators.
enemy_vel_window_s = 0.4 # velocity = displacement over this window
enemy_vel_jump_mm = 500.0 # a track jumping further than this resets its history
enemy_vel_min_span_s = 0.15 # need at least this much history to estimate


class EnemyVelocityTracker:
    """field-frame velocity per tracked enemy id."""

    def __init__(self):
        """no history for any id yet."""
        self._hist = {} # track id -> deque[(t, x, y)]

    def reset(self):
        """drop every track's history (e.g. on a re-localise or tracker reset)."""
        self._hist = {}

    def update(self, t, tracks):
        """feed one tick's confirmed enemy track list; returns {id: (vx, vy)}."""
        live = set()
        for trk in tracks:
            tid = trk.get("id")
            if tid is None:
                continue
            live.add(tid)
            h = self._hist.setdefault(tid, collections.deque())
            if h:
                t0, x0, y0 = h[-1]
                if (t - t0 > 0.5
                        or math.hypot(trk["x"] - x0, trk["y"] - y0)
                        > enemy_vel_jump_mm):
                    h.clear() # track swap / association glitch
            h.append((t, float(trk["x"]), float(trk["y"])))
            while h and t - h[0][0] > enemy_vel_window_s:
                h.popleft()
        for tid in [k for k in self._hist if k not in live]:
            del self._hist[tid] # dead track, drop its history with it
        return {tid: self._velocity(h) for tid, h in self._hist.items()}

    @staticmethod
    def _velocity(hist):
        """endpoint difference over one track's window (the tracker's EMA already smooths the
        track, so a least-squares fit adds little).
        """
        if len(hist) < 2:
            return 0.0, 0.0
        t0, x0, y0 = hist[0]
        t1, x1, y1 = hist[-1]
        dt = t1 - t0
        if dt < enemy_vel_min_span_s:
            return 0.0, 0.0
        return (x1 - x0) / dt, (y1 - y0) / dt


# Ball velocity: lead the capture point by the ball's field-frame velocity, not just
# its last-seen position.
ball_vel_window_s = 0.25 # velocity = displacement over this window
ball_vel_jump_mm = 400.0 # a fix jumping further than this resets history
ball_vel_min_span_s = 0.04 # need at least this much history to estimate

# Stationary-ball gate: report zero velocity if the last 5 fixes barely moved, since our
# own motion adds apparent jitter to a real fit.
ball_vel_stationary_std_mm = 10.0


def _stdev(vals):
    """population standard deviation (small enough not to need `statistics`)."""
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    return math.sqrt(sum((v - m) ** 2 for v in vals) / n)


class BallVelocityEstimator:
    """field-frame ball velocity from successive ball fixes."""

    def __init__(self):
        """no history yet: velocity() is (0, 0) until update() has had a few fixes."""
        self._hist = collections.deque() # (t, bx, by)

    def reset(self):
        """drop all history (e.g. after losing sight of the ball)."""
        self._hist.clear()

    def update(self, t, bx, by):
        """feed one fresh camera fix, projected into the field frame."""
        if self._hist:
            t0, x0, y0 = self._hist[-1]
            if (t - t0 > 0.5
                    or math.hypot(bx - x0, by - y0) > ball_vel_jump_mm):
                self._hist.clear() # detection glitch / stale history
        self._hist.append((t, bx, by))
        while self._hist and t - self._hist[0][0] > ball_vel_window_s:
            self._hist.popleft()

    def velocity(self):
        """(vx, vy) field mm/s, (0, 0) until enough history has accumulated."""
        n = len(self._hist)
        if n < 3:
            return 0.0, 0.0
        t0 = self._hist[0][0]
        if self._hist[-1][0] - t0 < ball_vel_min_span_s:
            return 0.0, 0.0
        recent = list(self._hist)[-5:]
        if len(recent) >= 5:
            std_sum = (_stdev([h[1] for h in recent])
                       + _stdev([h[2] for h in recent]))
            if std_sum < ball_vel_stationary_std_mm:
                return 0.0, 0.0
        ts = [h[0] - t0 for h in self._hist]
        tm = sum(ts) / n
        stt = sum((t - tm) ** 2 for t in ts)
        if stt < 1e-9:
            return 0.0, 0.0
        xm = sum(h[1] for h in self._hist) / n
        ym = sum(h[2] for h in self._hist) / n
        vx = sum((t - tm) * (h[1] - xm) for t, h in zip(ts, self._hist)) / stt
        vy = sum((t - tm) * (h[2] - ym) for t, h in zip(ts, self._hist)) / stt
        return vx, vy


# Kalman ball tracking: a constant-velocity filter per axis over the same field-frame
# fixes, and the controllers' live ball-velocity source. Over the windowed fit it adds two
# things: dead reckoning through occlusion (between sightings the state extrapolates along
# its velocity with growing uncertainty, so a ball that slips behind a robot can still be
# chased for a short window before BallMemory takes over), and innovation gating (a fix
# too far from the prediction is a glitch or a re-acquisition after a kick, and resets the
# filter; being prediction-relative, a fast ball still passes). Two independent
# 2-state filters (x/vx, y/vy): with axis-aligned measurement noise the 4x4 filter
# decouples exactly.

# camera blob-centroid noise (mm), the measurement standard deviation
ball_kalman_sigma_meas_mm = 40.0
# process noise: white-acceleration std (mm/s^2) standing in for kicks and friction; sets
# how fast the velocity estimate may bend (a kicked ball is re-learned within a few
# frames)
ball_kalman_sigma_accel_mmss = 2500.0
# initial velocity std (mm/s) when seeded from one fix: generous, so the first updates
# pull the velocity in quickly
ball_kalman_init_vel_std_mmss = 600.0
# innovation gate (mm): a fix this far from the predicted position re-seeds the filter
ball_kalman_r_jump_mm = 350.0
# confidence right after a fresh fix, decaying with age (gated like BallMemory's)
ball_kalman_conf0 = 0.55
# confidence decay time constant (s): drops below ball_mem_min_conf (0.2) at about 1.07 s,
# so the memory tier takes over just inside the coast cap
ball_kalman_conf_tau_s = 0.8
# hard coast cap (s): past this age est() reports conf 0, however smooth the extrapolation
ball_kalman_max_coast_s = 1.2
# velocity deadband (mm/s): a stationary ball's filtered velocity jitters at about this
# magnitude, so below it report exactly (0, 0)
ball_kalman_v_deadband_mmss = 120.0


class BallKalman:
    """constant-velocity Kalman filter on the ball's field-frame position, per axis."""

    def __init__(self):
        """no state yet, update() seeds it from the first fix."""
        self._last_t = None # monotonic time of the last update()
        # per-axis state/covariance: self._x[axis] = [p, v], self._P[axis] = [[pp, pv], [pv, vv]]
        self._x = [None, None]
        self._P = [None, None]

    def reset(self):
        """drop all state (e.g. after losing the ball or a re-localise)."""
        self._last_t = None
        self._x = [None, None]
        self._P = [None, None]

    def _predict(self, axis, dt):
        """advance one axis' [p, v] state and covariance by dt, in place."""
        x, P = self._x[axis], self._P[axis]
        p, v = x
        pp, pv, vv = P[0][0], P[0][1], P[1][1]
        # state
        p2 = p + v * dt
        # covariance: F P F^T for F = [[1, dt], [0, 1]]
        pp2 = pp + 2.0 * dt * pv + dt * dt * vv
        pv2 = pv + dt * vv
        vv2 = vv
        # + Q, the white-acceleration process noise
        q = ball_kalman_sigma_accel_mmss ** 2
        pp2 += q * dt ** 4 / 4.0
        pv2 += q * dt ** 3 / 2.0
        vv2 += q * dt ** 2
        self._x[axis] = [p2, v]
        self._P[axis] = [[pp2, pv2], [pv2, vv2]]

    def update(self, t, bx, by):
        """feed one fresh camera fix, projected into the field frame."""
        r0 = ball_kalman_sigma_meas_mm ** 2
        v0 = ball_kalman_init_vel_std_mmss ** 2
        if self._last_t is None or self._x[0] is None:
            # seed from this fix alone: position known to measurement noise,
            # velocity wide open
            self._x = [[bx, 0.0], [by, 0.0]]
            self._P = [[[r0, 0.0], [0.0, v0]],
                       [[r0, 0.0], [0.0, v0]]]
            self._last_t = t
            return
        dt = t - self._last_t
        if dt <= 0.0:
            dt = 1e-3 # duplicate/stale timestamp, still take the update
        for axis, z in enumerate((bx, by)):
            self._predict(axis, dt)
            p, v = self._x[axis]
            pp, pv, _vv = self._P[axis][0][0], self._P[axis][0][1], self._P[axis][1][1]
            innov = z - p
            if abs(innov) > ball_kalman_r_jump_mm:
                # glitch or re-acquisition after a kick: start over from this fix
                self._x[axis] = [z, 0.0]
                self._P[axis] = [[r0, 0.0], [0.0, v0]]
                continue
            s = pp + r0
            kp = pp / s
            kv = pv / s
            self._x[axis] = [p + kp * innov, v + kv * innov]
            # P = (I - K H) P, kept explicit and symmetric
            self._P[axis] = [[(1.0 - kp) * pp, (1.0 - kp) * pv],
                             [(1.0 - kp) * pv, _vv - kv * pv]]
        self._last_t = t

    def est(self, t):
        """best current belief at time t without updating: (x, y, vx, vy, conf), or None with
        no state. conf decays with age since the last fix and is 0 past the coast cap.
        """
        if self._last_t is None or self._x[0] is None:
            return None
        dt = max(0.0, t - self._last_t)
        if dt > ball_kalman_max_coast_s:
            return None
        (px, vx), (py, vy) = self._x
        conf = ball_kalman_conf0 * math.exp(-dt / ball_kalman_conf_tau_s)
        return (px + vx * dt, py + vy * dt, vx, vy, conf)

    def velocity(self):
        """(vx, vy) field mm/s from the last update, (0, 0) until seeded or below the deadband."""
        if self._last_t is None or self._x[0] is None:
            return 0.0, 0.0
        vx, vy = self._x[0][1], self._x[1][1]
        if math.hypot(vx, vy) < ball_kalman_v_deadband_mmss:
            return 0.0, 0.0
        return vx, vy


# Teammate velocity (for leading a pass), same jump-reset idiom.
pass_teammate_vel_window_s = 0.6 # teammate velocity: displacement window
pass_teammate_vel_jump_mm = 800.0 # a fix jumping further than this resets it
pass_teammate_vel_min_span_s = 0.2 # minimum history before trusting it


class TeammateVelocityEstimator:
    """field-frame teammate velocity from successive teammate_pos_bt fixes."""

    def __init__(self):
        """no history yet: velocity() is (0, 0) until update() has had a few fixes."""
        self._hist = collections.deque() # (t, tx, ty)

    def reset(self):
        """drop all history (e.g. after the teammate link goes stale)."""
        self._hist.clear()

    def update(self, t, tx, ty):
        """feed one fresh teammate_pos_bt fix."""
        if self._hist:
            t0, x0, y0 = self._hist[-1]
            if (t - t0 > 0.5
                    or math.hypot(tx - x0, ty - y0) > pass_teammate_vel_jump_mm):
                self._hist.clear() # link glitch / stale history
        self._hist.append((t, tx, ty))
        while self._hist and t - self._hist[0][0] > pass_teammate_vel_window_s:
            self._hist.popleft()

    def velocity(self):
        """(vx, vy) field mm/s, (0, 0) until enough history has accumulated."""
        if len(self._hist) < 3:
            return 0.0, 0.0
        t0, x0, y0 = self._hist[0]
        t1, x1, y1 = self._hist[-1]
        dt = t1 - t0
        if dt < pass_teammate_vel_min_span_s:
            return 0.0, 0.0
        return (x1 - x0) / dt, (y1 - y0) / dt


# Ball memory (anti ball-hiding): when the ball vanishes, hold its last spot, then
# attribute it to the nearest bot and ride that bot's track.
ball_mem_coast_s = 0.7 # keep the exact last spot this long
ball_mem_attach_mm = 350.0 # bot this close at disappearance surely took it
ball_mem_near_mm = 900.0 # ... this close is still the probable holder
ball_mem_rebind_mm = 400.0 # dead holder track hands the ball to a bot this close
ball_mem_sigma_mm = 500.0 # distance -> probability spread for candidates()
ball_mem_max_s = 6.0 # unattributed memories expire after this long
ball_mem_high_conf = 0.85
ball_mem_low_conf = 0.4 # decays to half of this across the near band
ball_mem_min_conf = 0.2 # seek acts on inferred balls at/above this


class BallMemory:
    """last-seen ball state and holder attribution."""

    def __init__(self):
        """start with no memory of the ball at all."""
        self._last = None # (x, y, t) last camera fix, field frame
        self._holder = None # (track_id, conf) once attributed
        self._holder_pos = None # holder's last known position

    def reset(self):
        """forget the ball entirely (e.g. after a goal or a field reset)."""
        self._last = None
        self._holder = None
        self._holder_pos = None

    def seen(self, x, y, t):
        """record a fresh direct sighting, clearing any attribution (a real sighting always
        wins).
        """
        self._last = (x, y, t)
        self._holder = None
        self._holder_pos = None

    def _attribute(self, bots):
        """pin the ball's disappearance on whichever tracked bot was closest to its last-seen
        spot.
        """
        lx, ly, _ = self._last
        best = min(bots, key=lambda b: math.hypot(b["x"] - lx, b["y"] - ly))
        d = math.hypot(best["x"] - lx, best["y"] - ly)
        if d <= ball_mem_attach_mm:
            self._holder = (best["id"], ball_mem_high_conf)
        elif d <= ball_mem_near_mm:
            # probability decays with how far the bot was from the ball
            frac = ((d - ball_mem_attach_mm)
                    / (ball_mem_near_mm - ball_mem_attach_mm))
            self._holder = (best["id"], ball_mem_low_conf * (1.0 - 0.5 * frac))
        if self._holder is not None:
            self._holder_pos = (best["x"], best["y"])

    def infer(self, t, bots):
        """bots: tracked robot dicts (field mm, stable 'id'). The holder can be followed while
        occluded, since the tracker keeps tracks alive behind known blockers.
        """
        if self._last is None:
            return None
        lx, ly, lt = self._last
        age = t - lt
        if age <= ball_mem_coast_s:
            return lx, ly, 0.9

        if self._holder is None and bots and age <= ball_mem_max_s:
            self._attribute(bots)

        if self._holder is not None:
            hid, conf = self._holder
            trk = next((b for b in bots if b.get("id") == hid), None)
            if trk is None and bots and self._holder_pos is not None:
                # holder track died: hand the ball to whoever stands there
                hx, hy = self._holder_pos
                near = min(bots,
                           key=lambda b: math.hypot(b["x"] - hx, b["y"] - hy))
                if math.hypot(near["x"] - hx, near["y"] - hy) <= ball_mem_rebind_mm:
                    self._holder = (near["id"], conf)
                    trk = near
            if trk is not None:
                self._holder_pos = (trk["x"], trk["y"])
                return trk["x"], trk["y"], conf # no expiry: stay on the bot
            # holder gone and nobody near where it was: attribution is void
            self._holder = None
            self._holder_pos = None

        # Once out of memory (no attribution, hold window long expired), don't
        # return None: that dropped callers into their "completely lost" retreat,
        # so an opponent hiding the ball made us abandon the play. Floor the
        # confidence at ball_mem_min_conf instead, the exact level every caller
        # gates "chase the estimate" on, so we keep pressing the last known spot.
        return lx, ly, ball_mem_min_conf if age > ball_mem_max_s else 0.25

    def candidates(self, t, bots):
        """probability over possible ball holders for the goalie: [(x, y, prob), ...], best
        first, at most two, normalised over the returned set.
        """
        est = self.infer(t, bots) # keeps attribution fresh
        if est is None:
            return []
        if not bots:
            return [(est[0], est[1], 1.0)]
        lx, ly, _ = self._last
        hid = self._holder[0] if self._holder is not None else None
        weighted = []
        for b in bots:
            d = math.hypot(b["x"] - lx, b["y"] - ly)
            w = math.exp(-(d * d) / (2.0 * ball_mem_sigma_mm ** 2))
            if b.get("id") == hid:
                w = max(w, self._holder[1]) # attributed bot dominates
            weighted.append((b["x"], b["y"], w))
        weighted.sort(key=lambda c: c[2], reverse=True)
        top = weighted[:2]
        tot = sum(w for _, _, w in top) or 1.0
        return [(x, y, w / tot) for x, y, w in top]
