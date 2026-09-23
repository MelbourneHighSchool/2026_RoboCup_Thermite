"""
bot/localisation.py: a real 2D Monte Carlo Localisation (particle filter) -
the robot's only pose estimator. Ported idea from a second reference project:
its own localisation is a native (C++) particle filter with IMU yaw fed in as
a soft heading prior, a structurally different (and generally more robust to
partial occlusion and the field's left-right symmetry) approach than the
point-to-line ICP scan-matching this codebase used to run instead.

Not a reskin of ICP: this keeps its own persistent particle population across
calls (an actual filter, not a single best-fit refined per call), uses a real
odometry-driven motion model (drive() the particles, THEN weight them against
the scan) and a likelihood-field sensor model (reusing
FieldModel.nearest_wall_batch, the same wall-distance primitive ICP's own
point-to-line residual was built on - the field geometry is shared, only how
the pose is estimated from it differs), low-variance resampling, and a genuine
soft IMU-yaw prior baked into the particle weights every tick (not a
heading-delta bolt-on carried between lidar revolutions the way the old ICP
path's IMU assist worked).

localise(points_local, init_pose, ...) and global_localise(points_local, ...)
are what bot/perception.py's Perception hands its own two localisation entry
points off to, so lidar_thread (bot/lidar.py) drives the same scan
acquisition/motion-deskew/quality-watchdog loop it always did; only the
pose-estimation algorithm underneath is this. One MCL instance persists between
calls (bot/perception.py owns it) since a particle filter's whole value is
carrying its belief forward, not refitting from scratch every tick.

Uses its own independent WheelOdometry instance for the motion model's
translation component, NOT state.wheel_odom - that instance is already polled
once per revolution by lidar_thread's own deskew step (sec 4.13's own comment
explains why two pollers sharing one WheelOdometry would each reset the other's
last-poll baseline), so this needs its own, exactly the same reasoning that
already gives odom_thread its own separate state.odom_wheel instance. The
rotation component reuses imu_turn_between directly (a read against a
timestamped history buffer, not a stateful poll - safe to call
independently).
"""


import math
import os
import sys
import time

import numpy as np

from bot.compass import _imu_turn_between as imu_turn_between
from bot.field import FieldModel, wrap_deg
from bot.odometry import WheelOdometry
from bot.state import _lock as lock, _state as shared_state

# native/mcl_native.{so,pyd}: C++ port of the three per-tick hot loops
# (motion update, likelihood-field sensor weighting, resampling) - see
# native/mcl_native.cpp's own header comment. Measured ~18x faster on
# sensor_weights alone (the dominant cost: O(n_particles * n_scan_points *
# n_wall_segments)), confirmed a real chunk of the localisation budget
# (~30ms/call in pure Python at 300 particles/60 points). Falls back to
# the pure-numpy implementations below (byte-for-byte / machine-precision
# parity tested, see native/test_mcl_native.py) if the extension isn't
# built for this platform.
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "native"))
    import mcl_native
except ImportError:
    mcl_native = None


def circular_mean_deg(angles_deg, weights=None):
    """weighted circular mean, see RollingYawSampler below (that project's
    own startup-yaw averaging) for why (atan2 of the weighted sin/cos sums,
    not a naive mean that breaks across +-180)."""
    angles_deg = np.asarray(angles_deg, dtype=np.float64)
    if weights is None:
        weights = np.ones_like(angles_deg)
    rad = np.radians(angles_deg)
    s = float(np.sum(weights * np.sin(rad)))
    c = float(np.sum(weights * np.cos(rad)))
    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(s, c))


class RollingYawSampler:
    """Ported from that second reference project's own RollingYawSampler: circularly
    average the latest complete window of yaw samples, for a startup
    heading reference that isn't just the first (possibly noisy) IMU
    reading. See circular_mean_deg's own comment for why circular."""

    def __init__(self, sample_count=12):
        self.samples = []
        self.maxlen = sample_count

    def reset(self):
        self.samples = []

    def add(self, yaw_deg):
        if yaw_deg is None:
            return None
        self.samples.append(float(yaw_deg))
        if len(self.samples) > self.maxlen:
            self.samples.pop(0)
        if len(self.samples) < self.maxlen:
            return None
        return circular_mean_deg(self.samples)


def capture_startup_yaw(sample_count=12, sample_interval=0.05, timeout_s=3.0):
    """Average a short burst of IMU headings so the startup heading isn't
    just the first reading - see RollingYawSampler's own comment. Reads the
    IMU heading history bot/compass.py publishes into the shared shared_state, the
    same sensor the pose fusion already runs off.
    Returns None if no IMU reading ever arrives within timeout_s (no IMU
    fitted, or it hasn't warmed up yet) - callers should treat that as
    "no soft prior available", not fail outright."""
    sampler = RollingYawSampler(sample_count)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with lock:
            yaw = shared_state.get("imu_heading")
        result = sampler.add(yaw)
        if result is not None:
            return result
        time.sleep(sample_interval)
    return None


class MCL:
    """Particle-filter localisation, behind Perception.localise/
    global_localise (bot/perception.py). See module docstring."""

    n_particles           = 300
    sensor_sigma_mm        = 70.0     # likelihood-field Gaussian width
    sensor_sample_stride   = 3         # subsample scan points for the per-particle sensor update (perf)
    trans_noise_frac       = 0.04      # motion noise proportional to distance moved
    trans_noise_floor_mm   = 1.5
    turn_noise_frac        = 0.03      # motion noise proportional to heading change
    turn_noise_floor_deg   = 0.25
    resample_ess_frac      = 0.5       # resample once effective sample size drops below this fraction of n_particles
    imu_prior_sigma_deg    = 10.0      # soft heading-prior width (wide: a hint, not a hard constraint)
    inlier_threshold_mm    = 60.0      # for the returned inliers/outliers/rms_mm, matching Perception's own field names

    def __init__(self, n_particles=None):
        if n_particles is not None:
            self.n_particles = n_particles
        self.particles    = None   # (N, 3): x, y, theta_deg
        self.weights      = None   # (N,)
        self.odom         = WheelOdometry()   # own instance, see module docstring
        self.imu_bias_deg = None         # set via set_imu_yaw_prior, optional soft prior centre offset
        self.rng          = np.random.default_rng()

    # Optional IMU soft prior (that project's "feed_imu_yaw_prior" idea)
    def set_imu_yaw_prior(self, startup_yaw_deg):
        """Call once at startup with capture_startup_yaw()'s result (or
        None to disable). The prior tracks IMU heading deltas from this
        reference each tick (see prior_heading_deg), not a fixed value."""
        self.imu_bias_deg = startup_yaw_deg
        self.imu_bias_t = time.monotonic()

    def prior_heading_deg(self):
        if self.imu_bias_deg is None:
            return None
        d = imu_turn_between(self.imu_bias_t, time.monotonic())
        if d is None:
            return None
        return wrap_deg(self.imu_bias_deg + d)

    # Particle population management
    def seed_gaussian(self, pose, spread_mm=180.0, spread_deg=20.0):
        x, y, h = pose
        n = self.n_particles
        xs = self.rng.normal(x, spread_mm, n)
        ys = self.rng.normal(y, spread_mm, n)
        hs = self.rng.normal(h, spread_deg, n) % 360.0
        self.particles = np.column_stack([xs, ys, hs])
        self.weights = np.full(n, 1.0 / n)

    def seed_uniform(self, rough_region=None, known_heading=None, heading_tolerance=45.0):
        n = self.n_particles
        if rough_region is not None:
            rx, ry = rough_region
            xs = self.rng.normal(rx, 400.0, n)
            ys = self.rng.normal(ry, 400.0, n)
            xs = np.clip(xs, 0.0, FieldModel.field_x)
            ys = np.clip(ys, 0.0, FieldModel.field_y)
        else:
            xs = self.rng.uniform(0.0, FieldModel.field_x, n)
            ys = self.rng.uniform(0.0, FieldModel.field_y, n)
        if known_heading is not None:
            hs = (known_heading
                  + self.rng.uniform(-heading_tolerance, heading_tolerance, n)) % 360.0
        else:
            hs = self.rng.uniform(0.0, 360.0, n)
        self.particles = np.column_stack([xs, ys, hs])
        self.weights = np.full(n, 1.0 / n)

    # Motion model
    def motion_update(self, now):
        fwd_right = self.odom.poll(now)
        dth = imu_turn_between(now - 0.12, now)
        if dth is None:
            dth = 0.0
        if fwd_right is None:
            fwd, right = 0.0, 0.0
        else:
            fwd, right = fwd_right

        n = len(self.particles)
        trans_mag = math.hypot(fwd, right)
        trans_noise = self.trans_noise_frac * trans_mag + self.trans_noise_floor_mm
        turn_noise = self.turn_noise_frac * abs(dth) + self.turn_noise_floor_deg

        if mcl_native is not None:
            # seeded from self.rng so a fixed numpy seed still gives
            # reproducible runs end to end, even though the noise draws
            # themselves happen in a separate (C++) RNG stream.
            seed = int(self.rng.integers(0, 2**63 - 1))
            self.particles = np.ascontiguousarray(self.particles, dtype=np.float64)
            mcl_native.motion_update(self.particles, fwd, right, dth,
                                     trans_noise, turn_noise,
                                     FieldModel.field_x, FieldModel.field_y, seed)
            return

        th = np.radians(self.particles[:, 2])
        fwd_n = self.rng.normal(fwd, trans_noise, n)
        right_n = self.rng.normal(right, trans_noise, n)
        dth_n = self.rng.normal(dth, turn_noise, n)

        # rotate each particle's own robot-frame (fwd, right) into ITS OWN
        # field-frame heading - the whole point of per-particle motion:
        # particles with different headings disperse differently for the
        # same measured body-frame displacement, exactly what lets the
        # sensor update later disambiguate them.
        ch, sh = np.cos(th), np.sin(th)
        dx = ch * right_n + sh * fwd_n
        dy = ch * fwd_n - sh * right_n
        self.particles[:, 0] += dx
        self.particles[:, 1] += dy
        self.particles[:, 2] = (self.particles[:, 2] + dth_n) % 360.0
        # keep particles on the field - a real robot can't be off it
        np.clip(self.particles[:, 0], -50.0, FieldModel.field_x + 50.0, out=self.particles[:, 0])
        np.clip(self.particles[:, 1], -50.0, FieldModel.field_y + 50.0, out=self.particles[:, 1])

    # Sensor model (likelihood field, FieldModel.nearest_wall_batch)
    def sensor_weights(self, points_local):
        n_particles = len(self.particles)
        if not len(points_local):
            return np.full(n_particles, 1.0 / n_particles)
        pts = np.asarray(points_local, dtype=np.float64).reshape(-1, 2)
        if self.sensor_sample_stride > 1 and len(pts) > 20:
            pts = pts[::self.sensor_sample_stride]

        if mcl_native is not None:
            seg_a = np.ascontiguousarray(FieldModel.seg_a, dtype=np.float64)
            seg_ex = np.ascontiguousarray(FieldModel.seg_ex, dtype=np.float64)
            seg_ey = np.ascontiguousarray(FieldModel.seg_ey, dtype=np.float64)
            w = mcl_native.sensor_weights(
                np.ascontiguousarray(self.particles, dtype=np.float64),
                np.ascontiguousarray(pts, dtype=np.float64),
                seg_a, seg_ex, seg_ey, self.sensor_sigma_mm)
        else:
            xl, yl = pts[:, 0], pts[:, 1]
            x = self.particles[:, 0][:, None]      # (N, 1)
            y = self.particles[:, 1][:, None]
            h = np.radians(self.particles[:, 2])[:, None]
            c, s = np.cos(h), np.sin(h)
            px = x + xl[None, :] * c + yl[None, :] * s     # (N, M)
            py = y - xl[None, :] * s + yl[None, :] * c

            flat = np.column_stack([px.ravel(), py.ravel()])
            dist, nx, ny = FieldModel.nearest_wall_batch(flat)
            dist = dist.reshape(n_particles, len(pts))

            mean_sq = np.mean(np.minimum(dist, 4.0 * self.sensor_sigma_mm) ** 2, axis=1)
            log_w = -mean_sq / (2.0 * self.sensor_sigma_mm ** 2)
            log_w -= log_w.max()   # numerically stable before exponentiating
            w = np.exp(log_w)
            total = w.sum()
            w = np.full(n_particles, 1.0 / n_particles) if total <= 1e-300 else w / total

        prior_h = self.prior_heading_deg()
        if prior_h is not None:
            dh = ((self.particles[:, 2] - prior_h + 180.0) % 360.0) - 180.0
            w = w * np.exp(-(dh ** 2) / (2.0 * self.imu_prior_sigma_deg ** 2))

        total = w.sum()
        if total <= 1e-300:
            return np.full(n_particles, 1.0 / n_particles)
        return w / total

    # Resampling (low-variance / systematic)
    def effective_sample_size(self):
        return 1.0 / np.sum(self.weights ** 2)

    def resample(self):
        n = len(self.weights)
        if self.effective_sample_size() >= self.resample_ess_frac * n:
            return
        u0 = float(self.rng.uniform())
        if mcl_native is not None:
            self.particles = mcl_native.resample(
                np.ascontiguousarray(self.particles, dtype=np.float64),
                np.ascontiguousarray(self.weights, dtype=np.float64), u0)
        else:
            positions = (u0 + np.arange(n)) / n
            cumsum = np.cumsum(self.weights)
            cumsum[-1] = 1.0   # guard float rounding
            idx = np.searchsorted(cumsum, positions)
            self.particles = self.particles[idx].copy()
        self.weights = np.full(n, 1.0 / n)

    # Pose estimate + Perception-shaped return value
    def estimate_pose(self):
        x = float(np.sum(self.weights * self.particles[:, 0]))
        y = float(np.sum(self.weights * self.particles[:, 1]))
        h = circular_mean_deg(self.particles[:, 2], self.weights) % 360.0
        return x, y, h

    def result_for(self, pose, points_local):
        X, Y, H = pose
        if not len(points_local):
            return {"pose": pose, "inliers": [], "outliers": [], "rms_mm": float("inf"),
                    "inlier_count": 0}
        pts = np.asarray(points_local, dtype=np.float64).reshape(-1, 2)
        xl, yl = pts[:, 0], pts[:, 1]
        h = math.radians(H)
        c, s = math.cos(h), math.sin(h)
        px = X + xl * c + yl * s
        py = Y - xl * s + yl * c
        dist, nx, ny = FieldModel.nearest_wall_batch(np.column_stack((px, py)))
        mask = dist <= self.inlier_threshold_mm
        inliers = list(zip(px[mask].tolist(), py[mask].tolist()))
        outliers = list(zip(px[~mask].tolist(), py[~mask].tolist()))
        rms = math.sqrt(float(np.mean(dist[mask] ** 2))) if inliers else float("inf")
        return {"pose": (X, Y, H % 360.0), "inliers": inliers, "outliers": outliers,
                "rms_mm": rms, "inlier_count": len(inliers)}

    # Perception-compatible entry points
    def localise(self, points_local, init_pose, inlier_threshold=None, max_iters=None):
        """Perception.localise's signature/return shape (max_iters is
        accepted and ignored - a particle filter has no per-call
        iteration count, its "iteration" is every tick's filter cycle)."""
        now = time.monotonic()
        if self.particles is None:
            self.seed_gaussian(init_pose)
            self.odom.poll(now)   # prime the odometry baseline, first delta is meaningless
        else:
            self.motion_update(now)
        self.weights = self.sensor_weights(points_local)
        pose = self.estimate_pose()
        self.resample()
        return self.result_for(pose, points_local)

    # Exploration jitter schedule for global_localise, decreasing per round
    # (simulated-annealing style): a single static scan gives resampling
    # nothing to diversify against (no real motion between rounds the way
    # normal tracking gets), so pure resample-only rounds collapse onto
    # whatever mode looks best in round 1 and get stuck there - confirmed
    # directly (a synthetic global search converged to a false mode ~700mm
    # off and stayed there every subsequent round, ESS never dropping low
    # enough to trigger another resample). Injecting shrinking jitter noise
    # into every particle each round (not just the resampled survivors)
    # keeps the population exploring long enough for the true mode to win
    # out before it's allowed to lock in.
    global_jitter_schedule = (
        (300.0, 60.0), (200.0, 40.0), (120.0, 25.0),
        (70.0, 12.0), (35.0, 6.0), (15.0, 3.0), (0.0, 0.0),
    )

    def global_localise(self, points_local, x_step=None, y_step=None, heading_step=None,
                        coarse_iters=None, top_k=None, refine_iters=None,
                        inlier_threshold=None, min_inliers=40,
                        rough_region=None, known_heading=None, heading_tolerance=20.0):
        """Perception.global_localise's signature/return shape. Scatter
        particles across the whole field (or rough_region/known_heading,
        the same hints Perception's version takes) and run several
        sensor+jitter+resample rounds (see global_jitter_schedule's own
        comment) to converge before reporting - a particle filter's own
        natural way of handling "no idea where we are"."""
        if len(points_local) < min_inliers:
            return None
        self.seed_uniform(rough_region, known_heading, heading_tolerance)
        self.odom.poll(time.monotonic())
        n = len(self.particles)
        for jitter_mm, jitter_deg in self.global_jitter_schedule:
            if jitter_mm or jitter_deg:
                self.particles[:, 0] += self.rng.normal(0.0, jitter_mm, n)
                self.particles[:, 1] += self.rng.normal(0.0, jitter_mm, n)
                self.particles[:, 2] = (self.particles[:, 2]
                                        + self.rng.normal(0.0, jitter_deg, n)) % 360.0
                np.clip(self.particles[:, 0], -50.0, FieldModel.field_x + 50.0, out=self.particles[:, 0])
                np.clip(self.particles[:, 1], -50.0, FieldModel.field_y + 50.0, out=self.particles[:, 1])
            self.weights = self.sensor_weights(points_local)
            u0 = float(self.rng.uniform())
            if mcl_native is not None:
                self.particles = mcl_native.resample(
                    np.ascontiguousarray(self.particles, dtype=np.float64),
                    np.ascontiguousarray(self.weights, dtype=np.float64), u0)
            else:
                positions = (u0 + np.arange(n)) / n
                cumsum = np.cumsum(self.weights)
                cumsum[-1] = 1.0
                idx = np.searchsorted(cumsum, positions)
                self.particles = self.particles[idx].copy()
            self.weights = np.full(n, 1.0 / n)
        pose = self.estimate_pose()
        result = self.result_for(pose, points_local)
        if result["inlier_count"] < min_inliers:
            return None
        result["ambiguous"] = False
        result["resolved"] = True
        return result
