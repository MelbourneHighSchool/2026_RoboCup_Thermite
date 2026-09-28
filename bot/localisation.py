"""Monte Carlo Localisation (particle filter): the robot's only pose estimator.

Wheel odometry and the IMU turn drive the motion update; the likelihood field against
FieldModel weights the particles. It copes better with partial occlusion and the field's
left-right symmetry than the point-to-line ICP it replaced.
"""


import math
import os
import sys
import time

import numpy as np

from bot.compass import heading_carry_max_deg
from bot.diagnostics import health_stale_s, _health_t
from bot.field import FieldModel, wrap_deg
from bot.odometry import WheelOdometry
from bot.state import _lock as lock, _state as shared_state

# Optional compiled core for the three hot loops (motion update, sensor weighting,
# resampling), about 70x faster on sensor_weights. Falls back to the numpy code below if
# not built; parity is tested in tests/test_runtime_integrity.py and
# native/test_mcl_native.py.
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "native"))
    import mcl_native
except ImportError:
    mcl_native = None


def circular_mean_deg(angles_deg, weights=None):
    """weighted circular mean (atan2 of the weighted sin/cos sums, so it survives the +-180
    wrap).
    """
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
    """circular mean of the latest full window of yaw samples: a startup baseline that isn't
    just the first reading.
    """

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
    """circular mean of a short burst of IMU headings from _state["imu_heading"], or None if no
    reading arrives within timeout_s (no IMU, or not warmed up): treat that as "no prior",
    not a failure.
    """
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
    """particle-filter localisation with Perception's localise/global_localise interface."""

    n_particles = 300
    sensor_sigma_mm = 70.0 # likelihood-field Gaussian width
    sensor_sample_stride = 3 # subsample scan points for the sensor update (perf)
    trans_noise_frac = 0.04 # motion noise proportional to distance moved
    trans_noise_floor_mm = 1.5
    turn_noise_frac = 0.03 # motion noise proportional to heading change
    turn_noise_floor_deg = 0.25
    resample_ess_frac = 0.5 # resample once ESS drops below this fraction of n_particles
    imu_prior_sigma_deg = 10.0 # soft heading-prior width (wide: a hint, not a constraint)
    # The prior is dropped once the last good IMU sample is older than health_stale_s.
    # That covers the imu_fault latch and also a compass thread that hung without
    # erroring, where imu_heading just freezes.
    imu_prior_max_age_s = health_stale_s
    # Off by default: this is the one absolute use of the IMU (every other consumer
    # takes deltas, so frames cancel). The BNO08x zeroes its yaw at boot, so the prior
    # assumes the robot's field heading at boot equals the captured baseline. Settle
    # that at the bench first (BENCH_TEST_CHECKLIST section 9).
    imu_prior_enabled = False
    # Motion-update turn from the IMU: the change between the live readings at consecutive
    # updates. Two live samples, so there is no history window to fall off the end of (the
    # old window ended at "now", which the history never covers, so the turn was always 0).
    # False reverts to sensor-only rotation tracking.
    imu_motion_enabled = True
    inlier_threshold_mm = 60.0 # for the returned inliers/outliers/rms_mm

    def __init__(self, n_particles=None):
        if n_particles is not None:
            self.n_particles = n_particles
        self.particles = None # (N, 3): x, y, theta_deg
        self.weights = None # (N,)
        # own instance: sharing one would reset another poller's baseline
        self.odom = WheelOdometry()
        self.imu_bias_deg = None # soft prior centre, set by set_imu_yaw_prior
        self.imu_ref_raw_deg = None # imu_heading at the moment that baseline was captured
        self.imu_bias_t = None # capture time, telemetry only
        self.imu_prior_seen = None # (raw, monotonic t) last used by prior_heading_deg
        self.imu_motion_last = None # imu_heading at the last motion update
        self.rng = np.random.default_rng()

    # Optional IMU soft prior: re-derived every sensor update from the live reading,
    # and cleared whenever the sensor is stale or missing.
    def set_imu_yaw_prior(self, startup_yaw_deg, startup_raw_yaw_deg=None):
        """set the prior's baseline once at startup (None disables it). The centre then tracks
        the IMU's turn since capture (see prior_heading_deg). startup_raw_yaw_deg is the
        raw reading at capture if the baseline is in another frame; it defaults to
        startup_yaw_deg.
        """
        self.imu_bias_deg = startup_yaw_deg
        self.imu_ref_raw_deg = (startup_yaw_deg if startup_raw_yaw_deg is None
                                else startup_raw_yaw_deg)
        self.imu_bias_t = time.monotonic()
        self.imu_prior_seen = None

    def clear_imu_yaw_prior(self):
        """drop the soft heading prior until set_imu_yaw_prior() is called again."""
        self.imu_bias_deg = None
        self.imu_prior_seen = None

    def prior_heading_deg(self):
        """soft prior centre (field-frame heading, deg) for this sensor update, or None to run
        without one.
        """
        if not self.imu_prior_enabled or self.imu_bias_deg is None:
            return None
        now = time.monotonic()
        last_good = _health_t.get("imu")
        if last_good is None or now - last_good > self.imu_prior_max_age_s:
            return None # no fresh good sample to derive a hint from
        with lock:
            raw_now = shared_state["imu_heading"]
        if raw_now is None:
            return None
        prev = self.imu_prior_seen
        if prev is not None and now - prev[1] > self.imu_prior_max_age_s:
            # Our last sample is stale too and the reading jumped across the
            # gap: the chip re-zeroed its yaw (brownout or restart), so the
            # baseline's frame is gone. Same rule as the heading carry: a
            # jump past heading_carry_max_deg is a stale baseline, not a
            # turn. Drop it.
            if abs(wrap_deg(raw_now - prev[0])) > heading_carry_max_deg:
                self.clear_imu_yaw_prior()
                return None
        self.imu_prior_seen = (raw_now, now)
        return wrap_deg(self.imu_bias_deg + wrap_deg(raw_now - self.imu_ref_raw_deg))

    def imu_turn_since_last(self):
        """IMU turn (deg, cw+) since the previous call, or None: no reading, a stale sensor,
        or a jump too big to be a turn (a re-zeroed chip). Every call moves the baseline.
        """
        with lock:
            raw = shared_state["imu_heading"]
        prev, self.imu_motion_last = self.imu_motion_last, raw
        if not self.imu_motion_enabled or raw is None or prev is None:
            return None
        last_good = _health_t.get("imu")
        if last_good is None or time.monotonic() - last_good > health_stale_s:
            return None
        d = wrap_deg(raw - prev)
        if abs(d) > heading_carry_max_deg:
            return None
        return d

    # Particle population
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
        dth = self.imu_turn_since_last()
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
            # seeded from self.rng so a fixed numpy seed still reproduces a
            # run end to end
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

        # rotate each particle's robot-frame (fwd, right) by its heading, so
        # particles with different headings spread differently and the sensor
        # update can tell them apart
        ch, sh = np.cos(th), np.sin(th)
        dx = ch * right_n + sh * fwd_n
        dy = ch * fwd_n - sh * right_n
        self.particles[:, 0] += dx
        self.particles[:, 1] += dy
        self.particles[:, 2] = (self.particles[:, 2] + dth_n) % 360.0
        # keep particles on the field
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
            seg_a = np.ascontiguousarray(FieldModel._seg_a, dtype=np.float64)
            seg_ex = np.ascontiguousarray(FieldModel._seg_ex, dtype=np.float64)
            seg_ey = np.ascontiguousarray(FieldModel._seg_ey, dtype=np.float64)
            w = mcl_native.sensor_weights(
                np.ascontiguousarray(self.particles, dtype=np.float64),
                np.ascontiguousarray(pts, dtype=np.float64),
                seg_a, seg_ex, seg_ey, self.sensor_sigma_mm)
        else:
            xl, yl = pts[:, 0], pts[:, 1]
            x = self.particles[:, 0][:, None] # (N, 1)
            y = self.particles[:, 1][:, None]
            h = np.radians(self.particles[:, 2])[:, None]
            c, s = np.cos(h), np.sin(h)
            px = x + xl[None, :] * c + yl[None, :] * s # (N, M)
            py = y - xl[None, :] * s + yl[None, :] * c

            flat = np.column_stack([px.ravel(), py.ravel()])
            dist, nx, ny = FieldModel.nearest_wall_batch(flat)
            dist = dist.reshape(n_particles, len(pts))

            mean_sq = np.mean(np.minimum(dist, 4.0 * self.sensor_sigma_mm) ** 2, axis=1)
            log_w = -mean_sq / (2.0 * self.sensor_sigma_mm ** 2)
            log_w -= log_w.max() # numerically stable before exponentiating
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
            cumsum[-1] = 1.0 # guard float rounding
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
        """Perception.localise's signature and return shape; one filter cycle per call
        (max_iters is ignored, and init_pose only seeds the first call).
        """
        now = time.monotonic()
        if self.particles is None:
            self.seed_gaussian(init_pose)
            self.odom.poll(now) # prime the odometry baseline, first delta is meaningless
            self.imu_turn_since_last() # and the IMU turn baseline
        else:
            self.motion_update(now)

        self.weights = self.sensor_weights(points_local)
        pose = self.estimate_pose()
        self.resample()
        return self.result_for(pose, points_local)

    # Exploration jitter for global_localise, shrinking each round. One static scan
    # gives resampling nothing to diversify against, so resample-only rounds lock onto
    # whatever looks best in round one (seen converging ~700 mm off and staying
    # there). Jittering every particle each round keeps the population exploring until
    # the true mode wins.
    global_jitter_schedule = (
        (300.0, 60.0), (200.0, 40.0), (120.0, 25.0),
        (70.0, 12.0), (35.0, 6.0), (15.0, 3.0), (0.0, 0.0),
    )

    def global_localise(self, points_local, x_step=None, y_step=None, heading_step=None,
                        coarse_iters=None, top_k=None, refine_iters=None,
                        inlier_threshold=None, min_inliers=40,
                        rough_region=None, known_heading=None, heading_tolerance=20.0):
        """Perception.global_localise's signature and return shape: scatter particles over the
        field (or the rough_region/known_heading hints) and run sensor + jitter + resample
        rounds to converge.
        """
        if len(points_local) < min_inliers:
            return None
        self.seed_uniform(rough_region, known_heading, heading_tolerance)
        self.odom.poll(time.monotonic())
        self.imu_turn_since_last() # fresh baseline for the first motion update
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
