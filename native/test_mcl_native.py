"""Parity + benchmark for mcl_native against bot/localisation.py's own
pure-numpy math. Run from the repo root: `python3 native/test_mcl_native.py`."""
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "native")
import numpy as np

import mcl_native
from bot.field import FieldModel


def py_sensor_weights(particles, points_local, sigma_mm):
    xl, yl = points_local[:, 0], points_local[:, 1]
    x = particles[:, 0][:, None]
    y = particles[:, 1][:, None]
    h = np.radians(particles[:, 2])[:, None]
    c, s = np.cos(h), np.sin(h)
    px = x + xl[None, :] * c + yl[None, :] * s
    py_ = y - xl[None, :] * s + yl[None, :] * c
    n_p, n_pts = px.shape
    flat = np.column_stack([px.ravel(), py_.ravel()])
    dist, _, _ = FieldModel.nearest_wall_batch(flat)
    dist = dist.reshape(n_p, n_pts)
    mean_sq = np.mean(np.minimum(dist, 4.0 * sigma_mm) ** 2, axis=1)
    log_w = -mean_sq / (2.0 * sigma_mm ** 2)
    log_w -= log_w.max()
    w = np.exp(log_w)
    return w / w.sum()


def py_resample(particles, weights, u0):
    n = len(weights)
    positions = (u0 + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    idx = np.searchsorted(cumsum, positions)
    return particles[idx].copy()


def test_sensor_weights_parity():
    rng = np.random.default_rng(3)
    seg_a, seg_ex, seg_ey = FieldModel.seg_a, FieldModel.seg_ex, FieldModel.seg_ey
    for trial in range(10):
        n_particles = rng.integers(20, 400)
        n_pts = rng.integers(5, 100)
        particles = np.column_stack([
            rng.uniform(0, FieldModel.field_x, n_particles),
            rng.uniform(0, FieldModel.field_y, n_particles),
            rng.uniform(0, 360, n_particles),
        ])
        points = np.column_stack([rng.normal(0, 800, n_pts), rng.normal(0, 800, n_pts)])
        sigma = rng.uniform(30, 150)
        w_py = py_sensor_weights(particles, points, sigma)
        w_cpp = mcl_native.sensor_weights(particles, points, seg_a, seg_ex, seg_ey, sigma)
        diff = np.max(np.abs(w_py - w_cpp))
        assert diff < 1e-9, f"trial {trial}: max diff {diff}"
    print("sensor_weights parity ok (machine precision, 10 randomized trials)")


def test_resample_parity():
    rng = np.random.default_rng(5)
    for trial in range(10):
        n = int(rng.integers(10, 500))
        particles = np.column_stack([
            rng.uniform(0, FieldModel.field_x, n),
            rng.uniform(0, FieldModel.field_y, n),
            rng.uniform(0, 360, n),
        ])
        weights = rng.random(n)
        weights /= weights.sum()
        u0 = float(rng.uniform())
        py_out = py_resample(particles, weights, u0)
        cpp_out = mcl_native.resample(particles, weights, u0)
        assert np.array_equal(py_out, cpp_out), f"trial {trial}: mismatch"
    print("resample parity ok (exact match, 10 randomized trials)")


def bench():
    rng = np.random.default_rng(1)
    n_particles = 300
    particles = np.column_stack([
        rng.uniform(0, FieldModel.field_x, n_particles),
        rng.uniform(0, FieldModel.field_y, n_particles),
        rng.uniform(0, 360, n_particles),
    ])
    points = np.column_stack([rng.normal(0, 800, 60), rng.normal(0, 800, 60)])
    seg_a, seg_ex, seg_ey = FieldModel.seg_a, FieldModel.seg_ex, FieldModel.seg_ey
    sigma = 70.0

    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        py_sensor_weights(particles, points, sigma)
    t1 = time.perf_counter()
    for _ in range(N):
        mcl_native.sensor_weights(particles, points, seg_a, seg_ex, seg_ey, sigma)
    t2 = time.perf_counter()
    py_t, cpp_t = t1 - t0, t2 - t1
    print(f"python sensor_weights: {py_t/N*1000:.3f} ms/call")
    print(f"cpp    sensor_weights: {cpp_t/N*1000:.3f} ms/call   speedup: {py_t/cpp_t:.1f}x")


if __name__ == "__main__":
    test_sensor_weights_parity()
    test_resample_parity()
    bench()
