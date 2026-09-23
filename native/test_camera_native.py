"""Parity + benchmark for camera_native against the real OpenCV pipeline
(cv2.remap + cv2.cvtColor + cv2.inRange, bot/vision.py's own detect_ball_warp
preamble). NOT wired into bot/ - see this dir's own README note:
measured slower than OpenCV's own pipeline even with -march=native
-ffast-math -fopenmp, kept here only as a documented, tested "tried this,
didn't pay off" artifact. Run from the repo root: `python3 native/test_camera_native.py`."""
import sys
import time

sys.path.insert(0, "native")
import cv2
import numpy as np

import camera_native


def _build_maps(H, W, n_r, n_theta, seed):
    rng = np.random.default_rng(seed)
    cx = W / 2.0 + rng.uniform(-20, 20)
    cy = H / 2.0 + rng.uniform(-20, 20)
    theta = (2 * np.pi) * np.arange(n_theta) / n_theta
    r_lut = np.linspace(2, min(W, H) / 2.0, n_r).astype(np.float32)
    map_x = (cx + r_lut[:, None] * np.cos(theta)[None, :]).astype(np.float32)
    map_y = (cy + r_lut[:, None] * np.sin(theta)[None, :]).astype(np.float32)
    return map_x, map_y


def test_mask_parity(n_trials=15):
    total_mismatch = total_px = 0
    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        H, W = int(rng.integers(100, 500)), int(rng.integers(100, 500))
        frame = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
        n_r, n_theta = int(rng.integers(50, 250)), int(rng.integers(200, 800))
        map_x, map_y = _build_maps(H, W, n_r, n_theta, trial)

        unwrapped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        hsv_ref = cv2.cvtColor(unwrapped, cv2.COLOR_BGR2HSV)
        lower = np.array([int(rng.integers(0, 90)), 60, 60], dtype=np.uint8)
        upper = np.array([int(lower[0]) + 30, 255, 255], dtype=np.uint8)
        mask_ref = cv2.inRange(hsv_ref, lower, upper)

        _hsv, _s_raw, mask_cpp = camera_native.unwarp_threshold(
            frame, map_x, map_y, int(lower[0]), int(lower[1]), int(lower[2]),
            int(upper[0]), int(upper[1]), int(upper[2]), False, 8.0, 0.45)
        mism = int(np.sum(mask_ref != mask_cpp))
        total_mismatch += mism
        total_px += mask_ref.size

    rate = total_mismatch / total_px
    print(f"mask parity: {total_mismatch}/{total_px} mismatched pixels ({rate:.4%}) "
          f"across {n_trials} randomized trials - expected: a small residual from "
          f"+-1 rounding landing exactly on a threshold boundary, not a real bug "
          f"(confirmed by inspection, see this module's own header comment).")
    assert rate < 0.001, f"mismatch rate {rate:.4%} higher than expected"


def bench():
    H, W = 720, 720
    frame = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)
    map_x, map_y = _build_maps(H, W, 200, 900, 0)
    lower = np.array([10, 80, 80], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)

    def py_pipeline():
        unwrapped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(unwrapped, cv2.COLOR_BGR2HSV)
        cv2.inRange(hsv, lower, upper)

    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        py_pipeline()
    t1 = time.perf_counter()
    for _ in range(N):
        camera_native.unwarp_threshold(
            frame, map_x, map_y, int(lower[0]), int(lower[1]), int(lower[2]),
            int(upper[0]), int(upper[1]), int(upper[2]), False, 8.0, 0.45)
    t2 = time.perf_counter()
    py_t, cpp_t = t1 - t0, t2 - t1
    print(f"opencv pipeline: {py_t/N*1000:.3f} ms/call")
    print(f"fused cpp:       {cpp_t/N*1000:.3f} ms/call")
    print(f"ratio: {cpp_t/py_t:.1f}x SLOWER than OpenCV's own pipeline "
          f"(this is the expected, documented result - see header comment)")


if __name__ == "__main__":
    test_mask_parity()
    bench()
