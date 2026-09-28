"""Tests for bot/dwibble_cam_calib.py: the piecewise ball-distance regression built from
bench_dwibble_cam.py's calib_points.json. Pins the interpolation/clamp/fallback behaviour
and the possession-only rule when uncalibrated.
"""

import json

import pytest

from bot.dwibble_cam_calib import DwibbleCamCalib


@pytest.fixture()
def calib_file(tmp_path):
    """a synthetic calibration matching the real geometry: the camera tilts up as the roller
    pulls the ball in, so a closer ball sits lower in the frame: dy_px +40 -> 700mm (far),
    +80 -> 300mm, +120 -> 100mm (nearly seated).
    """
    pts = [
        {"distance_mm": 700.0, "dy_px": 40.0, "dx_px": 0.0, "radius_px": 10.0, "cx": 160, "cy": 160, "frac": 0.2},
        {"distance_mm": 300.0, "dy_px": 80.0, "dx_px": 0.0, "radius_px": 18.0, "cx": 160, "cy": 160, "frac": 0.3},
        {"distance_mm": 100.0, "dy_px": 120.0, "dx_px": 0.0, "radius_px": 30.0, "cx": 160, "cy": 160, "frac": 0.4},
    ]
    p = tmp_path / "calib_points.json"
    p.write_text(json.dumps({"resolution": [320, 240], "points": pts}))
    return str(p)


class TestLoad:
    def test_load_counts_points(self, calib_file):
        c = DwibbleCamCalib()
        assert c.load(calib_file) == 3

    def test_missing_file_loads_zero(self, tmp_path):
        c = DwibbleCamCalib()
        assert c.load(str(tmp_path / "absent.json")) == 0
        assert c.distance_mm(50.0) is None # possession-only rule holds


class TestInterpolation:
    def test_midpoint_linear(self, calib_file):
        c = DwibbleCamCalib()
        c.load(calib_file)
        est, exact = c.distance_mm(60.0)
        assert est == pytest.approx(500.0) # halfway between 700 and 300
        assert exact is False

    def test_exact_point_flags_true(self, calib_file):
        c = DwibbleCamCalib()
        c.load(calib_file)
        est, exact = c.distance_mm(80.0)
        assert est == pytest.approx(300.0)
        assert exact is True

    def test_clamps_at_ends_no_extrapolation(self, calib_file):
        c = DwibbleCamCalib()
        c.load(calib_file)
        lo, _ = c.distance_mm(-500.0)
        hi, _ = c.distance_mm(900.0)
        assert lo == pytest.approx(700.0)
        assert hi == pytest.approx(100.0)


class TestRadiusFallback:
    def test_dy_outside_range_falls_back_to_radius(self, calib_file):
        """dy beyond the calibrated span, but the radius reading (10 px) sits inside its own
        range -> distance from the radius key.
        """
        c = DwibbleCamCalib()
        c.load(calib_file)
        est, exact = c.distance_mm(150.0, radius_px=10.0)
        assert est == pytest.approx(700.0)
        assert exact is True # the radius reading hits an exact key

    def test_no_radius_uses_dy_clamp(self, calib_file):
        c = DwibbleCamCalib()
        c.load(calib_file)
        est, _ = c.distance_mm(150.0, radius_px=None)
        assert est == pytest.approx(100.0)

    def test_monotone_decreasing_with_dy(self, calib_file):
        """ball lower in frame (larger dy) = closer: distance must decrease."""
        c = DwibbleCamCalib()
        c.load(calib_file)
        far, _ = c.distance_mm(50.0)
        near, _ = c.distance_mm(100.0)
        assert near < far
