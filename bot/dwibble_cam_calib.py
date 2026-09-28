"""Ball distance for the second (dwibbler-mouth) camera, from the points bench_dwibble_cam.py
collects.

The camera rides the dwibbler, so it swings back and tilts up once the ball touches the
roller: pixel position isn't a fixed function of distance without this regression. Until
calib_points.json loads, distance_mm() returns None and callers treat the camera as
possession-only.
"""

import bisect
import json
import os


# bench_dwibble_cam.py saves here (the working directory); DWIBBLE_CALIB_PATH overrides
_CALIB_FILENAME = "calib_points.json"


class DwibbleCamCalib:
    """piecewise-linear regression keyed on the ball's pixel y-offset, with pixel radius as the
    fallback key. Clamps at the ends, no extrapolation.
    """

    def __init__(self):
        self._dy_keys = [] # sorted dy_px
        self._dy_dist = [] # parallel distance_mm
        self._r_keys = [] # sorted radius_px fallback
        self._r_dist = [] # parallel distance_mm
        self.loaded_from = None

    def load(self, path=None):
        """load calib_points.json and return the point count, 0 if absent (possession-only
        stays in force).
        """
        path = path or os.environ.get("DWIBBLE_CALIB_PATH", _CALIB_FILENAME)
        if not os.path.exists(path):
            return 0
        with open(path) as f:
            data = json.load(f)
        pts = data.get("points", [])
        if not pts:
            return 0
        dy = sorted((p["dy_px"], p["distance_mm"]) for p in pts)
        self._dy_keys = [k for k, _ in dy]
        self._dy_dist = [d for _, d in dy]
        rr = sorted((p["radius_px"], p["distance_mm"]) for p in pts)
        self._r_keys = [k for k, _ in rr]
        self._r_dist = [d for _, d in rr]
        self.loaded_from = path
        return len(pts)

    def _interp(self, keys, dists, x):
        """piecewise-linear between collected points, clamped at the ends; returns (dist, exact)."""
        if not keys:
            return None
        if x <= keys[0]:
            return dists[0], x == keys[0]
        if x >= keys[-1]:
            return dists[-1], x == keys[-1]
        i = bisect.bisect_right(keys, x) - 1
        if keys[i] == x:
            return dists[i], True
        f = (x - keys[i]) / (keys[i + 1] - keys[i])
        return dists[i] + f * (dists[i + 1] - dists[i]), False

    def distance_mm(self, dy_px, radius_px=None):
        """estimated ball distance from dy_px (its offset below frame centre; the camera tilts
        with the roller, so that tracks distance), falling back to pixel radius when dy_px
        is outside the calibrated range. Returns (distance_mm, exact) or None when
        uncalibrated.
        """
        if not self._dy_keys:
            return None
        est, exact = self._interp(self._dy_keys, self._dy_dist, dy_px)
        # radius only when dy was clamped at an end; inside the range the dy
        # interpolation stands
        in_range = self._dy_keys[0] <= dy_px <= self._dy_keys[-1]
        if not in_range and radius_px is not None and self._r_keys:
            rest, rexit = self._interp(self._r_keys, self._r_dist, radius_px)
            if rest is not None:
                return rest, rexit
        return est, exact


# module singleton, loaded on first use
_calib = DwibbleCamCalib()
_calib_tried = False


def get_calib():
    """the shared DwibbleCamCalib, loaded once (re-loadable by calling load() again)."""
    global _calib_tried
    if not _calib_tried:
        _calib_tried = True
        n = _calib.load()
        if n:
            print(f"[dwibblecam] calib loaded: {n} points "
                  f"from {_calib.loaded_from}", flush=True)
        else:
            print("[dwibblecam] no calib_points.json, camera stays "
                  "possession-only", flush=True)
    return _calib


def reset_calib():
    """drop the cached calibration (bench reload path)."""
    global _calib, _calib_tried
    _calib = DwibbleCamCalib()
    _calib_tried = False
    return get_calib()
