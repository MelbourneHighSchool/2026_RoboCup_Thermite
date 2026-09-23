"""Camera thread + HSV-based ball/goal/enemy-bearing vision.

Extracted verbatim from mainrunbot1.py's camera/detection sections: the
sensor/crop/warp tunables and detection functions (originally around lines
3663-4475) plus the Picamera2 capture loop, `_camera_thread` (originally
around lines 5627-5817).

Per-bot camera constants gap (documented, intentional - do not "fix" by
guessing): `crop_top`, `crop_bottom`, `crop_left`, `exclusion_inner_frac`,
`exclusion_outer_frac`, `cam_bearing_offset_deg` and `handle_exclusion_deg`
are PER-BOT values. `crop_top`/`crop_bottom`/`crop_left`/
`exclusion_inner_frac`/`exclusion_outer_frac`/`cam_bearing_offset_deg` are
already present in `bot1_config.py`/`bot2_config.py` from stage 1;
`handle_exclusion_deg` is too (also per-bot there, same numeric value 4.0
in both today, but it is a per-bot knob, not a shared constant - see those
files). The `ROBOT_ID` selection mechanism that would pick which of
bot1_config/bot2_config to pull these from does not exist yet - wiring it
is a later stage's job (main.py). This module intentionally leaves all
seven names as undefined module globals, referenced only inside function
bodies (`_in_cam_handle_zone`, `detect_ball_warp`, `_camera_thread`) -
exactly like bot/logs.py's documented gap for its own not-yet-available
names. `import bot.vision` succeeds fine (Python does not evaluate a
function body at import time); calling a function that needs one of these
seven names raises NameError until a later stage's ROBOT_ID wiring supplies
them (e.g. `from bot1_config import crop_top, ...` chosen by ROBOT_ID in
main.py). Do NOT hardcode bot1's or bot2's values here as a default - that
would silently run the wrong bot with the wrong bot's crop/exclusion/handle
geometry.

orange_lower/orange_upper rebind hazard: both are shared HSV-tuner seed
constants AND rebound module globals - the --hsv tuner's "Save" button flow
reassigns them at runtime (mirrors the `_state["hsv_lower"/"hsv_upper"]`
live-tuning path already in bot.state). They live here in vision.py as
their natural home. Any OTHER module that needs the current saved values
must do `import bot.vision as vision; vision.orange_lower` - never
`from bot.vision import orange_lower`, which would freeze a stale copy at
import time and silently go stale after a Save. `_bot_candidate_mask`
below (in this same module) reads them as plain module globals, which is
fine since it's in the same module and always sees the live value.

enemy_goal_window / _enemy_goal_colour: `_enemy_goal_colour` is set only by
the Blue/Yellow colour-pick button handler that will live in a later
main.py stage; left here as a module-level global (`None` until that stage
wires it), same documented-gap pattern as the per-bot constants above.

Correctly-imported cross-module state used here: bot.state's `_lock`,
`_state`, `mode` and `_frame_cond`, accessed as `state.<name>` (never a
from-import - several of these are rebound at runtime; see bot/state.py's
own docstring), bot.field's `FieldModel` and `wrap_deg` (as `_wrap_deg`),
bot.dwibbler's `dwibble_mark_lower`, `dwibble_mark_upper`,
`dwibble_mark_min_px` (plain constants there, not rebound, so a normal
from-import is fine), and bot.diagnostics's `_mark_health_t`.
"""

import math
import time

import cv2
import numpy as np

# The Pi camera stack only exists on the Pi.
try:
    from picamera2 import Picamera2
except ImportError: # not on the Pi
    Picamera2 = None

import bot.state as state
from bot.field import FieldModel, wrap_deg as _wrap_deg
from bot.dwibbler import (
    dwibble_mark_lower, dwibble_mark_upper, dwibble_mark_min_px,
)
from bot.diagnostics import _mark_health_t


sensor_mode = (2028, 1080)
# post-rotation size (WxH used everywhere else), a net 270 (odd multiple of 90) does swap
# width/height versus sensor_mode
sensor_size = (1080, 2028)
sensor_fps  = 93.4 # measured 8-bit line rate at this mode

# Pre-processing crop applied to the rotated frame before any detection, post-rotation pixels, default 0 = no crop.
# crop_top/crop_bottom/crop_left are PER-BOT (see module docstring) -
# intentionally left undefined here.
crop_right  = 0

# Anything inside inner or outside outer is never processed -> only the ring is
# scanned, which is the bulk of the speed-up.
# exclusion_inner_frac/exclusion_outer_frac are PER-BOT (see module docstring).

# Mouth notch: the inner exclusion circle also blanked the dwibbler mouth, closing on the ball used to cycle approach/retreat/blind-spin.
mouth_notch_enabled  = True
mouth_notch_half_deg = 27.0 # half-width of the wedge about the dwibbler (70mm mouth)
mouth_notch_px       = 50.0 # how far inside inner_r the wedge reaches

# Fish-eye-aware "warp" detection: unwrap the annulus into (radius x angle) and space the radial axis to match the lens's own resolution.
warp_ntheta      = 540 # angular samples across the ring
ball_diameter_mm  = 43.0 # target ball diameter (sets radial resolution)
robot_diameter_mm = 220.0 # robot body diameter
ball_rows_target  = 6 # floor on radial rows across the ball, far out
# ceiling near in; spacing follows the sensor between the two (_warp_radial_targets),
# about 2.7x the old flat-spacing near-field detail for no measurable cost (see that
# function's own comment for the measured pipeline cost)
warp_max_rows_per_ball = 16

warp_nr_min      = 48 # clamp on the auto-chosen radial sample count
# was 384, forcing uniform ground-distance spacing mismatched to the sensor at both ends
warp_nr_max      = 900

# Provisional, re-run --hsv. Rebound at runtime by the --hsv tuner's Save
# button - see the "orange_lower/orange_upper rebind hazard" note above.
orange_lower = np.array([0, 201, 127], dtype=np.uint8)
orange_upper = np.array([11, 255, 255], dtype=np.uint8)
min_orange_px = 3

# Angular ROI tracking: the unwrap axis is angle, so a sector is a column window.
ball_sector_track    = True
ball_sector_half_deg = 40.0 # search +/- this around the last-seen ball angle
_ball_track_col = None # last accepted ball column (module state), or None

# Logistic saturation boost, pre-inRange.
sat_boost_enabled = False
sat_boost_k   = 8.0 # steepness, higher pushes the curve closer to a step
sat_boost_mid = 0.45 # sigmoid centre on [0,1]; pixels below tend toward 0

# mirrors the raw atan2(cos theta, sin theta) angle about the forward axis, without this
# the "clockwise positive" convention the rest of the file uses would come out mirrored
# (counter-clockwise) for this camera's own mount.
cam_y_sign = -1

# cam_bearing_offset_deg is PER-BOT (see module docstring).


def _cam_bearing(cos_th, sin_th):
    """robot-frame bearing (deg, 0 = dwibbler-forward, clockwise+) for a raw camera-frame direction given as (cos theta, sin theta); the one place cam_y_sign and cam_bearing_offset_deg get applied, keeping all call sites in sync."""
    raw = np.degrees(np.arctan2(cos_th, cam_y_sign * sin_th))
    return _wrap_deg(raw + cam_bearing_offset_deg)


def _cam_direction(bearing_deg):
    """inverse of _cam_bearing: unit image-direction (dx, dy) for a given robot-frame bearing, used only by bot/debug_server.py's render_camera_panel to draw the notch/handle wedges."""
    a = math.radians(bearing_deg) - math.radians(cam_bearing_offset_deg)
    return math.sin(a), -math.cos(a)


# Camera controls: lock the ISP so the image stops moving under the threshold.
cam_lock_auto     = True # freeze AE/AWB after they settle
cam_settle_s      = 1.5 # let them run this long first
# None = adopt whatever AE settled on; a number forces it (must be <= the frame duration,
# or libcamera drops the frame rate to fit)
cam_exposure_us   = None
# ISP saturation, applied in hardware before 8-bit quantisation, strictly better than software, and free.
cam_saturation    = 1.8

# Calibrated fish-eye lens model: pixel offset from the optical centre to ground mm via an odd-power polynomial in t = tan(r / _fisheye_b).
# Fitted directly against the current crop's pixel radii, re-run
# fisheye_calib.py rather than rescale if crop_* ever changes.
fisheye_px_scale = 1.0
fisheye_rmax     = math.sqrt(8387300) # about 2896 mm saturation distance
_fisheye_b       = -510.76033
_fisheye_c       = (-214.82028, -246.09173, 130.02845, 1.0, -42.09367)
# tan() blows up at r = |b|*pi/2; the polynomial is only meaningful below that.
_fisheye_pole    = abs(_fisheye_b) * math.pi / 2.0


def _fisheye_poly(r):
    """raw calibration polynomial: ground mm for a calibration-space px radius."""
    t = math.tan(r / _fisheye_b)
    a, c, d, e, f = _fisheye_c
    return (a * t + c * t ** 3 + d * t ** 5
            + e * t ** 7 + f * t ** 9)


def _solve_clamp_radius(target):
    """px radius at which _fisheye_poly reaches `target` (monotonic -> bisection)."""
    lo, hi = 0.0, _fisheye_pole - 0.001 # tan -> inf at the pole
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _fisheye_poly(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# Clamp point chosen so _fisheye_poly(fisheye_rclamp) == fisheye_rmax exactly:
# the saturation clamp is then smooth (continuous) rather than a hard step.
fisheye_rclamp = _solve_clamp_radius(fisheye_rmax)


def _sat_boost(hsv):
    """in-place logistic saturation boost on an HSV uint8 image."""
    if not sat_boost_enabled:
        return hsv
    s  = hsv[:, :, 1].astype(np.float32) * (1.0 / 255.0)
    s  = 1.0 / (1.0 + np.exp(-sat_boost_k * (s - sat_boost_mid)))
    lo = 1.0 / (1.0 + math.exp( sat_boost_k * sat_boost_mid))
    hi = 1.0 / (1.0 + math.exp(-sat_boost_k * (1.0 - sat_boost_mid)))
    s  = (s - lo) * (255.0 / (hi - lo))
    hsv[:, :, 1] = np.clip(s, 0, 255).astype(np.uint8)
    return hsv


# HSV init

def _init_hsv():
    """seed _state's live HSV thresholds from the saved orange_lower/orange_upper."""
    with state._lock:
        state._state["hsv_lower"] = orange_lower.copy()
        state._state["hsv_upper"] = orange_upper.copy()


# Detection

# Handle exclusion for the camera: same side-normal angles as LiDAR (+/-90deg), but expressed in atan2 output space [-180deg, 180deg].
_cam_handle_centres = (90.0, -90.0, 180.0)

def _in_cam_handle_zone(angle_deg):
    """True if a camera detection angle falls inside a handle exclusion wedge."""
    return any(
        abs(((angle_deg - c + 180.0) % 360.0) - 180.0) <= handle_exclusion_deg
        for c in _cam_handle_centres
    )


def _fisheye_radius(r_px):
    """real ground radius (mm) for a pixel radius, via the calibrated fish-eye polynomial."""
    r = r_px * fisheye_px_scale
    if r <= 0.0:
        return 0.0
    if r >= fisheye_rclamp:
        return fisheye_rmax
    return _fisheye_poly(r)


def _pixel_to_position(px, py):
    """calibrated fish-eye pixel->ground mapping (replaces the old tan() approx)."""
    r_px = math.hypot(px, py)
    if r_px < 1e-6:
        return 0.0, 0.0
    scale = _fisheye_radius(r_px) / r_px
    return px * scale, cam_y_sign * py * scale


def _warp_radial_targets(inner_r, outer_r, grid_r=None, grid_d=None):
    """ground distances (mm) to place the unwrap's radial samples at."""
    if grid_r is None or grid_d is None:
        grid_r = np.linspace(0.0, float(outer_r), 4096)
        grid_d = np.array([_fisheye_radius(r) for r in grid_r])
    d_in  = float(_fisheye_radius(inner_r))
    d_out = float(_fisheye_radius(outer_r))
    coarse_mm = ball_diameter_mm / max(1.0, float(ball_rows_target))
    fine_mm   = (1e-3 if warp_max_rows_per_ball is None
                 else ball_diameter_mm / float(warp_max_rows_per_ball))

    out, d = [d_in], d_in
    while d < d_out and len(out) < warp_nr_max:
        rp = float(np.interp(d, grid_d, grid_r)) # source px at this range
        per_px = (_fisheye_radius(rp + 0.5) - _fisheye_radius(rp - 0.5))
        d += min(coarse_mm, max(fine_mm, per_px))
        out.append(min(d, d_out))
    if out[-1] < d_out:
        out.append(d_out)
    if len(out) < warp_nr_min: # degenerate ring
        out = list(np.linspace(d_in, d_out, warp_nr_min))
    return np.asarray(out, dtype=np.float64)


def build_warp_maps(cx, cy, inner_r, outer_r, n_r, n_theta):
    """precompute the cv2.remap lookup tables that unwrap the annulus (inner_r, outer_r] around (cx, cy) into an (n_r x n_theta) image."""
    theta_lut = (2.0 * np.pi) * np.arange(n_theta, dtype=np.float32) / n_theta

    grid_r = np.linspace(0.0, float(outer_r), 4096)
    grid_d = np.array([_fisheye_radius(r) for r in grid_r])
    d_tgt  = _warp_radial_targets(inner_r, outer_r, grid_r, grid_d)
    r_lut  = np.interp(d_tgt, grid_d, grid_r).astype(np.float32)

    cos_t = np.cos(theta_lut)[None, :] # (1, n_theta)
    sin_t = np.sin(theta_lut)[None, :]
    rr    = r_lut[:, None] # (n_r, 1)
    map_x = (cx + rr * cos_t).astype(np.float32)
    map_y = (cy + rr * sin_t).astype(np.float32)
    return map_x, map_y, r_lut, theta_lut


def _biggest_blob(mask):
    """largest connected component of a binary mask, as (col, row, w, h, area) in mask coords, or None if none clears min_orange_px."""
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_orange_px:
        return None
    cx, cy = cents[k]
    return (cx, cy, int(stats[k, cv2.CC_STAT_WIDTH]),
            int(stats[k, cv2.CC_STAT_HEIGHT]), int(stats[k, cv2.CC_STAT_AREA]))


def _find_ball_columns(orange, n_theta, n_r):
    """ball blob column/row (col, row) + box (w, h), or None."""
    global _ball_track_col

    if ball_sector_track and _ball_track_col is not None:
        H = max(1, int(round(ball_sector_half_deg / 360.0 * n_theta)))
        # modular gather -> the window is contiguous even across the 0/2pi seam
        cols = (np.arange(_ball_track_col - H, _ball_track_col + H + 1)
                % n_theta)
        sub = np.ascontiguousarray(orange[:, cols])
        b = _biggest_blob(sub)
        if b is not None:
            cx, cy, w, h, _a = b
            # accept only if the blob is clear of both window edges (fully
            # inside the sector); if it touches an edge it may run past the
            # window, so fall back to the full scan below.
            if (cx - w / 2.0) > 0.5 and (cx + w / 2.0) < sub.shape[1] - 1.5:
                # cols is a contiguous modular range, so mapping the sub-window
                # column back is just an offset, no need to index it.
                col = (int(cols[0]) + cx) % n_theta
                _ball_track_col = int(round(col)) % n_theta
                return col, min(max(cy, 0.0), n_r - 1.0), w, h
        # window miss / ran off the edge -> full scan

    # Full 360 scan, wrap-padded so a ball on the seam stays one blob.
    margin = n_theta // 8
    padded = np.hstack([orange[:, -margin:], orange, orange[:, :margin]])
    b = _biggest_blob(padded)
    if b is None:
        _ball_track_col = None
        return None
    cx, cy, w, h, _a = b
    col = (cx - margin) % n_theta
    _ball_track_col = int(round(col)) % n_theta
    return col, min(max(cy, 0.0), n_r - 1.0), w, h


# Sub-pixel ball centre: at range the ball is only about 1-2 source pixels deep radially (fisheye compression), so a binary centroid quantises the range estimate hard.
ball_centre_fill_holes = True
# cells: fragments whose bounding boxes come within this of the main one are the same ball
ball_centre_merge_span = 6.0
ball_centre_subpixel = True
# cells of margin around the blob's box, so cells the hard threshold excluded still get a
# vote
ball_centre_pad      = 2
# S floor for the soft weight.
ball_centre_s_floor  = 85.0


def _ball_cells_at(row, r_lut, n_theta):
    """how many cells wide and tall a ball SHOULD be at this radial row."""
    n_r = len(r_lut)
    i = int(min(max(round(row), 0), n_r - 1))
    d = _fisheye_radius(float(r_lut[i]))
    if d <= ball_diameter_mm * 0.5:
        return n_theta, 8.0
    half = math.asin(min(1.0, (ball_diameter_mm * 0.5) / d))
    w = 2.0 * half / (2.0 * math.pi) * n_theta
    j = min(n_r - 1, i + 1)
    k = max(0, i - 1)
    step = abs(_fisheye_radius(float(r_lut[j]))
               - _fisheye_radius(float(r_lut[k]))) / max(1, j - k)
    h = ball_diameter_mm / step if step > 1e-6 else float(ball_rows_target)
    return w, h


def _subpixel_centre(hsv, s_raw, mask, col, row, w_box, h_box, lower, upper,
                     r_lut=None):
    """refine a blob's (col, row), the centroid from the binary mask, as floats, to a saturation-weighted centre."""
    if not ball_centre_subpixel:
        return col, row
    n_r, n_theta = s_raw.shape
    # Size the window off the ball's expected extent as well as the blob's, so
    # a partly-eaten or split blob still gets its whole neighbourhood weighed.
    w_exp, h_exp = ((w_box, h_box) if r_lut is None
                    else _ball_cells_at(row, r_lut, n_theta))
    half_w = max(1, int(math.ceil(max(w_box, w_exp) / 2.0)) + ball_centre_pad)
    half_h = max(1, int(math.ceil(max(h_box, h_exp) / 2.0)) + ball_centre_pad)
    half_w = min(half_w, max(1, n_theta // 4)) # never wrap onto itself
    c0, r0 = int(round(col)), int(round(row))

    # Columns wrap around the seam, rows clamp at the ring edges.
    cols = (np.arange(c0 - half_w, c0 + half_w + 1) % n_theta)
    r_lo = max(0, r0 - half_h)
    r_hi = min(n_r - 1, r0 + half_h)
    rows = np.arange(r_lo, r_hi + 1)
    if rows.size < 2 or cols.size < 2:
        return col, row

    win_h = hsv[r_lo:r_hi + 1, :, 0][:, cols].astype(np.int16)
    win_v = hsv[r_lo:r_hi + 1, :, 2][:, cols].astype(np.int16)
    win_s = s_raw[r_lo:r_hi + 1, :][:, cols].astype(np.float32)

    ok = ((win_h >= int(lower[0])) & (win_h <= int(upper[0]))
          & (win_v >= int(lower[2])) & (win_v <= int(upper[2])))
    wgt = np.where(ok, np.maximum(0.0, win_s - ball_centre_s_floor), 0.0)

    # Fill what the threshold punched out of the middle (see above).
    if ball_centre_fill_holes:
        win_m = np.ascontiguousarray(mask[r_lo:r_hi + 1, :][:, cols])
        cnts = cv2.findContours(win_m, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        if cnts:
            # Merge every fragment near the biggest one, then hull the union,
            # a highlight that splits the ball leaves pieces, not a hole.
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            keep = [cnts[0]]
            bx, by, bw, bh = cv2.boundingRect(cnts[0])
            for c in cnts[1:]:
                x, y, w, h = cv2.boundingRect(c)
                gap = math.hypot(max(0, max(bx - (x + w), x - (bx + bw))),
                                 max(0, max(by - (y + h), y - (by + bh))))
                if gap <= ball_centre_merge_span:
                    keep.append(c)
            pts = np.vstack(keep)
            hull = cv2.convexHull(pts) if len(pts) >= 3 else keep[0]
            solid = np.zeros_like(win_m)
            cv2.drawContours(solid, [hull], -1, 255, -1)
            lit = wgt[win_m > 0]
            if lit.size:
                wgt = np.maximum(wgt, np.where(solid > 0,
                                               float(np.median(lit)), 0.0))

    tot = float(wgt.sum())
    if tot <= 1e-6:
        return col, row # nothing orange, keep the blob

    row_w = wgt.sum(axis=1)
    col_w = wgt.sum(axis=0)
    new_row = float((row_w * rows).sum() / tot)
    # Column index is relative to the (modular) window, so average in window
    # space and map back, averaging wrapped absolute columns would land the
    # centre on the far side of the ring for a ball sitting on the seam.
    rel = np.arange(cols.size, dtype=np.float64)
    new_col = (float(cols[0]) + float((col_w * rel).sum() / tot)) % n_theta
    return new_col, new_row


def detect_ball_warp(frame, lower, upper, map_x, map_y, r_lut, theta_lut,
                     col_bearing, inner_blank=None):
    """detect the orange ball inside the unwrapped annulus produced by build_warp_maps()."""
    n_r     = len(r_lut)
    n_theta = len(theta_lut)

    unwrapped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
    hsv       = cv2.cvtColor(unwrapped, cv2.COLOR_BGR2HSV)
    # Keep the saturation as it came off the sensor before the boost rewrites
    # it: the boost is right for finding the ball and wrong for locating its
    # centre (see _subpixel_centre).
    s_raw     = hsv[:, :, 1].copy()
    _sat_boost(hsv)
    orange    = cv2.inRange(hsv, lower, upper)
    # No medianBlur here.
    if inner_blank is not None:
        orange[inner_blank] = 0 # robot body, outside the mouth notch

    # Columns inside handle exclusion wedges (shared by ball + line masks).
    handle_cols = None
    if handle_exclusion_deg > 0:
        for c in _cam_handle_centres:
            cols = np.abs(((col_bearing - c + 180.0) % 360.0) - 180.0) <= handle_exclusion_deg
            handle_cols = cols if handle_cols is None else (handle_cols | cols)
        orange[:, handle_cols] = 0

    blob = _find_ball_columns(orange, n_theta, n_r)
    if blob is None:
        return None, None, None, hsv
    col, row, w_box, h_box = blob
    raw_row = row
    col, row = _subpixel_centre(hsv, s_raw, orange, col, row, w_box, h_box,
                                lower, upper, r_lut)
    # Blob geometry, for the debug overlay.
    with state._lock:
        state._state["ball_blob"] = (float(w_box), float(h_box),
                               float(row), float(row - raw_row))

    # Interpolate the axes at the sub-cell centre rather than indexing them.
    r_px = float(np.interp(row, np.arange(n_r), r_lut))
    th   = (2.0 * math.pi) * (col / n_theta)
    px   = r_px * math.cos(th)
    py   = r_px * math.sin(th)

    rx, ry    = _pixel_to_position(px, py)
    dist_mm   = math.hypot(rx, ry)
    # Bearing, not distance, is what cam_bearing_offset_deg corrects, see its own comment.
    angle_deg = _cam_bearing(px, py)

    # Full-frame centroid, for the debug overlay only, whole cells are fine
    # here (the remap LUT already encodes cx + r*costheta, etc.)
    ir = int(min(max(round(row), 0), n_r - 1))
    it = int(round(col)) % n_theta
    cx_img = int(round(map_x[ir, it]))
    cy_img = int(round(map_y[ir, it]))

    # Rough display radius from the blob's extent: arc length spanned vs radial
    # span, whichever is larger (the unwrap distorts shape, so this is only a
    # visual guide for the debug overlay).
    arc   = r_px * (2.0 * math.pi) * (w_box / n_theta)
    r0    = float(r_lut[max(0, ir - h_box // 2)])
    r1    = float(r_lut[min(n_r - 1, ir + h_box // 2)])
    radius = max(3, int(0.5 * max(arc, abs(r1 - r0))))

    return (angle_deg, dist_mm), (cx_img, cy_img), radius, hsv


# Goal detection: which part of the enemy goal is least defended. The enemy goal (CMYK cyan or yellow) is at the far wall; an enemy robot in front of it hides that colour from the camera.
goal_cyan_lower   = np.array([85,  70,  70],  dtype=np.uint8)
goal_cyan_upper   = np.array([105, 255, 255], dtype=np.uint8)
goal_yellow_lower = np.array([20,  70,  70],  dtype=np.uint8)
goal_yellow_upper = np.array([40,  255, 255], dtype=np.uint8)

goal_open_enabled   = True
goal_min_px_per_col = 2 # goal-colour pixels in a column to call it "open"
goal_window_pad_deg = 6.0 # widen the geometric goal window this much each side

# Set only by the Blue/Yellow colour-pick button handler that will live in a
# later main.py stage - see module docstring.
_enemy_goal_colour = None # "cyan" | "yellow"


def _goal_colour_ranges(name):
    """HSV (lower, upper) thresholds for the goal-marker colour named "cyan" or "yellow"."""
    if name == "cyan":
        return goal_cyan_lower, goal_cyan_upper
    return goal_yellow_lower, goal_yellow_upper


def enemy_goal_window(pose, attack_low):
    """robot-frame bearing (deg, atan2(rx, ry) convention, 0 = +y) to the enemy goal centre and the half-span to its posts."""
    rx0, ry0, hdg = pose
    gy = 0.0 if attack_low else FieldModel.field_y
    def bearing(gx):
        """robot-frame bearing to a point at field x=gx, y=gy (the goal line)."""
        return _wrap_deg(math.degrees(math.atan2(gx - rx0, gy - ry0)) - hdg)
    centre = bearing(FieldModel.cx)
    half = max(abs(_wrap_deg(bearing(FieldModel.sx0) - centre)),
               abs(_wrap_deg(bearing(FieldModel.sx1) - centre)))
    return centre, half


def detect_goal_open(hsv, col_bearing, lower, upper, centre_deg, half_deg):
    """within the enemy-goal angular window, find the widest contiguous run of columns still showing goal colour (open) and return the aim bearing for its middle."""
    per_col = (cv2.inRange(hsv, lower, upper) > 0).sum(axis=0) # goal px / column
    win = np.abs(((col_bearing - centre_deg + 180.0) % 360.0) - 180.0) \
        <= (half_deg + goal_window_pad_deg)
    cols = np.flatnonzero(win)
    if cols.size == 0:
        return None
    off = ((col_bearing[cols] - centre_deg + 180.0) % 360.0) - 180.0
    order = np.argsort(off)
    cols, off = cols[order], off[order]
    is_open = per_col[cols] >= goal_min_px_per_col
    blocked_frac = 1.0 - float(is_open.sum()) / len(cols)

    best = None
    i, n = 0, len(cols)
    while i < n:
        if is_open[i]:
            j = i
            while j + 1 < n and is_open[j + 1]:
                j += 1
            width = off[j] - off[i]
            if best is None or width > best[0]:
                best = (width, 0.5 * (off[i] + off[j]))
            i = j + 1
        else:
            i += 1
    if best is None:
        return {"bearing_deg": None, "open_deg": 0.0, "blocked_frac": 1.0}
    return {"bearing_deg": _wrap_deg(centre_deg + best[1]),
            "open_deg": float(best[0]), "blocked_frac": blocked_frac}


cam_front_half_deg = 90.0 # "in front" for detect_front_goal
goal_front_min_px  = 40 # goal-colour pixels in the front half to count


def detect_front_goal(hsv, col_bearing, half_deg=cam_front_half_deg,
                      min_px=goal_front_min_px):
    """any goal-coloured (cyan or yellow) pixels within half_deg of dead ahead (bearing 0), no dependency on pose or attack_low."""
    win = np.abs(col_bearing) <= half_deg
    if not win.any():
        return None
    sub = hsv[:, win]
    cyan = int(cv2.inRange(sub, goal_cyan_lower, goal_cyan_upper).sum())
    yell = int(cv2.inRange(sub, goal_yellow_lower, goal_yellow_upper).sum())
    if cyan < min_px and yell < min_px:
        return None
    colour = "cyan" if cyan >= yell else "yellow"
    lower, upper = _goal_colour_ranges(colour)
    per_col = (cv2.inRange(sub, lower, upper) > 0).sum(axis=0)
    sel = per_col >= goal_min_px_per_col
    if not sel.any():
        return None
    cols = np.flatnonzero(win)
    bearing = float(np.mean(col_bearing[cols[sel]]))
    return colour, bearing


def detect_dwibble_mark(hsv, notch_region, lower, upper, min_px):
    """dwibbler-stall's visual second vote (sec 3.16): True if enough pixels in the mouth-notch wedge match the dwibble_mark colour; always False when mouth_notch_enabled is off."""
    if notch_region is None:
        return False
    mask = cv2.inRange(hsv, lower, upper) > 0
    return int(mask[notch_region].sum()) >= min_px


# Camera-bearing fusion onto lidar-tracked enemies: the lidar-confirmed
# enemies list (_state["enemies"]) only refreshes at scan rate (~10 Hz); the
# camera runs at ~30 Hz, so between scans a tracked enemy's *bearing* can be
# refreshed from the camera far more often than its range. RCJ bots aren't
# reliably colour-coded on the chassis the way the ball/goals are, so rather
# than look for "the enemy's colour" this hunts for "a body-width blob that
# is none of the known things" (not field green, not line white, not the
# ball's own orange, not either goal colour) - same blind spot the ball
# detector already has for an all-white/all-black chassis (indistinguishable
# from lines/walls under the same achromatic test), no way around that
# without the enemy actually wearing a distinguishing colour. This only ever
# nudges an already lidar-confirmed enemy's bearing (holding its lidar-
# derived range fixed); it can never spawn a phantom enemy or invent range.
cam_enemy_fix_enabled = True
field_green_lower = np.array([35,  40,  40],  dtype=np.uint8) # Provisional,
field_green_upper = np.array([95,  255, 255], dtype=np.uint8) # re-run --hsv
line_white_lower  = np.array([0,   0,   150], dtype=np.uint8) # against the
line_white_upper  = np.array([179, 60,  255], dtype=np.uint8) # real field/carpet before trusting.
bot_min_px_per_col   = 3    # candidate px in a column before it counts
bot_min_width_deg    = 4.0  # a run narrower than this is noise, not a body
bot_bearing_gate_deg = 8.0  # camera bearing must be this close to an
                             # enemy's own current bearing to be "the same one"
bot_bearing_blend    = 0.5  # EMA weight of the camera's own bearing fix,
                             # same idiom as RobotTracker.smooth


def _bot_candidate_mask(hsv):
    """px that are none of: field green, wall/line white, ball orange, either goal colour."""
    not_green = cv2.bitwise_not(cv2.inRange(hsv, field_green_lower, field_green_upper))
    not_white = cv2.bitwise_not(cv2.inRange(hsv, line_white_lower, line_white_upper))
    not_ball  = cv2.bitwise_not(cv2.inRange(hsv, orange_lower, orange_upper))
    not_cyan  = cv2.bitwise_not(cv2.inRange(hsv, goal_cyan_lower, goal_cyan_upper))
    not_yell  = cv2.bitwise_not(cv2.inRange(hsv, goal_yellow_lower, goal_yellow_upper))
    mask = cv2.bitwise_and(not_green, not_white)
    mask = cv2.bitwise_and(mask, not_ball)
    mask = cv2.bitwise_and(mask, not_cyan)
    return cv2.bitwise_and(mask, not_yell)


def _detect_bot_bearings(hsv, col_bearing):
    """bearings (deg, robot-frame) of every "none of the known things" blob
    wide enough to be a body, not noise. Contiguous-run grouping over
    columns, same idiom as detect_goal_open's own widest-run search, but
    circular (wraps past the last column back to the first, since bearing
    itself wraps at +-180) and reporting every run above bot_min_width_deg,
    not just the widest one. The mouth-notch wedge is dropped afterward,
    same as the ball detector's own _in_cam_handle_zone filter - our own
    plastic/cavity would otherwise read as a permanent phantom body sitting
    at a fixed bearing."""
    mask = _bot_candidate_mask(hsv)
    per_col = (mask > 0).sum(axis=0)
    is_bot_col = per_col >= bot_min_px_per_col
    n = len(is_bot_col)
    if n == 0 or not is_bot_col.any() or is_bot_col.all():
        return [] # nothing, or the whole ring (a bad frame either way)

    # Circular run-length grouping: rotate the column order to start right
    # after some gap (guaranteed to exist, not all-True), so every real run
    # - including one that spans the column-index wrap point - reads as
    # one contiguous stretch in the rotated order, no wraparound bookkeeping
    # needed in the scan itself.
    gap = int(np.argmin(is_bot_col))
    order = (np.arange(n) + gap + 1) % n

    bearings = []
    i = 0
    while i < n:
        if is_bot_col[order[i]]:
            j = i
            while j + 1 < n and is_bot_col[order[j + 1]]:
                j += 1
            cols = order[i:j + 1]
            width_deg = abs(_wrap_deg(float(col_bearing[cols[-1]] - col_bearing[cols[0]])))
            if width_deg >= bot_min_width_deg:
                ang = np.radians(col_bearing[cols].astype(np.float64))
                w = per_col[cols].astype(np.float64)
                s = float(np.sum(w * np.sin(ang)))
                c = float(np.sum(w * np.cos(ang)))
                bearing = _wrap_deg(math.degrees(math.atan2(s, c)))
                if not _in_cam_handle_zone(bearing):
                    bearings.append(bearing)
            i = j + 1
        else:
            i += 1
    return bearings


def _fuse_enemy_bearings(enemies, cam_bearings, rx, ry, hdg):
    """nudge each lidar-tracked enemy's (x, y) toward whatever camera bearing
    sits closest to its own current bearing, holding its lidar-derived range
    fixed (the camera has no range of its own to offer). Mutates and returns
    `enemies` in place; an enemy with no camera match within
    bot_bearing_gate_deg is left exactly as lidar reported it. With no
    enemies to nudge, or no camera blobs at all, this is a no-op - it never
    creates an enemy or a bearing/range value where lidar hadn't already
    confirmed one."""
    if not cam_bearings or not enemies:
        return enemies
    remaining = list(cam_bearings)
    for e in enemies:
        ex, ey = e["x"] - rx, e["y"] - ry
        rng = math.hypot(ex, ey)
        if rng < 1e-6:
            continue
        own_bearing = _wrap_deg(math.degrees(math.atan2(ex, ey)) - hdg)
        best_i, best_d = None, bot_bearing_gate_deg
        for i, cb in enumerate(remaining):
            d = abs(_wrap_deg(cb - own_bearing))
            if d < best_d:
                best_i, best_d = i, d
        if best_i is None:
            continue
        cb = remaining.pop(best_i)
        fixed_bearing = _wrap_deg(own_bearing
                                  + bot_bearing_blend * _wrap_deg(cb - own_bearing))
        rad = math.radians(fixed_bearing + hdg)
        e["x"] = rx + rng * math.sin(rad)
        e["y"] = ry + rng * math.cos(rad)
    return enemies


def _camera_thread(cap_res):
    """own the Picamera2 capture loop: grab frames, unwrap/detect the ball each tick, and publish ball/detection state for the play loops and debug UI."""
    if Picamera2 is None:
        print("[camera] picamera2 not installed, camera disabled "
              "(expected off the Pi; install python3-picamera2 on it).",
              flush=True)
        return
    frame_duration = int(1_000_000 / sensor_fps)
    picam2 = Picamera2()
    # Force the 2028x1080 8-bit mode (output_size + bit_depth select it); no
    # ScalerCrop, so the whole field of view of that mode is captured at its
    # full resolution.
    config = picam2.create_video_configuration(
        main={"size": sensor_mode, "format": "RGB888"},
        sensor={"output_size": sensor_mode, "bit_depth": 8},
        controls={"FrameDurationLimits": (frame_duration, frame_duration)},
    )
    picam2.configure(config)
    picam2.start()

    if cam_saturation is not None:
        picam2.set_controls({"Saturation": float(cam_saturation)})
    if cam_lock_auto:
        # Let the auto algorithms converge on this venue, then pin them.
        time.sleep(cam_settle_s)
        try:
            md = picam2.capture_metadata()
            exp  = int(cam_exposure_us if cam_exposure_us is not None
                       else md.get("ExposureTime", 8000))
            gain = float(md.get("AnalogueGain", 1.0))
            locked = {"AeEnable": False,
                      "ExposureTime": exp,
                      "AnalogueGain": gain}
            gains = md.get("ColourGains")
            if gains is not None:
                locked["AwbEnable"] = False
                locked["ColourGains"] = (float(gains[0]), float(gains[1]))
            picam2.set_controls(locked)
            print(f"[camera] locked: exposure {exp} us, gain {gain:.2f}"
                  + (f", colour gains {gains[0]:.2f}/{gains[1]:.2f}"
                     if gains is not None else " (AWB left auto)"),
                  flush=True)
        except Exception as e: # noqa: BLE001
            print(f"[camera] could not lock AE/AWB ({e}), running auto, "
                  "expect the ball fix to drift with the lighting", flush=True)

    # Annulus geometry in full cropped-frame pixels (centre = cropped centre).
    crop_w  = sensor_size[0] - crop_left - crop_right
    crop_h  = sensor_size[1] - crop_top  - crop_bottom
    ccx     = crop_w // 2
    ccy     = crop_h // 2
    max_r   = min(crop_w, crop_h) // 2
    inner_r = int(exclusion_inner_frac * max_r)
    outer_r = int(exclusion_outer_frac * max_r)
    # The unwrap starts at the notch depth so the mouth wedge is sampled at
    # all; the columns outside that wedge get those extra near rows blanked
    # again below (inner_blank), so only the wedge is really handed back.
    notch_px = int(round(mouth_notch_px)) if mouth_notch_enabled else 0
    ring_r   = max(1, inner_r - notch_px)

    # Fish-eye-aware unwrap: sample the full-res annulus into a (radius x angle) image whose radial axis is uniform in real-world mm, the row count falls out of the spacing rule (see _warp_radial_targets).
    map_x, map_y, r_lut, theta_lut = build_warp_maps(
        ccx, ccy, ring_r, outer_r, None, warp_ntheta)
    warp_nr = len(r_lut)

    # Ground distance per row, off r_lut: the radial axis is deliberately
    # Not uniform, so a linspace would mislabel every row. Used below only
    # for the startup log line's radial-spacing figures.
    gd = np.array([_fisheye_radius(float(r)) for r in r_lut], dtype=np.float32)

    # Robot-frame bearing (deg, 0 = dwibbler-forward, clockwise+) of each
    # unwrapped column, so the goal detector can pick the columns inside the
    # enemy-goal window. See _cam_bearing for cam_y_sign/cam_bearing_offset_deg.
    col_bearing = _cam_bearing(np.cos(theta_lut), np.sin(theta_lut))

    # Mouth notch mask: the unwrap above starts at ring_r for every column, so blank the rows inside the original inner_r again everywhere except the wedge about the dwibbler.
    inner_blank  = None
    # where detect_dwibble_mark looks, the wedge inner_blank above blanks out of ball
    # detection, i.e. its exact complement within notch_rows
    notch_region = None
    if notch_px:
        notch_cols = np.abs(col_bearing) <= mouth_notch_half_deg
        notch_rows = r_lut < inner_r
        inner_blank = np.zeros((warp_nr, warp_ntheta), dtype=bool)
        inner_blank[np.ix_(notch_rows, ~notch_cols)] = True
        notch_region = np.zeros((warp_nr, warp_ntheta), dtype=bool)
        notch_region[np.ix_(notch_rows, notch_cols)] = True
        print(f"[camera] mouth notch: +-{mouth_notch_half_deg:.0f} deg, "
              f"{notch_px} px ({_fisheye_radius(inner_r):.0f} -> "
              f"{_fisheye_radius(ring_r):.0f} mm nearest visible), "
              f"{int(notch_cols.sum())}/{warp_ntheta} columns", flush=True)

    _frame_times = []
    _printed_shape = False
    while True:
        frame = picam2.capture_array("main")
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) # net 270 CW
        h_rot, w_rot = frame.shape[:2]
        frame = frame[
            crop_top : h_rot - crop_bottom if crop_bottom else None,
            crop_left : w_rot - crop_right  if crop_right  else None,
        ]
        if not _printed_shape:
            mode_str = (f"warp {warp_nr}x{warp_ntheta} "
                        f"({_fisheye_radius(ring_r):.0f}-"
                        f"{_fisheye_radius(outer_r):.0f}mm, "
                        f"{gd[1]-gd[0]:.1f}mm/row near, "
                        f"{gd[-1]-gd[-2]:.1f} far)")
            print(f"[camera] frame {frame.shape} detect={mode_str}  "
                  f"ring px: inner={inner_r} outer={outer_r}", flush=True)
            if cam_saturation is not None and not sat_boost_enabled:
                print(f"[camera] saturation moved to the ISP ({cam_saturation}) "
                      "and _sat_boost is off, the S scale has changed, so "
                      "orange_lower/upper are provisional. Re-run --hsv against "
                      "the real ball before trusting the ball fix.", flush=True)
            _printed_shape = True

        now = time.monotonic()
        _frame_times.append(now)
        _frame_times = [t for t in _frame_times if now - t <= 1.0]
        cam_fps = len(_frame_times)

        with state._lock:
            lo = state._state["hsv_lower"]
            hi = state._state["hsv_upper"]

        result, centroid, radius, hsv_unwrap = detect_ball_warp(
            frame, lo, hi, map_x, map_y, r_lut, theta_lut, col_bearing, inner_blank)

        # Discard detections whose angle falls inside a handle exclusion wedge.
        if result is not None and _in_cam_handle_zone(result[0]):
            result, centroid, radius = None, None, None

        # least-defended part of the enemy goal (camera)
        # Geometry (pose) gives the goal's angular window; the camera says which
        # of it still shows goal colour (open) vs is blocked by an enemy body.
        open_goal = None
        if goal_open_enabled and state.mode == "run":
            with state._lock:
                gpose      = state._state["pose"]
                attack_low = state._state["attack_low"]
            if gpose is not None and attack_low is not None:
                # attack_low is only ever set once the goal-colour button has been pressed (_maybe_start requires it), so _enemy_goal_colour is always known here.
                gc, gh = enemy_goal_window(gpose, attack_low)
                lo_c, hi_c = _goal_colour_ranges(_enemy_goal_colour)
                open_goal = detect_goal_open(hsv_unwrap, col_bearing,
                                             lo_c, hi_c, gc, gh)
        with state._lock:
            state._state["open_goal"] = open_goal

        # Camera-rate bearing refresh for lidar-confirmed enemies: nudges each
        # tracked enemy's angle toward the nearest "none of the known things"
        # camera blob within gate, holding its lidar-derived range fixed -
        # keeps bearing fresher between the ~10 Hz lidar scans without ever
        # spawning a phantom enemy or inventing range. See
        # _fuse_enemy_bearings for the safety property.
        if cam_enemy_fix_enabled and state.mode == "run":
            with state._lock:
                epose = state._state["pose"]
                enemies_now = state._state["enemies"]
            if epose is not None and enemies_now:
                cam_bearings = _detect_bot_bearings(hsv_unwrap, col_bearing)
                if cam_bearings:
                    _fuse_enemy_bearings(enemies_now, cam_bearings,
                                        epose[0], epose[1], epose[2])
                    with state._lock:
                        state._state["enemies"] = enemies_now

        # Assumes whatever goal is in front is the one we're shooting at, computed unconditionally so _lidar_thread's global search always has a prior.
        front_goal = detect_front_goal(hsv_unwrap, col_bearing)
        with state._lock:
            state._state["front_goal"] = front_goal

        # Dwibbler-stall's visual second vote (sec 3.16), off until the real sticker is fitted, computed unconditionally so flipping it on doesn't need a camera-thread restart.
        mark_seen = detect_dwibble_mark(hsv_unwrap, notch_region,
                                        dwibble_mark_lower, dwibble_mark_upper,
                                        dwibble_mark_min_px)

        with state._lock:
            state._state["ball"]    = result
            state._state["frame"]   = frame
            state._state["mask"]    = None
            state._state["dwibble_mark_seen"] = mark_seen
            state._state["cx_px"]   = centroid[0] if centroid else None
            state._state["cy_px"]   = centroid[1] if centroid else None
            state._state["radius"]  = radius
            state._state["cam_fps"] = cam_fps
        _mark_health_t("camera")

        with state._frame_cond:
            state._frame_cond.notify_all()
