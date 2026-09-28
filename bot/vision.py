"""Camera thread + HSV-based ball/goal/enemy-bearing vision."""

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
# post-rotation size (WxH, used everywhere else): a net 270-degree rotation swaps width
# and height versus sensor_mode
sensor_size = (1080, 2028)
sensor_fps = 93.4 # measured 8-bit line rate at this mode

# Pre-processing crop applied to the rotated frame before any detection, in
# post-rotation pixels, 0 = no crop. crop_top/crop_bottom/crop_left are
# per-bot: robot_select publishes them from bot1_config/bot2_config at boot.
crop_top = None
crop_bottom = None
crop_left = None
crop_right = 0

# Anything inside inner or outside outer is never processed, only the ring
# is scanned, which is the bulk of the speed-up. Per-bot, set by robot_select.
exclusion_inner_frac = None
exclusion_outer_frac = None

# Mouth notch: the inner exclusion circle also blanked the dwibbler mouth, so closing on
# the ball used to cycle approach/retreat/blind-spin.
mouth_notch_enabled = True
mouth_notch_half_deg = 27.0 # half-width of the wedge about the dwibbler (70mm mouth)
mouth_notch_px = 50.0 # how far inside inner_r the wedge reaches

# Fish-eye-aware "warp" detection: unwrap the annulus into (radius x angle), with the
# radial spacing matched to the lens's resolution.
warp_ntheta = 540 # angular samples across the ring
ball_diameter_mm = 43.0 # target ball diameter (sets radial resolution)
ball_rows_target = 6 # floor on radial rows across the ball, far out
# ceiling near in; spacing follows the sensor in between (_warp_radial_targets), about
# 2.7x the old flat spacing's near-field detail at no measurable cost
warp_max_rows_per_ball = 16

warp_nr_min = 48 # clamp on the auto-chosen radial sample count
# was 384, forcing uniform ground-distance spacing mismatched to the sensor at both ends
warp_nr_max = 900

# Provisional: re-run --hsv and paste its printed values here.
orange_lower = np.array([0, 201, 127], dtype=np.uint8)
orange_upper = np.array([11, 255, 255], dtype=np.uint8)
min_orange_px = 3

# A blob has to fill this much of its minimum enclosing circle to count as the ball, so a
# red marking or stripe can't beat a smaller ball on pixel count. Measured with each axis
# scaled to the ball's expected size at that row (see _ball_candidates). On synthetic blobs
# a whole ball measures 0.85 near to 0.5 at 2.5 m, half a ball clipped at an edge about 0.45
# and a 3:1 stripe 0.35 to 0.4, so far balls sit close to the line. Tune it with
# tests/fillratio_calib.py.
ball_min_fill_ratio = 0.45

# Angular ROI tracking: the unwrap axis is angle, so a sector is a column window.
ball_sector_track = True
ball_sector_half_deg = 40.0 # search +/- this around the last-seen ball angle
_ball_track_col = None # last accepted ball column (module state), or None

# Logistic saturation boost, pre-inRange.
sat_boost_enabled = False
sat_boost_k = 8.0 # steepness, higher pushes the curve closer to a step
sat_boost_mid = 0.45 # sigmoid centre on [0,1]; pixels below tend toward 0

# mirror the raw atan2(cos theta, sin theta) angle about the forward axis, or this
# camera's mount comes out counter-clockwise against the file's clockwise-positive
# convention
cam_y_sign = -1

# Per-bot camera bearing trim and lidar handle wedge, set by robot_select.
cam_bearing_offset_deg = None
handle_exclusion_deg = None


def _cam_bearing(cos_th, sin_th):
    """robot-frame bearing (deg, 0 = dwibbler-forward, cw+) for a camera-frame direction (cos
    theta, sin theta); the one place cam_y_sign and cam_bearing_offset_deg are applied.
    """
    raw = np.degrees(np.arctan2(cos_th, cam_y_sign * sin_th))
    return _wrap_deg(raw + cam_bearing_offset_deg)


def _cam_direction(bearing_deg):
    """inverse of _cam_bearing: unit image direction (dx, dy) for a robot-frame bearing, used
    by the debug overlay to draw the notch and handle wedges.
    """
    a = math.radians(bearing_deg) - math.radians(cam_bearing_offset_deg)
    return math.sin(a), -math.cos(a)


# Camera controls: lock the ISP so the image stops moving under the threshold.
cam_lock_auto = True # freeze AE/AWB after they settle
cam_settle_s = 1.5 # let them run this long first
# None = adopt whatever AE settled on; a number forces it (must be <= the frame duration,
# or libcamera drops the frame rate to fit)
cam_exposure_us = None
# ISP saturation, applied in hardware before 8-bit quantisation: strictly better than
# software, and free
cam_saturation = 1.8

# Calibrated fish-eye model: pixel offset from the optical centre to ground mm via an
# odd-power polynomial in t = tan(r / _fisheye_b), fitted against the current crop's pixel
# radii. Re-fit (don't rescale) if crop_*, fisheye_px_scale or the frame size change:
# public-repo/tools/ fisheye_fit.py takes tape-measured samples, picks the model size by
# leave-one-out error, and prints the two lines below ready to paste. Its JSON records the
# camera geometry the samples were taken under, so a stale fit is detectable.
fisheye_px_scale = 1.0
fisheye_rmax = math.sqrt(8387300) # about 2896 mm saturation distance
_fisheye_b = -510.76033
_fisheye_c = (-214.82028, -246.09173, 130.02845, 1.0, -42.09367)
# tan() blows up at r = |b|*pi/2; the polynomial is only meaningful below that.
_fisheye_pole = abs(_fisheye_b) * math.pi / 2.0


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


# clamp point chosen so _fisheye_poly(fisheye_rclamp) == fisheye_rmax exactly, so the
# saturation is continuous rather than a step
fisheye_rclamp = _solve_clamp_radius(fisheye_rmax)


def _sat_boost(hsv):
    """in-place logistic saturation boost on an HSV uint8 image."""
    if not sat_boost_enabled:
        return hsv
    s = hsv[:, :, 1].astype(np.float32) * (1.0 / 255.0)
    s = 1.0 / (1.0 + np.exp(-sat_boost_k * (s - sat_boost_mid)))
    lo = 1.0 / (1.0 + math.exp( sat_boost_k * sat_boost_mid))
    hi = 1.0 / (1.0 + math.exp(-sat_boost_k * (1.0 - sat_boost_mid)))
    s = (s - lo) * (255.0 / (hi - lo))
    hsv[:, :, 1] = np.clip(s, 0, 255).astype(np.uint8)
    return hsv


# HSV init

def _init_hsv():
    """seed _state's live HSV thresholds from the saved orange_lower/orange_upper."""
    with state._lock:
        state._state["hsv_lower"] = orange_lower.copy()
        state._state["hsv_upper"] = orange_upper.copy()


# Detection

# Handle exclusion for the camera: the same side normals as the lidar (+/-90 deg), in
# atan2 output space [-180, 180].
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
    d_in = float(_fisheye_radius(inner_r))
    d_out = float(_fisheye_radius(outer_r))
    coarse_mm = ball_diameter_mm / max(1.0, float(ball_rows_target))
    fine_mm = (1e-3 if warp_max_rows_per_ball is None
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
    """precompute the cv2.remap tables that unwrap the annulus (inner_r, outer_r] around (cx,
    cy) into an (n_r x n_theta) image.
    """
    theta_lut = (2.0 * np.pi) * np.arange(n_theta, dtype=np.float32) / n_theta

    grid_r = np.linspace(0.0, float(outer_r), 4096)
    grid_d = np.array([_fisheye_radius(r) for r in grid_r])
    d_tgt = _warp_radial_targets(inner_r, outer_r, grid_r, grid_d)
    r_lut = np.interp(d_tgt, grid_d, grid_r).astype(np.float32)

    cos_t = np.cos(theta_lut)[None, :] # (1, n_theta)
    sin_t = np.sin(theta_lut)[None, :]
    rr = r_lut[:, None] # (n_r, 1)
    map_x = (cx + rr * cos_t).astype(np.float32)
    map_y = (cy + rr * sin_t).astype(np.float32)
    return map_x, map_y, r_lut, theta_lut


def _ring_geometry():
    """the unwrap for this bot's crop, exclusion and notch settings: (map_x, map_y, r_lut,
    theta_lut, col_bearing, inner_blank, notch_region, inner_r, outer_r, ring_r). The camera
    thread uses it, and so should any bench tool that measures what detect_ball_warp sees.
    """
    crop_w = sensor_size[0] - crop_left - crop_right
    crop_h = sensor_size[1] - crop_top - crop_bottom
    ccx = crop_w // 2
    ccy = crop_h // 2
    max_r = min(crop_w, crop_h) // 2
    inner_r = int(exclusion_inner_frac * max_r)
    outer_r = int(exclusion_outer_frac * max_r)
    # the unwrap starts at the notch depth so the mouth wedge gets sampled; outside
    # the wedge those extra near rows are blanked again below (inner_blank)
    notch_px = int(round(mouth_notch_px)) if mouth_notch_enabled else 0
    ring_r = max(1, inner_r - notch_px)

    # fish-eye-aware unwrap: sample the annulus into a (radius x angle) image whose
    # rows follow the lens's resolution (see _warp_radial_targets)
    map_x, map_y, r_lut, theta_lut = build_warp_maps(
        ccx, ccy, ring_r, outer_r, None, warp_ntheta)
    warp_nr = len(r_lut)

    # robot-frame bearing (deg, 0 = dwibbler-forward, cw+) of each unwrapped column
    col_bearing = _cam_bearing(np.cos(theta_lut), np.sin(theta_lut))

    # mouth notch mask: blank the rows inside the original inner_r everywhere except
    # the wedge about the dwibbler
    inner_blank = None
    # where detect_dwibble_mark looks: exactly the wedge inner_blank leaves out of
    # ball detection
    notch_region = None
    if notch_px:
        notch_cols = np.abs(col_bearing) <= mouth_notch_half_deg
        notch_rows = r_lut < inner_r
        inner_blank = np.zeros((warp_nr, warp_ntheta), dtype=bool)
        inner_blank[np.ix_(notch_rows, ~notch_cols)] = True
        notch_region = np.zeros((warp_nr, warp_ntheta), dtype=bool)
        notch_region[np.ix_(notch_rows, notch_cols)] = True

    return (map_x, map_y, r_lut, theta_lut, col_bearing, inner_blank, notch_region,
            inner_r, outer_r, ring_r)


def _ball_candidates(mask, r_lut=None, n_theta=None):
    """every blob in mask of at least min_orange_px pixels, as (cx, cy, x, y, w, h, area,
    fill_ratio). area is a pixel count; fill_ratio is the share of the blob's minimum enclosing
    circle it covers, with columns and rows scaled to the ball's expected width and height at
    its row. The unwrap isn't square: past a metre a ball is 2 columns wide and 6 rows tall,
    so an unscaled test would throw every far ball away. With no r_lut both axes count the
    same, as in a plain image.
    """
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for k in range(1, n):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < min_orange_px:
            continue
        x = int(stats[k, cv2.CC_STAT_LEFT])
        y = int(stats[k, cv2.CC_STAT_TOP])
        w = int(stats[k, cv2.CC_STAT_WIDTH])
        h = int(stats[k, cv2.CC_STAT_HEIGHT])
        cx, cy = cents[k]
        sx, sy = (1.0, 1.0) if r_lut is None else _ball_cells_at(cy, r_lut, n_theta)
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == k)
        pts = np.column_stack((xs / sx, ys / sy)).astype(np.float32)
        _c, enc_r = cv2.minEnclosingCircle(pts)
        # the points are pixel centres, so add half a pixel's extent
        enc_r += 0.5 * math.hypot(1.0 / sx, 1.0 / sy)
        fill = (area / (sx * sy)) / (math.pi * enc_r * enc_r)
        out.append((float(cx), float(cy), x, y, w, h, area, fill))
    return out


def _biggest_blob(mask, r_lut=None, n_theta=None):
    """the biggest blob that passes ball_min_fill_ratio, as (col, row, w, h, area), or None."""
    best = None
    for cand in _ball_candidates(mask, r_lut, n_theta):
        if cand[7] >= ball_min_fill_ratio and (best is None or cand[6] > best[6]):
            best = cand
    if best is None:
        return None
    cx, cy, _x, _y, w, h, area, _fill = best
    return cx, cy, w, h, area


def suggest_fill_ratio_threshold(samples):
    """the ball_min_fill_ratio that best splits logged (is_ball, fill_ratio) samples, for
    tests/fillratio_calib.py. Tries a cut halfway between each pair of neighbouring ratios,
    keeps the one that sorts the most samples right, and on a tie the one with the most room
    to its nearest sample. Returns (threshold, correct, total); threshold is None if every
    sample is the same kind.
    """
    balls = sorted(r for is_ball, r in samples if is_ball)
    decoys = sorted(r for is_ball, r in samples if not is_ball)
    total = len(balls) + len(decoys)
    if not balls or not decoys:
        return None, 0, total

    ratios = sorted(set(balls + decoys))
    cuts = ([0.0] + [(ratios[i] + ratios[i + 1]) / 2.0 for i in range(len(ratios) - 1)]
            + [1.0])
    best_t, best_correct, best_margin = cuts[0], -1, -1.0
    for t in cuts:
        margins = [b - t for b in balls if b >= t] + [t - d for d in decoys if d < t]
        correct = len(margins)
        margin = min(margins, default=0.0)
        if correct > best_correct or (correct == best_correct and margin > best_margin):
            best_t, best_correct, best_margin = t, correct, margin
    return round(best_t, 3), best_correct, total


def _find_ball_columns(orange, n_theta, n_r, r_lut=None):
    """ball blob column/row (col, row) + box (w, h), or None."""
    global _ball_track_col

    if ball_sector_track and _ball_track_col is not None:
        H = max(1, int(round(ball_sector_half_deg / 360.0 * n_theta)))
        # modular gather -> the window is contiguous even across the 0/2pi seam
        cols = (np.arange(_ball_track_col - H, _ball_track_col + H + 1)
                % n_theta)
        sub = np.ascontiguousarray(orange[:, cols])
        b = _biggest_blob(sub, r_lut, n_theta)
        if b is not None:
            cx, cy, w, h, _a = b
            # accept only a blob clear of both window edges; one touching an
            # edge may run past the window, so fall back to the full scan
            # below
            if (cx - w / 2.0) > 0.5 and (cx + w / 2.0) < sub.shape[1] - 1.5:
                # cols is a contiguous modular range, so mapping back is
                # just an offset
                col = (int(cols[0]) + cx) % n_theta
                _ball_track_col = int(round(col)) % n_theta
                return col, min(max(cy, 0.0), n_r - 1.0), w, h
        # window miss / ran off the edge -> full scan

    # Full 360 scan, wrap-padded so a ball on the seam stays one blob.
    margin = n_theta // 8
    padded = np.hstack([orange[:, -margin:], orange, orange[:, :margin]])
    b = _biggest_blob(padded, r_lut, n_theta)
    if b is None:
        _ball_track_col = None
        return None
    cx, cy, w, h, _a = b
    col = (cx - margin) % n_theta
    _ball_track_col = int(round(col)) % n_theta
    return col, min(max(cy, 0.0), n_r - 1.0), w, h


# Sub-pixel ball centre: at range the ball is only 1-2 source pixels deep radially
# (fisheye compression), so a binary centroid quantises the range hard.
ball_centre_fill_holes = True
# cells: fragments whose bounding boxes come within this of the main one are the same ball
ball_centre_merge_span = 6.0
ball_centre_subpixel = True
# cells of margin around the blob's box, so cells the hard threshold excluded still get a vote.
ball_centre_pad = 2
# S floor for the soft weight.
ball_centre_s_floor = 85.0


def _ball_cells_at(row, r_lut, n_theta):
    """how many cells wide and tall a ball should be at this radial row."""
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
    """refine a blob's (col, row) centroid to a saturation-weighted centre, as floats."""
    if not ball_centre_subpixel:
        return col, row
    n_r, n_theta = s_raw.shape
    # size the window off the ball's expected extent as well as the blob's, so a
    # partly eaten or split blob still gets its whole neighbourhood weighed
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
            # merge every fragment near the biggest one, then hull the union:
            # a highlight that splits the ball leaves pieces, not a hole
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
    # the column index is window-relative, so average in window space and map back;
    # averaging wrapped absolute columns would put a ball on the seam on the far side
    # of the ring
    rel = np.arange(cols.size, dtype=np.float64)
    new_col = (float(cols[0]) + float((col_w * rel).sum() / tot)) % n_theta
    return new_col, new_row


def _ball_mask(frame, lower, upper, map_x, map_y, col_bearing, inner_blank=None):
    """unwrap frame and threshold it: (orange mask, unwrapped HSV, raw saturation), with the
    robot body and handle wedges blanked. Shared with tests/fillratio_calib.py so the bench
    tool sees the same mask as the robot.
    """
    unwrapped = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(unwrapped, cv2.COLOR_BGR2HSV)
    # keep the saturation as it came off the sensor: the boost is right for finding
    # the ball and wrong for locating its centre
    s_raw = hsv[:, :, 1].copy()
    _sat_boost(hsv)
    orange = cv2.inRange(hsv, lower, upper)
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

    return orange, hsv, s_raw


def detect_ball_warp(frame, lower, upper, map_x, map_y, r_lut, theta_lut,
                     col_bearing, inner_blank=None):
    """detect the orange ball inside the unwrapped annulus produced by build_warp_maps()."""
    n_r = len(r_lut)
    n_theta = len(theta_lut)

    orange, hsv, s_raw = _ball_mask(frame, lower, upper, map_x, map_y, col_bearing, inner_blank)

    blob = _find_ball_columns(orange, n_theta, n_r, r_lut)
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
    th = (2.0 * math.pi) * (col / n_theta)
    px = r_px * math.cos(th)
    py = r_px * math.sin(th)

    rx, ry = _pixel_to_position(px, py)
    dist_mm = math.hypot(rx, ry)
    # Bearing, not distance, is what cam_bearing_offset_deg corrects, see its comment.
    angle_deg = _cam_bearing(px, py)

    # full-frame centroid, for the debug overlay only (whole cells are fine here)
    ir = int(min(max(round(row), 0), n_r - 1))
    it = int(round(col)) % n_theta
    cx_img = int(round(map_x[ir, it]))
    cy_img = int(round(map_y[ir, it]))

    # rough display radius from the blob's extent (the unwrap distorts shape; a visual
    # guide only)
    arc = r_px * (2.0 * math.pi) * (w_box / n_theta)
    r0 = float(r_lut[max(0, ir - h_box // 2)])
    r1 = float(r_lut[min(n_r - 1, ir + h_box // 2)])
    radius = max(3, int(0.5 * max(arc, abs(r1 - r0))))

    return (angle_deg, dist_mm), (cx_img, cy_img), radius, hsv


# Goal detection: which part of the enemy goal is least defended. An enemy robot in front
# of the goal hides its colour from the camera.
goal_cyan_lower = np.array([85, 70, 70], dtype=np.uint8)
goal_cyan_upper = np.array([105, 255, 255], dtype=np.uint8)
goal_yellow_lower = np.array([20, 70, 70], dtype=np.uint8)
goal_yellow_upper = np.array([40, 255, 255], dtype=np.uint8)

goal_open_enabled = True
goal_min_px_per_col = 2 # goal-colour pixels in a column to call it "open"
goal_window_pad_deg = 6.0 # widen the geometric goal window this much each side

# set by main.py's Blue/Yellow button (and cleared on reset)
_enemy_goal_colour = None # "cyan" | "yellow"


def _goal_colour_ranges(name):
    """HSV (lower, upper) thresholds for the goal-marker colour named "cyan" or "yellow"."""
    if name == "cyan":
        return goal_cyan_lower, goal_cyan_upper
    return goal_yellow_lower, goal_yellow_upper


def enemy_goal_window(pose, attack_low):
    """robot-frame bearing (deg, 0 = +y) to the enemy goal centre, and the half-span to its
    posts.
    """
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
    """within the enemy-goal window, the widest run of columns still showing goal colour
    (open), and the aim bearing at its middle.
    """
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
goal_front_min_px = 40 # goal-colour pixels in the front half to count


def detect_front_goal(hsv, col_bearing, half_deg=cam_front_half_deg,
                      min_px=goal_front_min_px):
    """any goal-coloured pixels within half_deg of dead ahead; no dependency on pose or
    attack_low.
    """
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
    """the roller-arm marker's visual vote: True if enough mouth-notch pixels match the
    dwibble_mark colour (always False with the notch off).
    """
    if notch_region is None:
        return False
    mask = cv2.inRange(hsv, lower, upper) > 0
    return int(mask[notch_region].sum()) >= min_px


# Camera-rate bearing fusion onto lidar-tracked enemies. The lidar list refreshes at about
# 10 Hz and the camera at about 30, so between scans an enemy's bearing can be refreshed
# from the camera. RCJ robots aren't reliably colour-coded, so this looks for a body-width
# blob that is none of the known things (not field green, line white, ball orange or goal
# colour). Like the ball detector, it can't see an all-white or all-black chassis. It only
# ever nudges an already lidar-confirmed enemy's bearing, holding its range; it never
# spawns an enemy or invents a range.
cam_enemy_fix_enabled = True
field_green_lower = np.array([35, 40, 40], dtype=np.uint8) # Provisional,
field_green_upper = np.array([95, 255, 255], dtype=np.uint8) # re-run --hsv
line_white_lower = np.array([0, 0, 150], dtype=np.uint8) # against the
line_white_upper = np.array([179, 60, 255], dtype=np.uint8) # real field/carpet before trusting.
bot_min_px_per_col = 3 # candidate px in a column before it counts
bot_min_width_deg = 4.0 # a run narrower than this is noise, not a body
bot_bearing_gate_deg = 8.0 # camera bearing must be this close to an
                             # enemy's current bearing to be "the same one"
bot_bearing_blend = 0.5 # EMA weight of the camera's bearing fix,
                             # same idiom as RobotTracker.smooth


def _bot_candidate_mask(hsv):
    """px that are none of: field green, wall/line white, ball orange, either goal colour."""
    not_green = cv2.bitwise_not(cv2.inRange(hsv, field_green_lower, field_green_upper))
    not_white = cv2.bitwise_not(cv2.inRange(hsv, line_white_lower, line_white_upper))
    not_ball = cv2.bitwise_not(cv2.inRange(hsv, orange_lower, orange_upper))
    not_cyan = cv2.bitwise_not(cv2.inRange(hsv, goal_cyan_lower, goal_cyan_upper))
    not_yell = cv2.bitwise_not(cv2.inRange(hsv, goal_yellow_lower, goal_yellow_upper))
    mask = cv2.bitwise_and(not_green, not_white)
    mask = cv2.bitwise_and(mask, not_ball)
    mask = cv2.bitwise_and(mask, not_cyan)
    return cv2.bitwise_and(mask, not_yell)


def _detect_bot_bearings(hsv, col_bearing):
    """robot-frame bearings of every "none of the known things" blob wide enough to be a body.
    Circular run-length grouping over the columns (bearing wraps at +-180), reporting every
    run above bot_min_width_deg. The mouth-notch wedge is dropped, or our cavity would
    read as a permanent phantom.
    """
    mask = _bot_candidate_mask(hsv)
    per_col = (mask > 0).sum(axis=0)
    is_bot_col = per_col >= bot_min_px_per_col
    n = len(is_bot_col)
    if n == 0 or not is_bot_col.any() or is_bot_col.all():
        return [] # nothing, or the whole ring (a bad frame either way)

    # rotate the column order to start just after a gap (one exists, not all True), so
    # a run that spans the index wrap reads as one stretch with no wrap bookkeeping
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
    """nudge each lidar-tracked enemy toward the camera bearing closest to its own, holding its
    lidar range. Mutates and returns `enemies`; an enemy with no camera match within
    bot_bearing_gate_deg is left as the lidar reported it. Never creates an enemy or a
    range.
    """
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
    """run the Picamera2 capture loop: grab frames, unwrap and detect each tick, and publish
    ball and detection state for the play loops and the debug page.
    """
    if Picamera2 is None:
        print("[camera] picamera2 not installed, camera disabled "
              "(expected off the Pi; install python3-picamera2 on it).",
              flush=True)
        return
    frame_duration = int(1_000_000 / sensor_fps)
    picam2 = Picamera2()
    # force the 2028x1080 8-bit mode (output_size + bit_depth); no ScalerCrop, so the
    # mode's whole field of view is captured at full resolution
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
            exp = int(cam_exposure_us if cam_exposure_us is not None
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

    (map_x, map_y, r_lut, theta_lut, col_bearing, inner_blank, notch_region,
     inner_r, outer_r, ring_r) = _ring_geometry()
    warp_nr, notch_px = len(r_lut), int(round(mouth_notch_px)) if mouth_notch_enabled else 0

    # ground distance per row, from r_lut (the radial axis is deliberately not
    # uniform); used only for the startup log line
    gd = np.array([_fisheye_radius(float(r)) for r in r_lut], dtype=np.float32)

    if notch_px:
        notch_cols = np.abs(col_bearing) <= mouth_notch_half_deg
        print(f"[camera] mouth notch: +-{mouth_notch_half_deg:.0f} deg, "
              f"{notch_px} px, {int(notch_cols.sum())}/{warp_ntheta} columns", flush=True)

    _frame_times = []
    _printed_shape = False
    while True:
        frame = picam2.capture_array("main")
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) # net 270 CW
        h_rot, w_rot = frame.shape[:2]
        frame = frame[
            crop_top : h_rot - crop_bottom if crop_bottom else None,
            crop_left : w_rot - crop_right if crop_right else None,
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

        # Least-defended part of the enemy goal: the pose gives the goal's angular
        # window, the camera says which of it still shows goal colour
        open_goal = None
        if goal_open_enabled and state.mode == "run":
            with state._lock:
                gpose = state._state["pose"]
                attack_low = state._state["attack_low"]
            if gpose is not None and attack_low is not None:
                # attack_low is only set once the goal-colour button has
                # been pressed, so _enemy_goal_colour is known here
                gc, gh = enemy_goal_window(gpose, attack_low)
                lo_c, hi_c = _goal_colour_ranges(_enemy_goal_colour)
                open_goal = detect_goal_open(hsv_unwrap, col_bearing,
                                             lo_c, hi_c, gc, gh)
        with state._lock:
            state._state["open_goal"] = open_goal

        # Camera-rate bearing refresh for lidar-confirmed enemies (see
        # _fuse_enemy_bearings). The fuse runs on a copy: every reader takes
        # _state["enemies"] under the lock, so an in-place edit could show them
        # half-updated dicts. It is written back only if the lidar thread hasn't
        # published a newer list meanwhile, or the stale copy would overwrite it.
        if cam_enemy_fix_enabled and state.mode == "run":
            with state._lock:
                epose = state._state["pose"]
                enemies_now = state._state["enemies"]
            if epose is not None and enemies_now:
                cam_bearings = _detect_bot_bearings(hsv_unwrap, col_bearing)
                if cam_bearings:
                    fused = _fuse_enemy_bearings([dict(e) for e in enemies_now],
                                                 cam_bearings,
                                                 epose[0], epose[1], epose[2])
                    with state._lock:
                        if state._state["enemies"] is enemies_now:
                            state._state["enemies"] = fused

        # assumes whatever goal is in front is the one we're shooting at; computed
        # every frame
        front_goal = detect_front_goal(hsv_unwrap, col_bearing)
        with state._lock:
            state._state["front_goal"] = front_goal

        # the roller-arm marker's vote, computed every frame so enabling it needs
        # no restart
        mark_seen = detect_dwibble_mark(hsv_unwrap, notch_region,
                                        dwibble_mark_lower, dwibble_mark_upper,
                                        dwibble_mark_min_px)

        with state._lock:
            state._state["ball"] = result
            state._state["frame"] = frame
            state._state["mask"] = None
            state._state["dwibble_mark_seen"] = mark_seen
            state._state["cx_px"] = centroid[0] if centroid else None
            state._state["cy_px"] = centroid[1] if centroid else None
            state._state["radius"] = radius
            state._state["cam_fps"] = cam_fps
        _mark_health_t("camera")

        with state._frame_cond:
            state._frame_cond.notify_all()
