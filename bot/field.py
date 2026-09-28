"""Field geometry: wall segments, FieldModel, and the wrap_deg angle helper."""

import math

import numpy as np


# Wall keep-out distances used by FieldModel.active_walls. They are owned by
# bot.motion, which publishes both here when it loads.
keep_min_mm = None
goal_flank_keep_mm = None


def wrap_deg(a):
    """wrap an angle to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def _build_field_segments(FX, FY, bx0, bx1, sx0, sx1, goal_depth, slot_depth):
    """build FieldModel.segments: both side walls, then each goal end as line-left, seven box
    segments, line-right.
    """
    def _goal_end(y_line, direction):
        """the nine segments of one goal end; y_line is the goal line, direction (+1/-1) the
        way the box extends into the field.
        """
        s = direction
        yf = y_line + s * goal_depth
        ys = y_line + s * (goal_depth - slot_depth)
        return [
            ((0.0, y_line), (bx0, y_line)),
            ((bx0, y_line), (bx0, yf)),
            ((bx0, yf), (sx0, yf)),
            ((sx0, yf), (sx0, ys)),
            ((sx0, ys), (sx1, ys)),
            ((sx1, ys), (sx1, yf)),
            ((sx1, yf), (bx1, yf)),
            ((bx1, yf), (bx1, y_line)),
            ((bx1, y_line), (FX, y_line)),
        ]
    segs = [((0.0, 0.0), (0.0, FY)),
             ((FX, 0.0), (FX, FY))]
    segs += _goal_end(0.0, +1) # y=0 end, box extends into the field (+y)
    segs += _goal_end(FY, -1) # y=FY end, box extends into the field (-y)
    return segs


class FieldModel:
    """RCJ soccer field geometry (mm), all static/class methods."""

    field_x = 1820.0 # short axis (x)
    field_y = 2430.0 # long axis (y)
    goal_width = 470.0 # goal-box width along x
    goal_depth = 300.0 # goal-box depth into the field along y
    slot_width = 450.0 # goal-mouth width along x
    slot_depth = 74.0 # goal-mouth depth along y

    cx = field_x / 2 # 910.0, the field's horizontal centre
    bx0 = cx - goal_width / 2 # 675.0, goal-box left edge
    bx1 = cx + goal_width / 2 # 1145.0, goal-box right edge
    sx0 = cx - slot_width / 2 # 685.0, slot left edge
    sx1 = cx + slot_width / 2 # 1135.0, slot right edge

    segments = _build_field_segments(
        field_x, field_y,
        bx0, bx1, sx0, sx1,
        goal_depth, slot_depth,
    )

    # Per-segment tags in _build_field_segments' order. The box's two outer flanks are
    # "goal-flank" rather than "goal": a robot passing a goal has no reason to touch
    # them, and an exempt flank right past the front corner is where the wheel cutouts
    # used to wedge. The mouth and back stay "goal" (exempt), so the robot can still
    # drive fully in to clear or score.
    _goal_end_tags = (["boundary", "goal-flank"] + ["goal"] * 5
                      + ["goal-flank", "boundary"])
    segment_tags = ["boundary", "boundary"] + _goal_end_tags + _goal_end_tags
    assert len(segment_tags) == len(segments), "segment tag/count mismatch"

    _seg_a = np.array([s[0] for s in segments], dtype=np.float64) # (S, 2)
    _seg_ex = np.array([s[1][0] - s[0][0] for s in segments], dtype=np.float64) # (S,)
    _seg_ey = np.array([s[1][1] - s[0][1] for s in segments], dtype=np.float64) # (S,)
    _seg_len2 = _seg_ex ** 2 + _seg_ey ** 2
    _seg_len2[_seg_len2 < 1e-9] = 1e-9 # guard a degenerate (zero-length) segment

    @classmethod
    def nearest_wall_batch(cls, pts):
        """vectorised nearest_wall: pts is an (N, 2) array-like of (x, y)."""
        P = np.atleast_2d(np.asarray(pts, dtype=np.float64)) # (N, 2)
        # (N, 1, 2) - (1, S, 2) -> (N, S, 2): every point against every segment
        # start in one subtraction
        rel = P[:, None, :] - cls._seg_a[None, :, :]
        t = (rel[:, :, 0] * cls._seg_ex + rel[:, :, 1] * cls._seg_ey) / cls._seg_len2
        np.clip(t, 0.0, 1.0, out=t)
        foot_x = cls._seg_a[None, :, 0] + t * cls._seg_ex[None, :]
        foot_y = cls._seg_a[None, :, 1] + t * cls._seg_ey[None, :]
        dx = P[:, 0:1] - foot_x
        dy = P[:, 1:2] - foot_y
        d2 = dx * dx + dy * dy # (N, S)
        j = np.argmin(d2, axis=1) # nearest segment per point
        rows = np.arange(P.shape[0])
        dist = np.sqrt(d2[rows, j])
        fdx, fdy = dx[rows, j], dy[rows, j]
        safe = dist > 1e-6
        nx = np.where(safe, np.divide(fdx, dist, out=np.zeros_like(dist), where=safe), 0.0)
        ny = np.where(safe, np.divide(fdy, dist, out=np.zeros_like(dist), where=safe), 0.0)
        return dist, nx, ny

    @classmethod
    def active_walls(cls, px, py, within_mm):
        """every keep-out wall (boundary or goal-flank) within within_mm of (px, py), as (dist,
        nx, ny, keep): the unit normal from wall to point and that wall's keep-out distance.
        """
        out = []
        for seg, tag in zip(cls.segments, cls.segment_tags):
            if tag not in ("boundary", "goal-flank"):
                continue
            fx, fy, dist = cls.closest_point_on_segment(px, py, seg)
            if dist < within_mm:
                if dist < 1e-6:
                    nx, ny = 0.0, 0.0
                else:
                    nx, ny = (px - fx) / dist, (py - fy) / dist
                keep = goal_flank_keep_mm if tag == "goal-flank" else keep_min_mm
                out.append((dist, nx, ny, keep))
        return out

    @staticmethod
    def closest_point_on_segment(px, py, seg):
        """nearest point on a finite segment to (px, py), as (foot_x, foot_y, distance_mm)."""
        (x1, y1), (x2, y2) = seg
        ex, ey = x2 - x1, y2 - y1
        seg_len2 = ex * ex + ey * ey
        if seg_len2 < 1e-9:
            fx, fy = x1, y1
        else:
            t = ((px - x1) * ex + (py - y1) * ey) / seg_len2
            t = max(0.0, min(1.0, t))
            fx = x1 + t * ex
            fy = y1 + t * ey
        return fx, fy, math.hypot(px - fx, py - fy)

    @classmethod
    def nearest_wall(cls, px, py):
        """nearest field wall to (px, py), as (dist, foot_x, foot_y, nx, ny)."""
        best = None
        for seg in cls.segments:
            fx, fy, dist = cls.closest_point_on_segment(px, py, seg)
            if best is None or dist < best[2]:
                best = (fx, fy, dist)
        fx, fy, dist = best
        if dist < 1e-6:
            nx, ny = 0.0, 0.0
        else:
            nx, ny = (px - fx) / dist, (py - fy) / dist
        return dist, fx, fy, nx, ny

    @staticmethod
    def ray_segment_dist(ox, oy, dx, dy, seg):
        """distance along the ray from (ox, oy) in direction (dx, dy) to a segment, or None on
        a miss.
        """
        (x1, y1), (x2, y2) = seg
        ex, ey = x2 - x1, y2 - y1
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-9:
            return None # ray is parallel to the segment
        t = ((x1 - ox) * ey - (y1 - oy) * ex) / denom
        u = ((x1 - ox) * dy - (y1 - oy) * dx) / denom
        if t > 0 and 0.0 <= u <= 1.0:
            return t
        return None

    @classmethod
    def raycast(cls, ox, oy, dir_x, dir_y):
        """nearest wall hit along unit direction (dir_x, dir_y) from (ox, oy), or None."""
        best = None
        for seg in cls.segments:
            d = cls.ray_segment_dist(ox, oy, dir_x, dir_y, seg)
            if d is not None and (best is None or d < best):
                best = d
        return best

    @classmethod
    def raycast_normal(cls, ox, oy, dir_x, dir_y):
        """raycast plus the unit normal of the segment hit: (dist_mm, nx, ny), or (None, 0.0,
        0.0).
        """
        best = None
        bnx = bny = 0.0
        for (x1, y1), (x2, y2) in cls.segments:
            ex, ey = x2 - x1, y2 - y1
            denom = dir_x * ey - dir_y * ex
            if -1e-9 < denom < 1e-9: # ray parallel to the segment
                continue
            t = ((x1 - ox) * ey - (y1 - oy) * ex) / denom
            u = ((x1 - ox) * dir_y - (y1 - oy) * dir_x) / denom
            if t <= 1e-9 or (best is not None and t >= best) or not (0.0 <= u <= 1.0):
                continue
            L = math.hypot(ex, ey)
            if L < 1e-9:
                continue
            best = t
            bnx, bny = -ey / L, ex / L
        if best is None:
            return None, 0.0, 0.0
        return best, bnx, bny
