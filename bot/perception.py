"""Point-to-line ICP against FieldModel, and lidar-based robot detection."""

import math

import numpy as np

from bot.field import FieldModel


class Perception:
    """localisation and robot detection against the RCJ field."""

    # localisation tuning
    inlier_threshold_mm = 120.0 # max wall-distance to count as an inlier
    max_iters = 8 # ICP iterations per call
    converge_mm = 1.0 # stop early if translation delta < this

    # object detection tuning
    robot_radius_mm = 105.0 # about 210 mm diameter robots
    max_robots_on_field = 3 # other bots visible to us (1-3 by rule)
    cluster_gap_mm = 150.0 # max gap between consecutive cluster points
    depth_split_mm = 120.0 # range jump that splits a cluster
    # object must sit this far in front of the wall its ray would otherwise hit
    occlude_margin_mm = 140.0
    min_span_mm = 35.0 # clusters narrower than this are noise
    max_cluster_span_mm = 280.0 # clusters wider than this get split
    # candidates this close are the same bot (an arc chopped by dropouts/occlusion)
    dup_merge_mm = 170.0
    circle_resid_max_mm = 45.0 # max circle-fit RMS for a valid robot
    field_margin_mm = 60.0 # robot centres must be this far in-field
    scan_step_rad = math.radians(0.9) # LiDAR angular step (about LD19)

    # coordinate transform
    @staticmethod
    def transform(points_local, X, Y, H_deg):
        """convert a list of robot-frame (xl, yl) pairs to field-frame."""
        h = math.radians(H_deg)
        c, s = math.cos(h), math.sin(h)
        return [(X + xl * c + yl * s,
                 Y - xl * s + yl * c) for xl, yl in points_local]

    # ICP internals
    @staticmethod
    def _solve3(A, b):
        """solve the 3x3 system Ax = b by Gaussian elimination, or None if singular."""
        M = [list(A[i]) + [b[i]] for i in range(3)]
        for col in range(3):
            piv = max(range(col, 3), key=lambda r: abs(M[r][col]))
            if abs(M[piv][col]) < 1e-12:
                return None
            M[col], M[piv] = M[piv], M[col]
            pv = M[col][col]
            for j in range(col, 4):
                M[col][j] /= pv
            for r in range(3):
                if r != col:
                    f = M[r][col]
                    for j in range(col, 4):
                        M[r][j] -= f * M[col][j]
        return [M[0][3], M[1][3], M[2][3]]

    # localisation
    @classmethod
    def localise(cls, points_local, init_pose,
                 inlier_threshold=None, max_iters=None):
        """refine init_pose (X, Y, H_deg) by point-to-line ICP of robot-frame scan points
        against the nearest field walls.
        """
        if inlier_threshold is None:
            inlier_threshold = cls.inlier_threshold_mm
        if max_iters is None:
            max_iters = cls.max_iters

        X, Y, H = init_pose

        # convert once; each iteration below only rotates the array
        if not len(points_local):
            L = np.zeros((0, 2), dtype=np.float64)
        else:
            L = np.asarray(points_local, dtype=np.float64).reshape(-1, 2)
        xl, yl = L[:, 0], L[:, 1]

        for _ in range(max_iters):
            h = math.radians(H)
            c, s = math.cos(h), math.sin(h)
            px = X + xl * c + yl * s
            py = Y - xl * s + yl * c
            dist, nx, ny = FieldModel.nearest_wall_batch(np.column_stack((px, py)))
            mask = (dist <= inlier_threshold) & (dist >= 1e-6)
            n_in = int(mask.sum())
            if n_in < 6:
                break
            nx_m, ny_m, dist_m = nx[mask], ny[mask], dist[mask]
            aH = nx_m * (py[mask] - Y) - ny_m * (px[mask] - X)
            J = np.stack((nx_m, ny_m, aH), axis=1) # (n_in, 3)
            AtA = J.T @ J # (3, 3)
            Atb = -(J.T @ dist_m) # (3,)
            delta = cls._solve3(AtA, Atb)
            if delta is None:
                break
            X += delta[0]
            Y += delta[1]
            H += math.degrees(delta[2])
            if (abs(delta[0]) < cls.converge_mm and
                    abs(delta[1]) < cls.converge_mm and
                    abs(math.degrees(delta[2])) < 0.05):
                break

        h = math.radians(H)
        c, s = math.cos(h), math.sin(h)
        px = X + xl * c + yl * s
        py = Y - xl * s + yl * c
        dist, _nx, _ny = FieldModel.nearest_wall_batch(np.column_stack((px, py)))
        inlier_mask = dist <= inlier_threshold
        inliers = list(zip(px[inlier_mask].tolist(), py[inlier_mask].tolist()))
        outliers = list(zip(px[~inlier_mask].tolist(), py[~inlier_mask].tolist()))
        rms = (math.sqrt(float(np.mean(dist[inlier_mask] ** 2)))
               if inliers else float("inf"))

        return {
            "pose": (float(X), float(Y), float(H % 360.0)),
            "inliers": inliers,
            "outliers": outliers,
            "rms_mm": float(rms),
            "inlier_count": len(inliers),
        }

    @staticmethod
    def _frange(start, stop, step):
        """range() for floats, inclusive of stop (within a small epsilon)."""
        vals, x = [], start
        while x <= stop + 1e-6:
            vals.append(x)
            x += step
        return vals

    @classmethod
    def global_localise(cls, points_local,
                        x_step=220.0, y_step=220.0, heading_step=20.0,
                        coarse_iters=4, top_k=8, refine_iters=None,
                        inlier_threshold=None, min_inliers=40,
                        rough_region=None,
                        known_heading=None, heading_tolerance=20.0):
        """the robot's pose from scratch: grid search, refine the best seeds, then test the
        180-degree twin.
        """
        if inlier_threshold is None:
            inlier_threshold = cls.inlier_threshold_mm
        if refine_iters is None:
            refine_iters = cls.max_iters

        if len(points_local) < min_inliers:
            return None
        sparse = points_local[::4] if len(points_local) > 200 else points_local

        if known_heading is not None:
            all_h = cls._frange(0.0, 330.0, heading_step)
            heading_candidates = [h for h in all_h
                                  if abs(((h - known_heading + 180) % 360) - 180)
                                  <= heading_tolerance] or [known_heading]
        else:
            heading_candidates = cls._frange(0.0, 330.0, heading_step)

        scored = []
        for x in cls._frange(0.0, FieldModel.field_x, x_step):
            for y in cls._frange(0.0, FieldModel.field_y, y_step):
                for h in heading_candidates:
                    r = cls.localise(sparse, (x, y, h),
                                     inlier_threshold, max_iters=coarse_iters)
                    scored.append((r["rms_mm"], r["inlier_count"], r["pose"]))
        scored.sort(key=lambda s: (s[1] < min_inliers // 2, s[0]))

        refined = []
        for _, _, seed in scored[:top_k]:
            r = cls.localise(points_local, seed, inlier_threshold,
                             max_iters=refine_iters)
            if r["inlier_count"] >= min_inliers:
                refined.append(r)
        if not refined:
            return None
        refined.sort(key=lambda r: r["rms_mm"])
        best = refined[0]

        bx, by, bh = best["pose"]
        twin = cls.localise(
            points_local,
            (FieldModel.field_x - bx, FieldModel.field_y - by,
             (bh + 180.0) % 360.0),
            inlier_threshold, max_iters=refine_iters)

        is_ambiguous = (twin["inlier_count"] >= min_inliers and
                        twin["rms_mm"] <= best["rms_mm"] * 1.3)

        result = dict(best)
        result["ambiguous"] = is_ambiguous
        result["alternate_pose"] = twin["pose"] if is_ambiguous else None
        result["resolved"] = not is_ambiguous

        # A heading prior beats a position guess: best came from a seed already
        # within heading_tolerance of a trusted heading, and the twin is 180 deg
        # off it. rough_region (which half the robot was last in) only decides on
        # a true cold start with no heading prior.
        if is_ambiguous and known_heading is not None:
            result["resolved"] = True
        elif is_ambiguous and rough_region is not None:
            rx, ry = rough_region
            if (math.hypot(twin["pose"][0] - rx, twin["pose"][1] - ry) <
                    math.hypot(bx - rx, by - ry)):
                result = dict(twin)
                result["ambiguous"] = True
                result["alternate_pose"] = best["pose"]
            result["resolved"] = True

        return result

    # Obstacle detection: ray-test candidate points against the wall model, cluster,
    # split on range and size, circle-fit, and keep at most max_robots_on_field.

    @classmethod
    def _object_points(cls, outliers, sensor_xy):
        """occlusion-gate outlier points against the field geometry."""
        sx, sy = sensor_xy
        kept = []
        for (px, py) in outliers:
            dx, dy = px - sx, py - sy
            rng = math.hypot(dx, dy)
            if rng < 2.0 * cls.robot_radius_mm * 0.3: # inside own footprint
                continue
            d_wall = FieldModel.raycast(sx, sy, dx / rng, dy / rng)
            if d_wall is None:
                continue # ray exits the model
            if rng > d_wall - cls.occlude_margin_mm:
                continue # wall-ish / beyond
            kept.append((px, py, rng))
        return kept

    @classmethod
    def _cluster(cls, pts, gap_mm=None):
        """group occlusion-gated (x, y, range) points into clusters."""
        if gap_mm is None:
            gap_mm = cls.cluster_gap_mm
        if not pts:
            return []
        clusters = [[pts[0]]]
        for p in pts[1:]:
            q = clusters[-1][-1]
            if (math.hypot(p[0] - q[0], p[1] - q[1]) <= gap_mm
                    and abs(p[2] - q[2]) <= cls.depth_split_mm):
                clusters[-1].append(p)
            else:
                clusters.append([p])
        if len(clusters) > 1:
            first, last = clusters[0][0], clusters[-1][-1]
            if (math.hypot(first[0] - last[0], first[1] - last[1]) <= gap_mm
                    and abs(first[2] - last[2]) <= cls.depth_split_mm):
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()
        return clusters

    @classmethod
    def _split_oversized(cls, cluster, depth=0):
        """split a cluster wider than max_cluster_span_mm at its largest internal gap (two
        adjacent bots).
        """
        if cls.cluster_span(cluster) <= cls.max_cluster_span_mm:
            return [cluster]
        if depth >= 2 or len(cluster) < 4:
            return []
        gaps = [math.hypot(cluster[i + 1][0] - cluster[i][0],
                           cluster[i + 1][1] - cluster[i][1])
                for i in range(len(cluster) - 1)]
        k = max(range(len(gaps)), key=lambda i: gaps[i]) + 1
        if k < 2 or k > len(cluster) - 2:
            k = len(cluster) // 2
        out = []
        for part in (cluster[:k], cluster[k:]):
            out.extend(cls._split_oversized(part, depth + 1))
        return out

    @classmethod
    def _inside_playable(cls, x, y):
        """True if (x, y) could be a robot centre: in the field and clear of the walls and goal
        structures.
        """
        m = cls.field_margin_mm
        if not (m <= x <= FieldModel.field_x - m
                and m <= y <= FieldModel.field_y - m):
            return False
        bx0 = FieldModel.bx0 - m
        bx1 = FieldModel.bx1 + m
        gd = FieldModel.goal_depth + m
        if bx0 <= x <= bx1 and (y <= gd or y >= FieldModel.field_y - gd):
            return False
        return True

    @classmethod
    def _circle_residual(cls, cluster, cx, cy, radius=None):
        """RMS of |point - centre| - radius over the cluster (mm): how well a fixed-radius
        circle fits.
        """
        if radius is None:
            radius = cls.robot_radius_mm
        sq = 0.0
        for p in cluster:
            sq += (math.hypot(p[0] - cx, p[1] - cy) - radius) ** 2
        return math.sqrt(sq / len(cluster))

    @staticmethod
    def cluster_span(cluster):
        """bounding-box diagonal of a cluster of (x, y[, ...]) points (mm)."""
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    @classmethod
    def fit_circle_centre(cls, cluster, sensor_xy, radius=None, iters=8):
        """fit a robot centre to its visible near-side arc by Gauss-Newton on |p_i - C| =
        radius, seen from sensor_xy.
        """
        if radius is None:
            radius = cls.robot_radius_mm
        cx = sum(p[0] for p in cluster) / len(cluster)
        cy = sum(p[1] for p in cluster) / len(cluster)
        sx, sy = sensor_xy
        dx, dy = cx - sx, cy - sy
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            cx += dx / dist * radius * 0.5
            cy += dy / dist * radius * 0.5

        for _ in range(iters):
            Sxx = Sxy = Syy = bx = by = 0.0
            for (px, py) in cluster:
                ex, ey = px - cx, py - cy
                d = math.hypot(ex, ey)
                if d < 1e-6:
                    continue
                ux, uy = ex / d, ey / d
                r = d - radius
                Sxx += ux * ux; Sxy += ux * uy; Syy += uy * uy
                bx += ux * r; by += uy * r
            det = Sxx * Syy - Sxy * Sxy
            if abs(det) < 1e-9:
                break
            dcx = (Syy * bx - Sxy * by) / det
            dcy = (Sxx * by - Sxy * bx) / det
            cx += dcx; cy += dcy
            if math.hypot(dcx, dcy) < 0.05:
                break
        return cx, cy

    @classmethod
    def detect_robots(cls, outliers, sensor_xy, teammate_pos=None,
                      max_span=None, min_points=None):
        """turn non-wall outlier points into robot detections."""
        if max_span is None:
            max_span = cls.max_cluster_span_mm

        candidates = []
        for raw in cls._cluster(cls._object_points(outliers, sensor_xy)):
            for cluster in cls._split_oversized(raw):
                if teammate_pos is not None:
                    ax = sum(p[0] for p in cluster) / len(cluster)
                    ay = sum(p[1] for p in cluster) / len(cluster)
                    if (math.hypot(ax - teammate_pos[0], ay - teammate_pos[1])
                            <= cls.robot_radius_mm * 2.0):
                        continue
                span = cls.cluster_span(cluster)
                if not (cls.min_span_mm <= span <= max_span):
                    continue
                rng = sum(p[2] for p in cluster) / len(cluster)
                # a bot-sized arc at this range should return about n_exp
                # points; demand 15% of them (and never fewer than the
                # caller's floor)
                n_exp = (2.0 * math.asin(min(1.0, cls.robot_radius_mm / rng))
                         / cls.scan_step_rad)
                floor = max(min_points or 2, int(0.15 * n_exp))
                if len(cluster) < floor:
                    continue
                xy = [(p[0], p[1]) for p in cluster]
                cx, cy = cls.fit_circle_centre(xy, sensor_xy)
                resid = cls._circle_residual(xy, cx, cy)
                if resid > cls.circle_resid_max_mm:
                    continue
                if not cls._inside_playable(cx, cy):
                    continue
                conf = (min(1.0, len(cluster) / max(1.0, n_exp))
                        * (1.0 - resid / cls.circle_resid_max_mm))
                candidates.append({
                    "x": cx, "y": cy,
                    "points": len(cluster),
                    "points_xy": xy, # boundary points, for EnemyProfile
                    "span_mm": span,
                    "resid_mm": resid,
                    "conf": conf,
                })

        # Merge fragments of one robot: a dropout, handle wedge or partial
        # occlusion chops an arc into pieces whose circle fits all land on one
        # centre.
        candidates.sort(key=lambda d: d["conf"], reverse=True)
        merged = []
        for d in candidates:
            for m in merged:
                if math.hypot(d["x"] - m["x"], d["y"] - m["y"]) <= cls.dup_merge_mm:
                    wm, wd = m["points"], d["points"]
                    w = wm + wd
                    m["x"] = (m["x"] * wm + d["x"] * wd) / w
                    m["y"] = (m["y"] * wm + d["y"] * wd) / w
                    m["points"] = w
                    m["points_xy"] = m.get("points_xy", []) + d.get("points_xy", [])
                    m["span_mm"] = max(m["span_mm"], d["span_mm"])
                    m["conf"] = max(m["conf"], d["conf"])
                    break
            else:
                merged.append(d)

        return merged[:cls.max_robots_on_field]
