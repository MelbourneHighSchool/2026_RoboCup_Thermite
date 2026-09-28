#!/usr/bin/env python3
# lidar_debug.py: localisation jitter logger for --lidarlog.
#
#     python3 -m bot.main --lidarlog # log 60 s, then stop
#     python3 -m bot.main --lidarlog 120 # log 120 s
#
# One plain-text file per run at logs/lidar_YYYYmmdd_HHMMSS.txt. The point
# is to answer "why does the pose jitter" from the log alone, so every scan
# records what went into the ICP (the prior, and where it came from) as well
# as what came out, plus the raw polar returns every raw_every scans. A
# summary block at the end does the arithmetic: pose spread, fit quality
# percentiles, and the biggest jumps.
#
# Nothing here runs unless --lidarlog is passed: bot.state._lidar_log stays
# None and every hook is behind an "is not None" check.
#
# Park the robot for a jitter log: with it stationary, every millimetre of
# pose movement in the summary is error. Log while idle (no button press) to
# keep the motors out of it; the lidar thread runs in idle too.

import math
import os
import threading
import time

# Sample period assumed when costing a hypothetical 100 Hz translation
# sensor against plain dead-reckoning between lidar revolutions, see the
# "dx/dy" note in header() below.
XLATE_PERIOD_S = 0.010


class LidarLogger:
    """text logger for one localisation debug session.

    Fed from the lidar thread (scan/event/global); every public method
    takes the lock. Stops by itself once duration_s or max_bytes is
    reached; is_done() tells the caller when to call close().
    """

    def __init__(self, path, duration_s=60.0, raw_every=10,
                 max_bytes=32 * 1024 * 1024, clock=time.monotonic):
        """open `path` for writing and start the clock.

        `clock` is injectable so the summary arithmetic can be tested without
        a lidar: the timing-derived figures (publish gap, speed, and the
        one-revolution window the deskew score is paired over) are meaningless
        if a test's scans all land in the same microsecond.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.duration_s = duration_s
        self.raw_every = max(0, int(raw_every))
        self.max_bytes = max_bytes
        self._clock = clock
        self._lock = threading.Lock()
        self._fh = open(path, "w", buffering=1 << 16)
        self._t0 = clock()
        self._bytes = 0
        self._closed = False
        self._done = False

        # per-scan history, kept for the summary
        self._n_scan = 0
        self._poses = [] # (t, x, y, h) published poses
        self._deltas = [] # (n, t, dx, dy, dh, dt) scan-to-scan change
        self._rms = []
        self._inliers = []
        self._outliers = []
        self._hz = []
        self._prior_src = {} # source -> count
        self._events = [] # (t, tag, text)
        self._n_raw_dump = 0
        self._drops = [] # (t, rms, n_in, streak) revolutions not published
        # (n, t, dx, dy, dth, src) deskew inputs, per revolution, keyed to the scan they
        # predict
        self._skew = []

    # writing

    def _write(self, line):
        """append one line, tracking size so max_bytes can stop the log."""
        if self._closed:
            return
        data = line + "\n"
        self._fh.write(data)
        self._bytes += len(data)
        if self._bytes >= self.max_bytes:
            self._done = True

    def _t(self):
        """seconds since the log opened."""
        return self._clock() - self._t0

    def is_done(self):
        """True once duration_s has elapsed or max_bytes was hit."""
        if self._done:
            return True
        return (self.duration_s is not None
                and self._t() >= self.duration_s)

    # records

    def header(self, consts):
        """opening block: wall clock, and every constant needed to read the log."""
        with self._lock:
            self._write("# lidar localisation debug log")
            self._write("# started " + time.strftime("%Y-%m-%dT%H:%M:%S"))
            self._write(f"# duration_s={self.duration_s} raw_every={self.raw_every}")
            self._write("#")
            self._write("# Record types:")
            self._write("#   scan: n t hz raw used, one lidar revolution")
            self._write("#   prior: src x y h, the pose fed to ICP and its origin")
            self._write("#   post: x y h rms in out, the pose after ICP and the fit quality")
            self._write("#   delta: dx dy dh dt, published pose change since the last scan")
            self._write("#   skew: dx dy dth src period, motion fed to the deskew before ICP")
            self._write("#   drop: rms in streak, a revolution rejected and its pose not published")
            self._write("#   raw: angle_deg:dist_mm:intensity ... (every raw_every scans)")
            self._write("#   event: tag text, for watchdog, re-search and global fix")
            self._write("#")
            self._write("# skew src values:")
            self._write("#   imu: rotation measured by the compass across this revolution")
            self._write("#   carry: rotation extrapolated from the previous revolution")
            self._write("#   dx/dy are always extrapolated, since there is no translation sensor.")
            self._write("#   The summary scores that extrapolation against what actually")
            self._write("#   happened; that error is what a 100 Hz translation fix would remove.")
            self._write("#")
            self._write("# prior src values:")
            self._write("#   carry: ICP output of the previous scan")
            self._write("#   imu: carry, plus an IMU heading delta")
            self._write("#   global: a fresh global_localise (startup or recovery)")
            self._write("#")
            for k in sorted(consts):
                self._write(f"[const] {k}={consts[k]}")
            self._write("")

    def scan(self, n_raw, n_used, hz, prior, prior_src, post, rms,
             n_in, n_out, scan_points=None):
        """one full revolution: prior, ICP result, and the published delta."""
        with self._lock:
            if self._closed:
                return
            t = self._t()
            self._n_scan += 1
            n = self._n_scan
            self._write(f"[scan] n={n} t={t:.3f} hz={hz:.2f} "
                        f"raw={n_raw} used={n_used}")
            if prior is not None:
                self._write(f"[prior] src={prior_src} x={prior[0]:.1f} "
                            f"y={prior[1]:.1f} h={prior[2]:.2f}")
            self._prior_src[prior_src] = self._prior_src.get(prior_src, 0) + 1
            self._write(f"[post] x={post[0]:.1f} y={post[1]:.1f} h={post[2]:.2f} "
                        f"rms={rms:.1f} in={n_in} out={n_out}")

            if self._poses:
                pt, px, py, ph = self._poses[-1]
                dx, dy = post[0] - px, post[1] - py
                dh = _wrap(post[2] - ph)
                dt = t - pt
                self._write(f"[delta] dx={dx:.1f} dy={dy:.1f} dh={dh:.2f} dt={dt:.3f}")
                self._deltas.append((n, t, dx, dy, dh, dt))

            self._poses.append((t, post[0], post[1], post[2]))
            self._rms.append(rms)
            self._inliers.append(n_in)
            self._outliers.append(n_out)
            self._hz.append(hz)

            if (self.raw_every and scan_points is not None
                    and n % self.raw_every == 0):
                self._dump_raw(scan_points)

    def _dump_raw(self, scan_points):
        """raw polar returns, 12 per line, call with the lock held."""
        self._n_raw_dump += 1
        chunk = []
        for p in scan_points:
            # 0.1 deg / 1 mm is finer than the sensor resolves, and keeps a
            # 60 s log around a third of a megabyte instead of a megabyte.
            chunk.append(f"{p.angle_deg:.1f}:{p.distance_mm:.0f}:{p.intensity}")
            if len(chunk) == 12:
                self._write("[raw] " + " ".join(chunk))
                chunk = []
        if chunk:
            self._write("[raw] " + " ".join(chunk))

    def skew(self, dx, dy, dth, src, period):
        """the motion fed to _deskew, recorded against the scan it precedes.

        Stored as the next scan number because deskew runs before ICP: if this
        revolution is then dropped, no scan record ever appears for it and the
        summary simply finds no match, which is the correct outcome.
        """
        with self._lock:
            if self._closed:
                return
            self._write(f"[skew] dx={dx:.1f} dy={dy:.1f} dth={dth:.2f} "
                        f"src={src} period={period:.3f}")
            self._skew.append((self._n_scan + 1, self._t(), dx, dy, dth,
                               src, period))

    def drop(self, rms, n_in, streak):
        """a revolution ICP could not fit, no pose published, controllers coast."""
        with self._lock:
            if self._closed:
                return
            self._write(f"[drop] rms={rms:.1f} in={n_in} streak={streak}")
            self._drops.append((self._t(), rms, n_in, streak))

    def event(self, tag, text):
        """watchdog trip, re-search, global fix, anything worth a line of its own."""
        with self._lock:
            if self._closed:
                return
            t = self._t()
            self._write(f"[event] {tag} {text}")
            self._events.append((t, tag, text))

    # summary

    def close(self):
        """write the summary block and close the file. safe to call twice."""
        with self._lock:
            if self._closed:
                return self.path
            self._write("")
            self._write("# summary")
            for line in self._summary_lines():
                self._write(line)
            self._closed = True
            self._fh.close()
            return self.path

    def _summary_lines(self):
        """the arithmetic, pose spread, fit quality."""
        out = []
        n = len(self._poses)
        out.append(f"[sum] scans={n} raw_dumps={self._n_raw_dump} "
                   f"elapsed_s={self._t():.1f}")
        if n == 0:
            out.append("[sum] no scans logged, lidar never produced a fix")
            return out

        xs = [p[1] for p in self._poses]
        ys = [p[2] for p in self._poses]
        hs = [p[3] for p in self._poses]
        out.append(f"[sum] hz mean={_mean(self._hz):.2f} min={min(self._hz):.2f}")
        out.append("[sum] pose spread (robot should be parked, all of this is error):")
        out.append(f"[sum]   x mean={_mean(xs):8.1f} sd={_sd(xs):7.2f} "
                   f"ptp={max(xs) - min(xs):7.1f} mm")
        out.append(f"[sum]   y mean={_mean(ys):8.1f} sd={_sd(ys):7.2f} "
                   f"ptp={max(ys) - min(ys):7.1f} mm")
        # Unwrapped so a heading sitting on the 0/360 seam does not read as a
        # 360 deg spread; the mean can therefore fall outside [0, 360).
        hs_u = _unwrap(hs)
        out.append(f"[sum]   h mean={_mean(hs_u) % 360.0:8.2f} sd={_sd(hs_u):7.2f} "
                   f"ptp={max(hs_u) - min(hs_u):7.2f} deg (unwrapped)")

        if self._deltas:
            adx = [abs(d[2]) for d in self._deltas]
            ady = [abs(d[3]) for d in self._deltas]
            adh = [abs(d[4]) for d in self._deltas]
            out.append("[sum] per-scan pose change (jitter rate):")
            out.append(f"[sum]   |dx| mean={_mean(adx):6.2f} p95={_pct(adx, 95):6.2f} "
                       f"max={max(adx):6.2f} mm")
            out.append(f"[sum]   |dy| mean={_mean(ady):6.2f} p95={_pct(ady, 95):6.2f} "
                       f"max={max(ady):6.2f} mm")
            out.append(f"[sum]   |dh| mean={_mean(adh):6.2f} p95={_pct(adh, 95):6.2f} "
                       f"max={max(adh):6.2f} deg")
            worst = sorted(self._deltas, key=lambda d: -math.hypot(d[2], d[3]))[:10]
            out.append("[sum] biggest position jumps (scan, t, dx, dy, dh):")
            for d in worst:
                out.append(f"[sum]   n={d[0]:<5} t={d[1]:7.3f} dx={d[2]:8.2f} "
                           f"dy={d[3]:8.2f} dh={d[4]:7.2f}")

        out.append("[sum] fit quality:")
        out.append(f"[sum]   rms mean={_mean(self._rms):6.2f} "
                   f"p50={_pct(self._rms, 50):6.2f} p95={_pct(self._rms, 95):6.2f} "
                   f"max={max(self._rms):6.2f} mm")
        out.append(f"[sum]   inliers mean={_mean(self._inliers):6.1f} "
                   f"min={min(self._inliers)} p05={_pct(self._inliers, 5):6.1f}")
        out.append(f"[sum]   outliers mean={_mean(self._outliers):6.1f} "
                   f"max={max(self._outliers)}")

        out.append("[sum] ICP prior source counts: " +
                   " ".join(f"{k}={v}" for k, v in sorted(self._prior_src.items())))

        out.extend(self._health_lines())

        out.append(f"[sum] events: {len(self._events)}")
        for t, tag, text in self._events[:40]:
            out.append(f"[sum]   t={t:7.3f} {tag} {text}")
        return out

    def _health_lines(self):
        """the numbers that decide whether a faster position source is needed.

        Three questions, in order of how much they matter:
          1. how often does ICP fail to publish at all,
          2. how long do the controllers go without a fresh pose, and
          3. how wrong is the constant-velocity translation the deskew relies
             on, since that is the one input with no sensor behind it.
        """
        out = ["[sum] localisation health"]

        # 1. publish rate
        n_ok = len(self._poses)
        n_drop = len(self._drops)
        n_rev = n_ok + n_drop
        if n_rev == 0:
            out.append("[sum]   no revolutions seen")
            return out
        pct = 100.0 * n_drop / n_rev
        out.append(f"[sum]   revolutions={n_rev} published={n_ok} "
                   f"dropped={n_drop} ({pct:.1f}%)")
        if self._drops:
            out.append(f"[sum]   worst drop streak={max(d[3] for d in self._drops)} "
                       f"consecutive revolutions")
            drms = [d[1] for d in self._drops]
            din = [d[2] for d in self._drops]
            out.append(f"[sum]   dropped-frame rms mean={_mean(drms):6.1f} "
                       f"min={min(drms):6.1f} | inliers mean={_mean(din):6.1f} "
                       f"max={max(din)}")

        # 2. how stale the pose gets
        # Nominal is one revolution. Every drop doubles the gap, and the gap
        # is what a faster sensor would fill.
        gaps = [d[5] * 1000.0 for d in self._deltas] # ms
        if gaps:
            out.append(f"[sum]   publish gap ms: mean={_mean(gaps):6.1f} "
                       f"p50={_pct(gaps, 50):6.1f} p95={_pct(gaps, 95):6.1f} "
                       f"max={max(gaps):6.1f}")

        # 3. speed, and therefore how far the robot travels un-localised
        # Measured from the published poses themselves, so it needs no motor
        # model and holds however the robot was actually moved,
        # driven, or pushed by hand across the field.
        speeds = [math.hypot(d[2], d[3]) / d[5]
                  for d in self._deltas if d[5] > 1e-6]
        if speeds and gaps:
            out.append(f"[sum]   measured speed mm/s: mean={_mean(speeds):7.1f} "
                       f"p95={_pct(speeds, 95):7.1f} max={max(speeds):7.1f}")
            # Lag is computed per revolution and then summarised, not as
            # (worst speed x worst gap), those two need not have happened on
            # the same revolution, and multiplying them invents a number the
            # robot never experienced. This is just the distance travelled
            # between one published pose and the next.
            lags = [math.hypot(d[2], d[3]) for d in self._deltas]
            lag95 = _pct(lags, 95)
            out.append(f"[sum]   position lag from staleness: "
                       f"p95={lag95:6.1f} mm worst={max(lags):6.1f} mm")
            # A 100 Hz sensor shortens the gap but does not close it: the
            # robot still moves for one sample period between readings. The
            # bare gap ratio ignores that, so take whichever is larger.
            resid = _pct(speeds, 95) * XLATE_PERIOD_S
            out.append(f"[sum]   a 100 Hz fix would leave about "
                       f"{max(lag95 * 10.0 / max(_mean(gaps), 1e-9), resid):.1f} mm"
                       f" (one {XLATE_PERIOD_S * 1000:.0f} ms sample period "
                       f"= {resid:.1f} mm at the p95 speed)")

        # 4. the deskew extrapolation, scored against reality
        # _deskew is handed last revolution's displacement as this
        # revolution's, on a constant-velocity assumption. Pair each
        # prediction with the displacement actually measured for that same
        # revolution: the residual is the error a measured translation would
        # delete outright.
        actual = {d[0]: (d[2], d[3], d[5]) for d in self._deltas}
        # A dropped revolution publishes nothing, so the next skew record
        # reuses the same scan number, last write wins, because that is the
        # one that actually preceded the published scan. The delta after a
        # drop also spans two or more revolutions, which would not be a fair
        # test of a one-revolution prediction, so those are skipped on dt.
        pending = {}
        for s in self._skew:
            pending[s[0]] = s
        errs, mags, skipped = [], [], 0
        for n, s in pending.items():
            if n not in actual:
                continue
            ax, ay, adt = actual[n]
            if abs(adt - s[6]) > 0.5 * s[6]:
                skipped += 1
                continue
            errs.append(math.hypot(ax - s[2], ay - s[3]))
            mags.append(math.hypot(ax, ay))
        src_n = {}
        for s in self._skew:
            src_n[s[5]] = src_n.get(s[5], 0) + 1
        out.append(f"[sum]   deskew rotation source: " +
                   (" ".join(f"{k}={v}" for k, v in sorted(src_n.items()))
                    or "none"))
        if errs:
            out.append(f"[sum]   deskew translation error mm: "
                       f"mean={_mean(errs):6.2f} p95={_pct(errs, 95):6.2f} "
                       f"max={max(errs):6.2f}")
            travel = _mean(mags)
            if travel > 1e-6:
                out.append(f"[sum]   ...against {travel:.2f} mm of mean actual "
                           f"travel per revolution "
                           f"({100.0 * _mean(errs) / travel:.0f}% of the motion "
                           "it is trying to undo)")
            out.append(f"[sum]   paired={len(errs)} skipped={skipped} "
                       "(skipped = delta spanned more than one revolution)")
        else:
            out.append("[sum]   deskew translation error: no paired samples "
                       "(robot never moved, or every frame was dropped)")

        # verdict
        out.append("[sum]   verdict:")
        if pct > 5.0:
            out.append(f"[sum]     ICP drops {pct:.1f}% of revolutions, an "
                       "independent fix is worth more than these numbers show,")
            out.append("[sum]     because during a drop there is no position "
                       "source at all.")
        elif (speeds and gaps
                and max(math.hypot(d[2], d[3]) for d in self._deltas) < 30.0):
            out.append("[sum]     ICP publishes reliably and the worst lag is "
                       "under 30 mm. A faster position")
            out.append("[sum]     source would buy little; spend the effort "
                       "elsewhere.")
        else:
            out.append("[sum]     ICP is reliable but the robot outruns it. "
                       "The lag above is the real")
            out.append("[sum]     size of what a 100 Hz fix removes, judge "
                       "it against your control tolerance.")
        return out


# small stats helpers (no numpy, this runs in the lidar thread's process
# on a Pi, and the sample counts are in the hundreds)

def _mean(v):
    """arithmetic mean, 0.0 for an empty sequence."""
    return sum(v) / len(v) if v else 0.0


def _sd(v):
    """population standard deviation, 0.0 for fewer than two samples."""
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _pct(v, p):
    """p-th percentile by nearest rank, 0.0 for an empty sequence."""
    if not v:
        return 0.0
    s = sorted(v)
    k = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def _wrap(d):
    """wrap a degree delta into (-180, 180]."""
    return (d + 180.0) % 360.0 - 180.0


def _unwrap(hs):
    """undo 0/360 wrapping so a heading series can be averaged and differenced."""
    if not hs:
        return []
    out = [hs[0]]
    for h in hs[1:]:
        out.append(out[-1] + _wrap(h - out[-1]))
    return out


def new_session_path(directory="logs"):
    """logs/lidar_YYYYmmdd_HHMMSS.txt, one file per run, never overwritten."""
    return os.path.join(directory,
                        time.strftime("lidar_%Y%m%d_%H%M%S.txt"))
