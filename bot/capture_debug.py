#!/usr/bin/env python3
# capture_debug.py: ball-capture logger for --capturelog.
#
#     python3 -m bot.main --capturelog # log 60 s, then stop
#     python3 -m bot.main --capturelog 120 # log 120 s
#
# One plain-text file per run, capture.txt in the debug session folder. One tick record per
# pass through StrikerController's ball-visible seek branch: bot pose, ball position, both
# velocities, and the final drive command, enough to plot the whole capture afterwards and
# see whether the ball-velocity lead is doing anything and how fast the ball really moved
# when the stationary gate should or shouldn't have zeroed it.
#
# Nothing here runs unless --capturelog is passed: bot.state._capture_log stays None and
# every hook is behind an "is not None" check, as with lidar_debug.py.

import math
import os
import threading
import time


class CaptureLogger:
    """text logger for one ball-capture debug session.

    Fed from the play loop's thread only, but keeps a lock anyway since close() can
    race the SIGTERM handler on another thread, as LidarLogger does.
    """

    def __init__(self, path, duration_s=60.0, max_bytes=32 * 1024 * 1024,
                 clock=time.monotonic):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.duration_s = duration_s
        self.max_bytes = max_bytes
        self._clock = clock
        self._lock = threading.Lock()
        self._fh = open(path, "w", buffering=1 << 16)
        self._t0 = clock()
        self._bytes = 0
        self._closed = False
        self._done = False

        self._n_tick = 0
        self._rows = [] # see tick() for the field order
        self._prev_bot = None # (t, rx, ry), for bot-velocity derivation

    # writing

    def _write(self, line):
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
            self._write("# ball-capture debug log")
            self._write("# started " + time.strftime("%Y-%m-%dT%H:%M:%S"))
            self._write(f"# duration_s={self.duration_s}")
            self._write("#")
            self._write("# Record types:")
            self._write("#   tick: state zone rx ry hdg bx by botvx botvy "
                        "ballvx ballvy drive_deg drive_cmd")
            self._write("#     rx/ry/hdg: bot pose (field mm / compass deg)")
            self._write("#     bx/by: ball position (field mm)")
            self._write("#     botvx/botvy: bot velocity (field mm/s, "
                        "differenced from consecutive ticks; no separate "
                        "odometry source at this layer)")
            self._write("#     ballvx/ballvy: ball velocity, "
                        "BallVelocityEstimator.velocity() (field mm/s), "
                        "after the stationary gate")
            self._write("#     drive_deg/cmd: final commanded bearing "
                        "(robot frame) and speed fraction, after the "
                        "velocity lead and the guards")
            self._write("#   event: tag text, for state transitions (captured, lost, ...)")
            self._write("#")
            for k in sorted(consts):
                self._write(f"[const] {k}={consts[k]}")
            self._write("")

    def tick(self, state, zone, rx, ry, hdg, bx, by, ball_vel,
             drive_deg, drive_cmd):
        """one control-loop pass through the capture logic; ball_vel is (vbx, vby) mm/s."""
        with self._lock:
            if self._closed:
                return
            t = self._t()
            self._n_tick += 1

            if self._prev_bot is not None:
                pt, px, py = self._prev_bot
                dt = t - pt
                vx, vy = ((rx - px) / dt, (ry - py) / dt) if dt > 1e-3 else (0.0, 0.0)
            else:
                vx, vy = 0.0, 0.0
            self._prev_bot = (t, rx, ry)

            vbx, vby = ball_vel

            self._write(f"[tick] t={t:.3f} state={state} zone={zone} "
                        f"rx={rx:.1f} ry={ry:.1f} hdg={hdg:.2f} "
                        f"bx={bx:.1f} by={by:.1f} "
                        f"botvx={vx:.1f} botvy={vy:.1f} "
                        f"ballvx={vbx:.1f} ballvy={vby:.1f} "
                        f"drive_deg={drive_deg:.1f} drive_cmd={drive_cmd:.3f}")
            self._rows.append((t, state, zone, rx, ry, hdg, bx, by,
                               vx, vy, vbx, vby, drive_deg, drive_cmd))

    def event(self, tag, text):
        """state transitions: captured, lost, yielded, etc."""
        with self._lock:
            if self._closed:
                return
            self._write(f"[event] t={self._t():.3f} {tag} {text}")

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
        out = []
        n = len(self._rows)
        out.append(f"[sum] ticks={n} elapsed_s={self._t():.1f}")
        if n == 0:
            out.append("[sum] no ticks logged (ball never seen with a pose)")
            return out

        zones = {}
        for r in self._rows:
            zones[r[2]] = zones.get(r[2], 0) + 1
        out.append("[sum] zone occupancy: "
                   + " ".join(f"{k}={v}" for k, v in sorted(zones.items())))

        dists = [math.hypot(r[6] - r[3], r[7] - r[4]) for r in self._rows]
        out.append(f"[sum] distance to ball mm: mean={_mean(dists):7.1f} "
                   f"p05={_pct(dists, 5):7.1f} min={min(dists):7.1f}")

        bspeed = [math.hypot(r[10], r[11]) for r in self._rows]
        out.append(f"[sum] ball speed mm/s: mean={_mean(bspeed):7.1f} "
                   f"p95={_pct(bspeed, 95):7.1f} max={max(bspeed):7.1f}")
        moving = sum(1 for s in bspeed if s > 1.0)
        out.append(f"[sum]   nonzero (post stationary-gate) on "
                   f"{moving}/{n} ticks ({100.0 * moving / n:.0f}%)")

        botspeed = [math.hypot(r[8], r[9]) for r in self._rows]
        out.append(f"[sum] bot speed mm/s: mean={_mean(botspeed):7.1f} "
                   f"p95={_pct(botspeed, 95):7.1f} max={max(botspeed):7.1f}")

        return out


def _mean(v):
    """arithmetic mean, 0.0 for an empty sequence."""
    return sum(v) / len(v) if v else 0.0


def _pct(v, p):
    """p-th percentile by nearest rank, 0.0 for an empty sequence."""
    if not v:
        return 0.0
    s = sorted(v)
    k = min(len(s) - 1, max(0, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]
