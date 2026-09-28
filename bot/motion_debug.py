#!/usr/bin/env python3
# motion_debug.py: combined motion/control debug logger for --motionlog,
# one row per sample covering motor commands, measured telemetry, and
# FSM/perception context.

import os
import threading
import time

_MOTORS = ("nw", "ne", "sw", "se", "dwibble")


class MotionLogger:
    """text logger for one --motionlog session: motor commands and measured
    pose/IMU/QDR/collision telemetry, one row per sample.
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
        # (t, cmds, rx, ry, hdg, imu, qdr, collision, fsm)
        self._rows = []

    def _write(self, line):
        if self._closed:
            return
        data = line + "\n"
        self._fh.write(data)
        self._bytes += len(data)
        if self._bytes >= self.max_bytes:
            self._done = True

    def _t(self):
        return self._clock() - self._t0

    def is_done(self):
        if self._done:
            return True
        return (self.duration_s is not None
                and self._t() >= self.duration_s)

    def header(self, consts):
        with self._lock:
            self._write("# combined motion/control debug log")
            self._write("# started " + time.strftime("%Y-%m-%dT%H:%M:%S"))
            self._write(f"# duration_s={self.duration_s}")
            self._write("#")
            self._write("# Record: [tick] t=<s> " +
                        " ".join(f"{m}=<frac|n/a>" for m in _MOTORS) +
                        " rx=<mm|n/a> ry=<mm|n/a> hdg=<deg|n/a> "
                        "imu=<deg|n/a> " +
                        " ".join(f"{m}_qdr=<frac|n/a>" for m in _MOTORS) +
                        " collision=<y|n> "
                        "state=<seek|has_ball|pass|flick_shot|goalie|n/a> "
                        "run=<idle|run|calib|n/a> role=<striker|goalie|n/a> "
                        "zone=<cone|search|n/a> ball=<y|n> "
                        "best=<cam|remote|mem|pass|n/a> bconf=<0..1|n/a> "
                        "drib=<y|n> dribok=<y|n>")
            self._write("#   <frac> (per motor, no suffix) is the -1..1 "
                        "command fraction last actually written to that "
                        "motor's driver (Motor._last_sent), what the bus "
                        "has right now, not just what was most recently staged.")
            self._write("#   rx/ry/hdg: fused localised pose "
                        "(_state[\"pose\"]), field mm / compass deg, "
                        "n/a until the lidar has a fix")
            self._write("#   imu: _state[\"imu_heading\"], BNO08x "
                        "relative yaw (deg cw+), n/a with no IMU")
            self._write("#   *_qdr: measured speed for that motor, "
                        "read_qdr()'s raw speed field / MOTOR_MAX_RAW, "
                        "same -1..1 scale as a command, but reported back "
                        "by the driver, not what was sent")
            self._write("#   collision: a COLLISION_ACCEL_G spike "
                        "(bot.odometry's WheelOdometry) landed before this "
                        "tick; that revolution's odometry delta was "
                        "dropped for last_delta.")
            self._write("#   state: _state[\"my_state\"], which "
                        "controller state produced this row's command. "
                        "n/a if the play loop hasn't set a role yet.")
            self._write("#   run/role: _state[\"run_mode\"]/[\"slot_role\"], "
                        "confirms this really is idle/run/calib and "
                        "striker/goalie, not an assumption from context.")
            self._write("#   zone: _state[\"capture_zone\"], \"cone\" "
                        "when the ball is centred/close, "
                        "\"search\" while blind-spinning with no usable "
                        "ball estimate (seek's last resort), n/a on any "
                        "tick not actively chasing.")
            self._write("#   ball: _state[\"ball\"] is not None, "
                        "our own camera has a sighting this tick, "
                        "independent of best/bconf below.")
            self._write("#   best/bconf: _state[\"ball_est\"]'s source and "
                        "confidence: cam (own sighting), remote (teammate's), "
                        "mem (BallMemory inference), pass (a teammate's "
                        "broadcast pass_target, lowest priority), n/a if "
                        "nothing is known right now.")
            self._write("#   drib/dribok: _possession.has_ball / .available "
                        "the dwibbler-stall possession flag "
                        "and its health gate (dribok=n means the roller "
                        "never proved it reaches its command, so drib "
                        "isn't trusted anywhere, see BallPossession's docstring).")
            self._write("#   'n/a' for a motor (command or QDR) means it "
                        "was never registered this run (e.g. --hsv, or a "
                        "motor that failed to init).")
            self._write("#")
            for k in sorted(consts):
                self._write(f"[const] {k}={consts[k]}")
            self._write("")

    def tick(self, cmds, pose, imu_heading, qdr, fsm, collision=False):
        """record one sample; cmds/qdr are {motor_name: frac_or_None}, pose is (rx, ry, hdg) or
        None, fsm is the dict described in header().
        """
        with self._lock:
            if self._closed:
                return
            t = self._t()
            self._n_tick += 1
            cmd_s = " ".join(
                f"{m}={cmds.get(m):.3f}" if cmds.get(m) is not None else f"{m}=n/a"
                for m in _MOTORS)
            if pose is not None:
                rx, ry, hdg = pose
                pose_s = f"rx={rx:.1f} ry={ry:.1f} hdg={hdg:.2f}"
            else:
                rx = ry = hdg = None
                pose_s = "rx=n/a ry=n/a hdg=n/a"
            imu_s = f"imu={imu_heading:.2f}" if imu_heading is not None else "imu=n/a"
            qdr_s = " ".join(
                f"{m}_qdr={qdr.get(m):.3f}" if qdr.get(m) is not None
                else f"{m}_qdr=n/a"
                for m in _MOTORS)
            collision_s = f"collision={'y' if collision else 'n'}"
            state = fsm.get("state") or "n/a"
            run_mode = fsm.get("run_mode") or "n/a"
            role = fsm.get("role") or "n/a"
            zone = fsm.get("zone") or "n/a"
            ball_s = "y" if fsm.get("ball_seen") else "n"
            best = fsm.get("ball_est_src") or "n/a"
            conf = fsm.get("ball_est_conf")
            bconf_s = f"{conf:.2f}" if conf is not None else "n/a"
            drib_s = "y" if fsm.get("drib") else "n"
            dribok_s = "y" if fsm.get("drib_avail") else "n"
            fsm_s = (f"state={state} run={run_mode} role={role} zone={zone} "
                    f"ball={ball_s} best={best} bconf={bconf_s} "
                    f"drib={drib_s} dribok={dribok_s}")
            self._write(f"[tick] t={t:.3f} {cmd_s} {pose_s} {imu_s} "
                        f"{qdr_s} {collision_s} {fsm_s}")
            self._rows.append((t, dict(cmds), rx, ry, hdg, imu_heading,
                               dict(qdr), collision, dict(fsm)))

    def close(self):
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
            out.append("[sum] no ticks logged")
            return out

        for m in _MOTORS:
            vals = [r[1].get(m) for r in self._rows if r[1].get(m) is not None]
            if not vals:
                out.append(f"[sum] {m}: never commanded (n/a every tick)")
            else:
                mags = [abs(v) for v in vals]
                reversed_n = sum(1 for v in vals if v < 0.0)
                out.append(f"[sum] {m}: |cmd| mean={_mean(mags):.3f} "
                           f"max={max(mags):.3f} reversed on {reversed_n}/"
                           f"{len(vals)} ticks")

        with_pose = [r for r in self._rows if r[2] is not None]
        out.append(f"[sum] pose available {len(with_pose)}/{n} ticks")
        with_imu = [r[5] for r in self._rows if r[5] is not None]
        out.append(f"[sum] imu available {len(with_imu)}/{n} ticks")
        for m in _MOTORS:
            vals = [r[6].get(m) for r in self._rows if r[6].get(m) is not None]
            if not vals:
                out.append(f"[sum] {m}: no QDR reads (n/a every tick)")
            else:
                mags = [abs(v) for v in vals]
                out.append(f"[sum] {m}: |measured| mean={_mean(mags):.3f} "
                           f"max={max(mags):.3f}")

        collision_n = sum(1 for r in self._rows if r[7])
        out.append(f"[sum] collision flagged {collision_n}/{n} ticks, "
                   "each is a revolution where WheelOdometry's delta was "
                   "dropped for last_delta (see COLLISION_ACCEL_G)")

        # FSM/perception breakdown: a dominant state/zone paired with near-zero
        # |cmd| above is the stuck branch.
        def _counts(key):
            c = {}
            for r in self._rows:
                v = r[8].get(key)
                c[v] = c.get(v, 0) + 1
            return c
        for key, label in (("state", "state"), ("run_mode", "run"),
                           ("role", "role"), ("zone", "zone")):
            c = _counts(key)
            parts = ", ".join(f"{k or 'n/a'}={v}" for k, v in
                              sorted(c.items(), key=lambda kv: -kv[1]))
            out.append(f"[sum] {label} breakdown: {parts}")
        ball_seen_n = sum(1 for r in self._rows if r[8].get("ball_seen"))
        out.append(f"[sum] ball directly seen (own camera) {ball_seen_n}/{n} ticks")
        best_c = _counts("ball_est_src")
        parts = ", ".join(f"{k or 'n/a'}={v}" for k, v in
                          sorted(best_c.items(), key=lambda kv: -kv[1]))
        out.append(f"[sum] ball_est source breakdown: {parts}")
        drib_n = sum(1 for r in self._rows if r[8].get("drib"))
        dribok_n = sum(1 for r in self._rows if r[8].get("drib_avail"))
        out.append(f"[sum] dwibbler-stall possession: drib=True {drib_n}/{n}, "
                   f"available {dribok_n}/{n} (available=False the whole "
                   "session means the roller never proved it reaches its "
                   "own command, check dwibble_calib, sec. 3.16)")
        return out


def _mean(v):
    return sum(v) / len(v) if v else 0.0
