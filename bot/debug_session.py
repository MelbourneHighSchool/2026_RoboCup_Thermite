"""Debug session: what the robot records for watching a match back, one folder per power-on.

logs/debug_r<id>_<YYYYmmdd_HHMMSS>/ holds:
    session.json: the robot, its clock and the row counts, rewritten every few seconds
    ticks.jsonl: one row per play tick (10 a second while idle), plus one per teammate
        message, which is how two robots' sessions get lined up
    lidar.txt, capture.txt, motion.txt: the --lidarlog, --capturelog and --motionlog logs

Turn it on with --debug, DEBUG_SESSION=1 or the debug page. Off pauses it and on again carries
on in the same folder. Files are appended and synced once a second, so a power cut loses at
most a second. Read the flag as bot.debug_session.recording, not a from-import copy. Watch a
session with simulator.py.
"""

import json
import os
import queue
import shutil
import threading
import time

import bot.state as state
from bot.dwibbler import _possession
from bot.hardware import Motor

SCHEMA_VERSION = 3

# True while ticks are being written. Flip it with set_recording().
recording = False

idle_tick_period_s = 0.1 # ticks recorded while not playing, 10 a second
min_free_mb = 500 # recording pauses itself below this much free space
checkpoint_period_s = 5.0

_RECORD_MOTORS = ("nw", "ne", "sw", "se", "dwibble")

# _state keys snapshotted each tick
_WORLD_KEYS = (
    "pose", "pose_t", "pose_live", "pose_imu", "imu_heading",
    "ball", "ball_est", "remote_ball", "enemies", "enemy_vel",
    "teammate_pos", "teammate_pos_bt",
    "slot_goal", "attack_low", "slot_role", "own_slot_role",
    "run_mode", "my_state", "kickoff_role", "kickoff_until_t",
    "capture_zone", "drib_has_ball",
    "dwibble_cam_seen", "dwibble_cam_frac",
    "loop_hz", "lidar_hz", "cam_fps", "wheel_slip_trust",
)

_session = None # the Session, made on first use and kept until finish()
_lock = threading.Lock()


def _jsonable(v):
    """json.dumps fallback: numpy scalars and arrays become plain numbers and lists, anything
    else its repr, so one odd value in the world can't end the recording.
    """
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (set, frozenset)):
        return list(v)
    return repr(v)


def _dumps(row):
    return json.dumps(row, default=_jsonable) + "\n"


class LineWriter:
    """appends lines to a file from its own thread so a slow SD card can't stall the loop.
    Syncs once a second, drops (and counts) lines if it falls behind, and keeps a disk error
    in .error.
    """

    def __init__(self, path, queue_size=4096):
        self.path = path
        self.written = 0
        self.dropped = 0
        self.error = None
        self._queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        # opened here so a bad path raises to the caller
        self._fh = open(path, "a", encoding="utf-8")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, line):
        if self._stop.is_set():
            return False
        try:
            self._queue.put_nowait(line)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self):
        try:
            with self._fh as fh:
                last_sync = time.monotonic()
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        line = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        line = None
                    if line is not None:
                        fh.write(line)
                        self.written += 1
                    now = time.monotonic()
                    if now - last_sync >= 1.0:
                        fh.flush()
                        os.fsync(fh.fileno())
                        last_sync = now
                fh.flush()
                os.fsync(fh.fileno())
        except Exception as e: # noqa: BLE001 (a dead SD card ends recording)
            self.error = e

    def close(self, timeout=3.0):
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=timeout)


def _session_folder(robot_id, root):
    """logs/debug_r<id>_<time>, with _2, _3 added if that folder already exists."""
    base = os.path.join(root, time.strftime(f"debug_r{robot_id}_%Y%m%d_%H%M%S"))
    path, n = base, 1
    while os.path.exists(path):
        n += 1
        path = f"{base}_{n}"
    return path


class Session:
    """one power-on's folder and its tick log."""

    def __init__(self, robot_id, root="logs"):
        self.robot_id = robot_id
        self.folder = os.path.abspath(_session_folder(robot_id, root))
        os.makedirs(self.folder)
        # the session clock: every "t" in every file is seconds since this moment
        self.mono_t0 = time.monotonic()
        self.meta = {
            "schema": SCHEMA_VERSION,
            "robot_id": robot_id,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            # this Pi's monotonic clock at t = 0, for lining up with the teammate's
            "mono_t0": self.mono_t0,
            "world_keys": list(_WORLD_KEYS),
            "motors": list(_RECORD_MOTORS),
            "ticks_file": "ticks.jsonl",
            "closed": False,
        }
        self._meta_lock = threading.Lock()
        self.checkpoint()
        self.ticks = LineWriter(self.path("ticks.jsonl"))
        self._last_idle_tick = None
        self._closed = threading.Event()
        threading.Thread(target=self._keeper_loop, daemon=True).start()

    def path(self, name):
        return os.path.join(self.folder, name)

    def now(self):
        return time.monotonic() - self.mono_t0

    def checkpoint(self, **extra):
        """rewrite session.json in one step (temp file, then rename), so it is never half
        written.
        """
        with self._meta_lock:
            self.meta.update(extra)
            if hasattr(self, "ticks"):
                self.meta.update({
                    "duration_s": round(self.now(), 2),
                    "ticks_written": self.ticks.written,
                    "ticks_dropped": self.ticks.dropped,
                })
            tmp = self.path("session.json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path("session.json"))

    def tick(self, cmds, world, drib, drib_avail):
        t = self.now()
        if world.get("run_mode") not in ("run", "camrun"):
            if (self._last_idle_tick is not None
                    and t - self._last_idle_tick < idle_tick_period_s):
                return
            self._last_idle_tick = t
        self.ticks.submit(_dumps({"type": "tick", "t": round(t, 4), "cmds": cmds,
                                  "drib": drib, "drib_avail": drib_avail, "world": world}))

    def event(self, kind, **fields):
        self.ticks.submit(_dumps({"type": kind, "t": round(self.now(), 4), **fields}))

    def _keeper_loop(self):
        """every few seconds: save session.json, and pause recording if the card is nearly
        full.
        """
        while not self._closed.wait(checkpoint_period_s):
            try:
                self.checkpoint()
                free_mb = shutil.disk_usage(self.folder).free / 1e6
                if recording and free_mb < min_free_mb:
                    print(f"[debug] only {free_mb:.0f} MB free, recording paused",
                          flush=True)
                    set_recording(False)
            except Exception as e: # noqa: BLE001
                print(f"[debug] checkpoint failed ({e!r})", flush=True)

    def close(self):
        self._closed.set()
        self.ticks.close()
        self.checkpoint(closed=True)
        return self.folder


def _get_session(create=True):
    """the running session, made on first call. Caller holds _lock."""
    global _session
    if _session is None and create:
        robot_id = os.environ.get("ROBOT_ID", "?")
        _session = Session(robot_id)
        print(f"[debug] session folder {_session.folder}", flush=True)
    return _session


def folder():
    """the session folder, made now if this is the first thing to need it (the bench logs
    use it even with recording off).
    """
    with _lock:
        return _get_session().folder


def path(name):
    """a file in the session folder: debug_session.path("lidar.txt")."""
    return os.path.join(folder(), name)


def set_recording(on):
    """switch recording on or off from any thread. Off pauses; on again carries on in the same
    session.
    """
    global recording
    with _lock:
        was = recording
        recording = bool(on)
        existed = _session is not None
        sess = _get_session() if recording else _get_session(create=False)
        if existed and was != recording:
            sess.event("resume" if recording else "pause")
    with state._lock:
        state._state["debug_recording"] = recording
    if was != recording:
        print(f"[debug] recording {'on' if recording else 'paused'}", flush=True)


def record_tick():
    """once per control tick: one flag check while off. Never raises into the control loop;
    a failure pauses recording.
    """
    if not recording:
        return
    try:
        # under the lock, so a pause can't land between the check and making the session
        with _lock:
            if not recording:
                return
            sess = _get_session()
        if sess.ticks.error is not None:
            print(f"[debug] tick log failed ({sess.ticks.error!r}), recording paused",
                  flush=True)
            set_recording(False)
            return
        with state._lock:
            world = {k: state._state.get(k) for k in _WORLD_KEYS}
        # what the controller asked for this tick (_last_sent lags a flush and is empty
        # off the robot)
        cmds = {m: Motor._targets.get(m) for m in _RECORD_MOTORS}
        # _possession is read outside the lock (atomic attribute reads), as logs.py does
        sess.tick(cmds, world, bool(_possession.has_ball), bool(_possession.available))
    except Exception as e: # noqa: BLE001 (never take down play for a log)
        print(f"[debug] recording failed ({e!r}), recording paused", flush=True)
        set_recording(False)


def link_rx(tx_mono, mtype):
    """a teammate message just arrived, sent at tx_mono on the teammate's clock. Logged so
    the viewer can line the two robots' sessions up.
    """
    if not recording or not isinstance(tx_mono, (int, float)):
        return
    sess = _session
    if sess is not None:
        sess.event("link", tx=tx_mono, msg=mtype)


def finish():
    """close the session (shutdown / SIGTERM path)."""
    global _session
    with _lock:
        sess = _session
        _session = None
    if sess is not None:
        print(f"[debug] session saved in {sess.close()}", flush=True)


class SessionReplay:
    """a session's ticks read back, skipping a torn last line. The helpers below give simple
    timelines for tests and analysis.
    """

    def __init__(self, folder_path):
        self.folder = folder_path
        with open(os.path.join(folder_path, "session.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.ticks = []
        self.links = []
        self.events = []
        self.skipped_rows = 0
        ticks_path = os.path.join(folder_path, self.meta.get("ticks_file", "ticks.jsonl"))
        with open(ticks_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    self.skipped_rows += 1 # torn tail from a power loss
                    continue
                kind = row.get("type")
                if kind == "tick":
                    self.ticks.append(row)
                elif kind == "link":
                    self.links.append(row)
                else:
                    self.events.append(row)

    def poses(self):
        """[(t, rx, ry, hdg)] for ticks with a pose fit."""
        return [(tk["t"], *tk["world"]["pose"][:3]) for tk in self.ticks
                if tk["world"].get("pose") is not None]

    def states(self):
        """[(t, fsm_state)]: the behaviour state machine timeline."""
        return [(tk["t"], tk["world"].get("my_state")) for tk in self.ticks]

    def ball_path(self):
        """[(t, bx, by)] from the fused field-frame ball estimate."""
        return [(tk["t"], tk["world"]["ball_est"][0], tk["world"]["ball_est"][1])
                for tk in self.ticks if tk["world"].get("ball_est") is not None]

    def possession_timeline(self):
        """[(t, has_ball)] from the dwibbler-stall detector."""
        return [(tk["t"], tk["drib"]) for tk in self.ticks]

    def bounds_violations(self, field_x, field_y, margin_mm=0.0):
        """ticks where the pose is more than margin_mm outside the field."""
        bad = []
        for t, x, y, _h in self.poses():
            if not (-margin_mm <= x <= field_x + margin_mm
                    and -margin_mm <= y <= field_y + margin_mm):
                bad.append((t, x, y))
        return bad
