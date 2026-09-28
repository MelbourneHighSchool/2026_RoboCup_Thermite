"""Match replay recording: one snapshot per play-loop tick to logs/match_<ts>.jsonl.

While record_match is True, record_tick() (called by the play loop) snapshots pose, ball,
enemies, teammate, FSM state and motor commands. An async writer keeps disk I/O off the
control thread. Read the flag as bot.match_recorder.record_match (never a from-import copy);
flip it with --record, RECORD_MATCH=1, the debug page, or set_record().
"""

import json
import os
import queue
import threading
import time

import bot.state as state
from bot.dwibbler import _possession
from bot.hardware import Motor

# The toggle: True = record every play tick to a JSONL match file. Flippable
# at runtime; the file is opened lazily on the first recorded tick, so
# leaving the flag on at boot costs nothing until play actually starts.
record_match = False

_RECORD_MOTORS = ("nw", "ne", "sw", "se", "dwibble")

# _state keys snapshotted verbatim each tick (the replay-relevant subset of
# the shared world, not frame/mask/opencv payloads (far too big)
# for a per-tick JSON row and are reproducible from the raw hardware log).
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

SCHEMA_VERSION = 1

_recorder = None # active _MatchRecorder while recording, None otherwise
_recorder_lock = threading.Lock()


def set_record(on):
    """flip recording on/off from any thread: on arms the recorder (the file opens on the first
    play tick), off closes the file and prints its path.
    """
    global record_match, _recorder
    record_match = bool(on)
    with state._lock:
        state._state["record_match"] = record_match
    if not on:
        with _recorder_lock:
            rec = _recorder
            _recorder = None
        if rec is not None:
            path = rec.close()
            print(f"[matchrec] wrote {path}", flush=True)
    else:
        print("[matchrec] recording ON, opens logs/match_<ts>.jsonl on "
              "the next play tick", flush=True)


class _MatchRecorder:
    """one JSONL match file: an async writer thread (a slow SD card never stalls the 50 Hz
    loop), a header row, and an fsync once a second so a power loss costs at most a second.
    """

    def __init__(self, path, robot_id, queue_size=4096):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._queue = queue.Queue(maxsize=queue_size)
        self._dropped = 0
        self._written = 0
        self._stop = threading.Event()
        self._t0 = time.monotonic()
        header = {"type": "header", "schema": SCHEMA_VERSION,
                  "robot_id": robot_id,
                  "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "world_keys": list(_WORLD_KEYS),
                  "motors": list(_RECORD_MOTORS)}
        # open synchronously so a bad path raises here (into record_tick's
        # failure handler, which disables recording) instead of silently
        # dying in the writer thread after the constructor returned.
        self._fh = open(self.path, "w", encoding="utf-8")
        self._fh.write(json.dumps(header) + "\n")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            with self._fh:
                fh = self._fh
                last_sync = time.monotonic()
                while True:
                    try:
                        row = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        row = None
                    if row is not None:
                        fh.write(row)
                        self._written += 1
                    now = time.monotonic()
                    if now - last_sync >= 1.0:
                        fh.flush()
                        os.fsync(fh.fileno())
                        last_sync = now
                    if self._stop.is_set() and self._queue.empty():
                        break
                # end row composed here, in the writer, so its tick count
                # is final (no race with close())
                fh.write(json.dumps(
                    {"type": "end", "ticks": self._written,
                     "dropped": self._dropped,
                     "elapsed_s": round(time.monotonic() - self._t0, 2)})
                    + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except Exception as e: # noqa: BLE001 (a dead SD card ends recording)
            print(f"[matchrec] writer failed ({e!r}), recording off",
                  flush=True)
            set_record(False)

    def tick(self, cmds, world, drib, drib_avail):
        """enqueue one tick snapshot; drops (counted) rather than blocks if the writer falls
        behind.
        """
        row = {"type": "tick",
               "t": round(time.monotonic() - self._t0, 4),
               "cmds": cmds, "drib": drib, "drib_avail": drib_avail,
               "world": world}
        try:
            self._queue.put_nowait(json.dumps(row) + "\n")
        except queue.Full:
            self._dropped += 1

    def close(self):
        """finalise: stop the writer (it drains the queue, then writes the end row itself) and
        return the path.
        """
        self._stop.set()
        self._thread.join(timeout=3.0)
        return self.path


def _match_path(robot_id):
    """logs/match_r<id>_<timestamp>.jsonl; a separate function so tests can redirect it."""
    return os.path.join("logs", time.strftime(
        f"match_r{robot_id}_%Y%m%d_%H%M%S.jsonl"))


def record_tick():
    """once per control tick: one flag check while off; while on, open the file on the first
    tick and enqueue a snapshot. Never raises into the control loop: any failure turns
    recording off.
    """
    global _recorder
    if not record_match:
        return
    try:
        if _recorder is None:
            robot_id = os.environ.get("ROBOT_ID", "?")
            path = _match_path(robot_id)
            _recorder = _MatchRecorder(path, robot_id)
            print(f"[matchrec] recording to {path}", flush=True)
        with state._lock:
            world = {k: state._state.get(k) for k in _WORLD_KEYS}
        cmds = {m: Motor._last_sent.get(m) for m in _RECORD_MOTORS}
        # _possession is read outside the lock (atomic attribute reads), as
        # logs.py does
        _recorder.tick(cmds, world,
                       bool(_possession.has_ball),
                       bool(_possession.available))
    except Exception as e: # noqa: BLE001 (never take down play for a log)
        print(f"[matchrec] recording failed ({e!r}), recording off",
              flush=True)
        set_record(False)


def finish():
    """close an active recording (shutdown / SIGTERM path)."""
    global _recorder
    with _recorder_lock:
        rec = _recorder
        _recorder = None
    if rec is not None:
        path = rec.close()
        print(f"[matchrec] wrote {path}", flush=True)


class MatchReplay:
    """read a recorded match back, skipping a torn final line from a power loss. poses(),
    states(), ball_path(), possession_timeline() and bounds_violations() give ready-made
    timelines for tests and offline analysis.
    """

    def __init__(self, path):
        self.path = path
        self.header = None
        self.ticks = []
        self.end = None
        self.skipped_rows = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    self.skipped_rows += 1 # torn tail from a power loss
                    continue
                if row.get("type") == "header":
                    self.header = row
                elif row.get("type") == "tick":
                    self.ticks.append(row)
                elif row.get("type") == "end":
                    self.end = row

    @property
    def duration_s(self):
        return self.end["elapsed_s"] if self.end else None

    def poses(self):
        """[(t, rx, ry, hdg)] for ticks with a pose fit."""
        out = []
        for tk in self.ticks:
            p = tk["world"].get("pose")
            if p is not None:
                out.append((tk["t"], p[0], p[1], p[2]))
        return out

    def states(self):
        """[(t, fsm_state)]: the behaviour state machine timeline."""
        return [(tk["t"], tk["world"].get("my_state")) for tk in self.ticks]

    def ball_path(self):
        """[(t, bx, by)] from the fused field-frame ball estimate."""
        out = []
        for tk in self.ticks:
            b = tk["world"].get("ball_est")
            if b is not None:
                out.append((tk["t"], b[0], b[1]))
        return out

    def possession_timeline(self):
        """[(t, has_ball)] from the dwibbler-stall detector."""
        return [(tk["t"], tk["drib"]) for tk in self.ticks]

    def bounds_violations(self, field_x, field_y, margin_mm=0.0):
        """ticks where the pose fit sits outside the field by margin_mm: the replay-level
        sanity check.
        """
        bad = []
        for tk in self.ticks:
            p = tk["world"].get("pose")
            if p is None:
                continue
            if not (-margin_mm <= p[0] <= field_x + margin_mm
                    and -margin_mm <= p[1] <= field_y + margin_mm):
                bad.append((tk["t"], p[0], p[1]))
        return bad
