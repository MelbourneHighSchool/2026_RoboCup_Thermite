"""Debug session viewer: plays a robot's recorded session back on the field.

    python simulator.py logs/debug_r1_20260928_140312
    python simulator.py logs/debug_r1_20260928_140312 logs/debug_r2_20260928_140305

Record with `python3 -m bot.main --debug` (or DEBUG_SESSION=1, or the debug page's switch). Each
robot keeps one session folder, logs/debug_r<id>_<time>/, for as long as it is powered on. Give
one folder to watch one robot, or one per robot to watch both on the same field. Two robots are
lined up by the teammate messages each one logged (the sender's clock against the receiver's);
if neither logged any, by the moment each started playing.

Our goal is always on the left, so a stretch where we defended the other end is turned round.
Each robot shows its pose and heading, a short trail, its camera line to the ball, the ball
estimate (coloured by where it came from), and the enemies and teammate it could see. The panel
lists that tick's inputs and outputs; the state shown as "in" is the previous tick's, since that
is the state the controller started the tick in.

Keys: space pauses and plays, a/d or the arrow keys step one tick, j/l jump 5 s, [ and ] halve or
double the speed, r goes back to the start of play, t toggles the trails, q or Esc quits. Click
or drag on the bar at the bottom to seek.
"""

# The idea of watching each match back like this came from Tech Support's session replays;
# this is our own implementation.

import bisect
import json
import math
import os
import statistics
import sys
import time

import cv2
import numpy as np

from bot.field import FieldModel

SCHEMA = 3 # the bot/debug_session.py schema this viewer reads

FIELD_X = FieldModel.field_x # short side
FIELD_Y = FieldModel.field_y # long side, drawn across the window

SCALE = 0.38 # pixels per mm
MARGIN = 30
FIELD_W = int(FIELD_Y * SCALE) + 2 * MARGIN
FIELD_H = int(FIELD_X * SCALE) + 2 * MARGIN
PANEL_W = 380
BAR_H = 46

ROBOT_R_MM = 90.0
ENEMY_R_MM = 105.0
BALL_R_MM = 21.5
TRAIL_S = 4.0
LINK_BIN_S = 30.0 # teammate messages are grouped this long when lining clocks up

FONT = cv2.FONT_HERSHEY_SIMPLEX
SMALL = 0.42
MED = 0.55

C_GRASS = (34, 100, 34)
C_LINE = (235, 235, 235)
C_OWN_GOAL = (80, 220, 80)
C_THEIR_GOAL = (70, 70, 230)
C_BALL = (30, 165, 255)
C_ENEMY = (60, 60, 230)
C_TEXT = (230, 230, 230)
C_DIM = (140, 140, 140)
C_PANEL = (28, 28, 28)
C_BAR = (60, 60, 60)
C_BAR_FILL = (90, 170, 90)
C_BAR_MARK = (0, 200, 255)
ROBOT_COLOURS = [(255, 255, 255), (230, 200, 40), (200, 120, 255), (40, 160, 255)]
BALL_SRC_COLOURS = {
    "cam": C_BALL,
    "remote": (0, 220, 220),
    "kal": (255, 180, 0),
    "mem": (160, 160, 160),
    "pass": (255, 0, 255),
}


def session_folder(path):
    """the session folder for a path to it, or to any file inside it."""
    path = os.path.abspath(path)
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.isfile(os.path.join(path, "session.json")):
        raise ValueError(f"{path}: not a debug session folder (no session.json)")
    return path


def _read_jsonl(path):
    """(rows, unreadable count) from a JSON-lines file; a torn last line is just skipped."""
    rows, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
            else:
                bad += 1
    return rows, bad


class Recording:
    """one robot's session: its ticks in time order and the teammate messages it logged."""

    def __init__(self, path):
        self.folder = session_folder(path)
        self.path = self.folder
        with open(os.path.join(self.folder, "session.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.schema = self.meta.get("schema")
        self.robot_id = str(self.meta.get("robot_id", "?"))
        self.mono_t0 = self.meta.get("mono_t0")
        rows, self.skipped = _read_jsonl(
            os.path.join(self.folder, self.meta.get("ticks_file", "ticks.jsonl")))
        self.ticks = []
        self.links = [] # (t, the sender's clock when it sent)
        for row in rows:
            kind = row.get("type")
            if not isinstance(row.get("t"), (int, float)):
                self.skipped += 1
            elif kind == "tick":
                if isinstance(row.get("world"), dict):
                    self.ticks.append(row)
                else:
                    self.skipped += 1
            elif kind == "link" and isinstance(row.get("tx"), (int, float)):
                self.links.append((row["t"], row["tx"]))
        if not self.ticks:
            raise ValueError(f"{self.folder}: no ticks recorded")
        self.ticks.sort(key=lambda tk: tk["t"])
        self.times = [tk["t"] for tk in self.ticks]
        self.offset = 0.0 # where this recording's t = 0 sits on the shared clock
        self.sync = "being the first file"
        # every state change and every change of end, worked out once so drawing a frame
        # never walks the whole session
        self.changes = []
        self.ends = [] # (t, True if we defended the y = FIELD_Y end)
        last_state, last_end = object(), None
        for tk in self.ticks:
            st = tk["world"].get("my_state")
            if st != last_state:
                self.changes.append((tk["t"], st))
                last_state = st
            goal = tk["world"].get("slot_goal")
            if goal in ("low", "high") and (goal == "high") != last_end:
                last_end = goal == "high"
                self.ends.append((tk["t"], last_end))
        self.change_times = [c[0] for c in self.changes]
        self.end_times = [e[0] for e in self.ends]

    def play_start(self):
        """t of the first tick in play, or of the first tick if play never started."""
        for tk in self.ticks:
            if tk["world"].get("run_mode") in ("run", "camrun"):
                return tk["t"]
        return self.ticks[0]["t"]

    def index_at(self, t_local):
        """index of the last tick at or before t_local, or None before the first tick."""
        i = bisect.bisect_right(self.times, t_local) - 1
        return i if i >= 0 else None

    def end_at(self, t_local):
        """True if we were defending the y = FIELD_Y end at t_local (the first known end
        before any is known), None if the session never says.
        """
        if not self.ends:
            return None
        i = bisect.bisect_right(self.end_times, t_local) - 1
        return self.ends[max(i, 0)][1]

    @property
    def first(self):
        return self.offset + self.times[0]

    @property
    def last(self):
        return self.offset + self.times[-1]


def link_offset(a, b):
    """seconds to add to b's session time to put it on a's, from the teammate messages each
    logged, or None if there are none or a session lacks its clock.

    A message b sent at b's clock tx and a logged at a's clock rx gives rx - tx = d + latency,
    where d is a's clock minus b's. One the other way gives -d + latency. Taking the smallest
    of each (the least delayed message) in each 30 s stretch and halving the difference cancels
    the latency; with messages only one way, the smallest delay is taken as zero.
    """
    if a.mono_t0 is None or b.mono_t0 is None:
        return None
    ab = [(a.mono_t0 + t, a.mono_t0 + t - tx) for t, tx in a.links] # d + latency
    ba = [(b.mono_t0 + t, b.mono_t0 + t - tx) for t, tx in b.links] # -d + latency
    if not ab and not ba:
        return None
    if ab and ba:
        d0 = (min(v for _, v in ab) - min(v for _, v in ba)) / 2
        lat = (min(v for _, v in ab) + min(v for _, v in ba)) / 2
    elif ab:
        d0, lat = min(v for _, v in ab), 0.0
    else:
        d0, lat = -min(v for _, v in ba), 0.0
    bins = {}
    for when, v in ab:
        k = int(when // LINK_BIN_S)
        bins.setdefault(k, [None, None])
        bins[k][0] = v if bins[k][0] is None else min(bins[k][0], v)
    for when, v in ba:
        k = int((when + d0) // LINK_BIN_S) # on a's clock, near enough to bin by
        bins.setdefault(k, [None, None])
        bins[k][1] = v if bins[k][1] is None else min(bins[k][1], v)
    ds = []
    for m_ab, m_ba in bins.values():
        if m_ab is not None and m_ba is not None:
            ds.append((m_ab - m_ba) / 2)
        elif m_ab is not None:
            ds.append(m_ab - lat)
        else:
            ds.append(lat - m_ba)
    d = statistics.median(ds)
    # b's session time t is b's clock b.mono_t0 + t, which is a's clock b.mono_t0 + t + d
    return b.mono_t0 + d - a.mono_t0


class Timeline:
    """every recording on one clock, zeroed where the earliest one starts."""

    def __init__(self, recordings):
        self.recs = recordings
        ref = self.recs[0]
        ref.offset = 0.0
        for r in self.recs[1:]:
            shift = link_offset(ref, r)
            if shift is None:
                r.offset = ref.play_start() - r.play_start()
                r.sync = "play start"
            else:
                r.offset = shift
                r.sync = "bluetooth"
        base = min(r.first for r in self.recs)
        for r in self.recs:
            r.offset -= base
        self.start = 0.0
        self.end = max(r.last for r in self.recs)
        # playback opens where play first started, not at power-on
        self.play_start = min(r.offset + r.play_start() for r in self.recs)
        self.steps = sorted({r.offset + t for r in self.recs for t in r.times})
        self._fields = {}
        self._bars = {}

    def flip_at(self, t):
        """True if the field is drawn turned round at t (we defended the y = FIELD_Y end)."""
        for r in self.recs:
            end = r.end_at(t - r.offset)
            if end is not None:
                return end
        return False

    def frame(self, t):
        """[(recording, tick, previous tick)] for every robot with a tick at time t."""
        out = []
        for r in self.recs:
            i = r.index_at(t - r.offset)
            if i is None or t > r.last + 0.5:
                continue
            out.append((r, r.ticks[i], r.ticks[i - 1] if i > 0 else None))
        return out

    def step(self, t, direction):
        """the next (+1) or previous (-1) tick time on the shared clock."""
        if direction > 0:
            i = bisect.bisect_right(self.steps, t + 1e-9)
            return self.steps[min(i, len(self.steps) - 1)]
        i = bisect.bisect_left(self.steps, t - 1e-9) - 1
        return self.steps[max(i, 0)]


# Field to screen: the long axis runs left to right and x runs down the screen, which keeps
# left and right the way the robot sees them. A heading h (0 = +y, clockwise) then points along
# (cos h, sin h) on screen. Turning the field round for the high end keeps our goal left.

def to_px(x, y, flip):
    if flip:
        x, y = FIELD_X - x, FIELD_Y - y
    return int(MARGIN + y * SCALE), int(MARGIN + x * SCALE)


def heading_vec(hdg_deg, flip):
    h = math.radians(hdg_deg + (180.0 if flip else 0.0))
    return math.cos(h), math.sin(h)


def _r(mm):
    return max(1, int(mm * SCALE))


def _put(img, text, xy, colour=C_TEXT, scale=SMALL, thick=1):
    cv2.putText(img, text, xy, FONT, scale, colour, thick, cv2.LINE_AA)


def _fmt(v, spec=".0f"):
    if v is None:
        return "-"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def draw_field_base(flip):
    """grass, walls and goal boxes from FieldModel, our goal box green, theirs red."""
    img = np.full((FIELD_H, FIELD_W, 3), C_GRASS, dtype=np.uint8)
    n_side = 2
    per_end = (len(FieldModel.segments) - n_side) // 2
    for i, (a, b) in enumerate(FieldModel.segments):
        colour = C_LINE
        if i >= n_side:
            low_end = i < n_side + per_end
            tag = FieldModel.segment_tags[i]
            if tag != "boundary":
                ours = low_end != flip
                colour = C_OWN_GOAL if ours else C_THEIR_GOAL
        cv2.line(img, to_px(*a, flip), to_px(*b, flip), colour, 2, cv2.LINE_AA)
    cx, cy = to_px(FIELD_X / 2, FIELD_Y / 2, flip)
    cv2.circle(img, (cx, cy), _r(300), C_LINE, 1, cv2.LINE_AA)
    cv2.line(img, to_px(0, FIELD_Y / 2, flip), to_px(FIELD_X, FIELD_Y / 2, flip), C_LINE, 1)
    return img


def draw_robot(img, x, y, hdg, flip, colour, label, has_ball):
    c = to_px(x, y, flip)
    r = _r(ROBOT_R_MM)
    cv2.circle(img, c, r, colour, 2, cv2.LINE_AA)
    hx, hy = heading_vec(hdg, flip)
    tip = (int(c[0] + hx * r), int(c[1] + hy * r))
    cv2.line(img, c, tip, colour, 2, cv2.LINE_AA)
    if has_ball:
        cv2.circle(img, (int(c[0] + hx * (r + 6)), int(c[1] + hy * (r + 6))), _r(BALL_R_MM),
                   C_BALL, -1, cv2.LINE_AA)
    _put(img, label, (c[0] + r + 4, c[1] - r))


def draw_field(timeline, t, show_trails=True):
    flip = timeline.flip_at(t)
    if flip not in timeline._fields:
        timeline._fields[flip] = draw_field_base(flip)
    img = timeline._fields[flip].copy()
    ball_drawn = False
    for n, (rec, tk, _prev) in enumerate(timeline.frame(t)):
        colour = ROBOT_COLOURS[n % len(ROBOT_COLOURS)]
        w = tk["world"]
        pose = w.get("pose")

        if show_trails:
            lo = rec.index_at(t - rec.offset - TRAIL_S) or 0
            hi = rec.index_at(t - rec.offset) or 0
            pts = [to_px(p[0], p[1], flip) for p in
                   (rec.ticks[i]["world"].get("pose") for i in range(lo, hi + 1)) if p]
            if len(pts) > 1:
                cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, colour, 1,
                              cv2.LINE_AA)

        for e in w.get("enemies") or []:
            if e.get("x") is None or e.get("y") is None:
                continue
            c = to_px(e["x"], e["y"], flip)
            cv2.circle(img, c, _r(ENEMY_R_MM), C_ENEMY, 1 if e.get("occluded") else 2,
                       cv2.LINE_AA)
            if e.get("id") is not None:
                _put(img, str(e["id"]), (c[0] + _r(ENEMY_R_MM) + 2, c[1] + 4), C_ENEMY)

        mate = w.get("teammate_pos_bt") or w.get("teammate_pos")
        if mate:
            cv2.drawMarker(img, to_px(mate[0], mate[1], flip), colour, cv2.MARKER_DIAMOND,
                           14, 1, cv2.LINE_AA)

        if pose:
            cam = w.get("ball")
            if cam:
                b = math.radians(cam[0] + pose[2])
                bx, by = pose[0] + cam[1] * math.sin(b), pose[1] + cam[1] * math.cos(b)
                cv2.line(img, to_px(pose[0], pose[1], flip), to_px(bx, by, flip), C_BALL, 1,
                         cv2.LINE_AA)
            label = f"R{rec.robot_id} {w.get('slot_role') or ''}".strip()
            draw_robot(img, pose[0], pose[1], pose[2], flip, colour, label, bool(tk.get("drib")))

        est = w.get("ball_est")
        if est:
            src = est[3] if len(est) > 3 else "cam"
            bc = BALL_SRC_COLOURS.get(src, C_BALL)
            c = to_px(est[0], est[1], flip)
            if src == "cam" and not ball_drawn:
                cv2.circle(img, c, _r(BALL_R_MM), C_BALL, -1, cv2.LINE_AA)
                ball_drawn = True
            cv2.circle(img, c, _r(BALL_R_MM) + 5, bc, 1, cv2.LINE_AA)
            _put(img, src, (c[0] + 10, c[1] - 8 + 14 * n), bc)
    return img


def _state_changes(rec, t_local, n=5):
    """the last n (t, state) changes up to t_local."""
    hi = bisect.bisect_right(rec.change_times, t_local)
    return rec.changes[max(0, hi - n):hi]


def _motor_bar(img, x, y, name, v):
    _put(img, name, (x, y + 9), C_DIM)
    x0, w = x + 60, 150
    cv2.rectangle(img, (x0, y), (x0 + w, y + 10), C_BAR, -1)
    mid = x0 + w // 2
    cv2.line(img, (mid, y - 1), (mid, y + 11), C_DIM, 1)
    if isinstance(v, (int, float)):
        end = mid + int(max(-1.0, min(1.0, v)) * w / 2)
        cv2.rectangle(img, (min(mid, end), y + 1), (max(mid, end), y + 9), C_BAR_FILL, -1)
        _put(img, f"{v:+.2f}", (x0 + w + 8, y + 9))
    else:
        _put(img, "n/a", (x0 + w + 8, y + 9), C_DIM)


def draw_panel(timeline, t, height):
    """the per-robot readout, at least height tall; it grows if the robots need more room."""
    img = np.full((max(height, 1400), PANEL_W, 3), C_PANEL, dtype=np.uint8)
    y = 20
    frame = timeline.frame(t)
    if not frame:
        _put(img, "no robot recorded at this time", (12, y), C_DIM)
    for n, (rec, tk, prev) in enumerate(frame):
        colour = ROBOT_COLOURS[n % len(ROBOT_COLOURS)]
        w = tk["world"]
        pose = w.get("pose")
        _put(img, f"robot {rec.robot_id}, {w.get('slot_role') or '-'} "
                  f"({w.get('run_mode') or '-'})", (12, y), colour, MED)
        y += 20
        state_in = prev["world"].get("my_state") if prev else None
        cam = w.get("ball")
        est = w.get("ball_est")
        rows = [
            ("state", f"{state_in or '-'} -> {w.get('my_state') or '-'}"),
            ("pose", f"{pose[0]:.0f}, {pose[1]:.0f} mm, {pose[2]:.1f} deg" if pose else "-"),
            ("imu", f"{_fmt(w.get('imu_heading'), '.1f')} deg"),
            ("camera ball", f"{cam[0]:+.1f} deg, {cam[1]:.0f} mm" if cam else "-"),
            ("ball estimate", f"{est[0]:.0f}, {est[1]:.0f} ({est[3] if len(est) > 3 else '?'}, "
                              f"{_fmt(est[2], '.2f')})" if est else "-"),
            ("possession", ("yes" if tk.get("drib") else "no")
             + ("" if tk.get("drib_avail") else " (roller unproven)")),
            ("mouth camera", f"{'seen' if w.get('dwibble_cam_seen') else 'no'}, "
                             f"{_fmt(w.get('dwibble_cam_frac'), '.2f')}"),
            ("capture zone", w.get("capture_zone") or "-"),
            ("enemies seen", str(len(w.get("enemies") or []))),
            ("rates", f"loop {_fmt(w.get('loop_hz'))} Hz, lidar "
                      f"{_fmt(w.get('lidar_hz'), '.1f')} Hz, cam {_fmt(w.get('cam_fps'))}"),
        ]
        for label, value in rows:
            _put(img, label, (12, y), C_DIM)
            _put(img, value, (118, y))
            y += 17
        y += 4
        for name in ("nw", "ne", "sw", "se", "dwibble"):
            _motor_bar(img, 12, y, name, (tk.get("cmds") or {}).get(name))
            y += 16
        y += 14
        _put(img, "recent states", (12, y), C_DIM)
        y += 16
        for when, st in _state_changes(rec, t - rec.offset):
            _put(img, f"{when + rec.offset:.2f} s: {st or '-'}", (22, y))
            y += 15
        y += 14
    return img[:max(height, y)]


def draw_bar(timeline, t, speed, playing, width):
    x0, x1 = 12, width - 200
    span = max(timeline.end - timeline.start, 1e-6)
    if width not in timeline._bars:
        # the empty bar with a mark at every state change, drawn once per window width
        base = np.full((BAR_H, width, 3), C_PANEL, dtype=np.uint8)
        cv2.rectangle(base, (x0, 17), (x1, 29), C_BAR, -1)
        for rec in timeline.recs:
            for when, _st in rec.changes[1:]:
                mx = x0 + int((rec.offset + when - timeline.start) / span * (x1 - x0))
                cv2.line(base, (mx, 14), (mx, 32), C_BAR_MARK, 1)
        timeline._bars[width] = base
    img = timeline._bars[width].copy()
    fx = x0 + int((t - timeline.start) / span * (x1 - x0))
    cv2.rectangle(img, (x0, 17), (fx, 29), C_BAR_FILL, -1)
    cv2.circle(img, (fx, 23), 7, C_TEXT, -1)
    _put(img, f"{t:.2f} / {timeline.end:.2f} s, {speed:g}x{'' if playing else ', paused'}",
         (x1 + 12, 28))
    return img, (x0, x1)


def _pad(img, height):
    if img.shape[0] >= height:
        return img
    pad = np.full((height - img.shape[0], img.shape[1], 3), C_PANEL, dtype=np.uint8)
    return np.vstack([img, pad])


def compose(timeline, t, speed=1.0, playing=True, show_trails=True):
    """one full frame (field, panel and seek bar), the bar's x range, and where the bar
    starts.
    """
    field = draw_field(timeline, t, show_trails)
    panel = draw_panel(timeline, t, field.shape[0])
    height = max(field.shape[0], panel.shape[0])
    top = np.hstack([_pad(field, height), _pad(panel, height)])
    bar, bar_x = draw_bar(timeline, t, speed, playing, top.shape[1])
    return np.vstack([top, bar]), bar_x, top.shape[0]


def render(timeline, t, speed=1.0, playing=True, show_trails=True):
    """one full frame as an image."""
    return compose(timeline, t, speed, playing, show_trails)[0]


class Viewer:
    def __init__(self, timeline):
        self.tl = timeline
        self.t = timeline.play_start
        self.speed = 1.0
        self.playing = True
        self.trails = True
        self.bar_x = (0, 1)
        self.bar_y = FIELD_H
        self.dragging = False

    def _seek_px(self, x):
        x0, x1 = self.bar_x
        frac = min(1.0, max(0.0, (x - x0) / max(x1 - x0, 1)))
        self.t = self.tl.start + frac * (self.tl.end - self.tl.start)

    def _on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and y >= self.bar_y:
            self.dragging = True
            self._seek_px(x)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self._seek_px(x)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def run(self):
        win = "debug session"
        try:
            # resizable, for screens narrower than the full frame
            cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        except cv2.error:
            print("[replay] this OpenCV has no window support (opencv-python-headless?); "
                  "install opencv-python on a computer with a screen")
            return 1
        cv2.setMouseCallback(win, self._on_mouse)
        sized = False
        last = time.monotonic()
        while True:
            now = time.monotonic()
            if self.playing and not self.dragging:
                self.t = min(self.tl.end, self.t + (now - last) * self.speed)
                if self.t >= self.tl.end:
                    self.playing = False
            last = now
            frame, self.bar_x, self.bar_y = compose(self.tl, self.t, self.speed, self.playing,
                                                    self.trails)
            if not sized:
                cv2.resizeWindow(win, frame.shape[1], frame.shape[0])
                sized = True
            cv2.imshow(win, frame)
            key = cv2.waitKeyEx(15)
            if key == -1:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            ch = key & 0xFF
            if ch in (ord("q"), 27):
                break
            if ch == ord(" "):
                if not self.playing and self.t >= self.tl.end:
                    self.t = self.tl.play_start
                self.playing = not self.playing
            elif ch == ord("d") or key in (65363, 2555904, 63235):
                self.playing = False
                self.t = self.tl.step(self.t, +1)
            elif ch == ord("a") or key in (65361, 2424832, 63234):
                self.playing = False
                self.t = self.tl.step(self.t, -1)
            elif ch == ord("l"):
                self.t = min(self.tl.end, self.t + 5.0)
            elif ch == ord("j"):
                self.t = max(self.tl.start, self.t - 5.0)
            elif ch == ord("]"):
                self.speed = min(16.0, self.speed * 2)
            elif ch == ord("["):
                self.speed = max(1 / 16, self.speed / 2)
            elif ch == ord("r"):
                self.t = self.tl.play_start
                self.playing = True
            elif ch == ord("t"):
                self.trails = not self.trails
        cv2.destroyAllWindows()
        return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 1
    try:
        timeline = Timeline([Recording(p) for p in argv])
    except (OSError, ValueError) as e:
        print(f"[replay] {e}")
        return 1
    for r in timeline.recs:
        print(f"[replay] {r.folder}: robot {r.robot_id}, {len(r.ticks)} ticks, "
              f"{r.times[-1] - r.times[0]:.1f} s, lined up by {r.sync}"
              + (f", {r.skipped} unreadable rows skipped" if r.skipped else ""))
        if isinstance(r.schema, int) and r.schema > SCHEMA:
            print(f"[replay] {r.folder} is schema {r.schema}, newer than this viewer ({SCHEMA}); "
                  "some fields may be missing")
    return Viewer(timeline).run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
