"""Team link: Bluetooth RFCOMM transports, TeamLink/TeamState, the UDP pose fallback, and the
master/slave negotiation.
"""

import json
import math
import queue
import random
import re
import socket
import subprocess
import threading
import time

import bot.debug_session as debug_session
import bot.state as state
from bot.field import FieldModel
from bot.state import _apply_slot_state


proto_version = 1
peer_timeout_s = 0.6 # no message for this long -> peer considered gone
reconnect_s = 1.0 # pause between transport reconnect attempts


# Transports

class LoopbackTransport:
    """in-process transport pair for tests: a.write() -> b.readline()."""

    @classmethod
    def pair(cls):
        """two LoopbackTransports wired together: a.write() shows up on b.readline() and vice
        versa.
        """
        a, b = cls(), cls()
        a._peer, b._peer = b, a
        return a, b

    def __init__(self):
        """one end of a pair, not usable alone until pair() links it to a peer."""
        self._q = queue.Queue()
        self._peer = None
        self._open = True

    def connect(self):
        """always succeeds, there's no real connection to make."""
        return True

    def readline(self, timeout=0.5):
        """next line written by the peer, "" if idle, None once the peer has closed."""
        try:
            item = self._q.get(timeout=timeout)
        except queue.Empty:
            return "" # idle, still connected
        return item # None = peer closed

    def write(self, line):
        """deliver `line` to the peer's readline() queue."""
        if not (self._open and self._peer and self._peer._open):
            raise OSError("loopback closed")
        self._peer._q.put(line)

    def close(self):
        """mark this end closed and wake the peer's readline() with None."""
        self._open = False
        if self._peer and self._peer._open:
            self._peer._q.put(None)


class _RfcommBase:
    """shared line-buffered recv over a stdlib AF_BLUETOOTH RFCOMM socket."""

    def __init__(self):
        """no socket yet, connect() (in a subclass) sets self._sock."""
        self._sock = None
        self._buf = ""

    def readline(self, timeout=0.5):
        """next newline-terminated line from the socket, "" if idle, None on a dead/closed
        socket.
        """
        if self._sock is None:
            return None
        if "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            return line
        self._sock.settimeout(timeout)
        try:
            chunk = self._sock.recv(1024)
        except Exception as e: # noqa: BLE001
            if "timed out" in str(e).lower():
                return "" # idle
            return None # dead socket
        if not chunk:
            return None
        self._buf += chunk.decode(errors="replace")
        if "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            return line
        return ""

    def write(self, line):
        """send `line` (already newline-terminated by the caller) over the socket."""
        if self._sock is None:
            raise OSError("not connected")
        self._sock.send(line.encode())

    def close(self):
        """close the socket if open and reset the read buffer."""
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception: # noqa: BLE001
                pass
            self._sock = None
        self._buf = ""


class RfcommServerTransport(_RfcommBase):
    """master side: listen on an RFCOMM channel, re-accept after disconnects."""

    def __init__(self, port=1):
        """port : RFCOMM channel to listen on."""
        super().__init__()
        self.port = port
        self._server = None

    def connect(self):
        """bind/listen once, then block until a subordinate connects; call again after a
        disconnect.
        """
        if self._server is None:
            self._server = socket.socket(socket.AF_BLUETOOTH,
                                         socket.SOCK_STREAM,
                                         socket.BTPROTO_RFCOMM)
            self._server.bind(("", self.port))
            self._server.listen(1)
        print(f"[team] RFCOMM listening on channel {self.port}...", flush=True)
        self._sock, addr = self._server.accept()
        print(f"[team] subordinate connected: {addr}", flush=True)
        return True


class RfcommClientTransport(_RfcommBase):
    """slave side: connect to the master's MAC address, retried by TeamLink."""

    def __init__(self, master_mac, port=1):
        """master_mac : the master Pi's Bluetooth address. port : its RFCOMM channel."""
        super().__init__()
        self.master_mac = master_mac
        self.port = port

    def connect(self):
        """dial the master. TeamLink retries this on failure/disconnect."""
        sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM,
                             socket.BTPROTO_RFCOMM)
        sock.connect((self.master_mac, self.port))
        self._sock = sock
        print(f"[team] connected to master {self.master_mac}", flush=True)
        return True


# Link

class TeamLink:
    """bidirectional newline-JSON message link over a pluggable transport."""

    def __init__(self, transport):
        """transport: any object with connect()/readline()/write()/close()."""
        self.transport = transport
        self._lock = threading.Lock()
        self._latest = {} # type -> (payload, rx_monotonic, seq)
        self._tx_seq = {}
        self._last_rx = None
        self._connected = False
        self._stop = False

    # lifecycle
    def start(self):
        """start the background connect/read/reconnect thread; returns self."""
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def stop(self):
        """stop the background thread and close the transport."""
        self._stop = True
        self.transport.close()

    def _run(self):
        """connect, read lines until disconnected, wait reconnect_s, repeat."""
        while not self._stop:
            try:
                self.transport.connect()
                self._connected = True
                while not self._stop:
                    line = self.transport.readline()
                    if line is None:
                        break # disconnected
                    if line == "":
                        continue # idle tick
                    self._ingest(line)
            except Exception as e: # noqa: BLE001
                if not self._stop:
                    print(f"[team] link error: {e}", flush=True)
            self._connected = False
            self.transport.close()
            if not self._stop:
                time.sleep(reconnect_s)

    # rx
    def _ingest(self, line):
        """parse one line and keep it as the latest of its type, dropping garbage and stale or
        duplicate sequence numbers.
        """
        try:
            msg = json.loads(line)
            mtype = msg["type"]
            seq = int(msg.get("seq", 0))
        except (ValueError, KeyError, TypeError):
            return # garbage line, skip
        now = time.monotonic()
        # sender's clock vs ours, so the viewer can line the two sessions up
        debug_session.link_rx(msg.get("t"), mtype)
        with self._lock:
            prev = self._latest.get(mtype)
            if prev is not None and seq <= prev[2]:
                return # stale / duplicate
            self._latest[mtype] = (msg.get("data", {}), now, seq)
            self._last_rx = now

    # tx
    def send(self, mtype, payload=None):
        """stamp `payload` with a sequence number + timestamp and write it out."""
        if not self._connected:
            return False
        seq = self._tx_seq.get(mtype, 0) + 1
        self._tx_seq[mtype] = seq
        line = json.dumps({"v": proto_version, "type": mtype, "seq": seq,
                           "t": time.monotonic(),
                           "data": payload or {}}) + "\n"
        try:
            self.transport.write(line)
            return True
        except Exception: # noqa: BLE001
            return False

    # queries
    def latest(self, mtype):
        """(payload, age_s) of the freshest message of type `mtype`, or None if none has arrived."""
        with self._lock:
            entry = self._latest.get(mtype)
        if entry is None:
            return None
        payload, rx_t, _seq = entry
        return payload, time.monotonic() - rx_t

    def peer_alive(self):
        """True if any message has arrived within peer_timeout_s."""
        with self._lock:
            last = self._last_rx
        return last is not None and time.monotonic() - last <= peer_timeout_s


# Master-side aggregation

class TeamState:
    """the master's view of the subordinate, with freshness gating baked in."""

    pose_max_age_s = 0.6 # teammate pose usable for ID/blocking this long
    ball_max_age_s = 0.5 # remote ball sighting usable this long

    def __init__(self, link):
        """link : a started TeamLink to read the subordinate's "status" messages from."""
        self.link = link

    def teammate_pos(self):
        """(x, y) of the subordinate, or None if silent/stale."""
        got = self.link.latest("status")
        if got is None:
            return None
        data, age = got
        pose = data.get("pose")
        if pose is None or age > self.pose_max_age_s:
            return None
        return float(pose[0]), float(pose[1])

    def remote_ball(self):
        """(x, y, conf, src) ball fix from the subordinate's camera, or None."""
        got = self.link.latest("status")
        if got is None:
            return None
        data, age = got
        ball = data.get("ball")
        if ball is None or age > self.ball_max_age_s:
            return None
        x, y, conf = float(ball[0]), float(ball[1]), float(ball[2])
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        src = str(ball[3]) if len(ball) > 3 else "cam"
        return x, y, conf, src

    def peer_state(self):
        """the subordinate's reported state ("seek", "has_ball", "goalie", ...), or None if
        silent/stale.
        """
        got = self.link.latest("status")
        if got is None:
            return None
        data, age = got
        st = data.get("state")
        if st is None or age > self.pose_max_age_s:
            return None
        return str(st)

    def peer_playing(self):
        """True if the subordinate's fresh status says it is in play (not idle, paused or
        lifted off for a penalty)."""
        got = self.link.latest("status")
        if got is None:
            return False
        data, age = got
        return age <= self.pose_max_age_s and bool(data.get("playing"))

    def peer_pass_target(self):
        """(x, y) the subordinate is passing towards, or None if it isn't or the status is stale."""
        got = self.link.latest("status")
        if got is None:
            return None
        data, age = got
        target = data.get("pass_target")
        if target is None or age > self.pose_max_age_s:
            return None
        return float(target[0]), float(target[1])

    def publish_world(self, pose, ball, enemies, role_for_slave,
                      my_state=None, pass_target=None):
        """one master broadcast tick: world snapshot + (re)assigned role."""
        # The slave never read "mate" or "solo" (it tracks its pose and solo
        # flag), so they are no longer sent.
        self.link.send("world", {
            "pose": list(pose) if pose is not None else None,
            "ball": ball,
            "enemies": [{"id": e.get("id"), "x": e["x"], "y": e["y"],
                         "occluded": bool(e.get("occluded"))}
                        for e in (enemies or [])],
            "state": my_state,
            "pass_target": list(pass_target) if pass_target is not None else None,
        })
        self.link.send("cmd", {"role": role_for_slave})


# Bluetooth team link: the master owns the fused world model and assigns the subordinate
# its role.
bt_team_enabled = True
# bot/main.py gates the UDP fallback threads on this alias
team_play_enabled = bt_team_enabled
bt_team_port = 1
bt_world_hz = 10.0 # world/cmd broadcast rate

# Dynamic role swap: every tick the master picks which robot strikes, and the other keeps
# goal. In order: if the teammate isn't playing, we strike; a dribbler holding the ball
# strikes; if only one robot sees the ball, it strikes; if both see it, the closer by
# role_swap_margin_mm strikes, or, within role_goalside_band_mm of each other, the one
# goal-side of the ball; if neither sees it, the one further upfield. False keeps each
# robot on its button pick.
bt_dynamic_roles = True
role_swap_margin_mm = 200.0
role_goalside_band_mm = 250.0
# Minimum time between swaps. A teammate dropping out and a change of possession skip it.
role_swap_hold_s = 2.0

udp_port = 5005
_my_udp_id = random.randint(0, 0xFFFF)


def _udp_send_thread():
    """broadcast our pose to the teammate at about 10 Hz (the fallback channel; Bluetooth
    supersedes it).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    while True:
        with state._lock:
            pose = state._state["pose"]
        if pose is not None:
            try:
                msg = json.dumps({"id": _my_udp_id,
                                  "pose": [round(pose[0], 1),
                                           round(pose[1], 1),
                                           round(pose[2], 2)]}).encode()
                sock.sendto(msg, ("255.255.255.255", udp_port))
            except Exception:
                pass
        time.sleep(0.1)


def _udp_recv_thread():
    """receive teammate pose over the UDP fallback channel; clear it if the teammate goes quiet."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", udp_port))
    sock.settimeout(0.5)
    while True:
        try:
            data, _ = sock.recvfrom(256)
            msg = json.loads(data)
            if msg.get("id") == _my_udp_id:
                continue # our broadcast, ignore
            tp = msg.get("pose")
            if tp and len(tp) == 3:
                with state._lock:
                    state._state["teammate_pos"] = (tp[0], tp[1])
        except socket.timeout:
            # No packet from teammate recently, assume gone
            with state._lock:
                if state._state["teammate_pos"] is not None:
                    state._state["teammate_pos"] = None
        except Exception:
            pass


def _discover_peer_mac(timeout=2.0):
    """MAC address of the one already-paired Bluetooth peer, or None."""
    try:
        out = subprocess.run(["bluetoothctl", "paired-devices"],
                             capture_output=True, text=True,
                             timeout=timeout).stdout
    except Exception: # noqa: BLE001
        return None
    match = re.search(r"Device ([0-9A-Fa-f:]{17})", out)
    return match.group(1) if match else None


def _striker_claim(me, peer):
    """"me" or "peer": which robot should strike, or None to keep the current pick. me and
    peer are dicts: playing, holds (dribbler has the ball), sees (own camera sighting), dist
    (to that sighting, mm), behind (goal-side of that sighting), depth (mm from our goal line).
    """
    if not peer["playing"]:
        return "me"
    if not me["playing"]:
        return "peer"
    if me["holds"] != peer["holds"]:
        return "me" if me["holds"] else "peer"
    if me["sees"] != peer["sees"]:
        return "me" if me["sees"] else "peer"
    if me["sees"]:
        if me["dist"] + role_swap_margin_mm < peer["dist"]:
            return "me"
        if peer["dist"] + role_swap_margin_mm < me["dist"]:
            return "peer"
        if (abs(me["dist"] - peer["dist"]) < role_goalside_band_mm
                and me["behind"] != peer["behind"]):
            return "me" if me["behind"] else "peer"
        return None
    # neither sees it: the robot further upfield. The margin stops two robots level with
    # each other trading roles every hold period.
    if me["depth"] > peer["depth"] + role_swap_margin_mm:
        return "me"
    if peer["depth"] > me["depth"] + role_swap_margin_mm:
        return "peer"
    return None


def _claim_view(playing, holds, pose, sighting, own_goal):
    """one robot's _striker_claim dict from its pose and its own camera sighting (x, y) or None."""
    into = 1.0 if own_goal[1] < FieldModel.field_y / 2 else -1.0
    depth = into * (pose[1] - own_goal[1]) if pose is not None else 0.0
    view = {"playing": playing, "holds": holds, "sees": False,
            "dist": math.inf, "behind": False, "depth": depth}
    if sighting is not None and pose is not None:
        view["sees"] = True
        view["dist"] = math.hypot(sighting[0] - pose[0], sighting[1] - pose[1])
        view["behind"] = depth <= into * (sighting[1] - own_goal[1])
    return view


def _run_master(link):
    """master side of the Bluetooth team link (RFCOMM server)."""
    # function-body import: bot.controllers imports this module, so a top-level import
    # would cycle
    from bot.calibration import goal_positions
    from bot.controllers import has_ball, passing
    team = TeamState(link)
    period = 1.0 / bt_world_hz
    striker_is_me = None # the current pick, None until the first tick
    last_swap_t = -math.inf
    was_running = False
    was_solo = None # None so the first pass always logs

    while True:
        with state._lock:
            if state._state["bt_is_master"] is not True:
                return # role reset, hand back
        # Solo = no live teammate.
        solo = not link.peer_alive()
        if solo != was_solo:
            print("[team] playing solo (no teammate)" if solo
                  else "[team] teammate connected, coordinating", flush=True)
            was_solo = solo

        tm = team.teammate_pos()
        rb = team.remote_ball()
        with state._lock:
            state._state["solo"] = solo
            state._state["teammate_pos_bt"] = tm
            state._state["remote_ball"] = rb
            state._state["peer_state"] = team.peer_state()
            state._state["peer_pass_target"] = team.peer_pass_target()
            pose = state._state["pose"]
            ball_est = state._state["ball_est"]
            enemies = state._state["enemies"]
            my_state = state._state["my_state"]
            peer_state = state._state["peer_state"]
            pass_target = state._state["pass_target"]
            goal = state._state["slot_goal"]
            my_run = state._state["run_mode"]
            # the role buttons can change this live, so re-read it every tick
            my_own_role = state._state["own_slot_role"] or state._state["slot_role"]
            kickoff_hold = (state._state["kickoff_role"] is not None
                            and state._state["kickoff_until_t"] is not None
                            and time.monotonic() < state._state["kickoff_until_t"])
        peer_playing = team.peer_playing()

        # Pick the striker for this tick (see bt_dynamic_roles). The first pick
        # follows the buttons; after that a swap needs role_swap_hold_s since the
        # last one, unless the teammate dropped out or possession changed hands.
        if my_run == "run" and not was_running:
            striker_is_me = None # play (re)started: back to the button picks
            last_swap_t = time.monotonic()
        was_running = my_run == "run"
        if striker_is_me is None:
            striker_is_me = my_own_role != "goalie"
        me_playing = my_run == "run"
        peer_playing = not solo and peer_playing and tm is not None
        pick, urgent = None, False
        if bt_dynamic_roles and not kickoff_hold:
            if me_playing != peer_playing:
                # one robot is out (idle, paused, lifted, link gone): the other strikes
                pick, urgent = ("me" if me_playing else "peer"), True
            elif pose is not None and goal is not None:
                own_goal, _ = goal_positions(goal)
                my_sight = ((ball_est[0], ball_est[1])
                            if ball_est is not None and ball_est[3] == "cam" else None)
                peer_sight = ((rb[0], rb[1])
                              if rb is not None and rb[3] == "cam" else None)
                me = _claim_view(True, my_state in (has_ball, passing),
                                   pose, my_sight, own_goal)
                peer = _claim_view(True, peer_state in (has_ball, passing),
                                   tm, peer_sight, own_goal)
                pick = _striker_claim(me, peer)
                urgent = me["holds"] != peer["holds"]
        now = time.monotonic()
        if (pick is not None and (pick == "me") != striker_is_me
                and (urgent or now - last_swap_t >= role_swap_hold_s)):
            striker_is_me = pick == "me"
            last_swap_t = now
            print(f"[team] role swap: {'we strike' if striker_is_me else 'teammate strikes'}",
                  flush=True)
        my_role = "striker" if striker_is_me else "goalie"
        slave_role = "goalie" if striker_is_me else "striker"
        if bt_dynamic_roles and my_run == "run":
            with state._lock:
                if state._state["slot_role"] != my_role:
                    state._state["slot_role"] = my_role
                    _apply_slot_state()

        ball_msg = None
        if ball_est is not None:
            ball_msg = {"x": ball_est[0], "y": ball_est[1],
                        "conf": ball_est[2], "src": ball_est[3]}
        team.publish_world(pose, ball_msg, enemies, slave_role,
                           my_state=my_state, pass_target=pass_target)
        time.sleep(period)


def _run_slave(link):
    """slave side of the Bluetooth team link (RFCOMM client)."""
    period = 1.0 / bt_world_hz
    was_solo = None

    while True:
        with state._lock:
            if state._state["bt_is_master"] is not False:
                return # role reset, hand back

        solo = not link.peer_alive()
        if solo != was_solo:
            print("[team] playing solo (no teammate)" if solo
                  else "[team] teammate connected, coordinating", flush=True)
            was_solo = solo

        world = link.latest("world")
        cmd = link.latest("cmd")
        with state._lock:
            state._state["solo"] = solo
            if world is not None and world[1] <= TeamState.pose_max_age_s:
                data = world[0]
                mpose = data.get("pose")
                state._state["teammate_pos_bt"] = (
                    (float(mpose[0]), float(mpose[1])) if mpose else None)
                mball = data.get("ball")
                # keep the master's source tag: its "ball" is the fused
                # estimate, which may be a memory inference rather than a
                # sighting
                state._state["remote_ball"] = (
                    (mball["x"], mball["y"], mball["conf"],
                     mball.get("src", "cam")) if mball else None)
                state._state["peer_state"] = data.get("state")
                mpass = data.get("pass_target")
                state._state["peer_pass_target"] = (
                    (float(mpass[0]), float(mpass[1])) if mpass else None)
            else:
                state._state["teammate_pos_bt"] = None
                state._state["remote_ball"] = None
                state._state["peer_state"] = None
                state._state["peer_pass_target"] = None
            if cmd is not None and cmd[1] <= TeamState.pose_max_age_s:
                role = cmd[0].get("role")
                if role in ("striker", "goalie"):
                    state._state["slot_role"] = role
                    _apply_slot_state()
            else:
                # No live cmd, master's gone (dead link, crashed, powered off).
                # With the swap on, the robot left playing strikes; off, back to
                # our button pick.
                own = ("striker" if bt_dynamic_roles
                       else state._state["own_slot_role"])
                if own is not None and state._state["slot_role"] != own:
                    print(f"[team] teammate link lost, playing {own}", flush=True)
                    state._state["slot_role"] = own
                    _apply_slot_state()
            pose = state._state["pose"]
            ball_est = state._state["ball_est"]
            my_state = state._state["my_state"]
            pass_target = state._state["pass_target"]
            playing = state._state["run_mode"] == "run"

        # Only our own camera sighting goes out as "ball".
        own_sighting = (list(ball_est) if ball_est is not None
                        and ball_est[3] == "cam" else None)
        link.send("status", {
            "pose": list(pose) if pose is not None else None,
            "ball": own_sighting,
            "state": my_state,
            "playing": playing,
            "pass_target": list(pass_target) if pass_target is not None else None,
        })
        time.sleep(period)


class _PreConnectedRfcomm(_RfcommBase):
    """wraps a socket _negotiate_bt_role already connected, so TeamLink's first connect()
    reuses it.
    """

    def __init__(self, sock, fallback):
        super().__init__()
        self._first_sock = sock
        self._fallback = fallback

    def connect(self):
        if self._first_sock is not None:
            self._sock, self._first_sock = self._first_sock, None
            return True
        ok = self._fallback.connect()
        self._sock = self._fallback._sock
        return ok


def _negotiate_bt_role(peer_mac):
    """one round of deciding master vs slave: accept() for a randomised window, then connect()
    to peer_mac if nobody called. Returns (is_master, socket), or (None, None) to retry.
    """
    window = random.uniform(0.4, 0.9)

    srv = None
    try:
        srv = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM,
                            socket.BTPROTO_RFCOMM)
        srv.bind(("", bt_team_port))
        srv.listen(1)
        srv.settimeout(window)
        sock, _addr = srv.accept()
        return True, sock
    except Exception: # noqa: BLE001
        pass # no adapter, or nobody called
    finally:
        if srv is not None:
            srv.close()

    if peer_mac is None:
        return None, None
    try:
        sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM,
                             socket.BTPROTO_RFCOMM)
        sock.settimeout(window)
        sock.connect((peer_mac, bt_team_port))
        return False, sock
    except Exception: # noqa: BLE001
        return None, None # no adapter, or peer not up yet


def _bt_link_thread():
    """decide master vs slave and run that side of the Bluetooth link, retrying forever with no
    adapter or peer.
    """
    peer_mac = None
    while True:
        with state._lock:
            role_picked = state._state["slot_role"] is not None

        if not role_picked:
            if peer_mac is None:
                peer_mac = _discover_peer_mac()
            time.sleep(0.5)
            continue

        if peer_mac is None:
            peer_mac = _discover_peer_mac()

        is_master, sock = _negotiate_bt_role(peer_mac)
        if is_master is None:
            time.sleep(0.2) # guards against a busy-loop if there's no adapter
            continue # nobody there yet, retry

        with state._lock:
            state._state["bt_is_master"] = is_master
        fallback = (RfcommServerTransport(bt_team_port) if is_master
                    else RfcommClientTransport(peer_mac, bt_team_port))
        transport = _PreConnectedRfcomm(sock, fallback)
        link = TeamLink(transport).start()

        (_run_master if is_master else _run_slave)(link)
        link.stop()
        peer_mac = None # re-discover next round
