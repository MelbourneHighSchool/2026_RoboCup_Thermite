"""Bluetooth team-link comms (RFCOMM transports, TeamLink/TeamState) plus the
UDP pose-broadcast fallback and the master/slave negotiation threads.

Extracted verbatim from mainrunbot1.py's team-link section (originally
around lines 1606-1918: proto_version/transports/TeamLink/TeamState) and its
runtime threads (originally around lines 5355-5624: UDP send/recv,
_discover_peer_mac, _run_master/_run_slave, _PreConnectedRfcomm,
_negotiate_bt_role, _bt_link_thread).

Documented gap (same pattern as bot/logs.py's for its own not-yet-extracted
names): `_run_master` references the bare name `has_ball`, one of a group of
behaviour-state-name string constants (`seek`, `has_ball`, `passing`,
`flick_shot`) that belong to a later controllers/motion-state-machine
extraction stage and do not exist as a bot.* module yet. It is left as an
undefined module global here, referenced only inside `_run_master`'s
function body, so `import bot.network` still succeeds (Python does not
evaluate a function body at import time); calling `_run_master` will
NameError on `has_ball` until that later stage supplies it. Do not invent a
local definition for it here - the real one belongs with its sibling state
names in whichever module extracts the state machine.

Correctly-imported cross-module state used here: bot.state's `_lock`,
`_state` (accessed as `state.<name>`, never a from-import - see
bot/state.py's own docstring) and its `_apply_slot_state` helper (a plain
function there, not rebound, so a normal from-import is fine).
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

import bot.state as state
from bot.state import _apply_slot_state


proto_version  = 1
peer_timeout_s = 0.6 # no message for this long -> peer considered gone
reconnect_s    = 1.0 # pause between transport reconnect attempts


# Transports

class LoopbackTransport:
    """in-process transport pair for tests: a.write() -> b.readline()."""

    @classmethod
    def pair(cls):
        """two LoopbackTransports wired to each other, (a, b) where a.write() shows up on b.readline() and vice versa."""
        a, b = cls(), cls()
        a._peer, b._peer = b, a
        return a, b

    def __init__(self):
        """one end of a pair, not usable on its own until pair() links it to a peer."""
        self._q     = queue.Queue()
        self._peer  = None
        self._open  = True

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
        self._buf  = ""

    def readline(self, timeout=0.5):
        """next newline-terminated line from the socket, "" if idle, None on a dead/closed socket."""
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
        self.port    = port
        self._server = None

    def connect(self):
        """bind/listen once, then block until a subordinate connects. call again after a disconnect to accept the next one."""
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
        self.port       = port

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
        """transport : any object with connect()/readline()/write()/close(), e.g. RfcommServerTransport."""
        self.transport  = transport
        self._lock      = threading.Lock()
        self._latest    = {} # type -> (payload, rx_monotonic, seq)
        self._tx_seq    = {}
        self._last_rx   = None
        self._connected = False
        self._stop      = False

    # lifecycle
    def start(self):
        """start the background connect/read/reconnect thread. returns self, so `link = TeamLink(t).start()` works."""
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def stop(self):
        """stop the background thread and close the transport."""
        self._stop = True
        self.transport.close()

    def _run(self):
        """background loop: connect, read lines until disconnected, then wait reconnect_s and try again."""
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
        """parse one received line and store it as the latest message of its type, dropping garbage lines and stale/duplicate sequence numbers."""
        try:
            msg = json.loads(line)
            mtype = msg["type"]
            seq   = int(msg.get("seq", 0))
        except (ValueError, KeyError, TypeError):
            return # garbage line, skip
        now = time.monotonic()
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
        """(payload, age_s) of the freshest message of type `mtype` received so far, or None if none has arrived."""
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
        """the subordinate's reported behaviour state ("seek", "has_ball", "goalie", ...), or None if silent/stale."""
        got = self.link.latest("status")
        if got is None:
            return None
        data, age = got
        st = data.get("state")
        if st is None or age > self.pose_max_age_s:
            return None
        return str(st)

    def peer_pass_target(self):
        """(x, y) the subordinate is currently passing towards, or None if it isn't passing / the status message is stale (sec 3.21)."""
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
        # "mate" (master's belief of the slave's own position, echoed back)
        # and "solo" used to be sent here too, but neither is ever read on
        # the slave side (it tracks its own pose and computes its own solo
        # flag from link.peer_alive()) - dropped as dead weight on the wire.
        self.link.send("world", {
            "pose":    list(pose) if pose is not None else None,
            "ball":    ball,
            "enemies": [{"id": e.get("id"), "x": e["x"], "y": e["y"],
                         "occluded": bool(e.get("occluded"))}
                        for e in (enemies or [])],
            "state":   my_state,
            "pass_target": list(pass_target) if pass_target is not None else None,
        })
        self.link.send("cmd", {"role": role_for_slave})


# Bluetooth team link (master side): this bot owns the fused world model and assigns the subordinate its role over RFCOMM.
bt_team_enabled = False
# alias, matching the monolith's own pattern (mainrunbot1.py's
# `team_play_enabled = bt_team_enabled`): bot/main.py gates the UDP fallback
# threads on this name.
team_play_enabled = bt_team_enabled
bt_team_port    = 1
bt_world_hz     = 10.0 # world/cmd broadcast rate

# Dynamic role handoff on our TeamLink: a bot chasing the ball tells its partner to defend, whoever is better placed takes it.
bt_dynamic_roles    = True
role_swap_margin_mm = 60.0 # peer must be this much closer before roles swap
yield_ball_near_mm  = 600.0 # ball inside this -> chase regardless of yield
yield_hold_mm       = 900.0 # yielding striker holds this far out, on the goal->ball line

udp_port    = 5005
_my_udp_id  = random.randint(0, 0xFFFF)


def _udp_send_thread():
    """broadcast our pose to the teammate at about 10 Hz (fallback pose channel, superseded by Bluetooth when connected)."""
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
                continue # our own broadcast, ignore
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


def _run_master(link):
    """master side of the Bluetooth team link (RFCOMM server)."""
    team         = TeamState(link)
    period       = 1.0 / bt_world_hz
    slave_has_it = False # hysteresis memory for the role handoff
    was_solo     = None # None so the first pass always logs

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
            state._state["solo"]              = solo
            state._state["teammate_pos_bt"]   = tm
            state._state["remote_ball"]       = rb
            state._state["peer_state"]        = team.peer_state()
            state._state["peer_pass_target"]  = team.peer_pass_target()
            pose        = state._state["pose"]
            ball_est    = state._state["ball_est"]
            enemies     = state._state["enemies"]
            my_state    = state._state["my_state"]
            pass_target = state._state["pass_target"]
            # The role buttons can change this live; re-read every tick
            # rather than freezing it at thread start.  Being master says
            # nothing about which role this bot is playing any more.
            my_role   = state._state["slot_role"] or "striker"

        static_role = "goalie" if my_role == "striker" else "striker"

        # Dynamic role handoff (master-decided): whoever is closer to the fused ball strikes, role_swap_margin_mm of hysteresis stops the roles flapping.
        slave_role = static_role
        if (bt_dynamic_roles and my_role == "striker" and pose is not None
                and tm is not None and ball_est is not None):
            d_me   = math.hypot(ball_est[0] - pose[0], ball_est[1] - pose[1])
            d_peer = math.hypot(ball_est[0] - tm[0],   ball_est[1] - tm[1])
            if d_me <= yield_ball_near_mm or my_state == has_ball:
                slave_has_it = False
            elif slave_has_it:
                slave_has_it = d_peer <= d_me + role_swap_margin_mm
            else:
                slave_has_it = d_peer + role_swap_margin_mm < d_me
            if slave_has_it:
                slave_role = "striker"
        else:
            slave_has_it = False
        with state._lock:
            state._state["yield_striker"] = slave_has_it

        ball_msg = None
        if ball_est is not None:
            ball_msg = {"x": ball_est[0], "y": ball_est[1],
                        "conf": ball_est[2], "src": ball_est[3]}
        team.publish_world(pose, ball_msg, enemies, slave_role,
                           my_state=my_state, pass_target=pass_target)
        time.sleep(period)


def _run_slave(link):
    """slave side of the Bluetooth team link (RFCOMM client)."""
    period   = 1.0 / bt_world_hz
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
        cmd   = link.latest("cmd")
        with state._lock:
            state._state["solo"] = solo
            if world is not None and world[1] <= TeamState.pose_max_age_s:
                data = world[0]
                mpose = data.get("pose")
                state._state["teammate_pos_bt"] = (
                    (float(mpose[0]), float(mpose[1])) if mpose else None)
                mball = data.get("ball")
                # Keep the master's own source tag: its "ball" is the fused estimate, so it can be an inference off ball memory rather than a sighting.
                state._state["remote_ball"] = (
                    (mball["x"], mball["y"], mball["conf"],
                     mball.get("src", "cam")) if mball else None)
                state._state["peer_state"] = data.get("state")
                mpass = data.get("pass_target")
                state._state["peer_pass_target"] = (
                    (float(mpass[0]), float(mpass[1])) if mpass else None)
            else:
                state._state["teammate_pos_bt"]  = None
                state._state["remote_ball"]      = None
                state._state["peer_state"]       = None
                state._state["peer_pass_target"] = None
            if cmd is not None and cmd[1] <= TeamState.pose_max_age_s:
                role = cmd[0].get("role")
                if role in ("striker", "goalie"):
                    state._state["slot_role"] = role
                    _apply_slot_state()
            else:
                # No live cmd, master's gone (dead link, crashed, powered off).
                own = state._state["own_slot_role"]
                if own is not None and state._state["slot_role"] != own:
                    print(f"[team] teammate link lost, reverting to our "
                          f"own {own} pick", flush=True)
                    state._state["slot_role"] = own
                    _apply_slot_state()
            pose        = state._state["pose"]
            ball_est    = state._state["ball_est"]
            my_state    = state._state["my_state"]
            pass_target = state._state["pass_target"]

        # Only our own camera sighting goes out as "ball".
        own_sighting = (list(ball_est) if ball_est is not None
                        and ball_est[3] == "cam" else None)
        link.send("status", {
            "pose":  list(pose) if pose is not None else None,
            "ball":  own_sighting,
            "state": my_state,
            "pass_target": list(pass_target) if pass_target is not None else None,
        })
        time.sleep(period)


class _PreConnectedRfcomm(_RfcommBase):
    """wraps a socket _negotiate_bt_role already connected, so TeamLink's first connect() call reuses it instead of dialling/listening again."""

    def __init__(self, sock, fallback):
        super().__init__()
        self._first_sock = sock
        self._fallback    = fallback

    def connect(self):
        if self._first_sock is not None:
            self._sock, self._first_sock = self._first_sock, None
            return True
        ok = self._fallback.connect()
        self._sock = self._fallback._sock
        return ok


def _negotiate_bt_role(peer_mac):
    """one round of deciding master vs. slave: accept() for a randomised window, then connect() to peer_mac if nobody called; returns (is_master, socket) on success or (None, None) to retry."""
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
    """decide master vs. slave and run the matching side of the Bluetooth team link; degrades gracefully, retrying forever with no adapter/peer."""
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
        fallback  = (RfcommServerTransport(bt_team_port) if is_master
                    else RfcommClientTransport(peer_mac, bt_team_port))
        transport = _PreConnectedRfcomm(sock, fallback)
        link      = TeamLink(transport).start()

        (_run_master if is_master else _run_slave)(link)
        link.stop()
        peer_mac = None # re-discover next round
