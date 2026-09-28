"""Tests for bot/debug_session.py: one session folder per power-on holding the tick log and the
bench logs. Covers the switch, round trips, pause and resume in the same session, power-loss
tolerance, the failure paths, and the teammate clock rows.
"""

import json
import os
import threading
import time

import numpy as np
import pytest

import bot.debug_session as ds
import bot.motion as M
import bot.state as state
from bot import controllers as C
from bot.controllers import StrikerController
from bot.debug_session import SessionReplay


@pytest.fixture(autouse=True)
def _session_off(monkeypatch, tmp_path):
    """each test records into its own logs/ and ends with the session closed, so a leaked
    flag can't start the rest of the suite writing files
    """
    monkeypatch.chdir(tmp_path)
    ds.set_recording(False)
    ds.finish()
    yield
    ds.set_recording(False)
    ds.finish()


def _world(**kw):
    with state._lock:
        state._state.update({"run_mode": "run", "pose": (910.0, 500.0, 0.0),
                             "ball_est": (900.0, 700.0, 0.9, "cam"), "my_state": "seek"})
        state._state.update(kw)


def _only_session(tmp_path):
    folders = os.listdir(tmp_path / "logs")
    assert len(folders) == 1
    return str(tmp_path / "logs" / folders[0])


class TestSwitch:
    def test_flag_and_debug_page_mirror(self):
        ds.set_recording(True)
        assert ds.recording is True
        with state._lock:
            assert state._state["debug_recording"] is True
        ds.set_recording(False)
        assert ds.recording is False
        with state._lock:
            assert state._state["debug_recording"] is False

    def test_record_tick_does_nothing_while_off(self, tmp_path):
        ds.record_tick()
        ds.record_tick()
        assert ds._session is None
        assert not (tmp_path / "logs").exists()


class TestRoundTrip:
    def test_record_and_replay(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROBOT_ID", "2")
        _world()
        ds.set_recording(True)
        ds.record_tick()
        _world(my_state="has_ball")
        ds.record_tick()
        ds.finish()
        folder = _only_session(tmp_path)
        assert os.path.basename(folder).startswith("debug_r2_")
        r = SessionReplay(folder)
        assert r.meta["robot_id"] == "2"
        assert r.meta["schema"] == ds.SCHEMA_VERSION
        assert r.meta["closed"] is True
        assert r.meta["ticks_written"] == 2
        assert len(r.ticks) == 2
        assert r.skipped_rows == 0
        assert r.poses()[0][1:] == (910.0, 500.0, 0.0)
        assert [s for _, s in r.states()] == ["seek", "has_ball"]
        assert r.ticks[0]["t"] <= r.ticks[1]["t"]

    def test_pause_and_resume_stay_in_one_session(self, tmp_path):
        """one power-on is one session: switching off and on again carries on in the same
        folder, with the pause marked.
        """
        _world()
        ds.set_recording(True)
        ds.record_tick()
        ds.set_recording(False)
        ds.record_tick() # paused, not written
        ds.set_recording(True)
        ds.record_tick()
        ds.finish()
        r = SessionReplay(_only_session(tmp_path))
        assert len(r.ticks) == 2
        assert [e["type"] for e in r.events] == ["pause", "resume"]

    def test_idle_ticks_are_thinned(self, tmp_path):
        _world(run_mode="idle")
        ds.set_recording(True)
        for _ in range(20):
            ds.record_tick()
        _world(run_mode="run")
        for _ in range(5):
            ds.record_tick()
        ds.finish()
        r = SessionReplay(_only_session(tmp_path))
        assert len(r.ticks) == 1 + 5

    def test_camera_only_play_is_not_thinned(self, tmp_path):
        _world(run_mode="camrun")
        ds.set_recording(True)
        for _ in range(5):
            ds.record_tick()
        ds.finish()
        assert len(SessionReplay(_only_session(tmp_path)).ticks) == 5

    def test_torn_final_line_skipped(self, tmp_path):
        _world()
        ds.set_recording(True)
        ds.record_tick()
        ds.finish()
        folder = _only_session(tmp_path)
        with open(os.path.join(folder, "ticks.jsonl"), "a", encoding="utf-8") as f:
            f.write('{"type": "tick", "t": 1.5, "cmds": {"nw"')
        r = SessionReplay(folder)
        assert len(r.ticks) == 1
        assert r.skipped_rows == 1

    def test_bounds_violations(self, tmp_path):
        from bot.field import FieldModel
        ds.set_recording(True)
        _world(pose=(-200.0, 500.0, 0.0))
        ds.record_tick()
        _world(pose=(910.0, 500.0, 0.0))
        ds.record_tick()
        ds.finish()
        r = SessionReplay(_only_session(tmp_path))
        bad = r.bounds_violations(FieldModel.field_x, FieldModel.field_y)
        assert len(bad) == 1 and bad[0][1] == -200.0

    def test_session_json_is_there_before_the_first_tick(self, tmp_path):
        ds.set_recording(True)
        meta = json.loads(open(os.path.join(_only_session(tmp_path), "session.json")).read())
        assert meta["closed"] is False
        assert isinstance(meta["mono_t0"], float)

    def test_bench_logs_share_the_folder(self, tmp_path):
        """--lidarlog and friends write into the session even with recording off."""
        p = ds.path("lidar.txt")
        assert os.path.dirname(p) == _only_session(tmp_path)
        assert ds.recording is False


class TestFailuresAndEdges:
    def test_disk_failure_pauses_recording(self, monkeypatch):
        """a write failing in the writer thread pauses recording on the next tick, and the
        writer doesn't crash
        """
        crashes = []
        monkeypatch.setattr(threading, "excepthook", lambda a: crashes.append(a.exc_value))
        _world()
        ds.set_recording(True)
        ds.record_tick()
        writer = ds._session.ticks

        def dead_disk(_row):
            raise OSError("card removed")

        writer._fh.write = dead_disk
        ds.record_tick()
        writer._thread.join(timeout=2.0)
        assert isinstance(writer.error, OSError)
        ds.record_tick()
        assert ds.recording is False
        assert crashes == []

    def test_numpy_values_are_recorded(self, tmp_path):
        _world(pose=(np.float32(900.0), np.int64(400), 0.0), ball_est=np.array([1.0, 2.0]))
        ds.set_recording(True)
        ds.record_tick()
        assert ds.recording is True
        ds.finish()
        w = SessionReplay(_only_session(tmp_path)).ticks[0]["world"]
        assert w["pose"] == [900.0, 400, 0.0]
        assert w["ball_est"] == [1.0, 2.0]

    def test_records_what_the_controller_commanded(self, tmp_path):
        M.Motor.drive(0.0, 0.3)
        _world()
        ds.set_recording(True)
        ds.record_tick()
        ds.finish()
        cmds = SessionReplay(_only_session(tmp_path)).ticks[0]["cmds"]
        assert all(isinstance(cmds[m], float) and cmds[m] != 0.0
                   for m in ("nw", "ne", "sw", "se"))

    def test_pause_between_check_and_open_leaves_no_session(self, monkeypatch, tmp_path):
        """a pause landing just after the play loop's flag check doesn't leave a session made"""
        monkeypatch.setattr(ds, "recording", True)
        real_lock = ds._lock

        class PauseFirst:
            def __enter__(self):
                ds._lock = real_lock
                ds.recording = False # the other thread wins the race
                return real_lock.__enter__()

            def __exit__(self, *exc):
                return real_lock.__exit__(*exc)

        monkeypatch.setattr(ds, "_lock", PauseFirst())
        ds.record_tick()
        assert ds._session is None
        assert not (tmp_path / "logs").exists()

    def test_second_session_in_the_same_second_gets_its_own_folder(self, tmp_path,
                                                                     monkeypatch):
        monkeypatch.setattr(ds.time, "strftime", lambda fmt, *a: fmt.replace(
            "%Y%m%d_%H%M%S", "20260928_120000").replace("%Y-%m-%dT%H:%M:%S", "x"))
        ds.set_recording(True)
        ds.finish()
        ds.set_recording(False)
        ds.set_recording(True)
        ds.finish()
        robot_id = os.environ["ROBOT_ID"]
        assert sorted(os.listdir(tmp_path / "logs")) == [
            f"debug_r{robot_id}_20260928_120000", f"debug_r{robot_id}_20260928_120000_2"]

    def test_low_disk_pauses_recording(self, monkeypatch):
        monkeypatch.setattr(ds, "checkpoint_period_s", 0.02)

        class Usage:
            free = 10e6

        monkeypatch.setattr(ds.shutil, "disk_usage", lambda p: Usage())
        ds.set_recording(True)
        deadline = time.monotonic() + 2.0
        while ds.recording and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ds.recording is False


class TestLinkRows:
    def test_teammate_messages_are_logged(self, tmp_path):
        ds.set_recording(True)
        ds.link_rx(1234.5, "world")
        ds.link_rx(None, "world") # an old message with no clock
        ds.set_recording(False)
        ds.link_rx(1300.0, "world") # paused
        ds.finish()
        r = SessionReplay(_only_session(tmp_path))
        assert [(row["tx"], row["msg"]) for row in r.links] == [(1234.5, "world")]

    def test_team_link_feeds_the_session(self, tmp_path):
        from bot.network import TeamLink
        ds.set_recording(True)
        link = TeamLink(transport=None)
        link._ingest(json.dumps({"v": 1, "type": "status", "seq": 1, "t": 99.25,
                                 "data": {}}))
        ds.finish()
        r = SessionReplay(_only_session(tmp_path))
        assert [(row["tx"], row["msg"]) for row in r.links] == [(99.25, "status")]


class TestSimDrivenRecording:
    def test_full_carry_cycle_recorded(self, tmp_path):
        """drive the striker with recording on: the replay shows the state timeline and every
        tick carries the motor commands.
        """
        M._last_slew_speed = 0.0
        ds.set_recording(True)
        sc = StrikerController()
        sc.state = C.seek
        for _ in range(40):
            with state._lock:
                state._state["run_mode"] = "run"
                state._state["slot_goal"] = "low"
                state._state["pose"] = (910.0, 300.0, 0.0)
                state._state["pose_t"] = time.monotonic()
                state._state["ball"] = (0.0, 800.0)
                state._state["enemies"] = []
            sc.tick()
            ds.record_tick()
        ds.finish()
        r = SessionReplay(_only_session(tmp_path))
        assert len(r.ticks) == 40
        assert all(tk["world"]["run_mode"] == "run" for tk in r.ticks)
        assert all(s == "seek" for _, s in r.states())
        assert all(isinstance(tk["cmds"]["nw"], float) for tk in r.ticks)
