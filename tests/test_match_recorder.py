"""Tests for bot/match_recorder.py: the match replay recording system (async writer, toggleable
global record flag, crash-safe replay loader). Covers round-trip record/replay, toggle
behaviour, power-loss tolerance, and a live sim-driven recording.
"""

import json
import time

import pytest

import bot.match_recorder as mr
import bot.motion as M
import bot.state as state
import bot.dwibbler as dw
from bot import controllers as C
from bot.controllers import StrikerController
from bot.match_recorder import MatchReplay


@pytest.fixture(autouse=True)
def _recorder_off():
    """every test starts and ends with recording off (the toggle is a module global; leak it
    and the whole suite starts writing files).
    """
    mr.set_record(False)
    yield
    mr.set_record(False)


class TestToggle:
    def test_set_record_flips_flag_and_state(self):
        mr.set_record(True)
        assert mr.record_match is True
        with state._lock:
            assert state._state["record_match"] is True
        mr.set_record(False)
        assert mr.record_match is False
        with state._lock:
            assert state._state["record_match"] is False

    def test_record_tick_noop_when_off(self):
        # must not create anything or raise while off
        mr.record_tick()
        mr.record_tick()
        assert mr._recorder is None


class TestRoundTrip:
    def test_record_and_replay(self, tmp_path, monkeypatch):
        """record some ticks with a published world, then read them back."""
        rec = mr._MatchRecorder(str(tmp_path / "m.jsonl"), "1")
        with state._lock:
            state._state["pose"] = (910.0, 500.0, 0.0)
            state._state["ball_est"] = (900.0, 700.0, 0.9, "cam")
            state._state["my_state"] = "seek"
        rec.tick({"nw": 0.1, "ne": -0.1, "sw": -0.1, "se": 0.1,
                  "dwibble": 1.0},
                 {k: state._state.get(k) for k in mr._WORLD_KEYS},
                 True, True)
        rec.tick({"nw": 0.0, "ne": 0.0, "sw": 0.0, "se": 0.0,
                  "dwibble": 0.0},
                 {k: state._state.get(k) for k in mr._WORLD_KEYS},
                 False, True)
        path = rec.close()

        r = MatchReplay(path)
        assert r.header["robot_id"] == "1"
        assert r.header["schema"] == mr.SCHEMA_VERSION
        assert len(r.ticks) == 2
        assert r.end["ticks"] == 2
        assert r.skipped_rows == 0
        assert r.poses()[0][1:] == (910.0, 500.0, 0.0)
        assert r.states()[0][1] == "seek"
        assert r.possession_timeline()[0][1] is True
        assert r.ticks[0]["cmds"]["dwibble"] == 1.0

    def test_torn_final_line_skipped(self, tmp_path):
        """a power loss mid-write leaves a torn last row: the loader skips it instead of failing."""
        rec = mr._MatchRecorder(str(tmp_path / "m.jsonl"), "1")
        rec.tick({"nw": 0.0, "ne": 0.0, "sw": 0.0, "se": 0.0, "dwibble": 0.0},
                 {k: None for k in mr._WORLD_KEYS}, False, False)
        rec.close()
        with open(str(tmp_path / "m.jsonl"), "a", encoding="utf-8") as f:
            f.write('{"type": "tick", "t": 1.5, "cmds": {"nw"') # torn
        r = MatchReplay(str(tmp_path / "m.jsonl"))
        assert len(r.ticks) == 1
        assert r.skipped_rows == 1

    def test_bounds_violations(self, tmp_path):
        from bot.field import FieldModel
        rec = mr._MatchRecorder(str(tmp_path / "m.jsonl"), "1")
        with state._lock:
            state._state["pose"] = (-200.0, 500.0, 0.0) # off-field
        rec.tick({m: 0.0 for m in mr._RECORD_MOTORS},
                 {k: state._state.get(k) for k in mr._WORLD_KEYS},
                 False, False)
        with state._lock:
            state._state["pose"] = (910.0, 500.0, 0.0) # fine
        rec.tick({m: 0.0 for m in mr._RECORD_MOTORS},
                 {k: state._state.get(k) for k in mr._WORLD_KEYS},
                 False, False)
        path = rec.close()
        r = MatchReplay(path)
        bad = r.bounds_violations(FieldModel.field_x, FieldModel.field_y)
        assert len(bad) == 1 and bad[0][1] == -200.0


class TestRecordTickPath:
    def test_record_tick_opens_and_writes(self, monkeypatch, tmp_path):
        """the play-loop entry point: flag on -> first tick opens a file, ticks accumulate,
        finish() closes it.
        """
        monkeypatch.setattr(mr, "record_match", True)
        with state._lock:
            state._state["run_mode"] = "run"
        mr.record_tick()
        assert mr._recorder is not None
        path = mr._recorder.path
        mr.record_tick()
        mr.finish()
        assert mr._recorder is None
        r = MatchReplay(path)
        assert len(r.ticks) == 2

    def test_writer_failure_disables_recording(self, monkeypatch, tmp_path):
        """a dead SD card must never take down play: the failure path flips recording off
        instead of raising.
        """
        mr.set_record(True)
        # a regular file where the log directory should be makes the open fail on every OS
        blocker = tmp_path / "blocker"
        blocker.write_text("")
        monkeypatch.setattr(mr, "_match_path",
                            lambda rid: str(blocker / "match.jsonl"))
        mr.record_tick()
        assert mr.record_match is False, "failure must disable recording"
        assert mr._recorder is None


class TestSimDrivenRecording:
    def test_full_carry_cycle_recorded(self, tmp_path, monkeypatch):
        """drive the striker through a carry with recording on: the replay must show the seek
        -> has_ball state timeline and a glued-ball world.
        """
        real = M.Motor.drive

        def spy(bearing, speed, **kw):
            real(bearing, speed, **kw)

        monkeypatch.setattr(M.Motor, "drive", spy)
        M._last_slew_speed = 0.0
        # record into tmp: patch the path hook so we don't pollute logs/
        monkeypatch.setattr(mr, "_match_path",
                            lambda rid: str(tmp_path / "sim_match.jsonl"))
        mr.set_record(True)
        sc = StrikerController()
        sc.state = C.seek
        rx, ry, hdg = 910.0, 300.0, 0.0
        for i in range(40):
            with state._lock:
                state._state["run_mode"] = "run"
                state._state["slot_goal"] = "low"
                state._state["pose"] = (rx, ry, hdg)
                state._state["pose_t"] = time.monotonic()
                state._state["ball"] = (0.0, 800.0)
                state._state["enemies"] = []
            sc.tick()
            mr.record_tick()
        mr.finish()
        r = MatchReplay(str(tmp_path / "sim_match.jsonl"))
        assert len(r.ticks) == 40
        assert all(tk["world"]["run_mode"] == "run" for tk in r.ticks)
        states = [s for _, s in r.states()]
        assert all(s == "seek" for s in states)
        # every tick carries the motor commands snapshot
        assert all("cmds" in tk and "nw" in tk["cmds"] for tk in r.ticks)
