"""Tests for simulator.py, the debug session viewer: loading a session folder, lining two robots
up on one clock (by their teammate messages, or by play start), the field-to-screen mapping, and
a headless render of a real recording.
"""

import json
import os
import time

import pytest

import bot.debug_session as ds
import bot.state as state
import simulator as sim
from bot import controllers as C
from bot.controllers import StrikerController


def _tick(t, run_mode="run", pose=(910.0, 600.0, 0.0), state_name="seek", **world):
    w = {"run_mode": run_mode, "pose": list(pose) if pose else None, "my_state": state_name,
         "slot_goal": "low", "slot_role": "striker"}
    w.update(world)
    return {"type": "tick", "t": t, "cmds": {"nw": 0.1, "ne": -0.1, "sw": 0.0, "se": None,
                                             "dwibble": 0.6},
            "drib": False, "drib_avail": True, "world": w}


def _session(folder, rows, robot_id="1", mono_t0=1000.0, torn_tail=False):
    os.makedirs(folder)
    meta = {"schema": 3, "robot_id": robot_id, "mono_t0": mono_t0}
    with open(os.path.join(folder, "session.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(folder, "ticks.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        if torn_tail:
            f.write('{"type": "tick", "t": 9')
    return str(folder)


def _link(t, tx):
    return {"type": "link", "t": t, "tx": tx, "msg": "world"}


class TestRecording:
    def test_loads_sorts_and_skips_torn_line(self, tmp_path):
        p = _session(tmp_path / "s", [_tick(0.2), _tick(0.0), _tick(0.1)], torn_tail=True)
        rec = sim.Recording(p)
        assert rec.times == [0.0, 0.1, 0.2]
        assert rec.skipped == 1
        assert rec.robot_id == "1"

    def test_any_file_in_the_folder_opens_it(self, tmp_path):
        p = _session(tmp_path / "s", [_tick(0.0)])
        assert sim.Recording(os.path.join(p, "ticks.jsonl")).folder == os.path.abspath(p)

    def test_not_a_session(self, tmp_path):
        with pytest.raises(ValueError):
            sim.Recording(str(tmp_path))

    def test_malformed_rows_are_skipped(self, tmp_path):
        p = _session(tmp_path / "s", [_tick(0.0), {"type": "tick", "world": {}},
                                      {"type": "tick", "t": 0.1, "world": None}, [1, 2],
                                      _tick(0.2)])
        rec = sim.Recording(p)
        assert rec.times == [0.0, 0.2]
        assert rec.skipped == 3

    def test_state_changes(self, tmp_path):
        rows = [_tick(0.0, state_name="seek"), _tick(0.1, state_name="seek"),
                _tick(0.2, state_name="approach"), _tick(0.3, state_name="has_ball"),
                _tick(0.4, state_name="seek")]
        rec = sim.Recording(_session(tmp_path / "s", rows))
        assert rec.changes == [(0.0, "seek"), (0.2, "approach"), (0.3, "has_ball"),
                               (0.4, "seek")]
        assert sim._state_changes(rec, 0.25) == [(0.0, "seek"), (0.2, "approach")]
        assert sim._state_changes(rec, 0.4, n=2) == [(0.3, "has_ball"), (0.4, "seek")]
        assert sim._state_changes(rec, -1.0) == []

    def test_no_ticks_is_an_error(self, tmp_path):
        with pytest.raises(ValueError):
            sim.Recording(_session(tmp_path / "s", []))

    def test_play_start_is_first_run_tick(self, tmp_path):
        rows = [_tick(0.0, run_mode="idle"), _tick(0.5, run_mode="idle"), _tick(1.0)]
        assert sim.Recording(_session(tmp_path / "s", rows)).play_start() == 1.0

    def test_camera_only_play_counts_as_play(self, tmp_path):
        rows = [_tick(0.0, run_mode="idle"), _tick(0.7, run_mode="camrun")]
        assert sim.Recording(_session(tmp_path / "s", rows)).play_start() == 0.7

    def test_play_start_falls_back_to_first_tick(self, tmp_path):
        rec = sim.Recording(_session(tmp_path / "s", [_tick(0.3, run_mode="idle")]))
        assert rec.play_start() == 0.3

    def test_index_at(self, tmp_path):
        rec = sim.Recording(_session(tmp_path / "s", [_tick(0.0), _tick(0.1), _tick(0.2)]))
        assert rec.index_at(-0.01) is None
        assert rec.index_at(0.15) == 1
        assert rec.index_at(5.0) == 2


class TestTimeline:
    def test_lined_up_by_play_start_without_links(self, tmp_path):
        r1 = [_tick(0.0, run_mode="idle"), _tick(1.0), _tick(2.0)]
        r2 = [_tick(0.0, run_mode="idle"), _tick(3.0), _tick(4.0)]
        tl = sim.Timeline([sim.Recording(_session(tmp_path / "a", r1, "1")),
                           sim.Recording(_session(tmp_path / "b", r2, "2"))])
        a, b = tl.recs
        assert b.sync == "play start"
        assert a.offset + 1.0 == pytest.approx(b.offset + 3.0)
        assert tl.start == 0.0
        assert tl.play_start == pytest.approx(a.offset + 1.0)
        assert [r.robot_id for r, _, _ in tl.frame(tl.play_start)] == ["1", "2"]

    def test_lined_up_by_teammate_messages(self, tmp_path):
        """robot 2 powered on 7 s after robot 1 and its clock reads 500 s more; messages take
        20 ms each way. Play start would give a different answer, and the messages win.
        """
        a_t0, b_t0 = 1000.0, 1507.0 # b's clock = a's clock + 500
        lat = 0.02
        a_links, b_links = [], []
        for k in range(100):
            b_send = b_t0 + k * 0.5
            a_links.append(_link(b_send - 500 + lat - a_t0, b_send))
            a_send = a_t0 + 7 + k * 0.5 + 0.25
            b_links.append(_link(a_send + 500 + lat - b_t0, a_send))
        r1 = [_tick(0.0, run_mode="idle"), _tick(20.0), _tick(30.0)] + a_links
        r2 = [_tick(0.0, run_mode="idle"), _tick(15.0), _tick(23.0)] + b_links
        a = sim.Recording(_session(tmp_path / "a", r1, "1", mono_t0=a_t0))
        b = sim.Recording(_session(tmp_path / "b", r2, "2", mono_t0=b_t0))
        assert sim.link_offset(a, b) == pytest.approx(7.0, abs=1e-6)
        sim.Timeline([a, b])
        assert b.sync == "bluetooth"
        assert b.offset - a.offset == pytest.approx(7.0, abs=1e-6)

    def test_one_way_messages_are_close_enough(self, tmp_path):
        a_t0, b_t0 = 1000.0, 1507.0
        a_links = [_link(b_t0 + k - 500 + 0.02 - a_t0, b_t0 + k) for k in range(50)]
        a = sim.Recording(_session(tmp_path / "a", [_tick(0.0)] + a_links, mono_t0=a_t0))
        b = sim.Recording(_session(tmp_path / "b", [_tick(0.0)], "2", mono_t0=b_t0))
        assert sim.link_offset(a, b) == pytest.approx(7.0, abs=0.03)

    def test_frame_gives_previous_tick(self, tmp_path):
        rows = [_tick(0.0, state_name="seek"), _tick(0.1, state_name="has_ball")]
        tl = sim.Timeline([sim.Recording(_session(tmp_path / "s", rows))])
        (_, tk, prev), = tl.frame(0.1)
        assert prev["world"]["my_state"] == "seek"
        assert tk["world"]["my_state"] == "has_ball"

    def test_robot_drops_out_after_its_last_tick(self, tmp_path):
        tl = sim.Timeline([sim.Recording(_session(tmp_path / "s", [_tick(0.0), _tick(1.0)]))])
        assert tl.frame(1.2)
        assert not tl.frame(2.0)

    def test_step(self, tmp_path):
        rows = [_tick(0.0), _tick(0.1), _tick(0.2)]
        tl = sim.Timeline([sim.Recording(_session(tmp_path / "s", rows))])
        assert tl.step(0.0, +1) == pytest.approx(0.1)
        assert tl.step(0.1, -1) == pytest.approx(0.0)
        assert tl.step(0.2, +1) == pytest.approx(0.2)
        assert tl.step(0.0, -1) == pytest.approx(0.0)

    def test_field_turns_round_when_we_change_ends(self, tmp_path):
        rows = [_tick(0.0, run_mode="idle", slot_goal=None), _tick(1.0),
                _tick(2.0, slot_goal="high"), _tick(3.0, slot_goal="high")]
        tl = sim.Timeline([sim.Recording(_session(tmp_path / "s", rows))])
        assert tl.flip_at(0.0) is False
        assert tl.flip_at(1.5) is False
        assert tl.flip_at(2.5) is True


class TestScreenMapping:
    def test_own_goal_on_the_left(self):
        left, _ = sim.to_px(sim.FIELD_X / 2, 0.0, False)
        right, _ = sim.to_px(sim.FIELD_X / 2, sim.FIELD_Y, False)
        assert left < right
        assert sim.to_px(0.0, 0.0, True) == sim.to_px(sim.FIELD_X, sim.FIELD_Y, False)

    def test_heading(self):
        assert sim.heading_vec(0.0, False) == pytest.approx((1.0, 0.0))
        assert sim.heading_vec(90.0, False) == pytest.approx((0.0, 1.0), abs=1e-9)
        assert sim.heading_vec(0.0, True) == pytest.approx((-1.0, 0.0), abs=1e-9)


class TestRender:
    def _shape(self):
        return (sim.FIELD_H + sim.BAR_H, sim.FIELD_W + sim.PANEL_W, 3)

    def test_render_handles_sparse_ticks(self, tmp_path):
        rows = [
            _tick(0.0, pose=None),
            _tick(0.1, ball=[12.0, 700.0], ball_est=[900.0, 1300.0, 0.8, "kal"],
                  enemies=[{"x": 900.0, "y": 2000.0, "id": 3, "occluded": True},
                           {"x": None, "y": None}],
                  teammate_pos=[400.0, 300.0]),
            _tick(0.2, state_name=None, ball_est=[900.0, 1300.0, 0.8]),
        ]
        tl = sim.Timeline([sim.Recording(_session(tmp_path / "s", rows))])
        for t in (0.0, 0.1, 0.2, 5.0):
            assert sim.render(tl, t).shape == self._shape()

    def test_panel_grows_instead_of_clipping(self, tmp_path):
        rows = [_tick(i * 0.1, state_name=f"s{i}") for i in range(8)]
        tl = sim.Timeline([sim.Recording(_session(tmp_path / "a", rows, "1")),
                           sim.Recording(_session(tmp_path / "b", rows, "2"))])
        img, (x0, x1), bar_y = sim.compose(tl, 0.7)
        assert img.shape[1] == sim.FIELD_W + sim.PANEL_W
        assert img.shape[0] == bar_y + sim.BAR_H
        assert bar_y >= sim.FIELD_H
        assert x0 < x1

    def test_cached_frames_match_fresh_ones(self, tmp_path):
        rows = [_tick(i * 0.1, state_name=("seek", "approach")[i % 2]) for i in range(20)]
        p = _session(tmp_path / "s", rows)
        warm = sim.Timeline([sim.Recording(p)])
        for t in (0.3, 1.1, 0.5):
            sim.render(warm, t)
        cold = sim.Timeline([sim.Recording(p)])
        assert (sim.render(warm, 1.5) == sim.render(cold, 1.5)).all()

    def test_render_a_real_session(self, tmp_path, monkeypatch):
        """record with the real debug session, then play it back."""
        monkeypatch.chdir(tmp_path)
        ds.set_recording(True)
        try:
            sc = StrikerController()
            sc.state = C.seek
            for _i in range(20):
                with state._lock:
                    state._state.update({
                        "run_mode": "run", "slot_goal": "high",
                        "pose": (910.0, 1800.0, 180.0), "pose_t": time.monotonic(),
                        "ball": (0.0, 800.0), "enemies": []})
                sc.tick()
                ds.record_tick()
                time.sleep(0.01)
        finally:
            ds.set_recording(False)
            ds.finish()
        folder = os.path.join(tmp_path, "logs", os.listdir(tmp_path / "logs")[0])
        tl = sim.Timeline([sim.Recording(folder)])
        assert tl.flip_at(0.0) is True
        assert len(tl.recs[0].ticks) == 20
        img = sim.render(tl, tl.recs[0].offset + tl.recs[0].times[-1], speed=2.0, playing=False)
        assert img.shape == self._shape()


class TestMain:
    def test_no_folder_prints_usage(self, capsys):
        assert sim.main([]) == 1
        assert "python simulator.py" in capsys.readouterr().out

    def test_opencv_without_windows(self, tmp_path, monkeypatch, capsys):
        def no_gui(*_a, **_k):
            raise sim.cv2.error("The function is not implemented")

        monkeypatch.setattr(sim.cv2, "namedWindow", no_gui)
        assert sim.main([_session(tmp_path / "s", [_tick(0.0)])]) == 1
        assert "opencv-python" in capsys.readouterr().out

    def test_missing_folder(self, tmp_path, capsys):
        assert sim.main([str(tmp_path / "nope")]) == 1
        assert "[replay]" in capsys.readouterr().out
