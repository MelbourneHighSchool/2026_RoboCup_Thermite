"""Shared fixtures for the bot test suite: a clean ROBOT_ID and a fresh shared-state world per
test, hardware-free (Motor is stubbed, no Pi needed).
"""

import os
import sys
from pathlib import Path

import pytest

# ROBOT_ID gates every bot.* import (wrong pins = wrong physical robot), and pytest may be
# invoked from anywhere, so anchor sys.path at public-repo/.
os.environ.setdefault("ROBOT_ID", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# tools/ holds the bench scripts and the fisheye fitter; their tests import them directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


@pytest.fixture()
def fresh_world(monkeypatch):
    """clean shared state + a no-hardware Motor stub, per test."""
    import bot.state as state

    clean = {
        "pose": None, "pose_t": None, "pose_live": None,
        "pose_imu": None, "imu_heading": None,
        "ball": None, "ball_blob": None,
        "ball_est": None, "remote_ball": None,
        "enemies": [], "enemy_vel": {},
        "teammate_pos": None, "teammate_pos_bt": None,
        "peer_state": None, "peer_pass_target": None,
        "pass_target": None, "my_state": None,
        "run_mode": "run", "slot_goal": "yellow",
        "capture_zone": None, "seek_util": None,
        "collision_t": None, "wheel_slip_trust": 1.0,
        "kickoff_role": None, "kickoff_until_t": None,
    }

    def world(**overrides):
        with state._lock:
            for k, v in clean.items():
                state._state[k] = v
            for k, v in overrides.items():
                state._state[k] = v
        return overrides

    # Motor needs no stub at all: Motor.drive/_stage with an empty cls.motors registry
    # records staged targets and writes nothing, and stopall/clear_faults iterate that
    # same empty registry. Stubbing the class instead would strip the constants
    # (DECEL_MAX_FRAC_PER_S, ...) a dozen motion-layer call sites read off it.

    yield world


@pytest.fixture(autouse=True)
def _reset_shared_state():
    """autouse: every test starts from a clean shared state, so leaked keys (a kickoff hold, a
    slot_goal, ...) can never order-depend a suite.
    """
    import bot.state as state

    with state._lock:
        for k in list(state._state):
            if not k.startswith("hsv_"):
                state._state[k] = None
        state._state["enemies"] = []
        state._state["enemy_vel"] = {}
        state._state["lidar_pts"] = []
        state._state["run_mode"] = "idle"
    yield
    with state._lock:
        for k in list(state._state):
            if not k.startswith("hsv_"):
                state._state[k] = None
        state._state["enemies"] = []
        state._state["enemy_vel"] = {}
        state._state["lidar_pts"] = []
        state._state["run_mode"] = "idle"
