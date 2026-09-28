"""Tests for the robot's boot-time and bench-flag paths that match play never
exercises: the --lidarlog/--capturelog/--motionlog loggers, the --kickoff start
path, the native C++ cores behind their real call sites, and a whole-package
undefined-name check that catches the next one of these before the robot does."""

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import bot.state as state

PUBLIC_REPO = Path(__file__).resolve().parents[1]


class TestUndefinedNames:
    def test_no_undefined_names_in_runtime_or_tools(self):
        """every name the robot reads must resolve: pyflakes finds the NameError before a boot
        flag does.
        """
        pytest.importorskip("pyflakes")
        out = subprocess.run(
            [sys.executable, "-m", "pyflakes", "bot", "tools"],
            cwd=PUBLIC_REPO, capture_output=True, text=True).stdout
        bad = [line for line in out.splitlines() if "undefined name" in line]
        assert not bad, "\n".join(bad)


class TestLogConsts:
    def test_lidar_log_consts_resolve(self):
        from bot.logs import _lidar_log_consts
        consts = _lidar_log_consts()
        assert consts["field_x"] > 0 and consts["robot_radius_mm"] > 0

    def test_capture_log_consts_resolve(self):
        from bot.logs import _capture_log_consts
        consts = _capture_log_consts()
        assert consts["capture_cone_half_deg"] > 0 and consts["base_speed"] > 0

    def test_motion_log_consts_resolve(self):
        from bot.logs import _motion_log_consts
        consts = _motion_log_consts()
        assert consts["dwibble_speed"] > 0 and consts["base_speed"] > 0


class TestBenchLoggers:
    def test_loggers_import_from_the_package(self):
        """main.py imports all three from bot/, so they must ship inside it."""
        from bot.capture_debug import CaptureLogger # noqa: F401
        from bot.lidar_debug import LidarLogger # noqa: F401
        from bot.motion_debug import MotionLogger # noqa: F401

    def test_capture_logger_round_trip(self, tmp_path):
        from bot.capture_debug import CaptureLogger
        from bot.logs import _capture_log_consts
        log = CaptureLogger(str(tmp_path / "cap.txt"), duration_s=60.0)
        log.header(_capture_log_consts())
        log.tick("seek", "cone", 900.0, 1200.0, 0.0, 900.0, 1500.0,
                 (0.0, -50.0), 0.0, 0.6)
        log.event("captured", "camera route")
        path = log.close()
        text = Path(path).read_text()
        assert "[tick]" in text and "ballvy=-50.0" in text and "drive_cmd=0.600" in text
        assert "[event]" in text and "captured camera route" in text
        assert "distance to ball mm: mean=  300.0" in text

    def test_motion_log_thread_runs_and_closes(self, tmp_path, monkeypatch):
        """one pass of the --motionlog poll thread with nothing registered: every field it
        reads must resolve.
        """
        import bot.logs as logs
        from bot.motion_debug import MotionLogger
        log = MotionLogger(str(tmp_path / "motion.txt"), duration_s=0.0)
        log.header(logs._motion_log_consts())
        monkeypatch.setattr(state, "_motion_log", log)
        logs._motion_log_thread()
        assert state._motion_log is None
        assert "[tick]" in (tmp_path / "motion.txt").read_text()


class TestKickoffStart:
    def test_main_reads_the_hold_through_controllers(self):
        """maybe_start stamps kickoff_until_t from controllers.kickoff_hold_s; without the
        module import, --kickoff never held.
        """
        import bot.main as main
        import bot.controllers as controllers
        assert main.controllers is controllers
        assert controllers.kickoff_hold_s > 0


class TestNativeParity:
    """the native cores, tested through the same calls the runtime makes."""

    def test_mcl_sensor_weights_native_matches_numpy(self, monkeypatch):
        import bot.localisation as loc
        if loc.mcl_native is None:
            pytest.skip("mcl_native not built on this machine (native/build.sh)")
        mcl = loc.MCL()
        mcl.seed_gaussian((900.0, 1200.0, 30.0))
        monkeypatch.setattr(mcl, "prior_heading_deg", lambda: None)
        rng = np.random.default_rng(7)
        pts = np.column_stack([rng.normal(0, 800, 60), rng.normal(0, 800, 60)])
        w_native = mcl.sensor_weights(pts)
        monkeypatch.setattr(loc, "mcl_native", None)
        w_numpy = mcl.sensor_weights(pts)
        assert np.max(np.abs(w_native - w_numpy)) < 1e-9

    def test_lidar_parse_native_matches_python(self, monkeypatch):
        import struct
        import bot.lidar as lidar
        if lidar.lidar_native is None:
            pytest.skip("lidar_native not built on this machine (native/build.sh)")
        buf = bytearray(47)
        buf[0], buf[1] = 0x54, 0x2C
        struct.pack_into("<H", buf, 2, 3600)
        struct.pack_into("<H", buf, 4, 1000)
        for i in range(12):
            struct.pack_into("<HB", buf, 6 + i * 3, 500 + 37 * i, 200)
        struct.pack_into("<H", buf, 42, 2100)
        buf[46] = lidar.LidarReader._crc8(bytes(buf[:-1]))
        n_pts, n_speed = lidar.LidarReader._parse_packet(bytes(buf))
        monkeypatch.setattr(lidar, "lidar_native", None)
        p_pts, p_speed = lidar.LidarReader._parse_packet(bytes(buf))
        assert n_speed == p_speed and len(n_pts) == len(p_pts) == 12
        for a, b in zip(n_pts, p_pts):
            assert abs(a.angle_deg - b.angle_deg) < 1e-9
            assert (a.distance_mm, a.intensity) == (b.distance_mm, b.intensity)


class TestWheelOdometryWrap:
    def test_counter_wrap_reads_as_a_small_step(self, monkeypatch):
        """a 32-bit QDR position wrapping (or a wheel reversing past zero) is a step of a few
        units, not a 4-billion-unit jump into the lidar deskew."""
        import bot.odometry as od
        from bot.hardware import Motor

        class FakeMotor:
            def __init__(self, pos):
                self.pos = pos

            def read_qdr(self):
                return (self.pos, 0, 0, 0)

        near_top = 2 ** 32 - 1000
        motors = {n: FakeMotor(near_top) for n in od.WheelOdometry.NAMES}
        monkeypatch.setattr(Motor, "motors", motors)
        odom = od.WheelOdometry()
        assert odom.poll(0.0) is None # first call only sets the baseline
        for m in motors.values():
            m.pos = 2000 # 3000 units later, past the wrap
        fwd, right = odom.poll(0.1)
        step_mm = 3000 / od.POS_RAW_PER_OUTPUT_REV * math.pi * od.wheel_diameter_mm
        assert abs(fwd) < 1.0 and abs(right) <= 2.0 * step_mm

    def test_step_helper(self):
        from bot.odometry import _qdr_step
        assert _qdr_step(5, 2 ** 32 - 5) == 10
        assert _qdr_step(2 ** 32 - 5, 5) == -10
        assert _qdr_step(-3, 4) == -7 # signed counters diff the same way


class TestImuHealthReport:
    def test_absent_imu_is_not_reported_ok(self, monkeypatch):
        """the compass thread reports "disabled" once when no BNO08x answers; the health poll
        used to overwrite it with "ok" every 0.2 s."""
        import bot.diagnostics as diag
        monkeypatch.setattr(diag, "_health", {})
        diag._report_health("imu", "disabled (no BNO08x)")

        def stop(_):
            raise StopIteration
        monkeypatch.setattr(diag.time, "sleep", stop)
        with pytest.raises(StopIteration):
            diag._health_thread()
        assert diag._health["imu"] == "disabled (no BNO08x)"


class TestRobotConfigs:
    def test_both_robots_define_every_motor(self):
        """main.py and motorcheck.py take pins from the selected config and calibration from
        the shared drive_config, so each config must carry the full pin set (motorcheck used
        to hardcode bot 1's)."""
        from bot import bot1_config, bot2_config, drive_config
        for cfg in (bot1_config, bot2_config):
            assert set(cfg.MOTOR_PINS) == {"nw", "ne", "sw", "se", "dwibble"}
            assert len(set(cfg.MOTOR_PINS.values())) == 5 # no address used twice
        assert set(drive_config.MOTOR_CALIB) == {"nw", "ne", "sw", "se"}

    def test_robots_differ_only_in_addresses_and_camera(self):
        """the two robots are the same build: their configs may differ in motor I2C
        addresses and camera calibration, plus the boot role, and nothing else."""
        from bot import bot1_config, bot2_config
        names = lambda m: {k for k in vars(m) if not k.startswith("_")}
        assert names(bot1_config) == names(bot2_config) == {
            "handle_exclusion_deg", "crop_top", "crop_bottom", "crop_left",
            "exclusion_inner_frac", "exclusion_outer_frac", "cam_bearing_offset_deg",
            "MOTOR_PINS", "DEFAULT_ROLE"}


class TestPossessionTwoFeeders:
    def test_out_of_order_sample_never_drains_evidence(self):
        """the roller monitor and the mouth camera both feed BallPossession; a sample stamped
        before the last one used to produce a negative dt that drained loaded evidence."""
        from bot.dwibbler import BallPossession
        p = BallPossession()
        p.update(10.00, True)
        p.update(10.05, True)
        before = p._load_s
        p.update(10.03, True) # the other thread's sample, stamped slightly earlier
        assert p._load_s >= before
