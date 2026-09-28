"""Tests for bench_drive_sweep's measurement helpers.

The bench script itself runs on the robot, but its loops and its unit conversion are
pure enough to pin here against a scripted drive: a wrong conversion or a roll-out
that drops samples would quietly corrupt the very numbers BENCH_TEST_CHECKLIST
section 7 asks the bench run to produce.
"""

import pytest


class FakeDrive:
    """a scripted stand-in for WheelDrive: one measured speed per commanded fraction."""

    def __init__(self, speed_for):
        self.speed_for = speed_for # fraction -> measured mm/s
        self.commanded = []
        self.stopped = 0
        self._reading = None

    def command(self, frac, bearing=0.0):
        self.commanded.append(frac)
        self._reading = self.speed_for(frac)

    def stop(self):
        self.stopped += 1
        self._reading = None

    def speed_mms(self):
        return self._reading


class RolloutDrive(FakeDrive):
    """a drive whose wheels travel a fixed displacement per poll, so the roll-out integral is
    exact.
    """

    def __init__(self, steps, cruise_mms=800.0):
        super().__init__(lambda frac: cruise_mms)
        self.steps = list(steps) # (forward_mm, right_mm, dt_s) per poll

    def delta(self):
        if self.steps:
            return self.steps.pop(0)
        return 0.0, 0.0, 0.01


def _bench():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    import bench_drive_sweep
    return bench_drive_sweep


def test_conversion_round_trips():
    bench = _bench()

    assert bench.frac_to_mms(0.5, 7346.0) == 3673.0
    assert bench.mms_to_frac(3673.0, 7346.0) == 0.5


def test_ramp_steps_the_command_and_reports_the_peak(capsys):
    bench = _bench()

    drive = FakeDrive(lambda frac: frac * 1000.0)

    peak = bench.ramp(drive, 0.2, 0.6, 0.2, 0.0, 4000.0)

    assert [round(frac, 6) for frac in drive.commanded] == [0.2, 0.4, 0.6]
    assert peak == 600.0
    assert drive.stopped == 1
    out = capsys.readouterr().out
    assert "peak measured: 600 mm/s" in out
    assert "0.150 of full scale" in out


def test_ramp_stops_the_robot_when_the_wheels_cannot_be_read(capsys):
    bench = _bench()

    drive = FakeDrive(lambda frac: None)

    peak = bench.ramp(drive, 0.5, 0.5, 0.1, 0.0, 4000.0)

    assert peak == 0.0
    assert drive.stopped == 1 # a failed read must never leave the command running
    assert "no QDR read" in capsys.readouterr().out


def test_stopping_distance_integrates_every_rollout_sample(capsys):
    bench = _bench()

    # 30 mm at 600 mm/s, 20 mm at 400 mm/s, then 1 mm at 20 mm/s == the stop threshold
    drive = RolloutDrive([(30.0, 0.0, 0.05), (20.0, 0.0, 0.05), (1.0, 0.0, 0.05)])

    rolled, cruise = bench.stopping_distance(drive, 0.8, 0.0, 20.0, 5.0)

    assert rolled == pytest.approx(51.0)
    assert cruise == 800.0
    assert drive.stopped == 1
    assert "roll-out=51.0 mm" in capsys.readouterr().out


def test_stopping_distance_gives_up_at_the_timeout():
    bench = _bench()

    drive = RolloutDrive([(30.0, 0.0, 0.05)]) # never slows below threshold

    rolled, _ = bench.stopping_distance(drive, 0.8, 0.0, 20.0, 0.05)

    assert rolled > 0.0 # partial travel is reported, not silently dropped


def test_hold_speed_passes_within_tolerance_and_fails_otherwise():
    bench = _bench()

    reaching = FakeDrive(lambda frac: 0.8 * 4000.0 * 0.99)
    assert bench.hold_speed(reaching, 0.8, 0.2, 0.05, 0.0, 4000.0) is True
    assert reaching.stopped == 1

    crawling = FakeDrive(lambda frac: 100.0)
    assert bench.hold_speed(crawling, 0.8, 0.1, 0.05, 0.0, 4000.0) is False
    assert crawling.stopped == 1


def test_set_drive_current_limits_the_drive_motors_only(monkeypatch):
    bench = _bench()
    from bot.MotorFuncs_Proto1 import FOC_LSB_PER_AMP, Motor

    written = []

    class _Md:
        def set_current_limit_foc(self, raw):
            written.append(raw)

    class _Motor:
        def __init__(self):
            self.md = _Md()
            self.current_limit_amps = 8.0

    motors = {name: _Motor() for name in bench.DRIVE_NAMES}
    motors["dwibble"] = _Motor()
    monkeypatch.setattr(Motor, "motors", motors)

    bench.set_drive_current(2.0)

    assert written == [int(2.0 * FOC_LSB_PER_AMP)] * 4
    assert motors["nw"].current_limit_amps == 2.0
    assert motors["dwibble"].current_limit_amps == 8.0 # the roller is not part of the sweep


def test_help_and_bad_arguments_work_without_the_robot_import_chain(capsys):
    """the robot imports are deferred until the arguments are known good, so this runs on a
    desktop.
    """
    bench = _bench()

    with pytest.raises(SystemExit) as help_exit:
        bench.main(["--help"])
    assert help_exit.value.code == 0
    assert "top speed vs command" in capsys.readouterr().out

    with pytest.raises(SystemExit) as bad_exit:
        bench.main(["ramp", "--start-frac", "0"])
    assert bad_exit.value.code == 2

    with pytest.raises(SystemExit) as bad_current_exit:
        bench.main(["current", "--max-current", "9"])
    assert bad_current_exit.value.code == 2
