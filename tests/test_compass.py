"""Tests for bot/compass.py: the gyroscope rate's sign and units, and what _compass_poll
publishes, including when only the gyro read fails.
"""

import math

import pytest

import bot.compass as compass
import bot.state as state


def _bare_compass(gyro=(0.0, 0.0, 0.0), quaternion=(0.0, 0.0, 0.0, 1.0)):
    """a Compass with a fake bno, skipping the hardware constructor"""
    c = compass.Compass.__new__(compass.Compass)

    class FakeBno:
        pass

    c.bno = FakeBno()
    c.bno.gyro = gyro
    c.bno.game_quaternion = quaternion
    c.bno.linear_acceleration = (0.0, 0.0, 0.0)
    return c


class TestReadGyroRate:
    def test_matches_reads_own_cw_convention(self):
        """read() negates the ccw+ quaternion yaw, so the gyro rate is negated the same way"""
        c = _bare_compass(gyro=(0.0, 0.0, math.radians(45.0))) # 45 deg/s raw, ccw+
        assert c.read_gyro_rate() == pytest.approx(-45.0)

    def test_zero_rate(self):
        assert _bare_compass(gyro=(0.0, 0.0, 0.0)).read_gyro_rate() == pytest.approx(0.0)

    def test_negative_raw_reads_positive_cw(self):
        c = _bare_compass(gyro=(0.0, 0.0, math.radians(-90.0)))
        assert c.read_gyro_rate() == pytest.approx(90.0)

    def test_only_the_z_axis_is_used(self):
        c = _bare_compass(gyro=(math.radians(200.0), math.radians(-300.0),
                                math.radians(10.0)))
        assert c.read_gyro_rate() == pytest.approx(-10.0)


class TestCompassPollPublishesGyroRate:
    """One poll at a time, called directly, so nothing runs in the background between tests."""

    def test_a_good_read_publishes_rate_and_timestamp(self):
        fake = _bare_compass(gyro=(0.0, 0.0, math.radians(-30.0)))
        compass._compass_poll(fake, fail_since=None)
        with state._lock:
            rate = state._state["imu_yaw_rate_dps"]
            t = state._state["imu_yaw_rate_t"]
        assert rate == pytest.approx(30.0)
        assert t is not None

    def test_a_failed_gyro_read_does_not_cost_the_heading_read(self):
        """a gyro read failing on its own still publishes the heading, which is all imu_fault
        and heading carry use
        """
        def always_fails():
            raise RuntimeError("bus error")

        fake = _bare_compass(quaternion=(0.0, 0.0, 0.0, 1.0))
        fake.read_gyro_rate = always_fails
        compass._compass_poll(fake, fail_since=None)
        with state._lock:
            heading = state._state["imu_heading"]
            rate = state._state["imu_yaw_rate_dps"]
        assert heading is not None
        assert rate is None

    def test_a_stale_rate_from_an_earlier_good_read_is_not_overwritten_by_a_glitch(self):
        """a failed gyro read leaves the last good rate alone; the staleness check in motion
        decides when it's too old
        """
        fake = _bare_compass(gyro=(0.0, 0.0, math.radians(-55.0)))
        compass._compass_poll(fake, fail_since=None) # first cycle: a good gyro read
        with state._lock:
            rate_after_first = state._state["imu_yaw_rate_dps"]
            t_after_first = state._state["imu_yaw_rate_t"]

        def always_fails():
            raise RuntimeError("bus error")

        fake.read_gyro_rate = always_fails
        compass._compass_poll(fake, fail_since=None) # second cycle: gyro glitches
        with state._lock:
            assert state._state["imu_yaw_rate_dps"] == rate_after_first == pytest.approx(55.0)
            assert state._state["imu_yaw_rate_t"] == t_after_first

    def test_returns_fail_since_unset_on_a_good_read(self):
        assert compass._compass_poll(_bare_compass(), fail_since=123.0) is None

    def test_a_bad_read_starts_the_fail_since_clock(self):
        def always_fails():
            raise RuntimeError("bus error")

        fake = _bare_compass()
        fake.read = always_fails
        before = compass.time.monotonic()
        fail_since = compass._compass_poll(fake, fail_since=None)
        assert fail_since is not None and fail_since >= before
