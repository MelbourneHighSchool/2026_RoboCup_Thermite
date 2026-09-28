"""Byte-level parity tests for the motor driver protocol, and for the motor bring-up sequence.

bot/rotom.py is a Python port of the vendor's PowerfulBLDC driver library, checked against the vendor's C++
source, and bot/MotorFuncs_Proto1.py brings the motors up in a fixed order. These tests pin the wire format
opcode by opcode and the bring-up order step by step, so a drift fails here instead of on the robot.

The hardware is faked: micropython/busio/adafruit_bus_device are the Pi-only import chain the driver needs, so
they are stubbed into sys.modules for the duration of this module's import.
"""

import os
import struct
import sys
import types

import pytest


def _stub_micropython_stack():
    """inject the Pi-only import chain rotom.py needs, so the driver can be imported and driven
    on a dev box.
    """
    if "micropython" not in sys.modules:
        micropython = types.ModuleType("micropython")
        micropython.const = lambda value: value
        sys.modules["micropython"] = micropython
    if "busio" not in sys.modules:
        busio = types.ModuleType("busio")

        class I2C: # noqa: D401 (inert placeholder, the tests use a separate bus below)
            """stand-in for busio.I2C; real instances are never created in this file."""

        busio.I2C = I2C
        sys.modules["busio"] = busio
    if "adafruit_bus_device" not in sys.modules:
        package = types.ModuleType("adafruit_bus_device")
        i2c_device = types.ModuleType("adafruit_bus_device.i2c_device")

        class I2CDevice:
            """records what the driver writes and replays canned reads."""

            def __init__(self, bus, address):
                self.bus = bus
                self.address = address

            def write(self, buffer, *, end=0, start=0):
                self.bus.writes.append(bytes(buffer[start:end]))

            def readinto(self, buffer, *, end=0, start=0):
                data = self.bus.next_read(len(buffer[start:end]))
                buffer[start:end] = data

        i2c_device.I2CDevice = I2CDevice
        package.i2c_device = i2c_device
        sys.modules["adafruit_bus_device"] = package
        sys.modules["adafruit_bus_device.i2c_device"] = i2c_device


_stub_micropython_stack()

import bot.rotom as rotom # noqa: E402


class FakeBus:
    """the smallest I2C bus the driver's interface needs: a write log and a read queue."""

    def __init__(self, reads=()):
        self.writes = []
        self.reads = list(reads) # each entry: a bytes object as it appears on the wire

    def next_read(self, length):
        payload = self.reads.pop(0) if self.reads else bytes(length)
        assert len(payload) == length, f"read of {length} bytes, canned payload {payload!r}"
        return payload


def _driver(reads=()):
    bus = FakeBus(reads)
    return rotom.PowerfulBLDCDriver(bus, 0x19), bus


def _le32(value):
    return struct.pack("<i", value)


def _ule32(value):
    return struct.pack("<I", value)


class TestWireFormat:
    """opcode + payload for every command the driver exposes, little endian, as the vendor's
    C++ writes them.
    """

    def test_firmware_version_read(self):
        driver, bus = _driver([_ule32(3)])
        assert driver.get_firmware_version() == 3
        assert bus.writes == [b"\x00"]

    def test_clear_faults(self):
        driver, bus = _driver()
        driver.clear_faults()
        assert bus.writes == [b"\x01"]

    @pytest.mark.parametrize("opcode, method, value", [
        (0x10, "set_voltage", 300),
        (0x11, "set_torque", -1),
        (0x12, "set_speed", 123456),
        (0x33, "set_current_limit_foc", 524288),
        (0x34, "set_speed_limit", 546133333),
        (0x30, "set_ELECANGLEOFFSET", 1451095040),
        (0x31, "set_EAOPERSPEED", -7),
        (0x32, "set_SINCOSCENTRE", 1227),
    ])
    def test_32bit_commands(self, opcode, method, value):
        driver, bus = _driver()
        getattr(driver, method)(value)
        assert bus.writes == [bytes([opcode]) + _le32(value)]

    def test_position_includes_the_electrical_angle_byte(self):
        driver, bus = _driver()
        driver.set_position(0xDEADBEEF, 200)
        assert bus.writes == [b"\x13" + _ule32(0xDEADBEEF) + bytes([200])]

    def test_operating_mode_packs_sensor_into_the_high_nibble(self):
        driver, bus = _driver()
        driver.configure_operating_mode_and_sensor(3, 1)
        driver.configure_operating_mode_and_sensor(15, 1)
        assert bus.writes == [b"\x20\x13", b"\x20\x1f"]

    def test_command_mode(self):
        driver, bus = _driver()
        driver.configure_command_mode(2)
        driver.configure_command_mode(12)
        assert bus.writes == [b"\x21\x02", b"\x21\x0c"]

    def test_calibration_start_stop(self):
        driver, bus = _driver()
        driver.start_calibration()
        driver.stop_calibration()
        assert bus.writes == [b"\x38\x01", b"\x38\x00"]

    def test_calibration_options_four_fields(self):
        driver, bus = _driver()
        driver.set_calibration_options(300, 2097152, 50000, 500000)
        assert bus.writes == [b"\x3a" + _ule32(300) + _le32(2097152)
                              + _ule32(50000) + _ule32(500000)]

    def test_pid_commands(self):
        driver, bus = _driver()
        driver.set_iq_pid_constants(1500, 200)
        driver.set_id_pid_constants(1500, 200)
        driver.set_speed_pid_constants(4e-2, 4e-4, 3e-2)
        driver.set_position_pid_constants(275, 0, 0)
        driver.set_position_region_boundary(250000)
        assert bus.writes[0] == b"\x40" + _le32(1500) + _le32(200)
        assert bus.writes[1] == b"\x41" + _le32(1500) + _le32(200)
        assert bus.writes[2] == b"\x42" + struct.pack("<fff", 4e-2, 4e-4, 3e-2)
        assert bus.writes[3] == b"\x43" + struct.pack("<fff", 275.0, 0.0, 0.0)
        assert bus.writes[4] == b"\x44" + struct.pack("<f", 250000.0)

    def test_quick_data_readout_takes_no_command_byte(self):
        """The QDR is a bare read: the vendor's updateQuickDataReadout does requestFrom() with
        no write at all.
        """
        driver, bus = _driver([_ule32(5) + _le32(-6) + b"\x00\x00"])
        driver.update_quick_data_readout()
        assert bus.writes == []

    def test_calibration_read_sends_its_command_byte(self):
        """isCalibrationFinished isn't a bare read: the vendor writes 0x39 first, then reads 9
        bytes.
        """
        driver, bus = _driver([b"\xff" + _ule32(7) + _le32(-8)])
        driver.is_calibration_finished()
        assert bus.writes == [b"\x39"]


class TestQuickDataReadout:
    """format 0 layout: uint32 position, int32 speed, ERROR1, ERROR2: the vendor's types, not
    ours.
    """

    def test_position_is_unsigned_and_speed_is_signed(self):
        driver, _ = _driver([_ule32(0xFFFFFFFF) + _le32(-1) + bytes([3, 4])])
        driver.update_quick_data_readout()
        assert driver.get_position_QDR() == 4294967295 # not -1: the counter is unsigned
        # speed is signed, so a reversal reads negative
        assert driver.get_speed_QDR() == -1
        assert driver.get_ERROR1_QDR() == 3
        assert driver.get_ERROR2_QDR() == 4

    def test_position_past_two_to_the_31_stays_positive(self):
        """a signed read of this same value was a bug the register layout explains: 2^31 reads
        back negative.
        """
        driver, _ = _driver([_ule32(2 ** 31 + 5) + _le32(0) + b"\x00\x00"])
        driver.update_quick_data_readout()
        assert driver.get_position_QDR() == 2 ** 31 + 5


class TestCalibrationRead:
    def test_unfinished_calibration_reports_zero_not_false(self):
        """the read returns numbers, so an unfinished calibration is 0, and returning False made
        an int function lie.
        """
        driver, _ = _driver([bytes([0]) + _ule32(1451095040) + _le32(1227)])
        assert driver.get_calibration_ELECANGLEOFFSET() == 0
        assert driver.get_calibration_SINCOSCENTRE() == 0

    def test_finished_calibration_reports_the_values(self):
        payload = bytes([255]) + _ule32(1451095040) + _le32(1227)
        driver, _ = _driver([payload, payload, payload])
        assert driver.is_calibration_finished() is True
        assert driver.get_calibration_ELECANGLEOFFSET() == 1451095040
        assert driver.get_calibration_SINCOSCENTRE() == 1227

    def test_sincoscentre_is_signed(self):
        """the vendor's getter returns int32; its centre voltage is a signed field, so
        0x80000000 is negative.
        """
        driver, _ = _driver([bytes([255]) + _ule32(1) + _le32(-2147483648)])
        assert driver.get_calibration_SINCOSCENTRE() == -2147483648

    def test_sincoscentre_defaults_to_zero_before_any_read(self):
        driver, _ = _driver()
        assert driver.SINCOSCENTRE == 0


class TestMotorBringUp:
    """Motor.__init__'s configuration order and firmware gate, against a recording driver stub."""

    @pytest.fixture()
    def stub(self, monkeypatch):
        import bot.MotorFuncs_Proto1 as MotorFuncs
        import bot.rotom

        calls = []

        class StubDriver:
            def __init__(self, i2c, address):
                self.address = address
                self.firmware = 3

            def __getattr__(self, item):
                def record(*args, **kwargs):
                    calls.append((item, args))
                    if item == "get_firmware_version":
                        if self.firmware is None:
                            raise OSError("bus error")
                        return self.firmware
                    if item in ("get_position_QDR", "get_speed_QDR",
                                "get_ERROR1_QDR", "get_ERROR2_QDR"):
                        return 0
                    return None
                return record

        monkeypatch.setattr(bot.rotom, "PowerfulBLDCDriver", StubDriver)
        yield calls, StubDriver
        MotorFuncs.Motor.motors.clear()

    def test_configuration_order(self, stub):
        """firmware read first, then stop-in-torque-mode, then every limit/PID, then speed mode
        last.
        """
        calls, _ = stub
        import bot.MotorFuncs_Proto1 as MotorFuncs
        MotorFuncs.Motor(0x19, object(), name="nw")
        order = [name for name, _ in calls]
        assert order == [
            "get_firmware_version",
            "configure_command_mode", # torque mode: safe to reconfigure from
            "configure_operating_mode_and_sensor", # FOC + sin/cos, stopped
            "set_torque", "set_speed", # both command registers zeroed
            "set_current_limit_foc",
            "set_id_pid_constants", "set_iq_pid_constants",
            "set_speed_pid_constants", "set_position_pid_constants",
            "set_position_region_boundary", "set_speed_limit",
            "set_calibration_options",
            "configure_command_mode", # ...and only now the speed mode the drive runs in
        ]
        assert calls[1][1] == (2,)
        assert order.count("configure_command_mode") == 2
        assert calls[-1][1] == (12,)

    def test_drive_motors_get_the_drive_current_limit(self, stub):
        calls, _ = stub
        import bot.MotorFuncs_Proto1 as MotorFuncs
        MotorFuncs.Motor(0x19, object(), name="nw")
        limit = next(args[0] for name, args in calls if name == "set_current_limit_foc")
        assert limit == int(MotorFuncs.drive_current_limit_amps * MotorFuncs.FOC_LSB_PER_AMP)
        assert limit == 8 * 65536

    def test_roller_keeps_its_own_current_limit(self, stub):
        """the roller is still driven in speed mode (see bot/dwibbler.py's mode note), so it
        keeps its limit.
        """
        calls, _ = stub
        import bot.MotorFuncs_Proto1 as MotorFuncs
        motor = MotorFuncs.Motor(0x19, object(), name="dwibble")
        limit = next(args[0] for name, args in calls if name == "set_current_limit_foc")
        assert limit == int(MotorFuncs.roller_current_limit_amps * MotorFuncs.FOC_LSB_PER_AMP)
        assert motor.current_limit_amps == MotorFuncs.roller_current_limit_amps

    def test_explicit_current_limit_overrides_the_default(self, stub):
        calls, _ = stub
        import bot.MotorFuncs_Proto1 as MotorFuncs
        MotorFuncs.Motor(0x19, object(), name="nw", current_limit_amps=2.5)
        limit = next(args[0] for name, args in calls if name == "set_current_limit_foc")
        assert limit == int(2.5 * MotorFuncs.FOC_LSB_PER_AMP)

    def test_unknown_firmware_refuses_to_configure(self, stub):
        calls, StubDriver = stub
        import bot.MotorFuncs_Proto1 as MotorFuncs

        original = StubDriver.__init__

        def wrong_firmware(self, i2c, address):
            original(self, i2c, address)
            self.firmware = 2

        StubDriver.__init__ = wrong_firmware
        with pytest.raises(MotorFuncs.MotorFirmwareError):
            MotorFuncs.Motor(0x19, object(), name="nw")
        # the gate runs before any configuration write
        assert [name for name, _ in calls] == ["get_firmware_version"]

    def test_unreadable_firmware_warns_and_still_brings_the_motor_up(self, stub, capsys):
        calls, StubDriver = stub
        import bot.MotorFuncs_Proto1 as MotorFuncs

        original = StubDriver.__init__

        def silent(self, i2c, address):
            original(self, i2c, address)
            self.firmware = None

        StubDriver.__init__ = silent
        motor = MotorFuncs.Motor(0x19, object(), name="nw")
        assert motor.firmware_version is None
        assert "firmware version unreadable" in capsys.readouterr().out
        assert calls[-1][0] == "configure_command_mode"


class _CalibDriver:
    """records every driver call; is_calibration_finished() answers from a scripted list."""

    def __init__(self, i2c, address):
        self._address = address
        self.answers = []
        self.calls = []
        self.fail_exit = False

    def is_calibration_finished(self):
        self.calls.append(("is_calibration_finished", ()))
        return self.answers.pop(0) if self.answers else False

    def get_calibration_ELECANGLEOFFSET(self):
        self.calls.append(("get_calibration_ELECANGLEOFFSET", ()))
        return 1451095040

    def get_calibration_SINCOSCENTRE(self):
        self.calls.append(("get_calibration_SINCOSCENTRE", ()))
        return 1227

    def set_torque(self, value):
        self.calls.append(("set_torque", (value,)))
        if self.fail_exit:
            raise OSError("bus write failed")

    def __getattr__(self, item):
        def record(*args, **kwargs):
            self.calls.append((item, args))
            if item == "get_firmware_version":
                return 3
            return None
        return record


@pytest.fixture()
def calib_driver(monkeypatch):
    """install _CalibDriver as the motor driver and hand back the last instance built."""
    import bot.rotom
    import bot.MotorFuncs_Proto1 as MotorFuncs

    built = []

    def factory(i2c, address):
        driver = _CalibDriver(i2c, address)
        built.append(driver)
        return driver

    monkeypatch.setattr(bot.rotom, "PowerfulBLDCDriver", factory)
    MotorFuncs.Motor.motors.clear()
    yield built
    MotorFuncs.Motor.motors.clear()


class TestMotorCalibration:
    """Motor.calib()'s timeout, its guaranteed exit from calibration mode, and the saved-file
    fallback.
    """

    def test_calibration_returns_the_pair_and_restores_speed_mode(self, calib_driver):
        import bot.MotorFuncs_Proto1 as MotorFuncs
        motor = MotorFuncs.Motor(0x1A, object(), name="nw")
        driver = calib_driver[-1]
        driver.answers = [False, False, True]

        result = motor.calib(timeout_s=5.0)

        assert result == [1451095040, 1227]
        # the calibration exit sequence, then the speed mode the drive runs in
        assert driver.calls[-5:] == [
            ("set_torque", (0,)),
            ("configure_command_mode", (2,)),
            ("configure_operating_mode_and_sensor", (3, 1)),
            ("set_torque", (0,)),
            ("configure_command_mode", (12,)),
        ]

    def test_calibration_timeout_raises_and_still_leaves_calibration_mode(self, calib_driver):
        import bot.MotorFuncs_Proto1 as MotorFuncs
        motor = MotorFuncs.Motor(0x1A, object(), name="nw")
        driver = calib_driver[-1]
        driver.answers = [] # never reports finished

        with pytest.raises(TimeoutError):
            motor.calib(timeout_s=0.03)

        after_start = driver.calls[[name for name, _ in driver.calls].index("start_calibration"):]
        assert ("configure_operating_mode_and_sensor", (3, 1)) in after_start
        assert ("configure_command_mode", (2,)) in after_start
        assert ("set_torque", (0,)) in after_start

    def test_failed_calibration_exit_is_reported_not_swallowed(self, calib_driver):
        import bot.MotorFuncs_Proto1 as MotorFuncs
        motor = MotorFuncs.Motor(0x1A, object(), name="nw")
        driver = calib_driver[-1]
        driver.answers = [True]
        driver.fail_exit = True # the exit write itself now fails (set_torque)

        with pytest.raises(MotorFuncs.MotorCommunicationError):
            motor.calib(timeout_s=5.0)

    def test_calibrate_all_saves_results_under_the_driver_address(self, calib_driver, tmp_path):
        import bot.MotorFuncs_Proto1 as MotorFuncs
        path = str(tmp_path / "motor_calibration.json")
        MotorFuncs.Motor(0x1A, object(), name="nw").calibset([1, 2])
        calib_driver[-1].answers = [True]

        configs = MotorFuncs.Motor.calibrate_all(path=path)

        assert configs["nw"] == [1451095040, 1227]
        assert MotorFuncs.load_calibration(path) == {0x1A: [1451095040, 1227]}


class TestCalibrationFile:
    """save_calibration/load_calibration/resolve_calibration: the address-keyed file the
    motor init loads its calibration from.
    """

    def test_round_trips_and_leaves_no_temp_file(self, tmp_path):
        import bot.MotorFuncs_Proto1 as MotorFuncs
        path = str(tmp_path / "motor_calibration.json")

        MotorFuncs.save_calibration({30: [3, 4], 26: [1, 2]}, path)

        assert not os.path.exists(path + ".tmp")
        assert MotorFuncs.load_calibration(path) == {26: [1, 2], 30: [3, 4]}

    def test_missing_or_malformed_file_is_not_a_dependency(self, tmp_path):
        import bot.MotorFuncs_Proto1 as MotorFuncs
        assert MotorFuncs.load_calibration(str(tmp_path / "absent.json")) == {}

        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert MotorFuncs.load_calibration(str(bad)) == {}

        odd = tmp_path / "odd.json"
        odd.write_text('{"motors": [{"address": "x"}, {"elecangleoffset": 1}]}', encoding="utf-8")
        assert MotorFuncs.load_calibration(str(odd)) == {}

    def test_resolve_prefers_the_saved_file_per_motor(self, tmp_path):
        import bot.MotorFuncs_Proto1 as MotorFuncs
        path = str(tmp_path / "motor_calibration.json")
        MotorFuncs.save_calibration({30: [11, 12]}, path)

        resolved = MotorFuncs.Motor.resolve_calibration(
            {"nw": [1, 2], "se": [3, 4]}, {"nw": 26, "se": 30}, path)

        assert resolved == {"nw": [1, 2], "se": [11, 12]}

    def test_resolve_falls_back_to_every_default_without_a_file(self, tmp_path):
        import bot.MotorFuncs_Proto1 as MotorFuncs
        resolved = MotorFuncs.Motor.resolve_calibration(
            {"nw": [1, 2], "se": [3, 4]}, {"nw": 26, "se": 30},
            str(tmp_path / "absent.json"))

        assert resolved == {"nw": [1, 2], "se": [3, 4]}
