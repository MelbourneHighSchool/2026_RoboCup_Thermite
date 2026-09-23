# SPDX-FileCopyrightText: 2017 Scott Shawcroft, written for Adafruit Industries
# SPDX-FileCopyrightText: Copyright (c) 2024 Emily Trau for Steel Bar Robotics
#
# SPDX-License-Identifier: MIT
"""I2C driver for the SteelBar powerful BLDC motor controller (steelbar_powerful_bldc_driver)."""

import struct
from micropython import const
from busio import I2C
from adafruit_bus_device import i2c_device

__version__ = "0.0.0+auto.0"
__repo__ = "https://github.com/SteelBarRobotics/SteelBar_CircuitPython_powerful_bldc_driver.git"


_DEFAULT_I2C_ADDR = const(0x19)

OPERATING_MODE_TRAPEZOIDAL = const(1)
OPERATING_MODE_SINUSOIDAL = const(2)
OPERATING_MODE_FIELD_ORIENTED_CONTROL = const(3)
OPERATING_MODE_CALIBRATION = const(15)

SENSOR_TYPE_SIN_COS_ENCODER = const(1)
SENSOR_TYPE_INCREMENTAL_ENCODER = const(2)
SENSOR_TYPE_SENSORLESS = const(3)

COMMAND_MODE_VOLTAGE = const(1)
COMMAND_MODE_TORQUE = const(2)
COMMAND_MODE_SPEED = const(12)
COMMAND_MODE_POSITION = const(13)
COMMAND_MODE_CALIBRATION = const(15)


class PowerfulBLDCDriver:
    def __init__(self, i2c_bus: I2C, address: int = _DEFAULT_I2C_ADDR) -> None:
        self._i2c_device = i2c_device.I2CDevice(i2c_bus, address)
        self._address = address
        self._send_buffer = bytearray(64)
        self._receive_buffer = bytearray(64)
        self.calibration_state = 0
        self.ELECANGLEOFFSET = 0
        self.SINCOSCENTRE = 0
        self.QDRformat = 0
        self.QDRposition = 0
        self.QDRspeed = 0
        self.QDRERROR1 = 0
        self.QDRERROR2 = 0

    def _pack_float(self, offset: int, data: float) -> None:
        struct.pack_into("<f", self._send_buffer, offset, data)

    def _pack_uint8(self, offset: int, data: int) -> None:
        self._send_buffer[offset] = data

    def _pack_uint16(self, offset: int, data: int) -> None:
        self._send_buffer[offset] = data & 0xFF
        self._send_buffer[offset + 1] = (data >> 8) & 0xFF

    def _pack_uint32(self, offset: int, data: int) -> None:
        self._send_buffer[offset] = data & 0xFF
        self._send_buffer[offset + 1] = (data >> 8) & 0xFF
        self._send_buffer[offset + 2] = (data >> 16) & 0xFF
        self._send_buffer[offset + 3] = (data >> 24) & 0xFF

    def _pack_int8(self, offset: int, value: int):
        if value < 0:
            value += 256
        self._send_buffer[offset] = value & 0xFF

    def _pack_int16(self, offset: int, value: int):
        if value < 0:
            value += 65536
        self._send_buffer[offset] = value & 0xFF
        self._send_buffer[offset + 1] = (value >> 8) & 0xFF

    def _pack_int32(self, offset: int, value: int):
        if value < 0:
            value += 4294967296
        self._send_buffer[offset] = value & 0xFF
        self._send_buffer[offset + 1] = (value >> 8) & 0xFF
        self._send_buffer[offset + 2] = (value >> 16) & 0xFF
        self._send_buffer[offset + 3] = (value >> 24) & 0xFF

    def _unpack_float(self, offset: int) -> float:
        return struct.unpack_from("<f", self._receive_buffer, offset)[0]

    def _unpack_uint8(self, offset: int) -> int:
        return self._receive_buffer[offset]

    def _unpack_uint16(self, offset: int) -> int:
        return self._receive_buffer[offset] | (self._receive_buffer[offset + 1] << 8)

    def _unpack_uint32(self, offset: int) -> int:
        return (
            self._receive_buffer[offset]
            | (self._receive_buffer[offset + 1] << 8)
            | (self._receive_buffer[offset + 2] << 16)
            | (self._receive_buffer[offset + 3] << 24)
        )

    def _unpack_int8(self, offset: int) -> int:
        value = self._receive_buffer[offset]
        return value - 256 if value > 127 else value

    def _unpack_int16(self, offset: int) -> int:
        value = self._receive_buffer[offset] | (self._receive_buffer[offset + 1] << 8)
        return value - 65536 if value > 32767 else value

    def _unpack_int32(self, offset: int) -> int:
        value = (
            self._receive_buffer[offset]
            | (self._receive_buffer[offset + 1] << 8)
            | (self._receive_buffer[offset + 2] << 16)
            | (self._receive_buffer[offset + 3] << 24)
        )
        return value - 4294967296 if value > 2147483647 else value

    def get_firmware_version(self) -> int:
        self._send_buffer[0] = 0x00
        self._i2c_device.write(self._send_buffer, end=1)
        self._i2c_device.readinto(self._receive_buffer, end=4)
        return self._unpack_uint32(0)

    def set_iq_pid_constants(self, kp: int, ki: int) -> None:
        self._send_buffer[0] = 0x40
        self._pack_int32(1, kp)
        self._pack_int32(5, ki)
        self._i2c_device.write(self._send_buffer, end=9)

    def set_id_pid_constants(self, kp: int, ki: int) -> None:
        self._send_buffer[0] = 0x41
        self._pack_int32(1, kp)
        self._pack_int32(5, ki)
        self._i2c_device.write(self._send_buffer, end=9)

    def set_speed_pid_constants(self, kp: float, ki: float, kd: float) -> None:
        self._send_buffer[0] = 0x42
        self._pack_float(1, kp)
        self._pack_float(5, ki)
        self._pack_float(9, kd)
        self._i2c_device.write(self._send_buffer, end=13)

    def set_position_pid_constants(self, kp: float, ki: float, kd: float) -> None:
        self._send_buffer[0] = 0x43
        self._pack_float(1, kp)
        self._pack_float(5, ki)
        self._pack_float(9, kd)
        self._i2c_device.write(self._send_buffer, end=13)

    def set_position_region_boundary(self, boundary: float) -> None:
        self._send_buffer[0] = 0x44
        self._pack_float(1, boundary)
        self._i2c_device.write(self._send_buffer, end=5)

    def configure_operating_mode_and_sensor(
        self, operating_mode: int, sensor_type: int
    ) -> None:
        self._send_buffer[0] = 0x20
        self._send_buffer[1] = operating_mode + (sensor_type << 4)
        self._i2c_device.write(self._send_buffer, end=2)

    def configure_command_mode(self, command_mode: int) -> None:
        self._send_buffer[0] = 0x21
        self._send_buffer[1] = command_mode
        self._i2c_device.write(self._send_buffer, end=2)

    def set_voltage(self, voltage: int) -> None:
        self._send_buffer[0] = 0x10
        self._pack_int32(1, voltage)
        self._i2c_device.write(self._send_buffer, end=5)

    def set_torque(self, torque: int) -> None:
        self._send_buffer[0] = 0x11
        self._pack_int32(1, torque)
        self._i2c_device.write(self._send_buffer, end=5)

    def set_speed(self, speed: int) -> None:
        self._send_buffer[0] = 0x12
        self._pack_int32(1, speed)
        self._i2c_device.write(self._send_buffer, end=5)

    def set_position(self, position: int, elecangle: int) -> None:
        self._send_buffer[0] = 0x13
        self._pack_uint32(1, position)
        self._pack_uint8(5, elecangle)
        self._i2c_device.write(self._send_buffer, end=6)

    def set_current_limit_foc(self, current: int) -> None:
        self._send_buffer[0] = 0x33
        self._pack_int32(1, current)
        self._i2c_device.write(self._send_buffer, end=5)

    def set_speed_limit(self, speed: int) -> None:
        self._send_buffer[0] = 0x34
        self._pack_int32(1, speed)
        self._i2c_device.write(self._send_buffer, end=5)

    def clear_faults(self) -> None:
        self._send_buffer[0] = 0x01
        self._i2c_device.write(self._send_buffer, end=1)

    def set_ELECANGLEOFFSET(self, value: int) -> None:
        self._send_buffer[0] = 0x30
        self._pack_uint32(1, value)
        self._i2c_device.write(self._send_buffer, end=5)

    def set_EAOPERSPEED(self, value: int) -> None:
        self._send_buffer[0] = 0x31
        self._pack_int32(1, value)
        self._i2c_device.write(self._send_buffer, end=5)

    def set_SINCOSCENTRE(self, value: int) -> None:
        self._send_buffer[0] = 0x32
        self._pack_int32(1, value)
        self._i2c_device.write(self._send_buffer, end=5)

    def set_calibration_options(
        self, voltage: int, speed: int, scycles: int, cycles: int
    ) -> None:
        self._send_buffer[0] = 0x3A
        self._pack_uint32(1, voltage)
        self._pack_int32(5, speed)
        self._pack_uint32(9, scycles)
        self._pack_uint32(13, cycles)
        self._i2c_device.write(self._send_buffer, end=17)

    def start_calibration(self) -> None:
        self._send_buffer[0] = 0x38
        self._pack_uint8(1, 1)
        self._i2c_device.write(self._send_buffer, end=2)

    def stop_calibration(self) -> None:
        self._send_buffer[0] = 0x38
        self._pack_uint8(1, 0)
        self._i2c_device.write(self._send_buffer, end=2)

    def is_calibration_finished(self) -> bool:
        self._send_buffer[0] = 0x39
        self._i2c_device.write(self._send_buffer, end=1)
        self._i2c_device.readinto(self._receive_buffer, end=9)
        self.calibration_state = self._unpack_uint8(0)
        self.ELECANGLEOFFSET = self._unpack_uint32(1)
        self.SINCOSCENTRE = self._unpack_uint32(5)
        if self.calibration_state == 255:
            return True
        return False

    def get_calibration_ELECANGLEOFFSET(self) -> int:
        self._send_buffer[0] = 0x39
        self._i2c_device.write(self._send_buffer, end=1)
        self._i2c_device.readinto(self._receive_buffer, end=9)
        self.calibration_state = self._unpack_uint8(0)
        self.ELECANGLEOFFSET = self._unpack_uint32(1)
        self.SINCOSCENTRE = self._unpack_uint32(5)
        if self.calibration_state == 255:
            return self.ELECANGLEOFFSET
        return 0

    def get_calibration_SINCOSCENTRE(self) -> int:
        self._send_buffer[0] = 0x39
        self._i2c_device.write(self._send_buffer, end=1)
        self._i2c_device.readinto(self._receive_buffer, end=9)
        self.calibration_state = self._unpack_uint8(0)
        self.ELECANGLEOFFSET = self._unpack_uint32(1)
        self.SINCOSCENTRE = self._unpack_uint32(5)
        if self.calibration_state == 255:
            return self.SINCOSCENTRE
        return False

    def update_quick_data_readout(self) -> None:
        if self.QDRformat == 0:
            self._i2c_device.readinto(self._receive_buffer, end=10)
            self.QDRposition = self._unpack_int32(0)
            self.QDRspeed = self._unpack_int32(4)
            self.QDRERROR1 = self._unpack_uint8(8)
            self.QDRERROR2 = self._unpack_uint8(9)

    def get_position_QDR(self) -> int:
        return self.QDRposition

    def get_speed_QDR(self) -> int:
        return self.QDRspeed

    def get_ERROR1_QDR(self) -> int:
        return self.QDRERROR1

    def get_ERROR2_QDR(self) -> int:
        return self.QDRERROR2
