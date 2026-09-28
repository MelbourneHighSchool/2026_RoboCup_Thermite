# motorcalib.py: encoder calibration and manual command bench for the BLDC drivers.
#
#     python3 tests/motorcalib.py
#
# Asks for the number of drivers and each one's I2C address (bot 1: nw 26, se 30, sw 31,
# ne 32, dwibbler 29; bot 2: nw 28, se 31, sw 26, ne 25, dwibbler 27), calibrates each in
# turn and prints its ELECANGLEOFFSET and SINCOSCENTRE: the [elecangle, sincos] pair that
# goes in MOTOR_CALIB (bot/drive_config.py) or dwibble_calib (bot/dwibbler.py). Wheels off
# the floor: calibration spins every motor.
#
# Not a pytest file. After calibration it leaves each motor in FOC speed mode and reads
# single-line commands:
#     n<i> selects motor i
#     m<mode> sets the command mode: 2 torque, 12 speed, 13 position
#     s<v> sets the speed limit, and the speed in speed mode
#     p<v> sets the position target (position mode)
#     c<v> sets the current limit, and the torque in torque mode
#     k<v> sets the position P constant
#     b<v> sets the position region boundary
#     f clears faults
#     d<ms> waits ms milliseconds

import select
import sys
import time

import board
import busio

from steelbar_powerful_bldc_driver import PowerfulBLDCDriver

motor = [None] * 8
motormode = [0] * 8
motorcount = 0
selectedmotor = 0


def read_input():
    """one line from stdin if one is waiting, else None."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None


i2c = busio.I2C(board.SCL, board.SDA)

print("Please enter the number of motor drivers you want to control:")
motorcount = int(input())
if motorcount == 0 or motorcount > 8:
    print("Error motor count out of range, please reboot microcontroller to try again.")
    quit()

for i in range(motorcount):
    print(f"Please enter the i2c address of motor driver number {i}:")
    address = int(input())
    if address <= 7 or address >= 120:
        print("Error invalid i2c address, please reboot microcontroller to try again.")
        quit()
    motor[i] = PowerfulBLDCDriver(i2c, address)
    print(f"The firmware version of motor driver number {i} is: {motor[i].get_firmware_version()}")
    if motor[i].get_firmware_version() != 3:
        print("Error unsupported motor driver version, please check for updates, maybe check "
              "wiring and i2c configuration, reboot microcontroller to try again.")
        quit()

for i in range(motorcount):
    motor[i].set_current_limit_foc(65536) # 1 A (FOC mode only)
    motor[i].set_id_pid_constants(1500, 200)
    motor[i].set_iq_pid_constants(1500, 200)
    # speed PID from the driver's tuning document, for FOC and the M2006 P36 only
    motor[i].set_speed_pid_constants(4e-2, 4e-4, 3e-2)
    motor[i].set_position_pid_constants(275, 0, 0)
    motor[i].set_position_region_boundary(250000)
    motor[i].set_speed_limit(10000000)
    motor[i].configure_operating_mode_and_sensor(15, 1) # calibration mode, sin/cos encoder
    motor[i].configure_command_mode(15) # calibration command mode
    # 300/6798 of vcc, 2097152/65536 elecangle/s, 1 s settle, 10 s calibrate
    motor[i].set_calibration_options(300, 2097152, 50000, 500000)
    motor[i].start_calibration()
    print(f"Starting calibration of motor {i}")
    # no other driver calls until it finishes
    while not motor[i].is_calibration_finished():
        print(".", end="")
        sys.stdout.flush()
        time.sleep(0.5)
    print()
    print(f"ELECANGLEOFFSET: {motor[i].get_calibration_ELECANGLEOFFSET()}")
    print(f"SINCOSCENTRE: {motor[i].get_calibration_SINCOSCENTRE()}")

    motor[i].configure_operating_mode_and_sensor(3, 1) # FOC, sin/cos encoder
    motor[i].configure_command_mode(12) # speed command mode
    motormode[i] = 12

while True:
    userinput = read_input()
    if userinput:
        command = userinput[0]
        param = userinput[1:].strip()
        if command == 'n' and param:
            try:
                number = int(param)
                if number < motorcount:
                    selectedmotor = number
                    print(f"Selected motor number {selectedmotor}")
                else:
                    raise ValueError('Invalid motor number')
            except ValueError:
                print("Invalid motor number")
        elif command == 'm' and param:
            try:
                mode = int(param)
                if mode in (2, 12, 13):
                    motor[selectedmotor].configure_command_mode(mode)
                    motormode[selectedmotor] = mode
                    print(f"Command Mode {mode}")
                else:
                    raise ValueError('Invalid command mode')
            except ValueError:
                print("Invalid command mode")
        elif command == 's' and param:
            try:
                maxspeed = int(param)
                motor[selectedmotor].set_speed_limit(abs(maxspeed))
                if motormode[selectedmotor] == 12:
                    motor[selectedmotor].set_speed(maxspeed)
                print(f"Speed {maxspeed}")
            except ValueError:
                print("Invalid speed value")
        elif command == 'p' and param:
            try:
                postarget = float(param)
                posmsb = int(postarget)
                poslsb = int((postarget * 256) % 256)
                if motormode[selectedmotor] == 13:
                    motor[selectedmotor].set_position(posmsb, poslsb)
                else:
                    print("Motor is not in position mode")
                print(f"Position {postarget}")
            except ValueError:
                print("Invalid position value")
        elif command == 'c' and param:
            try:
                currentlimit = int(param)
                motor[selectedmotor].set_current_limit_foc(abs(currentlimit))
                if motormode[selectedmotor] == 2:
                    motor[selectedmotor].set_torque(currentlimit)
                print(f"Current (Torque) {currentlimit}")
            except ValueError:
                print("Invalid current value")
        elif command == 'k' and param:
            try:
                motor[selectedmotor].set_position_pid_constants(float(param), 0, 0)
            except ValueError:
                print("Invalid PID constant value")
        elif command == 'b' and param:
            try:
                motor[selectedmotor].set_position_region_boundary(float(param))
            except ValueError:
                print("Invalid boundary value")
        elif command == 'f':
            motor[selectedmotor].clear_faults()
            print("Clear Faults")
        elif command == 'd' and param:
            try:
                delay_ms = int(param)
                print(f"Delay {delay_ms}")
                time.sleep(delay_ms / 1000)
            except ValueError:
                print("Invalid delay value")
        else:
            print("Unknown command or missing parameter")

    time.sleep(0.001)
    for i in range(motorcount):
        motor[i].update_quick_data_readout()
