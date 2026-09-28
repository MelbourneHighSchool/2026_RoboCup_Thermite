# motorcheck.py: manual drive check. Drives straight at full speed until q is pressed.
#
#     ROBOT_ID=1 python3 -m bot.motorcheck
#
# Uses the selected robot's pins and the shared calibration (the same resolution
# main.py does), so it can't drive the other robot's addresses.

import select
import sys
import termios
import time
import tty

import board
import busio

from bot.drive_config import MOTOR_CALIB
from bot.dwibbler import dwibble_calib
from bot.MotorFuncs_Proto1 import Motor
from bot.robot_select import select_and_publish_config

SPEED = -1 # full scale; negative drives backward

cfg = select_and_publish_config()
i2c = busio.I2C(board.SCL, board.SDA)
pins = cfg.MOTOR_PINS
calib = Motor.resolve_calibration(
    dict(MOTOR_CALIB, dwibble=dwibble_calib),
    {name: pins[name] for name in ("nw", "se", "sw", "ne", "dwibble")})
for name in ("nw", "se", "sw", "ne", "dwibble"):
    Motor(pins[name], i2c, name=name).calibset(calib[name])


def get_key():
    """the key pressed since the last call, or None."""
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None


def restore_terminal():
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


# cbreak mode, so a single q stops the motors without waiting for Enter
old_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())

try:
    Motor.drive(0, SPEED)
    while get_key() != "q":
        time.sleep(0.01)
    print("\n stopped (q pressed)")
finally:
    # always restore the terminal, stop the motors and clear any latched fault
    restore_terminal()
    Motor.stopall()
    Motor.clear_faults()
