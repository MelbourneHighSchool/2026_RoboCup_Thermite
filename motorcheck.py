from bot.MotorFuncs_Proto1 import Motor
import time, board, busio, sys, select, termios, tty

i2c = busio.I2C(board.SCL, board.SDA)

Motor(26, i2c, name="nw").calibset([1450399744, 1229])
Motor(30, i2c, name="se").calibset([1588163584, 1232])
Motor(31, i2c, name="sw").calibset([1233465856, 1242])
Motor(32, i2c, name="ne").calibset([1428708608, 1259])
Motor(29, i2c, name="dwibble").calibset([1489797888, 1236])

SPEED = -1

# key check
def get_key():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None

def check_quit():
    key = get_key()
    if key == "q":
        print("\n emergency stop (q pressed)")
        Motor.stopall()
        restore_terminal()
        sys.exit(0)

# terminal setup
old_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())

def restore_terminal():
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def sleep_with_quit(t):
    start = time.time()
    while time.time() - start < t:
        check_quit()
        time.sleep(0.01)



def pause():
    Motor.stopall()
    sleep_with_quit(2)
try:
    Motor.drive(0, SPEED)
    while True:
        check_quit()
        time.sleep(0.01)

finally:
    # always restore terminal + stop motors and clear any latched fault
    restore_terminal()
    Motor.stopall()
    Motor.clear_faults()
