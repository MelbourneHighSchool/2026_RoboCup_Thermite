#!/usr/bin/env python3
# bench_drive_sweep.py: the drive-limit measurements BENCH_TEST_CHECKLIST section 7
# asks for, run on the robot with the wheels on the floor.
#
# Three measurements: ramp a command and read the speed back, cruise then stop and
# integrate the roll-out, and sweep the current limit until a target speed is held.
# Everything here reads the wheels' own QDR, so it measures what the wheels
# turn, not what the chassis does. Wheel slip is invisible to it, and a run on a
# slick floor reads faster than the robot actually travelled.
#
# Run on the Pi from public-repo/, robot on a clear floor with room to run:
#   python3 tools/bench_drive_sweep.py ramp # top speed vs command
#   python3 tools/bench_drive_sweep.py stop --speed-frac 0.8 # roll-out from cruise
#   python3 tools/bench_drive_sweep.py current --speed-frac 0.8 # lowest current that holds it
#
# The ramp report is the one the checklist wants first: if measured speed stops rising
# while the command keeps climbing, the wheels are slipping and the extra command buys
# motor heat rather than distance. The stop report gives the real roll-out to compare
# against Motor.DECEL_MAX_FRAC_PER_S, which the final-approach braking model currently
# only estimates.

import argparse
import math
import os
import sys
import time

# the bot package lives one level up; anchor it so the script runs from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DRIVE_NAMES = ("nw", "ne", "sw", "se")


def frac_to_mms(frac, full_scale_mms):
    """the drive model's conversion: command fraction -> mm/s at full scale
    (bot.motion.vmax_full_cmd_mms, uncapped).
    """
    return float(frac) * float(full_scale_mms)


def mms_to_frac(mms, full_scale_mms):
    """inverse of frac_to_mms."""
    return float(mms) / float(full_scale_mms)


class WheelDrive:
    """everything the measurement loops need from the robot: command a fraction, stop, and read
    the wheels' own motion.
    """

    def __init__(self, odom):
        self.odom = odom
        self._t = time.monotonic()
        # baseline: the first delta would otherwise cover every read since boot
        self.odom.poll(self._t)

    def command(self, speed_frac, bearing=0.0):
        """drive at `speed_frac` (-1..1) along `bearing` (0deg = forward)."""
        from bot.hardware import Motor
        Motor.drive(bearing, speed_frac)

    def stop(self):
        """stop every motor."""
        from bot.hardware import Motor
        Motor.stopall()

    def delta(self):
        """(forward_mm, right_mm, dt_s) since the previous call, or None if a QDR read failed."""
        now = time.monotonic()
        dt = now - self._t
        self._t = now
        since = self.odom.poll(now)
        if since is None:
            return None
        return since[0], since[1], dt

    def speed_mms(self):
        """measured body speed (mm/s) over the last interval, or None when the wheels cannot be
        read.
        """
        d = self.delta()
        if d is None or d[2] <= 1e-9:
            return None
        return math.hypot(d[0], d[1]) / d[2]


def ramp(drive, start_frac, limit_frac, step, settle_s, full_scale_mms, log=print):
    """step the command up and report the speed the wheels measure at each step; returns the
    peak measured speed.
    """
    peak, frac = 0.0, start_frac
    while frac <= limit_frac + 1e-9:
        drive.command(frac)
        time.sleep(settle_s)
        speed = drive.speed_mms()
        if speed is None:
            log(f"  command={frac:.3f} no QDR read")
        else:
            peak = max(peak, speed)
            log(f"  command={frac:.3f} ({frac_to_mms(frac, full_scale_mms):.0f} mm/s uncapped)  "
                f"measured={speed:.0f} mm/s")
        frac = round(frac + step, 6)
    drive.stop()
    log(f"  peak measured: {peak:.0f} mm/s "
        f"({mms_to_frac(peak, full_scale_mms):.3f} of full scale)")
    return peak


def stopping_distance(drive, speed_frac, cruise_s, settle_mms, timeout_s, log=print):
    """cruise at speed_frac, command a stop, then integrate the wheels' own displacement until
    they are slow; returns (rollout_mm, cruise_mms).
    """
    drive.command(speed_frac)
    time.sleep(cruise_s)
    cruise = drive.speed_mms() or 0.0
    drive.stop()
    rolled, started = 0.0, time.monotonic()
    while time.monotonic() - started < timeout_s:
        d = drive.delta()
        if d is None:
            break
        travelled = math.hypot(d[0], d[1])
        rolled += travelled
        if d[2] > 0 and travelled / d[2] <= settle_mms:
            break
        time.sleep(0.01)
    log(f"  cruise={cruise:.0f} mm/s roll-out={rolled:.1f} mm")
    return rolled, cruise


def hold_speed(drive, target_frac, duration, tolerance, hold, full_scale_mms, log=print):
    """command target_frac and report whether the measured speed reaches tolerance and stays
    there for hold seconds.
    """
    drive.command(target_frac)
    target_mms = frac_to_mms(target_frac, full_scale_mms)
    started, reached_at, best = time.monotonic(), None, 0.0
    while time.monotonic() - started < duration:
        speed = drive.speed_mms()
        if speed is None:
            break
        best = max(best, speed)
        error = abs(speed - target_mms) / target_mms
        if error <= tolerance:
            if reached_at is None:
                reached_at = time.monotonic()
            if time.monotonic() - reached_at >= hold:
                log(f"  pass measured={speed:.0f} mm/s target={target_mms:.0f} mm/s "
                    f"error={error:.1%}")
                drive.stop()
                return True
        else:
            reached_at = None
        time.sleep(0.02)
    drive.stop()
    log(f"  fail peak={best:.0f} mm/s target={target_mms:.0f} mm/s")
    return False


def set_drive_current(amps):
    """re-limit every drive motor in FOC amps, over the Motor registry.
    """
    from bot.MotorFuncs_Proto1 import FOC_LSB_PER_AMP, Motor
    raw = int(amps * FOC_LSB_PER_AMP)
    for name in DRIVE_NAMES:
        motor = Motor.motors.get(name)
        if motor is not None:
            motor.current_limit_amps = amps
            motor.md.set_current_limit_foc(raw)


def _start_motors(calib_path=None):
    """bring the drive motors and the roller up the same way main.py does, on this robot's own
    pins. Calibration comes from the saved file, address-keyed; a motor with no saved entry
    warns rather than being guessed at, because a wrong sin/cos centre drives badly and
    reads like a fault.
    """
    import board
    import busio
    import bot.robot_select
    from bot.MotorFuncs_Proto1 import Motor, load_calibration

    cfg = bot.robot_select.select_and_publish_config()
    i2c = busio.I2C(board.SCL, board.SDA)
    pins = cfg.MOTOR_PINS
    saved = load_calibration(calib_path)
    for name in tuple(DRIVE_NAMES) + ("dwibble",):
        motor = Motor(pins[name], i2c, name=name)
        if pins[name] in saved:
            motor.calibset(saved[pins[name]])
        else:
            print(f"[bench] no saved calibration for {name} (address {pins[name]}), "
                  "it comes up uncalibrated; calibrate it first", flush=True)
    return i2c


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("ramp", "stop", "current"),
                        help="ramp: top speed vs command; stop: roll-out from cruise; "
                             "current: lowest current that holds a speed")
    parser.add_argument("--start-frac", type=float, default=0.1, help="ramp: first command fraction")
    parser.add_argument("--limit-frac", type=float, default=1.0, help="ramp: last command fraction")
    parser.add_argument("--step", type=float, default=0.1, help="ramp: command increment")
    parser.add_argument("--settle", type=float, default=0.5, help="seconds to hold each command before reading")
    parser.add_argument("--speed-frac", type=float, default=0.8, help="stop/current: command fraction to cruise or hold")
    parser.add_argument("--cruise", type=float, default=0.5, help="stop: seconds at speed before the stop command")
    parser.add_argument("--settle-mms", type=float, default=20.0, help="stop: wheel speed treated as stopped")
    parser.add_argument("--timeout", type=float, default=5.0, help="stop: seconds to wait for the roll-out")
    parser.add_argument("--start-current", type=float, default=0.25, help="current: first FOC limit to try (amps)")
    parser.add_argument("--max-current", type=float, default=8.0, help="current: last FOC limit to try (amps)")
    parser.add_argument("--current-step", type=float, default=0.25, help="current: limit increment (amps)")
    parser.add_argument("--duration", type=float, default=10.0, help="current: seconds allowed per limit")
    parser.add_argument("--tolerance", type=float, default=0.05, help="current: fractional speed error that counts as held")
    parser.add_argument("--hold", type=float, default=0.25, help="current: seconds within tolerance")
    parser.add_argument("--calib-json", default=None, help="calibration file to load (default: the bench's own)")
    args = parser.parse_args(argv)

    if not 0.0 < args.start_frac <= args.limit_frac <= 1.0 or args.step <= 0:
        parser.error("require 0 < start-frac <= limit-frac <= 1 and step > 0")
    if args.settle <= 0 or args.cruise <= 0 or args.timeout <= 0 or args.settle_mms < 0:
        parser.error("require positive settle, cruise and timeout")
    if not 0.0 < args.speed_frac <= 1.0:
        parser.error("require 0 < speed-frac <= 1")
    if not 0.0 < args.start_current <= args.max_current <= 8.0:
        parser.error("require 0 < start-current <= max-current <= 8 amps")
    if not 0.0 <= args.tolerance < 1.0 or not 0.0 <= args.hold < args.duration:
        parser.error("require tolerance in [0, 1) and 0 <= hold < duration")

    # Import the robot's chain only once the arguments are known good, so --help
    # still works on a desktop.
    from bot.MotorFuncs_Proto1 import FOC_LSB_PER_AMP, Motor
    from bot.motion import vmax_full_cmd_mms
    from bot.odometry import WheelOdometry

    if args.current_step < 1 / FOC_LSB_PER_AMP:
        parser.error("current-step must be at least 1/65536 A")

    _start_motors(args.calib_json)
    drive = WheelDrive(WheelOdometry())
    try:
        if args.mode == "ramp":
            print(f"[bench] ramp {args.start_frac:.2f} -> {args.limit_frac:.2f} "
                  f"step {args.step:.2f}, settle {args.settle:.1f}s (Ctrl+C stops)")
            ramp(drive, args.start_frac, args.limit_frac, args.step, args.settle,
                 vmax_full_cmd_mms)
        elif args.mode == "stop":
            print(f"[bench] cruise command {args.speed_frac:.2f} for {args.cruise:.1f}s, "
                  "then stop and integrate the roll-out (Ctrl+C stops)")
            stopping_distance(drive, args.speed_frac, args.cruise,
                              args.settle_mms, args.timeout)
        else:
            print(f"[bench] current sweep {args.start_current:g} -> {args.max_current:g} A "
                  f"step {args.current_step:g} A, holding {args.speed_frac:.2f} "
                  f"({frac_to_mms(args.speed_frac, vmax_full_cmd_mms):.0f} mm/s) "
                  f"for {args.duration:.0f}s each")
            current = args.start_current
            while True:
                print(f"  testing {current:g} A")
                set_drive_current(current)
                if hold_speed(drive, args.speed_frac, args.duration, args.tolerance,
                              args.hold, vmax_full_cmd_mms):
                    print(f"  lowest passing tested current: {current:g} A at "
                          f"{frac_to_mms(args.speed_frac, vmax_full_cmd_mms):.0f} mm/s")
                    break
                if current >= args.max_current:
                    print("  no tested current held the target within the trial duration")
                    break
                current = min(round(current + args.current_step, 10), args.max_current)
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[bench] stopped.")
    finally:
        Motor.stopall()
        Motor.clear_faults()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
