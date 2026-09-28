# MotorFuncs_Proto1.py: whole-robot motor helpers.
#
# The low-level driver lives in rotom.py; this wraps it in a Motor class with a holonomic
# drive() model and a dribbler. Async mode (start_async()) moves I2C speed writes onto a flush
# thread: drive()/dwibble() only stage targets, so the control loop never blocks on the bus,
# and only changed speeds are written.
import json, os, time, threading

import bot.kinematics as kinematics

# Bring-up constants for this driver board, applied in the order below.
DRIVER_FIRMWARE_VERSION = 3 # any other firmware version is refused
FOC_LSB_PER_AMP = 65536 # set_current_limit_foc's unit: 1 LSB = 2^-16 A
# four drive motors in FOC (was a 4 A guess)
drive_current_limit_amps = 8.0
# our roller's limit; see bot/dwibbler.py before trying torque mode
roller_current_limit_amps = 4.0


# calibrate_all() writes here; resolve_calibration() reads it back
CALIBRATION_FILE = "motor_calibration.json"
CALIB_TIMEOUT_S = 120.0 # without a bound, a driver that never finishes blocks the bench forever


class MotorFirmwareError(RuntimeError):
    """a motor reported a driver firmware version this code was not written against."""


class MotorCommunicationError(RuntimeError):
    """a motor bus write failed in a way a caller must not mistake for success."""


class Motor:
    # registry of created motors, keyed by name ("nw", "ne", "sw", "se",'dwibble')
    motors = {}

    # async flush thread (opt-in via start_async)
    _async = False # when True, spee-via-drive is staged, not written
    _targets = {} # name -> staged command (int)
    _last_sent = {} # name -> last command actually written
    _stage_lock = threading.Lock()
    FLUSH_PERIOD_S = 0.002
    KEEPALIVE_S = 0.5 # rewrite an unchanged command this often

    def __init__(self, address, i2c, name=None, angleoffset=None, sincos = None,
                 current_limit_amps=None):
        from bot.rotom import PowerfulBLDCDriver # driver (hardware only)
        self.md = PowerfulBLDCDriver(i2c, address)
        # Firmware gate before any configuration write: every opcode below is only
        # guaranteed on version 3. A wrong version is a wiring or board mismatch
        # and raises; a motor that doesn't answer only warns, so a transient bus
        # error at boot can't take startup down (the fault shows up in the
        # driver's error bytes).
        try:
            version = self.md.get_firmware_version()
        except Exception as e: # noqa: BLE001
            version = None
            print(f"[motor] {name or address}: firmware version unreadable ({e}), "
                  "continuing (check the bus if the motor does not respond)", flush=True)
        if version is not None and version != DRIVER_FIRMWARE_VERSION:
            raise MotorFirmwareError(
                f"motor {name or address} at address {address} reports driver firmware "
                f"{version}; this code is written for {DRIVER_FIRMWARE_VERSION}")
        self.firmware_version = version
        # Reconfigure in order: leave the power-up mode and stop the
        # motor first (the vendor requires a stop before a mode change), zero both
        # command registers, set limits, PIDs and calibration, and only then
        # select the speed command mode the drive runs in.
        self.md.configure_command_mode(2) # torque command mode: a safe mode to be reconfigured from
        self.md.configure_operating_mode_and_sensor(3, 1) # FOC, sin/cos encoder
        self.md.set_torque(0)
        self.md.set_speed(0)
        if current_limit_amps is None:
            current_limit_amps = (roller_current_limit_amps if name == "dwibble"
                                  else drive_current_limit_amps)
        self.current_limit_amps = current_limit_amps
        self.md.set_current_limit_foc(int(current_limit_amps * FOC_LSB_PER_AMP))
        self.md.set_id_pid_constants(1500, 200)
        self.md.set_iq_pid_constants(1500, 200)
        self.md.set_speed_pid_constants(4e-2, 4e-4, 3e-2)# FOC + M2006 P36 only
        self.md.set_position_pid_constants(275, 0, 0)
        self.md.set_position_region_boundary(250000)
        # only bounds speed/position mode, so the command stays in charge and
        # spee() stays one I2C transaction
        self.md.set_speed_limit(int(Motor.DRIVER_MAX_RAW))
        self.md.set_calibration_options(300, 2097152, 50000, 500000)
        if angleoffset != None and sincos != None: # skip calib if known already
            self.md.set_ELECANGLEOFFSET(angleoffset)
            self.md.set_SINCOSCENTRE(sincos)
        self.md.configure_command_mode(12) # speed command mode, the mode the drive runs in
        self.current_speed = 0.0 # -1..1, only used by accel()
        self.name = name
        if name != None: # register so the whole-robot helpers can find this motor
            Motor.motors[name] = self
    def disable_calibration(self):
        # Leave calibration mode on zero torque. Writing speed zero alone doesn't
        # change command mode 15, so this walks torque -> command mode 2 -> FOC +
        # sin/cos -> torque. Failures come back as a list, so a failed exit can't
        # pass for a clean one.
        errors = []
        for method, args in (("set_torque", (0,)),
                             ("configure_command_mode", (2,)),
                             ("configure_operating_mode_and_sensor", (3, 1)),
                             ("set_torque", (0,))):
            try:
                getattr(self.md, method)(*args)
            except Exception as e: # noqa: BLE001, collected for the caller, not raised from here
                errors.append(f"motor {self.name}: {method}: {e}")
        return errors

    def calib(self, name=None, timeout_s=CALIB_TIMEOUT_S):
        # run the driver's encoder calibration; returns [elecangleoffset,
        # sincoscentre]. The exit sequence runs in a finally, so a timeout or bus
        # error still leaves the driver out of calibration mode on zero torque.
        if name is None:
            name = self.name
        # start_calibration() only works in calibration mode; __init__ leaves the
        # driver in FOC
        self.md.configure_operating_mode_and_sensor(15, 1) # calibration mode, sin/cos encoder
        self.md.configure_command_mode(15) # calibration command mode
        # set again here, though init already has; harmless
        self.md.set_calibration_options(300, 2097152, 50000, 500000)
        self.md.start_calibration()
        started = time.monotonic()
        print(f"[calib] {name}: calibrating (timeout {timeout_s:.0f}s)...", flush=True)
        try:
            while not self.md.is_calibration_finished():
                if time.monotonic() - started >= timeout_s:
                    raise TimeoutError(
                        f"motor {name}: calibration did not finish within {timeout_s:.0f}s")
                time.sleep(0.05)
            elecang = self.md.get_calibration_ELECANGLEOFFSET()
            sincos = self.md.get_calibration_SINCOSCENTRE()
        finally:
            errors = self.disable_calibration()
        if errors:
            raise MotorCommunicationError("calibration exit failed: " + "; ".join(errors))
        # ...and only then back into the mode the drive actually runs in: FOC +
        # sin/cos, speed command.
        self.md.configure_command_mode(12) # speed command mode
        print(f"[calib] {name}: elecangle {elecang} sincos {sincos}", flush=True)
        return [elecang, sincos]
    @classmethod
    def calibrate_all(cls, path=None):
        # calibrate every registered motor, apply the result, and save it
        # address-keyed for the next boot; returns name -> config
        configs = {}
        for name, motor in cls.motors.items():
            config = motor.calib(name)
            motor.calibset(config)
            configs[name] = config
        entries = {getattr(motor.md, "_address", None): configs[name]
                   for name, motor in cls.motors.items()
                   if name in configs and getattr(motor.md, "_address", None) is not None}
        if entries:
            save_calibration(entries, path)
        return configs
    @classmethod
    def resolve_calibration(cls, defaults, addresses, path=None):
        # per-motor startup calibration. `defaults` and `addresses` are
        # name-keyed; an address in the saved file wins over its default, so a
        # bench re-run takes effect without editing source, and a missing or
        # malformed file leaves every default in place.
        saved = load_calibration(path)
        out, used = {}, []
        for name, fallback in defaults.items():
            address = addresses.get(name)
            if address is not None and address in saved:
                out[name] = list(saved[address])
                used.append(name)
            else:
                out[name] = list(fallback)
        if used:
            print(f"[calib] loaded saved calibration for {', '.join(sorted(used))}", flush=True)
        return out
    def calibset(self,config):
        self.md.set_ELECANGLEOFFSET(config[0])
        self.md.set_SINCOSCENTRE(config[1])
    def spee(self, frac):
        # signed fraction of full scale, -1..1
        self.md.set_speed(Motor._to_raw(frac))

    def read_qdr(self):
        # (position, speed, error1, error2), or None if the read fails. speed is
        # raw driver units (2^-16 electrical rev/s): divide by MOTOR_MAX_RAW to
        # compare with a command fraction.
        try:
            self.md.update_quick_data_readout()
            return (self.md.get_position_QDR(), self.md.get_speed_QDR(),
                    self.md.get_ERROR1_QDR(), self.md.get_ERROR2_QDR())
        except Exception:
            return None

    @classmethod
    def start_async(cls):
        # switch drive()/dwibble() to staged writes and start the flush thread.
        if cls._async:
            return
        cls._async = True
        threading.Thread(target=cls._flush_loop, daemon=True).start()

    @classmethod
    def _stage(cls, name, val):
        if not cls._async:
            # No motors registered (bench/no-hardware run): nothing to write.
            m = cls.motors.get(name)
            if m is not None:
                m.spee(val)
            cls._targets[name] = float(val) # still record, debug parity checks read it
            return
        with cls._stage_lock:
            cls._targets[name] = float(val)

    @classmethod
    def _flush_loop(cls):
        last_write = {} # name -> monotonic time
        while True:
            with cls._stage_lock:
                targets = dict(cls._targets)
            now = time.monotonic()
            for name, val in targets.items():
                m = cls.motors.get(name)
                if m is None:
                    continue
                stale = now - last_write.get(name, 0.0) >= cls.KEEPALIVE_S
                if val != cls._last_sent.get(name) or stale:
                    try:
                        m.spee(val)
                        cls._last_sent[name] = val
                        last_write[name] = now
                    except Exception as e:
                        print(f"[motor] {name} write failed: {e}", flush=True)
            time.sleep(cls.FLUSH_PERIOD_S)
    def accel(self, target, step=0.01, delay=0.2):
        # ramp this motor to `target` (-1..1) in steps. The robot doesn't use it
        # (drive() commands speed directly); kept for bench scripts.
        while abs(self.current_speed - target) > step:
            measured = self.md.get_speed_QDR() / Motor.MOTOR_MAX_RAW
            self.current_speed = measured
            if self.current_speed < target:
                self.current_speed += step
            elif self.current_speed > target:
                self.current_speed -= step
            self.spee(self.current_speed)
        self.spee(target)
    debug = False

    # Speed units: every speed here (spee, drive, dwibble) is -1..1, a signed fraction
    # of the motor's no-load top speed. The driver's setSpeed unit is 2^-16 electrical
    # rev/s.
    RPM_TO_RAW = 7 * 36 / 60 * 2 ** 16
    DRIVER_MAX_RAW = 546_133_333
    MOTOR_MAX_RAW = 546133333
    MAX_RPM = MOTOR_MAX_RAW / RPM_TO_RAW
    MAX_REV_PER_S = MAX_RPM / 60.0
    # MAX_RPM works out at about 1984 rpm, the driver's full command range. The wheels
    # top out well below it (about 0.5 measured); BENCH_TEST_CHECKLIST section 7.

    _legacy_warned = False

    @classmethod
    def _to_raw(cls, frac):
        # -1..1 -> the driver's raw signed speed command
        try:
            f = float(frac)
        except (TypeError, ValueError):
            return 0
        if f != f:
            return 0
        if abs(f) > 1.5:
            if not cls._legacy_warned:
                cls._legacy_warned = True
                print(f"[motor] speed {f:g} is outside -1..1, speeds are "
                      "normalised now (see MOTOR_MAX_RAW); stopping instead", flush=True)
            return 0
        return int(max(-1.0, min(1.0, f)) * cls.MOTOR_MAX_RAW)

    @classmethod
    def rpm_to_frac(cls, rpm):
        return rpm * cls.RPM_TO_RAW / cls.MOTOR_MAX_RAW

    # Drive model: no wheel runs past the cap; if any would, the whole command set
    # scales down together, preserving the motion's shape.
    MOTOR_CAP_FRAC = 0.9
    # Centre-to-wheel contact distance, the orbit radius of a spin (kinematics divides
    # by it). Tape-measured at 94-95 mm; the old 110 guess made every orbit spiral
    # about 16% inside the requested radius.
    TURN_RADIUS_MM = 94.5

    # No software accel/decel ramp: speed and bearing apply the instant they're set,
    # reversals included (torque and traction still bound real motion). These two
    # remain as conservative achievable-accel estimates for the braking model in
    # bot/motion.py, not an active ramp.
    ACCEL_MAX_FRAC_PER_S = 5.0
    DECEL_MAX_FRAC_PER_S = 6.0

    @classmethod
    def drive(cls, bearing, speed, rot_r=0.0, rot_theta=0.0, rot_speed=0.0):
        # bearing, speed: robot-frame bearing (0 = forward, 90 = right) and
        # magnitude 0..1. rot_speed: spin, -1..1, positive = clockwise; it sums on
        # top before the cap, so a quick spin leaves less for translation.
        # rot_r/rot_theta place an off-centre rotation. The math lives in
        # bot.kinematics.solve, which returns commands already capped and
        # polarity-fixed.
        sent = kinematics.solve(bearing, speed, rot_r=rot_r,
                                rot_theta=rot_theta, rot_speed=rot_speed,
                                cap_frac=cls.MOTOR_CAP_FRAC,
                                turn_radius_mm=cls.TURN_RADIUS_MM)
        if cls.debug:
            scale = max(abs(speed), 1)
            print(f"[motor] bear={bearing:+.1f}deg "
                  f"nw={sent['nw']/scale:+.2f} ne={sent['ne']/scale:+.2f} "
                  f"sw={sent['sw']/scale:+.2f} se={sent['se']/scale:+.2f}",
                  flush=True)
        for name, val in sent.items():
            cls._stage(name, val)

    @classmethod
    def stopall(cls):
        # safety path
        with cls._stage_lock:
            for name in cls.motors:
                cls._targets[name] = 0
        for name, motor in cls.motors.items():
            try:
                motor.spee(0)
                cls._last_sent[name] = 0
            except Exception:
                pass
        # Reset bot.motion's slew state too, so the first command after a stop
        # ramps from 0 instead of springing from a stale level. Lazy import:
        # motion imports Motor, so a module-level import would cycle.
        try:
            import bot.motion as _motion
            _motion._last_slew_speed = 0.0
        except Exception:
            pass # motion not importable (bare-bench scripts), nothing to reset
    @classmethod
    def clear_faults(cls):
        # call after stopall() on quit, so a fault from this run doesn't swallow
        # the next start's commands
        for motor in cls.motors.values():
            try:
                motor.md.clear_faults()
            except Exception:
                pass
    @classmethod
    def dwibble(cls, speed):
        # dribbler
        if 'dwibble' in cls.motors:
            cls._stage('dwibble', float(speed))


# Calibration persistence: one address-keyed JSON file, so re-calibrating never means
# editing code. Written atomically (temp file + os.replace) so a crash mid-write can't
# leave a half file. Readers treat it as an optimisation, never a dependency: absent or
# unreadable, the compiled-in defaults stand.

def save_calibration(entries, path=None):
    """write {i2c_address: [elecangleoffset, sincoscentre]} to JSON atomically; returns the
    serialised dict.
    """
    path = path or CALIBRATION_FILE
    data = {"motors": [{"address": int(address),
                        "elecangleoffset": int(config[0]),
                        "sincoscentre": int(config[1])}
                       for address, config in sorted(entries.items())]}
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return data


def load_calibration(path=None):
    """{i2c_address: [elecangleoffset, sincoscentre]} from the saved file, or {} when absent or
    malformed. Never raises: a bad bench file must not block boot.
    """
    path = path or CALIBRATION_FILE
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {}
    for entry in data.get("motors", []):
        try:
            out[int(entry["address"])] = [int(entry["elecangleoffset"]),
                                          int(entry["sincoscentre"])]
        except (KeyError, TypeError, ValueError):
            continue
    return out
