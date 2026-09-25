# MotorFuncs_Proto1.py: whole-robot motor helpers.
# The low-level device driver lives in rotom.py
# this file wraps it into a Motor class with a holonomic drive() model and a dribbler
#
# Async mode (start_async()) moves I2C speed writes onto a dedicated flush
# thread, drive()/dwibble() just stage targets, so the control loop never
# blocks on the bus, and only changed speeds are written
import time, sys, math, struct, threading

class Motor:
    # registry of created motors, keyed by name ("nw", "ne", "sw", "se",'dwibble')
    motors = {}

    # async flush thread (opt-in via start_async)
    _async          = False # when True, spee-via-drive is staged, not written
    _targets        = {} # name -> staged command (int)
    _last_sent      = {} # name -> last command actually written
    _stage_lock     = threading.Lock()
    FLUSH_PERIOD_S  = 0.002
    KEEPALIVE_S     = 0.5 # rewrite an unchanged command this often

    def __init__(self, address, i2c, name=None, angleoffset=None, sincos = None):
        from bot.rotom import PowerfulBLDCDriver # driver (hardware only)
        self.md = PowerfulBLDCDriver(i2c, address)
        self.md.set_current_limit_foc(65536*4) # 4A, FOC mode only
        self.md.set_id_pid_constants(1500, 200)
        self.md.set_iq_pid_constants(1500, 200)
        self.md.set_speed_pid_constants(4e-2, 4e-4, 3e-2)# FOC + M2006 P36 only
        self.md.set_position_pid_constants(275, 0, 0)
        self.md.set_position_region_boundary(250000)
        # only bounds Speed/Position mode, so this leaves the command in charge and
        # keeps spee() to a single I2C transaction
        self.md.set_speed_limit(int(Motor.DRIVER_MAX_RAW))
        self.md.configure_operating_mode_and_sensor(3, 1) # FOC, sin/cos encoder
        self.md.configure_command_mode(12) # speed command mode
        self.md.set_calibration_options(300, 2097152, 50000, 500000)
        if angleoffset != None and sincos != None: # skip calib if known already
            self.md.set_ELECANGLEOFFSET(angleoffset)
            self.md.set_SINCOSCENTRE(sincos)
        self.current_speed = 0.0 # -1..1, only used by accel()
        self.name = name
        if name != None: # register so the whole-robot helpers can find this motor
            Motor.motors[name] = self
    def calib(self, name=None):
        if name is None:
            name = self.name
        # Must be in calibration mode for start_calibration() to do
        # anything, __init__ leaves it in FOC mode
        self.md.configure_operating_mode_and_sensor(15, 1) # calibration mode, sin/cos encoder
        self.md.configure_command_mode(15) # calibration command mode
        self.md.start_calibration()

        while not self.md.is_calibration_finished():
            print("so uh, ya like jazz?")
            sys.stdout.flush()
            time.sleep(0.5)
        elecang = self.md.get_calibration_ELECANGLEOFFSET()
        sincos = self.md.get_calibration_SINCOSCENTRE()
        # return to FOC / speed command mode so the motor can be driven
        self.md.configure_operating_mode_and_sensor(3, 1) # FOC mode, sin/cos encoder
        self.md.configure_command_mode(12) # speed command mode
        print("name " + str(name) + " elecangle " + str(elecang) + " sincos " + str(sincos))
        return [elecang, sincos]
    @classmethod
    def calibrate_all(cls):
        # calibrate every registered motor and apply the result; returns name->config
        configs = {}
        for name, motor in cls.motors.items():
            config = motor.calib(name)
            motor.calibset(config)
            configs[name] = config
        return configs
    def calibset(self,config):
        self.md.set_ELECANGLEOFFSET(config[0])
        self.md.set_SINCOSCENTRE(config[1])
    def spee(self, frac):
        # signed fraction of full scale, -1..1
        self.md.set_speed(Motor._to_raw(frac))

    def read_qdr(self):
        # (position, speed, error1, error2), or None if the read fails
        # (device busy / unplugged).
        # Note that speed is raw driver units (2^-16 electrical rev/s) and not spee()'s -1..1
        # scale, divide by MOTOR_MAX_RAW to compare against a commanded fraction
        try:
            self.md.update_quick_data_readout()
            return (self.md.get_position_QDR(), self.md.get_speed_QDR(),
                    self.md.get_ERROR1_QDR(), self.md.get_ERROR2_QDR())
        except Exception:
            return None

    @classmethod
    def start_async(cls):
        # switch drive()/dwibble() to staged writes and start the flush
        # thread
        if cls._async:
            return
        cls._async = True
        threading.Thread(target=cls._flush_loop, daemon=True).start()

    @classmethod
    def _stage(cls, name, val):
        if not cls._async:
            m = cls.motors.get(name)
            if m is not None:
                m.spee(val)
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
        # ramp this motor to "target" (-1..1) in step increments. Unused by
        # the robot, drive() commands speed directly, kept working in
        # case a bench script wants it
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

    # speed command units
    # Every speed here, spee(), drive(), dwibble(), is -1..1, a signed
    # fraction of the motor's own no-load top speed
    #
    # Driver's setSpeed unit is 2^-16 electrical rps
    RPM_TO_RAW     = 7 * 36 / 60 * 2 ** 16
    DRIVER_MAX_RAW = 546_133_333
    MOTOR_MAX_RAW  = 546133333
    MAX_RPM        = MOTOR_MAX_RAW / RPM_TO_RAW
    MAX_REV_PER_S  = MAX_RPM / 60.0

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

    # drive model
    # No wheel runs past MOTOR_CAP_FRAC of full scale, if any would, the
    # whole command set scales down together so the motion shape is
    # preserved
    MOTOR_CAP_FRAC = 0.9
    TURN_RADIUS_MM = 110.0

    # No software accel/decel ramp: commanded speed and bearing both apply
    # the instant they're set, direction reversals included - real motion is
    # still bounded by actual motor torque/wheel traction, just not throttled
    # further on top of that. These two stay only as a conservative achievable
    # -accel estimate for mainrunbot*.py's _brake_speed_frac (braking-distance
    # cap) and COLLISION_ACCEL_G's own comment, not an active ramp.
    ACCEL_MAX_FRAC_PER_S = 5.0
    DECEL_MAX_FRAC_PER_S = 6.0

    @classmethod
    def drive(cls, bearing, speed, rot_r=0.0, rot_theta=0.0, rot_speed=0.0):
        # bearing, speed: robot-frame compass bearing (0deg=forward,
        # 90deg=right), magnitude 0-1. Rotation sums on top before the
        # one cap is applied, so quick spin leaves less for translation
        # rot_r, rot_theta need to be removed
        # rot_speed: spin command, -1..1 (positive = clockwise).
        rad = math.radians(bearing)
        f = math.cos(rad) * speed
        s = math.sin(rad) * speed
        trans = {"nw": f + s, "ne": -f + s, "sw": -f + s, "se": f + s}

        # rotation about (rot_r, rot_theta): pure spin + induced translation
        fi = si = 0.0
        if rot_speed and rot_r:
            ind_b = math.radians(rot_theta - math.copysign(90.0, rot_speed))
            ind_c = abs(rot_speed) * rot_r / (math.sqrt(2.0) * cls.TURN_RADIUS_MM)
            fi = math.cos(ind_b) * ind_c
            si = math.sin(ind_b) * ind_c

        speeds = {
            "nw": trans["nw"] + fi + si + rot_speed,
            "ne": trans["ne"] - fi + si + rot_speed,
            "sw": trans["sw"] - fi + si - rot_speed,
            "se": trans["se"] + fi + si - rot_speed,
        }

        # per-motor ceiling, scale everything down together
        peak = max(abs(v) for v in speeds.values())
        if peak > cls.MOTOR_CAP_FRAC:
            k = cls.MOTOR_CAP_FRAC / peak
            speeds = {n: v * k for n, v in speeds.items()}

        # se and sw are mounted with reversed polarity, negate the others
        sent = {n: float(v) if n in ("se", "sw") else float(-v)
                for n, v in speeds.items()}
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
        # Reset bot.motion's acceleration-slew state too, so the first command
        # after a stop ramps from 0 instead of springing from wherever the
        # stale _last_slew_speed happened to be. Imported lazily: importing
        # bot.motion at module scope would be circular (motion imports Motor
        # from bot.hardware). A plain name lookup on the module object also
        # sees any later rebinding of the module global, matching how the
        # rest of the code reads rebindable globals through their module.
        try:
            import bot.motion as _motion
            _motion._last_slew_speed = 0.0
        except Exception:
            pass  # motion not importable (bare-bench scripts) - nothing to reset
    @classmethod
    def clear_faults(cls):
        # call after stopall() on quit so a fault from this run doesn't ignore the next
        # start's commands.
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
