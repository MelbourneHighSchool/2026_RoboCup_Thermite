"""Hardware-adjacent stubs and the one canonical import of `Motor`.

`Motor` itself lives in the external `MotorFuncs_Proto1` module (not part of
this package). mainrunbot1.py imports it once at module scope
(`from bot.MotorFuncs_Proto1 import Motor`) and every other section just uses
the name. To give every later-stage bot.* module (odometry, compass, motion,
controllers, ...) one place to import Motor from instead of each doing its
own `from bot.MotorFuncs_Proto1 import Motor`, this module does that import and
re-exports it. Later modules should do `from bot.hardware import Motor`.
bot/main.py (stage 3+) is expected to do the same rather than importing
MotorFuncs_Proto1 directly.

The rest of this module is the commented-out solenoid-kicker scaffold,
carried over byte-for-byte from mainrunbot1.py. It is dead code - never
enabled, no hardware fitted - preserved exactly as a comment so a future
session can wire it up by uncommenting rather than re-deriving it.
"""

from bot.MotorFuncs_Proto1 import Motor

__all__ = ["Motor"]

# Solenoid kicker (not fitted): wire at the "# shoot hooks" markers and
# uncomment to enable. One GPIO pin, active-low, pulse_s per kick. Deps: gpiozero.
#
# import threading
#
# kick_pin    = 23      # BCM pin driving the kicker
# pulse_s     = 0.05    # coil on-time per kick
# cooldown_s  = 0.35    # min gap between kicks (coil thermal + cap recharge)
# active_high = False   # flip if your driver runs active-low instead
#
# class Solenoid:
#     """pulse a GPIO pin to fire the kicker, thread-safe, self-cooldown."""
#     def __init__(self, pin=kick_pin, active_high=active_high):
#         from gpiozero import DigitalOutputDevice   # lazy: load without libs
#         self.device = DigitalOutputDevice(pin, active_high=active_high,
#                                           initial_value=False)
#         self._lock = threading.Lock()
#         self._last_kick = 0.0
#     def kick(self):
#         with self._lock:
#             now = time.monotonic()
#             if now - self._last_kick < cooldown_s:
#                 return False
#             self._last_kick = now
#             self.device.on()
#             time.sleep(pulse_s)
#             self.device.off()
#             return True
#     def off(self):
#         self.device.off()
