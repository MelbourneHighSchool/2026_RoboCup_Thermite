"""Shared drive constants for every robot."""

# Command levels, as fractions of the full motor command range (1.0 = about 1984 wheel rpm,
# 7.3 m/s body speed on paper). base_speed is normal open-field driving; everything
# situational (carry creep, duel strafe, hide, wall approach, the braking curves) is a
# fraction of it or tapers below it, so play runs 0.1-0.3. The keeper's positioning runs
# to 0.41 (motion.goalie_max_frac) and a contest for a held ball at 0.4.
#
# rush_speed is the clear-lane burst ("send it"). The match code ran 0.25 / 0.4, and 0.4
# already out-ran and shoved opponents. On the robot the measured top speed is about 0.5
# (1000 rpm), so rush sits at the top. Commands past it
# saturate the speed loop: full effort but no more speed, and the wheel ratios that steer
# and hold heading distort.
base_speed = 0.3
rush_speed = 0.5
wall_slide_zone_mm = 300.0

# Encoder calibration [elecangle, sincos] per drive motor. Both robots share one set:
# same motors, drivers and wheels. A saved motor_calibration.json (Motor.calibrate_all)
# overrides every motor it covers; tests/motorcalib.py prints fresh values.
MOTOR_CALIB = {
    "nw": [1451095040, 1227],
    "se": [1588074752, 1232],
    "sw": [1234744320, 1241],
    "ne": [1428907008, 1258],
}

# Each half of the visible boot check: rotate, then rotate back to the start.
BOOT_SWIVEL_SPEED = 0.08
BOOT_SWIVEL_SECONDS = 0.35
