"""Per-bot constants for bot1 (mainrunbot1.py), extracted from the
mainrunbot1.py/mainrunbot2.py diff.

Pure leaf module: no imports from anything in bot.*. The calibset([...])
payload arrays are shared hardware-SKU constants, not per-bot, and belong
in a later stage's bot/hardware.py, not here.
"""

handle_exclusion_deg = 4.0

base_speed     = 0.1 # normal open-field driving; see wall_safe_speed_cmd
rush_speed     = 0.4

# damp the into-wall component within this band outside the keep line. Must
# cover one pose-staleness interval (~100ms) of travel at rush_speed, or a
# fast, close approach can cross keep_min_mm before the guard ever sees it;
# 400 assumes rush_speed around 0.4, widen further if rush_speed goes up.
wall_slide_zone_mm   = 400.0

crop_top    = 448
crop_bottom = 445
crop_left   = 38

exclusion_inner_frac = 0.58 # inner 50% of radius ignored (robot body)
exclusion_outer_frac = 1.01 # outer 100% of radius ignored (field clutter)

# Was 90.0 (bench-confirmed at the time), now 0.0 after the debug overlay showed the handle/mouth wedges rotated 90deg CCW.
cam_bearing_offset_deg = 0.0

MOTOR_PINS = {
    "nw": 26,
    "se": 30,
    "sw": 31,
    "ne": 32,
    "dwibble": 29,
}

# This bot waits for the role buttons; it does not pre-select striker at startup.
ALWAYS_ATTACKER = False
