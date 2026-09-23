"""Per-bot constants for bot2 (mainrunbot2.py), extracted from the
mainrunbot1.py/mainrunbot2.py diff.

Pure leaf module: no imports from anything in bot.*. The calibset([...])
payload arrays are shared hardware-SKU constants, not per-bot, and belong
in a later stage's bot/hardware.py, not here.
"""

# 5.5deg: each handle's apparent width works out to about 2.5deg (handle size/mount offset), rounded up with margin.
handle_exclusion_deg = 5.5

base_speed     = 0.45 # normal open-field driving; see wall_safe_speed_cmd
rush_speed     = 1.0

# damp the into-wall component within this band outside the keep line
wall_slide_zone_mm   = 200.0

# Carried over from the old 180-rotation pipeline's own crop, transposed through the extra 90CW this mount needs.
crop_top    = 455
crop_bottom = 432
crop_left   = 52

exclusion_inner_frac = 0.54 # inner 50% of radius ignored (robot body)
exclusion_outer_frac = 1.03 # outer 100% of radius ignored (field clutter)

# Not yet bench-checked on this bot (ball at the physical mouth should read 0); happens to match mainrunbot1.py's current value but isn't verified either way.
cam_bearing_offset_deg = 0.0

MOTOR_PINS = {
    "nw": 28,
    "se": 31,
    "sw": 26,
    "ne": 25,
    "dwibble": 27,
}

# This bot always attacks: pre-picks the striker role at startup so only the
# goal-colour button is needed to start play (the role buttons still work,
# e.g. to switch it to goalie by hand).
ALWAYS_ATTACKER = True
