"""Per-bot constants for bot1 (mainrunbot1.py), extracted from the
mainrunbot1.py/mainrunbot2.py diff.

Pure leaf module: no imports from anything in bot.*. The calibset([...])
payload arrays are shared hardware-SKU constants, not per-bot, and belong
in a later stage's bot/hardware.py, not here.
"""

handle_exclusion_deg = 4.0

base_speed     = 0.2 # normal open-field driving; see wall_safe_speed_cmd
rush_speed     = 0.5


wall_slide_zone_mm   = 200.0

crop_top    = 448
crop_bottom = 445
crop_left   = 38

exclusion_inner_frac = 0.58 # inner 58% of radius ignored (robot body)
exclusion_outer_frac = 1.01 # detections past 101% of radius ignored (field clutter)

# Was 90.0 (bench-confirmed at the time), now 0.0 after the debug overlay showed the handle/mouth wedges rotated 90deg CCW.
cam_bearing_offset_deg = 0.0

MOTOR_PINS = {
    "nw": 26,
    "se": 30,
    "sw": 31,
    "ne": 32,
    "dwibble": 29,
}

# bot1_config.py
DEFAULT_ROLE = "goalie"
