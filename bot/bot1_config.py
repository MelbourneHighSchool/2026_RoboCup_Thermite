"""Per-bot constants for bot1."""

handle_exclusion_deg = 4.0

crop_top = 448
crop_bottom = 445
crop_left = 38

exclusion_inner_frac = 0.58 # inner 58% of radius ignored (robot body)
exclusion_outer_frac = 1.01 # detections past 101% of radius ignored (field clutter)

# 0.0 after the debug overlay showed the handle/mouth wedges rotated 90 deg CCW at 90.0.
cam_bearing_offset_deg = 0.0

MOTOR_PINS = {
    "nw": 26,
    "se": 30,
    "sw": 31,
    "ne": 32,
    "dwibble": 29,
}

# Bot 1 defends by default; the role buttons can still switch it later.
DEFAULT_ROLE = "goalie"
