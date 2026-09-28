"""Per-bot constants for bot2."""

# Each handle reads about 2.5 deg wide at this mount; 5.5 leaves margin.
handle_exclusion_deg = 5.5

# Crop carried over from the old 180-degree pipeline, transposed through this mount's
# extra 90 CW.
crop_top = 455
crop_bottom = 432
crop_left = 52

exclusion_inner_frac = 0.54 # inner 54% of radius ignored (robot body)
exclusion_outer_frac = 1.03 # detections past 103% of radius ignored (field clutter)

# Not bench-checked on this bot yet: a ball at the mouth should read 0.
cam_bearing_offset_deg = 0.0

MOTOR_PINS = {
    "nw": 28,
    "se": 31,
    "sw": 26,
    "ne": 25,
    "dwibble": 27,
}

# Bot 2 attacks by default; the role buttons can still switch it later.
DEFAULT_ROLE = "striker"
