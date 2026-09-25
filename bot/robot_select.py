"""bot/robot_select.py: pick bot1_config/bot2_config by the ROBOT_ID env var
and publish every per-bot constant gap onto the bot.* modules that reference
it as an undefined name (documented in each of those modules' own
docstrings/comments - crop_top/crop_bottom/crop_left, exclusion_inner_frac/
exclusion_outer_frac, cam_bearing_offset_deg and handle_exclusion_deg on
bot.vision (and handle_exclusion_deg also on bot.debug_server), base_speed/
rush_speed on bot.controllers and bot.motion, wall_slide_zone_mm on
bot.motion, DEFAULT_ROLE on bot.main's startup role pre-pick).

A single reusable function so this doesn't get duplicated between the real
entry point (bot/main.py) and the mainrunbot1.py/mainrunbot2.py shims: all
three call select_and_publish_config() before anything else that could touch
these names.
"""

import os
import sys


def select_and_publish_config():
    """Read ROBOT_ID ("1" or "2", no other value/unset ever silently
    defaults - wrong motor pins would drive the wrong physical hardware),
    import the matching bot1_config/bot2_config, publish its per-bot
    constants as module attributes on every bot.* module that references
    them as an undefined name, and return the selected config module."""
    robot_id = os.environ.get("ROBOT_ID")
    if robot_id not in ("1", "2"):
        sys.exit(
            "ROBOT_ID env var must be set to \"1\" or \"2\" (got "
            f"{robot_id!r}) - refusing to guess, wrong motor pins would "
            "drive the wrong physical robot."
        )

    if robot_id == "1":
        import bot1_config as cfg
    else:
        import bot2_config as cfg

    import bot.vision as vision
    import bot.motion as motion
    import bot.controllers as controllers
    import bot.debug_server as debug_server

    # bot.vision: camera crop/exclusion/bearing geometry, plus the handle
    # exclusion wedge angle (also consumed by bot.debug_server's overlay).
    vision.crop_top              = cfg.crop_top
    vision.crop_bottom           = cfg.crop_bottom
    vision.crop_left             = cfg.crop_left
    vision.exclusion_inner_frac  = cfg.exclusion_inner_frac
    vision.exclusion_outer_frac  = cfg.exclusion_outer_frac
    vision.cam_bearing_offset_deg = cfg.cam_bearing_offset_deg
    vision.handle_exclusion_deg  = cfg.handle_exclusion_deg

    debug_server.handle_exclusion_deg = cfg.handle_exclusion_deg

    # bot.motion / bot.controllers: drive-speed and wall-slide constants.
    motion.base_speed            = cfg.base_speed
    motion.rush_speed            = cfg.rush_speed
    motion.wall_slide_zone_mm    = cfg.wall_slide_zone_mm

    controllers.base_speed       = cfg.base_speed
    controllers.rush_speed       = cfg.rush_speed

    return cfg
