"""Pick the per-robot config from ROBOT_ID and publish its constants onto the modules that use
them.
"""

import os
import sys


def select_and_publish_config():
    """read ROBOT_ID ("1" or "2", never a silent default: wrong pins drive the wrong robot),
    import that bot's config, publish its per-bot constants onto bot.vision and
    bot.debug_server, and return the config module.
    """
    robot_id = os.environ.get("ROBOT_ID")
    if robot_id not in ("1", "2"):
        sys.exit(
            "ROBOT_ID env var must be set to \"1\" or \"2\" (got "
            f"{robot_id!r}), refusing to guess: wrong motor pins would "
            "drive the wrong physical robot."
        )

    if robot_id == "1":
        from bot import bot1_config as cfg
    else:
        from bot import bot2_config as cfg

    import bot.vision as vision
    import bot.debug_server as debug_server

    # camera crop/exclusion/bearing geometry and the lidar handle wedge (also drawn by
    # the debug overlay)
    vision.crop_top = cfg.crop_top
    vision.crop_bottom = cfg.crop_bottom
    vision.crop_left = cfg.crop_left
    vision.exclusion_inner_frac = cfg.exclusion_inner_frac
    vision.exclusion_outer_frac = cfg.exclusion_outer_frac
    vision.cam_bearing_offset_deg = cfg.cam_bearing_offset_deg
    vision.handle_exclusion_deg = cfg.handle_exclusion_deg

    debug_server.handle_exclusion_deg = cfg.handle_exclusion_deg

    return cfg
