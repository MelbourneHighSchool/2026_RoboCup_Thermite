"""
bot/calibration.py: ButtonCalib - drive one careful lap of the perimeter,
match every raw lidar hit against FieldModel's nominal geometry, and save the
offsets (a practice field isn't always exactly on-spec, e.g. a goal that isn't
dead-centre on x). goal_positions is where the rest of the code reads the
result back out.
"""

import json
import math
import time

# bot.motion first: importing it publishes keep_min_mm/goal_flank_keep_mm
# onto bot.field (see bot/motion.py's own docstring for that pattern), which
# the FieldModel import right below relies on already being in place.
from bot.motion import _wall_guard as wall_guard, wall_safe_speed_cmd, keep_min_mm
from bot.field import FieldModel, wrap_deg
from bot.hardware import Motor
from bot.perception import Perception
from bot.state import _lock as lock, _state as shared_state
import bot.state as state
# turn_gain is bot.controllers' constant (0.0012, shared across bots, not a
# per-bot gap); imported lazily inside _calib_routine below rather than at
# module scope to avoid a circular import (bot.controllers imports
# calib/goal_positions from this module).


# Field calibration (ButtonCalib): map the real field against FieldModel's nominal geometry, since a practice field isn't always exactly on-spec (e.g. a goal that isn't dead-centre on x).

# 150mm, same "never touch a wall" clearance used everywhere else
calib_standoff_mm    = keep_min_mm
# Pinned to wall_safe_speed_cmd, not base_speed: this routine drives precisely along a wall it's actively measuring.
calib_approach_speed = wall_safe_speed_cmd * 0.35
calib_follow_speed   = wall_safe_speed_cmd * 0.30
# how far ahead along the wall the steering target leads, each tick
calib_tangent_lead_mm = 250.0
# cumulative turn that means "back where we started" (about 1 lap, CW)
calib_lap_turn_deg   = 350.0
calib_timeout_s      = 90.0 # give up a stuck lap rather than loop forever
# a raw lidar hit counts as "on the wall" only this close to a nominal boundary segment
calib_hit_gate_mm    = 80.0
# how close to x=0/field_x or y=0/field_y a nominal point has to be to bucket as that wall
calib_edge_mm        = 10.0
# narrower gaps than this in the goal-line hits aren't trusted as "the goal mouth" (could
# just be a sparse-data hole)
calib_min_gap_mm     = 200.0
calib_file           = "field_calib.json"


def calib_still_active():
    """True while ButtonCalib's run_mode == "calib" still holds, checked by calib_routine to notice an abort."""
    with lock:
        return shared_state["run_mode"] == "calib"


def goal_centre_from_gap(xs, min_gap_mm=calib_min_gap_mm):
    """xs: sorted x-coordinates of boundary-wall hits along one goal-end wall."""
    if len(xs) < 4:
        return None
    best_gap, best_mid = 0.0, None
    for a, b in zip(xs, xs[1:]):
        if b - a > best_gap:
            best_gap, best_mid = b - a, (a + b) / 2.0
    return best_mid if best_gap >= min_gap_mm else None


def analyse_calib_samples(samples):
    """samples: list of (nominal_x, nominal_y, measured_x, measured_y), nominal is the FieldModel point the hit was matched against, measured is where the lidar actually saw it."""
    result = {"side_x0_offset_mm": None, "side_x1_offset_mm": None,
              "goal_low_centre_x": None, "goal_high_centre_x": None}
    if not samples:
        return result

    west  = [mx for nx, ny, mx, my in samples if nx < calib_edge_mm]
    east  = [mx for nx, ny, mx, my in samples
             if nx > FieldModel.field_x - calib_edge_mm]
    south = sorted(mx for nx, ny, mx, my in samples if ny < calib_edge_mm)
    north = sorted(mx for nx, ny, mx, my in samples
                   if ny > FieldModel.field_y - calib_edge_mm)

    if west:
        result["side_x0_offset_mm"] = sum(west) / len(west) - 0.0
    if east:
        result["side_x1_offset_mm"] = sum(east) / len(east) - FieldModel.field_x
    result["goal_low_centre_x"]  = goal_centre_from_gap(south)
    result["goal_high_centre_x"] = goal_centre_from_gap(north)
    return result


calib = {} # last-saved/loaded calibration, see load_calib() below


def load_calib():
    """load a previously-saved calibration from calib_file into the module-level calib dict, or {} if there isn't one."""
    global calib
    try:
        with open(calib_file) as f:
            calib = json.load(f)
    except (OSError, ValueError):
        calib = {}
    return calib


def save_calib(result):
    """apply `result` as the live calibration and persist it to calib_file."""
    global calib
    calib = result
    try:
        with open(calib_file, "w") as f:
            json.dump(result, f, indent=2)
    except OSError as e:
        print(f"[calib] could not save {calib_file}: {e}", flush=True)


load_calib() # pick up a previous calibration run, if any, at import time


def goal_positions(goal):
    """(own_goal, enemy_goal) field-mm centres for the given side, nudged by ButtonCalib's measured goal-centre-x if we have one, practice fields aren't always exactly on-spec."""
    low_cx  = calib.get("goal_low_centre_x")  or FieldModel.cx
    high_cx = calib.get("goal_high_centre_x") or FieldModel.cx
    if goal == "low":
        return (low_cx, 0.0), (high_cx, FieldModel.field_y)
    return (high_cx, FieldModel.field_y), (low_cx, 0.0)


def calib_routine():
    """runs to completion (or aborts early if a button takes us out of "calib"), then drops back to idle."""
    from bot.controllers import turn_gain
    print("[calib] starting, looking for the nearest wall...", flush=True)
    samples = [] # (nominal_x, nominal_y, measured_x, measured_y)

    # phase 1: approach
    while calib_still_active():
        with lock:
            pose = shared_state["pose"]
        if pose is None:
            Motor.stopall()
            time.sleep(0.05)
            continue
        rx, ry, hdg = pose
        dist, fx, fy, nx, ny = FieldModel.nearest_wall(rx, ry)
        if dist <= calib_standoff_mm + 20.0:
            break
        drive_rel = wrap_deg(math.degrees(math.atan2(fx - rx, fy - ry)) - hdg)
        speed = calib_approach_speed * min(1.0, (dist - calib_standoff_mm) / 300.0)
        drive_rel, speed = wall_guard(rx, ry, hdg, drive_rel, speed)
        Motor.drive(drive_rel, speed, rot_speed=0)
        time.sleep(0.02)
    Motor.stopall()
    if not calib_still_active():
        print("[calib] aborted during approach", flush=True)
        return

    # phase 2: follow the perimeter, sampling as we go
    print("[calib] wall found, following the perimeter...", flush=True)
    heading_turned = 0.0 # cumulative signed turn -> about 1 lap at +/-360
    hdg_prev       = None
    t_start        = time.monotonic()
    while (calib_still_active() and abs(heading_turned) < calib_lap_turn_deg
           and time.monotonic() - t_start < calib_timeout_s):
        with lock:
            pose = shared_state["pose"]
            pts  = list(shared_state["lidar_pts"])
        if pose is None:
            time.sleep(0.02)
            continue
        rx, ry, hdg = pose
        if hdg_prev is not None:
            heading_turned += wrap_deg(hdg - hdg_prev)
        hdg_prev = hdg

        dist, fx, fy, nx, ny = FieldModel.nearest_wall(rx, ry)
        tx, ty = ny, -nx # tangent, 90deg from the wall normal (CW lap)
        target_x = fx + nx * calib_standoff_mm + tx * calib_tangent_lead_mm
        target_y = fy + ny * calib_standoff_mm + ty * calib_tangent_lead_mm
        drive_rel = wrap_deg(math.degrees(math.atan2(target_x - rx,
                                                       target_y - ry)) - hdg)
        drive_rel, speed = wall_guard(rx, ry, hdg, drive_rel, calib_follow_speed)
        face = wrap_deg(math.degrees(math.atan2(tx, ty)) - hdg) # face along
        Motor.drive(drive_rel, speed, rot_speed=face * turn_gain) # the wall

        for wx, wy in Perception.transform(pts, rx, ry, hdg):
            wdist, wfx, wfy, wnx, wny = FieldModel.nearest_wall(wx, wy)
            if wdist <= calib_hit_gate_mm:
                samples.append((wfx, wfy, wx, wy))
        time.sleep(0.02)
    Motor.stopall()
    if not calib_still_active():
        print("[calib] aborted mid-lap", flush=True)
        return

    # phase 3: analyse
    result = analyse_calib_samples(samples)
    save_calib(result)
    print(f"[calib] done ({len(samples)} wall hits): {result}", flush=True)
    # Same as a colour/role button clearing its own set: back to idle means
    # every pick is cleared too, so a later button press can't silently
    # resume a combination chosen before calibration ran.
    state.enemy_goal_colour = None
    with lock:
        shared_state["slot_goal"]  = None
        shared_state["slot_role"]  = None
        shared_state["attack_low"] = None
        shared_state["run_mode"]   = "idle"
    if state.status_led is not None:
        state.status_led.off()
