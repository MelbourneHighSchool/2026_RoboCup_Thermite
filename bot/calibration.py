"""Field calibration: drive one careful lap of the perimeter, match raw lidar hits against the
nominal FieldModel, and save the offsets (practice fields aren't always on-spec, e.g. a goal
that isn't dead-centre on x). goal_positions reads the result back.

Nothing in bot/ starts calib_routine today (no button sets run_mode "calib"); the saved
field_calib.json is still loaded and used by goal_positions.
"""

import json
import math
import time

# bot.motion first: importing it publishes keep_min_mm/goal_flank_keep_mm onto bot.field
from bot.motion import _wall_guard as wall_guard, wall_safe_speed_cmd, keep_min_mm
from bot.field import FieldModel, wrap_deg
from bot.hardware import Motor
from bot.perception import Perception
from bot.state import _lock as lock, _state as shared_state
# the heading law (motion._heading_spin) is imported inside calib_routine: bot.controllers
# imports this module, so a top-level import would cycle



# the same 150 mm "never touch a wall" clearance used everywhere else
calib_standoff_mm = keep_min_mm
# Wall-safe speeds, not base_speed: this drives right along the wall it is measuring.
calib_approach_speed = wall_safe_speed_cmd * 0.35
calib_follow_speed = wall_safe_speed_cmd * 0.30
# how far ahead along the wall the steering target leads, each tick
calib_tangent_lead_mm = 250.0
# cumulative turn that means "back where we started" (about one CW lap)
calib_lap_turn_deg = 350.0
calib_timeout_s = 90.0 # give up a stuck lap rather than loop forever
# a raw lidar hit counts as on the wall only this close to a nominal boundary segment
calib_hit_gate_mm = 80.0
# how close to a field edge a nominal point must be to bucket as that wall
calib_edge_mm = 10.0
# narrower gaps in the goal-end hits could be a sparse-data hole, not the goal
calib_min_gap_mm = 200.0
calib_file = "field_calib.json"


def calib_still_active():
    """True while run_mode is still "calib"; calib_routine polls it to notice an abort."""
    with lock:
        return shared_state["run_mode"] == "calib"


def goal_centre_from_gap(xs, min_gap_mm=calib_min_gap_mm):
    """centre of the widest gap in the sorted x of one goal-end wall's hits, or None if too
    narrow.
    """
    if len(xs) < 4:
        return None
    best_gap, best_mid = 0.0, None
    for a, b in zip(xs, xs[1:]):
        if b - a > best_gap:
            best_gap, best_mid = b - a, (a + b) / 2.0
    return best_mid if best_gap >= min_gap_mm else None


def analyse_calib_samples(samples):
    """per-wall offsets and goal centres from (nominal_x, nominal_y, measured_x, measured_y)
    samples.
    """
    result = {"side_x0_offset_mm": None, "side_x1_offset_mm": None,
              "goal_low_centre_x": None, "goal_high_centre_x": None}
    if not samples:
        return result

    west = [mx for nx, ny, mx, my in samples if nx < calib_edge_mm]
    east = [mx for nx, ny, mx, my in samples
             if nx > FieldModel.field_x - calib_edge_mm]
    south = sorted(mx for nx, ny, mx, my in samples if ny < calib_edge_mm)
    north = sorted(mx for nx, ny, mx, my in samples
                   if ny > FieldModel.field_y - calib_edge_mm)

    if west:
        result["side_x0_offset_mm"] = sum(west) / len(west) - 0.0
    if east:
        result["side_x1_offset_mm"] = sum(east) / len(east) - FieldModel.field_x
    result["goal_low_centre_x"] = goal_centre_from_gap(south)
    result["goal_high_centre_x"] = goal_centre_from_gap(north)
    return result


calib = {} # last saved/loaded calibration


def load_calib():
    """load calib_file into the module-level calib dict, or {} if there isn't one."""
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


load_calib() # a previous calibration run, if any


def goal_positions(goal):
    """(own_goal, enemy_goal) centres in field mm for the given side, using the calibrated goal
    centre x when there is one.
    """
    low_cx = calib.get("goal_low_centre_x") or FieldModel.cx
    high_cx = calib.get("goal_high_centre_x") or FieldModel.cx
    if goal == "low":
        return (low_cx, 0.0), (high_cx, FieldModel.field_y)
    return (high_cx, FieldModel.field_y), (low_cx, 0.0)


def calib_routine():
    """drive the calibration lap, save the result, then drop back to idle (aborts if run_mode
    leaves "calib").
    """
    from bot.motion import _heading_spin
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
    heading_turned = 0.0 # cumulative signed turn, about one lap at +/-360
    hdg_prev = None
    t_start = time.monotonic()
    while (calib_still_active() and abs(heading_turned) < calib_lap_turn_deg
           and time.monotonic() - t_start < calib_timeout_s):
        with lock:
            pose = shared_state["pose"]
            pts = list(shared_state["lidar_pts"])
        if pose is None:
            time.sleep(0.02)
            continue
        rx, ry, hdg = pose
        if hdg_prev is not None:
            heading_turned += wrap_deg(hdg - hdg_prev)
        hdg_prev = hdg

        dist, fx, fy, nx, ny = FieldModel.nearest_wall(rx, ry)
        tx, ty = ny, -nx # tangent, 90 deg from the wall normal (CW lap)
        target_x = fx + nx * calib_standoff_mm + tx * calib_tangent_lead_mm
        target_y = fy + ny * calib_standoff_mm + ty * calib_tangent_lead_mm
        drive_rel = wrap_deg(math.degrees(math.atan2(target_x - rx,
                                                       target_y - ry)) - hdg)
        drive_rel, speed = wall_guard(rx, ry, hdg, drive_rel, calib_follow_speed)
        face = wrap_deg(math.degrees(math.atan2(tx, ty)) - hdg) # face along
        Motor.drive(drive_rel, speed, rot_speed=_heading_spin(face)) # the wall

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
    # Back to idle clears every pick too, so a later button press can't silently
    # resume a combination chosen before calibration ran.
    import bot.vision as vision
    vision._enemy_goal_colour = None
    with lock:
        shared_state["slot_goal"] = None
        shared_state["slot_role"] = None
        shared_state["attack_low"] = None
        shared_state["run_mode"] = "idle"
