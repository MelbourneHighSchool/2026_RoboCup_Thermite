#!/usr/bin/env python3
"""
bot/main.py: the entry point - parse the CLI flags, start every thread
(motors, lidar, camera, IMU, odometry, team link, debug server), wire up the
mode-select buttons, and hand off to play_loop, the one thread that's ever
allowed to command the drive motors during play.

Run it from the repo root: `python3 -m bot.main [--hsv]`.
"""

import argparse
import signal
import sys
import threading
import time
from collections import deque

from bot.calibration import calib_routine
from bot.compass import _compass_thread, imu_fusion_enabled
from bot.controllers import (CamRunController, GoalieController,
                             StrikerController, loop_dt)
from bot.debug_server import http_server_thread, render_thread, debug_port
import bot.diagnostics as diagnostics
from bot.dwibbler import _dwibble_camera_thread, _dwibbler_monitor_thread, _set_dwibbler, dwibble_calib
from bot.hardware import Motor
from bot.lidar import _lidar_thread, mcl
from bot.localisation import capture_startup_yaw
from bot.logs import (_capture_log_consts, _finish_capture_log,
                      _finish_lidar_log, _finish_motion_log,
                      _lidar_log_consts, _motion_log_consts,
                      _motion_log_thread)
from bot.odometry import WheelOdometry
from bot.state import _lock as lock, _state as shared_state
from bot.vision import _camera_thread, _init_hsv, sensor_size
# These are read through their modules, not from-imported: spiketypeshi.py
# forces feature flags off with setattr on the owning module, and main() has to
# see that (bot.network.bt_team_enabled, bot.odometry.wheel_odom_enabled), same
# reason bot.state's own rebindable globals are always read as state.X, and
# bot.vision's own _enemy_goal_colour (set here, read by the camera thread
# and CamRunController's front-goal check).
import bot.network as network
import bot.odometry as odometry
import bot.state as state
import bot.vision as vision
from bot.robot_select import select_and_publish_config

# ROBOT_ID selects bot1_config/bot2_config and publishes every per-bot
# constant gap (crop/exclusion/bearing geometry, drive speeds, wall-slide
# zone, handle exclusion, motor pins, DEFAULT_ROLE) onto the modules that
# reference them, before any thread starts or any function that could need
# them gets called. sys.exit()s loudly if ROBOT_ID isn't "1" or "2" - never
# silently default, wrong motor pins would drive the wrong physical robot.
_bot_cfg = select_and_publish_config()


def startup_yaw_worker():
    """Seed MCL's soft heading prior with a circular mean over a short burst
    of IMU yaw samples (that second reference project's own technique, see
    bot/localisation.py's capture_startup_yaw), rather than the first raw
    reading. Runs on its own daemon thread - see main()'s own comment for
    why it can't sample synchronously."""
    startup_yaw = capture_startup_yaw(timeout_s=15.0)
    if startup_yaw is not None:
        mcl.set_imu_yaw_prior(startup_yaw)
        print(f"[mcl] startup yaw reference: {startup_yaw:.1f}deg", flush=True)
    else:
        print("[mcl] no IMU yaw reference within timeout, proceeding without "
              "a soft prior", flush=True)


def play_loop():
    """the one thread that's ever allowed to command the drive motors during play."""
    goalie  = GoalieController()
    striker = StrikerController()
    camrun  = CamRunController()
    was_imu_pause_latched = False
    # Measured tick rate, not the nominal 1/loop_dt: a slow tick (e.g. a
    # localisation hiccup) shows up here same as it would in cam_fps/lidar_hz,
    # published for the debug page's /status readout (see bot/debug_server.py).
    tick_times = deque()
    try:
        while True:
            with lock:
                run_mode  = shared_state["run_mode"]
                slot_role = shared_state["slot_role"]
                imu_pause_latched = shared_state["imu_pause_latched"]

            # IMU sustained-fault forced stop (see bot.compass._compass_thread
            # / imu_fault_hold_s). Once latched, driving is refused outright - not just
            # this tick's Motor.stopall() below, but run_mode is forced back
            # to "idle" so the goalie/striker/camrun branches can't restart
            # anything - until the operator re-picks colour/role
            # (maybe_start) with the IMU healthy again, which is this
            # file's own equivalent of "pause and re-zero".
            if imu_pause_latched:
                if not was_imu_pause_latched:
                    print("[imu] sustained fault: forcing stop, re-pick "
                          "colour/role once the IMU is healthy again to "
                          "resume", flush=True)
                was_imu_pause_latched = True
                with lock:
                    if shared_state["run_mode"] != "idle":
                        shared_state["run_mode"] = "idle"
                Motor.stopall()
                _set_dwibbler(False)
                time.sleep(loop_dt)
                continue
            was_imu_pause_latched = False

            if run_mode == "calib":
                calib_routine()
            elif run_mode == "camrun":
                camrun.tick()
            elif slot_role == "goalie":
                goalie.tick()
            elif slot_role == "striker":
                striker.tick()
            else:
                Motor.stopall()
                _set_dwibbler(False)

            now = time.monotonic()
            tick_times.append(now)
            while tick_times and now - tick_times[0] > 1.0:
                tick_times.popleft()
            shared_state["loop_hz"] = len(tick_times)

            time.sleep(loop_dt)
    except KeyboardInterrupt:
        print("\n[main] stopped.")
        Motor.stopall()
        time.sleep(0.2)
        Motor.clear_faults()


# Main

def handle_sigterm(signum, frame):
    """systemd sends SIGTERM on stop/restart; the default handler would just kill us mid-tick, potentially leaving a driver holding its last commanded speed."""
    Motor.stopall()
    time.sleep(0.2)
    Motor.clear_faults()
    _finish_lidar_log()
    _finish_capture_log()
    _finish_motion_log()
    sys.exit(0)


def main():
    """entry point: parse args, start the motors/lidar/camera/team-link/http threads, wire up the mode-select button, and hand off to play_loop()."""
    signal.signal(signal.SIGTERM, handle_sigterm)

    ap = argparse.ArgumentParser()
    ap.add_argument("--hsv", action="store_true",
                    help="HSV ball-threshold tuner (no motors/lidar)")
    ap.add_argument("--lidarlog", nargs="?", type=float, const=60.0,
                    default=None, metavar="SECONDS",
                    help="log lidar localisation to logs/lidar_*.txt for "
                         "SECONDS (default 60), then stop logging and carry "
                         "on running. Park the robot first, with it parked, "
                         "every mm of pose movement in the log is error.")
    ap.add_argument("--capturelog", nargs="?", type=float, const=60.0,
                    default=None, metavar="SECONDS",
                    help="log bot/ball position+velocity and the yellow-zone "
                         "PD's own measures to logs/capture_*.txt for "
                         "SECONDS (default 60), then stop logging and carry "
                         "on running. Play normally, unlike --lidarlog this "
                         "one wants the robot actually chasing the ball.")
    ap.add_argument("--motionlog", nargs="?", type=float, const=60.0,
                    default=None, metavar="SECONDS",
                    help="log the capture-zone PD's own internals, the "
                         "resulting per-motor commands, and measured pose/"
                         "IMU/QDR telemetry to logs/motion_*.txt for "
                         "SECONDS (default 60) at motionlog_hz. Play "
                         "normally, see motion_debug.py.")
    args = ap.parse_args()

    _init_hsv()

    # --lidarlog: opened before any thread starts, so scan 1 is captured.
    # The lidar thread closes it once the duration is up and the robot keeps
    # running normally afterwards, logging is not a mode of its own.
    if args.lidarlog is not None:
        from lidar_debug import LidarLogger, new_session_path
        state._lidar_log = LidarLogger(new_session_path(),
                                       duration_s=args.lidarlog)
        state._lidar_log.header(_lidar_log_consts())
        print(f"[lidarlog] logging {args.lidarlog:.0f}s to {state._lidar_log.path}, "
              "leave the robot parked", flush=True)

    # --capturelog: same shape, but StrikerController.tick() closes it
    # (there is no separate capture thread, it's all in the play loop).
    if args.capturelog is not None:
        from capture_debug import CaptureLogger, new_session_path as new_capture_path
        state._capture_log = CaptureLogger(new_capture_path(),
                                           duration_s=args.capturelog)
        state._capture_log.header(_capture_log_consts())
        print(f"[capturelog] logging {args.capturelog:.0f}s to "
              f"{state._capture_log.path}, play normally", flush=True)

    # --motionlog is opened here (a no-op in hsv mode, no motors), its polling thread only starts once Motor is ready.
    if args.motionlog is not None:
        from motion_debug import MotionLogger, new_session_path as new_motion_path
        state._motion_log = MotionLogger(new_motion_path(),
                                         duration_s=args.motionlog)
        state._motion_log.header(_motion_log_consts())
        print(f"[motionlog] logging {args.motionlog:.0f}s to "
              f"{state._motion_log.path}, play normally", flush=True)

    state.mode = "hsv" if args.hsv else "run"
    # Neither goal colour nor role is a launch-time flag any more, both are picked live at the field with the mode-select buttons.
    cap_res = sensor_size

    # Motors only needed when actually playing (not the HSV tuner)
    if state.mode == "run":
        import board, busio
        i2c = busio.I2C(board.SCL, board.SDA)
        pins = _bot_cfg.MOTOR_PINS
        Motor(pins["nw"], i2c, name="nw").calibset([1451095040, 1227])
        Motor(pins["se"], i2c, name="se").calibset([1588074752, 1232])
        Motor(pins["sw"], i2c, name="sw").calibset([1234744320, 1241])
        Motor(pins["ne"], i2c, name="ne").calibset([1428907008, 1258])
        Motor(pins["dwibble"], i2c, name="dwibble").calibset(dwibble_calib) # ball roller
        # Staged motor writes on a flush thread (only deltas hit the bus, +
        # a keepalive rewrite) so the control loop never blocks on I2C.
        Motor.start_async()
        # Dwibbler-stall possession monitor (BallPossession, about 50 Hz QDR poll)
        threading.Thread(target=_dwibbler_monitor_thread, daemon=True).start()
        # Second camera aimed into the dwibbler mouth (sec 3.16), a direct
        # visual possession check; no-op until diagnostics.dwibble_cam_enabled
        # and the camera are both actually there.
        if diagnostics.dwibble_cam_enabled:
            threading.Thread(target=_dwibble_camera_thread, daemon=True).start()
        # --motionlog polling thread (see motion_debug.py), only worth
        # starting if the logger actually got opened above; it closes its
        # own logger when done.
        if state._motion_log is not None:
            threading.Thread(target=_motion_log_thread, daemon=True).start()
        # No solenoid fitted - see bot/hardware.py's own commented-out
        # Solenoid scaffold, uncomment there and here together to enable.

    threading.Thread(target=_camera_thread, args=(cap_res,), daemon=True).start()
    threading.Thread(target=render_thread,                   daemon=True).start()
    threading.Thread(target=http_server_thread,              daemon=True).start()
    print(f"[main] mode={state.mode}  http://10.98.141.100:{debug_port}/", flush=True)

    if state.mode == "hsv":
        print("[main] adjust sliders in browser, click Save when done.", flush=True)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[main] stopped.")
        return

    # Run mode
    if odometry.wheel_odom_enabled:
        odometry._wheel_odom = WheelOdometry()
    threading.Thread(target=_lidar_thread,                 daemon=True).start()
    threading.Thread(target=_compass_thread,               daemon=True).start()
    threading.Thread(target=diagnostics._health_thread,    daemon=True).start()
    # MCL's soft heading prior, on its own daemon thread: _compass_thread has
    # only just started, so shared_state["imu_heading"] isn't populated yet -
    # capture_startup_yaw's own retry loop waits for the first readings to
    # arrive rather than blocking startup here.
    threading.Thread(target=startup_yaw_worker, daemon=True).start()
    # Teammate channels, both off while playing solo (team_play_enabled).
    if network.team_play_enabled:
        threading.Thread(target=network._udp_send_thread,  daemon=True).start()
        threading.Thread(target=network._udp_recv_thread,  daemon=True).start()

    # Mode-select buttons (colour + role, auto-start): four momentary gpiozero Buttons, no separate run/idle toggle, play starts once both a goal colour and role are picked.
    from gpiozero import Button, LED
    button_blue_pin     = 20 # Placeholder, confirm real wiring
    button_yellow_pin   = 21 # Placeholder
    button_attacker_pin = 22 # Placeholder
    button_goalie_pin   = 23 # Placeholder
    button_calib_pin    = 19 # not fitted
    status_led_pin      = 16

    # slot_goal/attack_low, fixed the instant play starts (maybe_start), never re-derived from the camera afterward: a single-frame "goal seen ahead" check used to decide this and would occasionally misfire, sending the whole match's own-goal/enemy-goal sense backward with no way to self-correct.
    default_slot_goal = "low" # "low" | "high"

    button_blue     = Button(button_blue_pin)
    button_yellow   = Button(button_yellow_pin)
    button_attacker = Button(button_attacker_pin)
    button_goalie   = Button(button_goalie_pin)
    # button_calib    = Button(button_calib_pin)
    state._status_led = LED(status_led_pin)

    def led_for_run_mode():
        """set the status LED to match shared_state["run_mode"]: solid for "run", fast blink for "calib", slow blink for "camrun" (no lidar), off for "idle"."""
        with lock:
            mode = shared_state["run_mode"]
        if mode == "run":
            state._status_led.on()
        elif mode == "calib":
            state._status_led.blink(0.15, 0.15)
        elif mode == "camrun":
            state._status_led.blink(0.5, 0.5)
        else:
            state._status_led.off()

    def stop_and_idle():
        """stop the motors and clear any latched driver faults, then force run_mode back to idle."""
        Motor.stopall()
        time.sleep(0.2)
        Motor.clear_faults()
        with lock:
            shared_state["run_mode"]     = "idle"
            shared_state["bt_is_master"] = None # re-open the master/slave pick
        led_for_run_mode()

    def maybe_start():
        """call after any colour/role pick: start playing the moment both sets are filled, judged purely on the button selections. An IMU forced-stop latch (see "imu_pause_latched" / play_loop) only clears once the operator confirms and re-zeroes: both button slots being (re-)picked here IS the operator's confirmation, but it only actually clears the latch and starts play if the IMU is healthy again right now - a still-faulted IMU keeps the robot idle no matter how many times the buttons are pressed, and says so, rather than starting on stale/bad orientation data."""
        with lock:
            role      = shared_state["slot_role"]
            imu_fault = shared_state["imu_fault"]
        if vision._enemy_goal_colour is not None and role is not None:
            if imu_fault:
                print("[imu] still faulted, cannot resume yet - wait for "
                      "\"[health] imu: ok\" and press the buttons again",
                      flush=True)
                stop_and_idle()
                return
            with lock:
                if shared_state["slot_goal"] is None:
                    shared_state["slot_goal"]  = default_slot_goal
                    shared_state["attack_low"] = default_slot_goal == "high"
                shared_state["force_relocalise"]  = True
                shared_state["run_mode"]          = "run"
                shared_state["imu_pause_latched"] = False # operator confirmed with the IMU healthy, clear the latch
            led_for_run_mode()
        else:
            stop_and_idle()

    def select_colour(colour):
        """Blue/Yellow button."""
        with lock:
            running = shared_state["run_mode"] == "run"
        if running:
            vision._enemy_goal_colour = None
            with lock:
                shared_state["slot_goal"]        = None
                shared_state["attack_low"]       = None
                shared_state["force_relocalise"] = True
            stop_and_idle()
            print("[main] goal colour cleared, press Blue or Yellow "
                  "again to resume", flush=True)
            return
        vision._enemy_goal_colour = colour
        print(f"[main] enemy goal colour set: {colour}", flush=True)
        maybe_start()

    def select_role(role):
        """Attacker/Goalie button, same clear-vs-pick shape as select_colour, for the role slot."""
        with lock:
            running                       = shared_state["run_mode"] == "run"
            shared_state["slot_role"]     = None if running else role
            shared_state["own_slot_role"] = None if running else role
        if running:
            stop_and_idle()
            print("[main] role cleared, press Attacker or Goalie again "
                  "to resume", flush=True)
            return
        print(f"[main] role set: {role}", flush=True)
        maybe_start()

    def select_calib():
        """ButtonCalib: jump straight to run_mode "calib", bypassing the slots."""
        with lock:
            shared_state["run_mode"] = "calib"
        led_for_run_mode()

    def select_camrun():
        """sec 4.9: jump straight to run_mode "camrun", bypassing the colour/role slots the same way select_calib bypasses them for "calib", CamRunController needs neither (no enemy_goal_positions maths, no dynamic role handoff)."""
        with lock:
            running = shared_state["run_mode"] == "camrun"
        if running:
            stop_and_idle()
            print("[main] camera-only mode stopped", flush=True)
            return
        with lock:
            fwd0                                   = shared_state["imu_heading"]
            shared_state["run_mode"]               = "camrun"
            shared_state["camrun_forward_heading"] = fwd0
        led_for_run_mode()
        print("[main] camera-only mode: no lidar/pose, camera + dwibbler "
              "possession only, forward heading "
              + (f"locked at {fwd0:.1f} deg" if fwd0 is not None
                 else "not locked (no IMU), push direction will just be "
                      "whatever way the robot is currently facing"),
              flush=True)

    import sys, tty, termios

    def key_watch():
        """keyboard stand-ins for the four buttons, for testing without hardware: 1=Blue 2=Yellow 3=Attacker 4=Goalie 5=camera-only mode (sec 4.9, no real button wired for this one yet)."""
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        keymap = {
            "1": lambda: select_colour("cyan"),
            "2": lambda: select_colour("yellow"),
            "3": lambda: select_role("striker"),
            "4": lambda: select_role("goalie"),
            "5": select_camrun,
        }
        try:
            while True:
                fn = keymap.get(sys.stdin.read(1))
                if fn is not None:
                    fn()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    threading.Thread(target=key_watch, daemon=True).start()

    button_blue.when_pressed     = lambda: select_colour("cyan")
    button_yellow.when_pressed   = lambda: select_colour("yellow")
    button_attacker.when_pressed = lambda: select_role("striker")
    button_goalie.when_pressed   = lambda: select_role("goalie")
    # Uncomment alongside the matching Button() line above once fitted:
    # button_calib.when_pressed    = select_calib

    # DEFAULT_ROLE (per-bot, see bot1_config.py/bot2_config.py, published
    # through bot.robot_select): pre-pick the bot's configured default role at
    # startup - bot2 pre-picks striker (reproducing its original
    # _select_role("striker") pre-pick), bot1 pre-picks goalie - so only the
    # goal-colour button is then needed to start play (the role buttons still
    # work afterward, e.g. to switch roles by hand).
    if _bot_cfg.DEFAULT_ROLE is not None:
        select_role(_bot_cfg.DEFAULT_ROLE)

    srcs = "lidar" + ("+imu" if imu_fusion_enabled else "")
    print(f"[main] idle, GPIO {button_blue_pin}/{button_yellow_pin} pick "
          f"the enemy goal colour, {button_attacker_pin}/{button_goalie_pin}"
          f" pick the role; default role = {_bot_cfg.DEFAULT_ROLE}; "
          f"play starts once both are picked"
          f"{'' if network.team_play_enabled else ' (solo)'}; "
          f"localisation = {srcs}; key 5 = camera-only mode, no lidar "
          "(sec 4.9)", flush=True)

    if network.bt_team_enabled:
        threading.Thread(target=network._bt_link_thread, daemon=True).start()

    play_loop()


if __name__ == "__main__":
    main()
