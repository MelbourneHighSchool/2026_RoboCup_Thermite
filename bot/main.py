#!/usr/bin/env python3
"""Entry point: parse the CLI flags, start every thread (motors, lidar, camera, IMU, odometry,
team link, debug server), wire up the mode buttons, and run play_loop, the only thread that
commands the drive motors during play.
"""

import argparse
import os
import signal
import sys
import threading
import time
from collections import deque

import bot.compass as compass
from bot.compass import _compass_thread
from bot.controllers import (CamRunController, GoalieController,
                             StrikerController, loop_dt)
from bot.debug_server import http_server_thread, render_thread, debug_port
import bot.diagnostics as diagnostics
from bot.dwibbler import (_dwibble_camera_thread, _dwibbler_monitor_thread, _possession,
                          _set_dwibbler, dwibble_calib)
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
# Read through their modules, not from-imported, so a test or harness that flips a flag on
# the owning module (network.bt_team_enabled, odometry.wheel_odom_enabled) is seen here;
# likewise vision._enemy_goal_colour, set here and read by the camera thread and camrun.
import bot.controllers as controllers
import bot.drive_config as drive_config
import bot.debug_session as debug_session
import bot.network as network
import bot.odometry as odometry
import bot.state as state
import bot.vision as vision
from bot.robot_select import select_and_publish_config

# ROBOT_ID picks bot1_config/bot2_config and publishes its per-bot constants before any
# thread starts. Exits loudly on anything but "1" or "2": wrong motor pins drive the wrong
# robot.
_bot_cfg = select_and_publish_config()


def startup_yaw_worker():
    """seed MCL's soft heading prior from a circular mean over a short burst of IMU yaw
    samples; runs in a separate thread because the compass thread has only just started.
    """
    startup_yaw = capture_startup_yaw(timeout_s=15.0)
    if startup_yaw is not None:
        mcl.set_imu_yaw_prior(startup_yaw)
        print(f"[mcl] startup yaw baseline: {startup_yaw:.1f}deg", flush=True)
    else:
        print("[mcl] no IMU yaw baseline within timeout, proceeding without "
              "a soft prior", flush=True)


def _local_ip():
    """this Pi's LAN address for the debug-page hint (a UDP connect sends nothing)."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
            sk.connect(("10.255.255.255", 1))
            return sk.getsockname()[0]
    except OSError:
        return "localhost"


def run_boot_swivel():
    """Give a brief, in-place movement indication once the motor bus is ready."""
    speed = drive_config.BOOT_SWIVEL_SPEED
    duration_s = drive_config.BOOT_SWIVEL_SECONDS
    if speed <= 0 or duration_s <= 0:
        return
    print(f"[main] boot swivel: {duration_s:.2f}s each way", flush=True)
    try:
        Motor.drive(0, 0, rot_speed=speed)
        time.sleep(duration_s)
        Motor.drive(0, 0, rot_speed=-speed)
        time.sleep(duration_s)
    finally:
        Motor.stopall()


def play_loop():
    """the one thread that's ever allowed to command the drive motors during play."""
    goalie = GoalieController()
    striker = StrikerController()
    camrun = CamRunController()
    prev_role = None
    was_imu_pause_latched = False
    # measured tick rate, not the nominal 1/loop_dt, for the debug page's /status
    tick_times = deque()
    try:
        while True:
            with lock:
                run_mode = shared_state["run_mode"]
                slot_role = shared_state["slot_role"]
                imu_pause_latched = shared_state["imu_pause_latched"]

            # IMU sustained-fault forced stop: once latched, driving is
            # refused and run_mode is forced to idle, so no controller can
            # restart anything until the operator re-picks colour/role with
            # the IMU healthy again
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
                # keep recording through the stop, it's worth watching back
                debug_session.record_tick()
                time.sleep(loop_dt)
                continue
            was_imu_pause_latched = False

            # A role swap starts the incoming role from scratch: a striker that last
            # ran minutes ago would otherwise resume mid-carry or mid-pass. A
            # keeper handing over with the ball in its roller counts as a close
            # ball just now, so the striker's roller-stall override takes it on
            # its first tick.
            if slot_role != prev_role:
                if prev_role is not None:
                    if slot_role == "striker":
                        striker = StrikerController()
                        if _possession.has_ball:
                            striker.close_ball_t = time.monotonic()
                    elif slot_role == "goalie":
                        goalie = GoalieController()
                prev_role = slot_role

            if run_mode == "camrun":
                camrun.tick()
            elif slot_role == "goalie":
                goalie.tick()
            elif slot_role == "striker":
                striker.tick()
            else:
                Motor.stopall()
                _set_dwibbler(False)

            # debug session recording: one snapshot per tick while it is on, a single
            # flag check otherwise
            debug_session.record_tick()

            now = time.monotonic()
            tick_times.append(now)
            while tick_times and now - tick_times[0] > 1.0:
                tick_times.popleft()
            shared_state["loop_hz"] = len(tick_times)

            time.sleep(loop_dt)
    except KeyboardInterrupt:
        print("\n[main] stopped.")
        debug_session.finish()
        Motor.stopall()
        time.sleep(0.2)
        Motor.clear_faults()


# Main

def handle_sigterm(signum, frame):
    """systemd sends SIGTERM on stop/restart; the default handler would kill us mid-tick and
    could leave a driver holding its last speed.
    """
    Motor.stopall()
    time.sleep(0.2)
    Motor.clear_faults()
    _finish_lidar_log()
    _finish_capture_log()
    _finish_motion_log()
    debug_session.finish()
    sys.exit(0)


def main():
    """entry point: parse args, start the threads, wire up the mode buttons, and hand off to
    play_loop().
    """
    signal.signal(signal.SIGTERM, handle_sigterm)

    ap = argparse.ArgumentParser()
    ap.add_argument("--hsv", action="store_true",
                    help="HSV ball-threshold tuner (no motors/lidar)")
    ap.add_argument("--no-boot-swivel", action="store_true",
                    help="skip the brief motor boot self-test")
    ap.add_argument("--lidarlog", nargs="?", type=float, const=60.0,
                    default=None, metavar="seconds",
                    help="log lidar localisation to lidar.txt in the debug session for "
                         "seconds (default 60), then stop logging and carry "
                         "on running. Park the robot first, with it parked, "
                         "every mm of pose movement in the log is error.")
    ap.add_argument("--capturelog", nargs="?", type=float, const=60.0,
                    default=None, metavar="seconds",
                    help="log bot/ball position+velocity and the yellow-zone "
                         "PD's measures to capture.txt in the debug session for "
                         "seconds (default 60), then stop logging and carry "
                         "on running. Play normally, unlike --lidarlog this "
                         "one wants the robot actually chasing the ball.")
    ap.add_argument("--motionlog", nargs="?", type=float, const=60.0,
                    default=None, metavar="seconds",
                    help="log the capture-zone PD's internals, the "
                         "resulting per-motor commands, and measured pose/"
                         "IMU/QDR telemetry to motion.txt in the debug session for "
                         "seconds (default 60) at motionlog_hz. Play "
                         "normally, see motion_debug.py.")
    ap.add_argument("--kickoff", choices=("kicking", "receiving"),
                    default=None,
                    help="RCJA pre-match placement for the next play start: "
                         "'kicking' (striker to the centre ball, goalie in "
                         "front of goal) or 'receiving' (both bots lined up "
                         "in/behind the box). The hold runs kickoff_hold_s "
                         "from play start, or ends early once the ball moves "
                         "toward us. Omit for no kickoff hold.")
    ap.add_argument("--debug", action="store_true",
                    help="record this power-on as a debug session in "
                         "logs/debug_r<id>_<time>/: every control tick (pose, "
                         "ball, enemies, state, motor commands), for "
                         "simulator.py. Also DEBUG_SESSION=1 or the switch on "
                         "the debug page.")
    args = ap.parse_args()
    # which side of the kick-off we're on is an operator fact the robot can't sense
    kickoff_side = args.kickoff

    _init_hsv()

    # --lidarlog: opened before any thread starts, so scan 1 is captured. The lidar
    # thread closes it when the duration is up and play carries on normally.
    if args.lidarlog is not None:
        from bot.lidar_debug import LidarLogger
        state._lidar_log = LidarLogger(debug_session.path("lidar.txt"),
                                       duration_s=args.lidarlog)
        state._lidar_log.header(_lidar_log_consts())
        print(f"[lidarlog] logging {args.lidarlog:.0f}s to {state._lidar_log.path}, "
              "leave the robot parked", flush=True)

    # --capturelog: same shape, closed by StrikerController.tick()
    if args.capturelog is not None:
        from bot.capture_debug import CaptureLogger
        state._capture_log = CaptureLogger(debug_session.path("capture.txt"),
                                           duration_s=args.capturelog)
        state._capture_log.header(_capture_log_consts())
        print(f"[capturelog] logging {args.capturelog:.0f}s to "
              f"{state._capture_log.path}, play normally", flush=True)

    # --motionlog: opened here (a no-op in hsv mode); its poll thread starts once the
    # motors are up
    if args.motionlog is not None:
        from bot.motion_debug import MotionLogger
        state._motion_log = MotionLogger(debug_session.path("motion.txt"),
                                         duration_s=args.motionlog)
        state._motion_log.header(_motion_log_consts())
        print(f"[motionlog] logging {args.motionlog:.0f}s to "
              f"{state._motion_log.path}, play normally", flush=True)

    state.mode = "hsv" if args.hsv else "run"
    # Debug session: --debug, or DEBUG_SESSION=1 (set it in systemd's Environment= to
    # record every match). One session per power-on.
    if args.debug or os.environ.get("DEBUG_SESSION", "") == "1":
        debug_session.set_recording(True)
    # goal colour and role are both picked live at the field with the mode buttons
    cap_res = sensor_size

    # Motors only needed when actually playing (not the HSV tuner)
    if state.mode == "run":
        import board, busio
        i2c = busio.I2C(board.SCL, board.SDA)
        pins = _bot_cfg.MOTOR_PINS
        # Encoder calibration: the shared defaults, overridden by any motor a saved
        # motor_calibration.json covers, so re-calibrating never means editing source.
        calib = Motor.resolve_calibration(
            dict(drive_config.MOTOR_CALIB, dwibble=dwibble_calib),
            {_motor: pins[_motor] for _motor in ("nw", "se", "sw", "ne", "dwibble")})
        for _motor in ("nw", "se", "sw", "ne", "dwibble"):
            Motor(pins[_motor], i2c, name=_motor).calibset(calib[_motor])
        # staged motor writes on a flush thread (only deltas, plus a keepalive) so
        # the control loop never blocks on I2C
        Motor.start_async()
        if not args.no_boot_swivel:
            run_boot_swivel()
        # Dwibbler-stall possession monitor (BallPossession, about 50 Hz QDR poll)
        threading.Thread(target=_dwibbler_monitor_thread, daemon=True).start()
        # second camera aimed into the dwibbler mouth: a direct visual possession
        # check
        if diagnostics.dwibble_cam_enabled:
            threading.Thread(target=_dwibble_camera_thread, daemon=True).start()
        # --motionlog poll thread, only if the logger opened above; it closes the
        # logger when done
        if state._motion_log is not None:
            threading.Thread(target=_motion_log_thread, daemon=True).start()
        # no solenoid fitted: see bot/hardware.py's commented-out Solenoid,
        # uncomment there and here together

    threading.Thread(target=_camera_thread, args=(cap_res,), daemon=True).start()
    threading.Thread(target=render_thread, daemon=True).start()
    threading.Thread(target=http_server_thread, daemon=True).start()
    print(f"[main] mode={state.mode}  http://{_local_ip()}:{debug_port}/", flush=True)

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
    threading.Thread(target=_lidar_thread, daemon=True).start()
    # PosePropagator: carry the lidar fit forward with wheel odometry and the fused
    # IMU heading, publishing "pose_live" at about 50 Hz for the white-line gate
    if odometry.pose_propagate_enabled:
        threading.Thread(target=odometry._pose_propagator_thread, daemon=True).start()
    threading.Thread(target=_compass_thread, daemon=True).start()
    threading.Thread(target=diagnostics._health_thread, daemon=True).start()
    # MCL's soft heading prior in a separate thread: imu_heading isn't populated yet, and
    # capture_startup_yaw waits for the first readings rather than blocking startup
    threading.Thread(target=startup_yaw_worker, daemon=True).start()
    # Teammate channels, both off while playing solo (team_play_enabled).
    if network.team_play_enabled:
        threading.Thread(target=network._udp_send_thread, daemon=True).start()
        threading.Thread(target=network._udp_recv_thread, daemon=True).start()

    # Four momentary buttons select goal colour or role. A role is loaded at boot, so
    # a goal-colour pick is the final confirmation that starts play.
    from gpiozero import Button
    button_blue_pin = 19
    button_yellow_pin = 20
    button_attacker_pin = 21
    button_goalie_pin = 22

    # slot_goal/attack_low are fixed when play starts and never re-derived from the
    # camera: a single-frame "goal seen ahead" check used to decide this and
    # occasionally flipped the whole match's sense of which goal is ours
    default_slot_goal = "low" # "low" | "high"

    button_blue = Button(button_blue_pin)
    button_yellow = Button(button_yellow_pin)
    button_attacker = Button(button_attacker_pin)
    button_goalie = Button(button_goalie_pin)

    def stop_and_idle():
        """Stop the motors and return to the sensing-only idle state."""
        Motor.stopall()
        time.sleep(0.2)
        Motor.clear_faults()
        with lock:
            shared_state["run_mode"] = "idle"
            shared_state["bt_is_master"] = None # re-open the master/slave pick

    def reset_to_fresh_state():
        """Discard a running match selection and restore this bot's default role."""
        stop_and_idle()
        vision._enemy_goal_colour = None
        with lock:
            shared_state["slot_goal"] = None
            shared_state["attack_low"] = None
            shared_state["slot_role"] = _bot_cfg.DEFAULT_ROLE
            shared_state["own_slot_role"] = _bot_cfg.DEFAULT_ROLE
            shared_state["force_relocalise"] = True
        print(f"[main] fresh state; default role: {_bot_cfg.DEFAULT_ROLE}",
              flush=True)

    def reset_if_running():
        """A button press during play only resets; it does not select itself."""
        with lock:
            running = shared_state["run_mode"] == "run"
        if running:
            reset_to_fresh_state()
        return running

    def maybe_start():
        """call after any colour/role pick: start playing once both are set. Picking them again
        is also the operator's confirmation that clears an IMU forced-stop latch, but only
        if the IMU is healthy right now; a still-faulted IMU keeps the robot idle, and says
        so.
        """
        with lock:
            role = shared_state["slot_role"]
            imu_fault = shared_state["imu_fault"]
        if vision._enemy_goal_colour is not None and role is not None:
            if imu_fault:
                print("[imu] still faulted, cannot resume yet. Wait for "
                      "\"[health] imu: ok\" and press the buttons again",
                      flush=True)
                stop_and_idle()
                return
            with lock:
                if shared_state["slot_goal"] is None:
                    shared_state["slot_goal"] = default_slot_goal
                    shared_state["attack_low"] = default_slot_goal == "high"
                shared_state["force_relocalise"] = True
                shared_state["run_mode"] = "run"
                # operator confirmed with the IMU healthy, clear the latch
                shared_state["imu_pause_latched"] = False
                # Kickoff hold (see kickoff_hold_s in bot/controllers.py):
                # which side of the kick-off we're on is known only to the
                # operator. Set only here at a fresh play start, so a
                # mid-match reset replays the hold like a real restart;
                # the controllers clear the keys when the window expires
                # or the ball moves.
                if kickoff_side is not None:
                    import time as _t
                    shared_state["kickoff_role"] = kickoff_side
                    shared_state["kickoff_until_t"] = (_t.monotonic()
                                                       + controllers.kickoff_hold_s)
        else:
            stop_and_idle()

    def select_colour(colour):
        """Blue/Yellow button."""
        if reset_if_running():
            return
        vision._enemy_goal_colour = colour
        print(f"[main] enemy goal colour set: {colour}", flush=True)
        maybe_start()

    def select_role(role):
        """Attacker/Goalie button, same clear-vs-pick shape as select_colour, for the role slot."""
        if reset_if_running():
            return
        with lock:
            shared_state["slot_role"] = role
            shared_state["own_slot_role"] = role
        print(f"[main] role set: {role}", flush=True)
        maybe_start()

    def select_camrun():
        """Enter camera-only mode without a goal-colour or role selection."""
        with lock:
            running = shared_state["run_mode"] == "camrun"
        if running:
            stop_and_idle()
            print("[main] camera-only mode stopped", flush=True)
            return
        with lock:
            fwd0 = shared_state["imu_heading"]
            shared_state["run_mode"] = "camrun"
            shared_state["camrun_forward_heading"] = fwd0
        print("[main] camera-only mode: no lidar/pose, camera + dwibbler "
              "possession only, forward heading "
              + (f"locked at {fwd0:.1f} deg" if fwd0 is not None
                 else "not locked (no IMU), push direction will just be "
                      "whatever way the robot is currently facing"),
              flush=True)

    import sys, tty, termios

    def key_watch():
        """keyboard stand-ins for the four buttons, for testing without hardware: 1=Blue
        2=Yellow 3=Attacker 4=Goalie 5=camera-only mode (no real button for that one yet).
        """
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

    # the keyboard helper is for the bench; under systemd stdin isn't a terminal, so a
    # setup error here must not take down boot
    if sys.stdin.isatty():
        threading.Thread(target=key_watch, daemon=True).start()
    else:
        print("[main] keyboard controls disabled (no interactive terminal)",
              flush=True)

    button_blue.when_pressed = lambda: select_colour("cyan")
    button_yellow.when_pressed = lambda: select_colour("yellow")
    button_attacker.when_pressed = lambda: select_role("striker")
    button_goalie.when_pressed = lambda: select_role("goalie")
    # select the configured role at boot; the goal colour is still picked at the
    # field, so the robot stays idle until then
    if _bot_cfg.DEFAULT_ROLE is not None:
        select_role(_bot_cfg.DEFAULT_ROLE)

    srcs = "lidar" + ("+imu" if compass.imu_fusion_enabled else "")
    print(f"[main] idle, GPIO {button_blue_pin}/{button_yellow_pin} pick "
          f"the enemy goal colour, {button_attacker_pin}/{button_goalie_pin}"
          f" pick the role; default role = {_bot_cfg.DEFAULT_ROLE}; goal colour starts play; any button during play resets to this fresh state"
          f"{'' if network.team_play_enabled else ' (solo)'}; "
          f"localisation = {srcs}; key 5 = camera-only mode, no lidar "
          "(sec 4.9)", flush=True)

    if network.bt_team_enabled:
        threading.Thread(target=network._bt_link_thread, daemon=True).start()

    play_loop()


if __name__ == "__main__":
    main()
