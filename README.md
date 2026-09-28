# Thermite

Control code for Thermite, a two-robot RoboCup Junior Australia Open Soccer team. Each robot is
a Raspberry Pi 5 driving four omni wheels, with a 360-degree lidar, a fisheye camera, an IMU
and a roller dribbler (we call it the dwibbler). Both robots run this same code; a `ROBOT_ID`
environment variable picks which one it is.

This code is shared so other teams can learn from it. Please don't copy it into your own
robot: the [RCJA General Rules](https://www.robocupjunior.org.au/wp-content/uploads/2026/02/2026-RCJA-General-Rules.pdf),
rule 4.3.2, say:

> Teams may not directly use designs and programs that has been passed down to them by the teams before them

## Contents

- [Hardware](#hardware)
- [Setup on a Pi](#setup-on-a-pi)
- [Running](#running)
- [Playing a match](#playing-a-match)
- [The debug page](#the-debug-page)
- [Debug sessions: watching a match back](#debug-sessions-watching-a-match-back)
- [How it works](#how-it-works)
- [Files](#files)
- [Configuration](#configuration)
- [Tools and calibration](#tools-and-calibration)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)

## Hardware

| Part | What it does | Where in the code |
|---|---|---|
| Raspberry Pi 5 | Runs everything | |
| 4 x brushless drive motors on PowerfulBLDC I2C drivers, 50 mm omni wheels | Holonomic drive | `bot/MotorFuncs_Proto1.py`, `bot/rotom.py`, `bot/kinematics.py` |
| 1 x the same motor and driver on the roller | Grabs and holds the ball | `bot/dwibbler.py` |
| 360-degree serial lidar (230400 baud) | Localisation and robot detection | `bot/lidar.py`, `bot/localisation.py`, `bot/perception.py` |
| Main camera with fisheye lens (Picamera2) | Ball, goals, enemy bearings | `bot/vision.py` |
| Second camera in the mouth (Picamera2 camera 1) | Confirms we have the ball | `bot/dwibbler.py`, `bot/dwibble_cam_calib.py` |
| BNO08x IMU on I2C | Heading between lidar fixes, collision detection | `bot/compass.py` |
| Four push buttons on GPIO 19, 20, 21, 22 | Pick goal colour and role | `bot/main.py` |
| Bluetooth (built into the Pi) | Link between the two robots | `bot/network.py` |

The two robots are the same build:

| | Bot 1 | Bot 2 |
|---|---|---|
| Login | `thermite@Ironhusk` | `thermite@Coppersentry` |
| `ROBOT_ID` | 1 | 2 |

Motor I2C addresses (`MOTOR_PINS` in the per-bot configs):

| Motor | Bot 1 | Bot 2 |
|---|---|---|
| nw | 26 | 28 |
| se | 30 | 31 |
| sw | 31 | 26 |
| ne | 32 | 25 |
| dwibbler | 29 | 27 |

## Setup on a Pi

1. Clone the repo onto the Pi.
2. Make a virtual environment that can see the system Picamera2:
   ```
   python3 -m venv --system-site-packages ~/env
   source ~/env/bin/activate
   ```
3. Install the Python packages:
   ```
   sudo apt install python3-picamera2 python3-opencv
   pip install numpy pyserial gpiozero adafruit-blinka adafruit-circuitpython-bno08x \
       adafruit-circuitpython-busdevice pybind11 pytest
   ```
   `tests/motorcalib.py` also needs the vendor package `steelbar_powerful_bldc_driver`; the
   robot itself uses the copy in `bot/rotom.py`.
4. Build the C++ speed-ups (optional, but the particle filter is much faster with them). This
   has to be done on the Pi itself:
   ```
   bash native/build.sh
   ```
   Without them everything falls back to numpy and still works.
5. Check the motor encoder calibration (see [Tools and calibration](#tools-and-calibration)).
6. To start on boot, install the service. Edit `systemd/lidar-robot.service` first: set `User`,
   the two paths to wherever you cloned the repo, and `ROBOT_ID` (1 or 2).
   ```
   sudo cp systemd/lidar-robot.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now lidar-robot
   ```
7. For the robot-to-robot link, pair the two Pis once with `bluetoothctl` (`pair`, then
   `trust`, on each side).

## Running

Always set `ROBOT_ID`. With the wrong ID the robot drives the other robot's motor addresses, so
the code refuses to start without one.

```
ROBOT_ID=1 ./start.sh # normal play (activates ~/env, then runs bot.main)
ROBOT_ID=1 python3 -m bot.main # the same, without start.sh
```

`start.sh` uses the virtual environment at `~/env`; set `VENV_DIR` to use another.

Command-line options for `bot.main`:

| Option | What it does |
|---|---|
| `--hsv` | Colour tuner only: camera and debug page, no motors or lidar |
| `--kickoff kicking` / `--kickoff receiving` | Hold the kick-off placement at the start of play (4 s, or until the ball moves toward us) |
| `--debug` | Record this power-on as a debug session for replay (also `DEBUG_SESSION=1`, or the debug page) |
| `--lidarlog [seconds]` | Log localisation to `lidar.txt` in the debug session folder (default 60 s). Park the robot: any pose movement in the log is error |
| `--capturelog [seconds]` | Log ball capture (robot and ball positions and velocities) to `capture.txt` in the debug session folder |
| `--motionlog [seconds]` | Log motor commands and pose, IMU and wheel telemetry to `motion.txt` in the debug session folder |
| `--no-boot-swivel` | Skip the small left-right wiggle the robot does at boot to show the motors are alive |

Other entry points:

```
ROBOT_ID=1 python3 -m bot.motorcheck # drive straight at full speed until you press q
python3 tests/motorcalib.py # calibrate motor encoders (interactive)
python simulator.py logs/debug_r1_<time> # watch a debug session back (on a laptop)
```

## Playing a match

1. Power on. The robot boots idle, with its default role loaded (bot 1 goalie, bot 2 striker).
2. Press a role button if you want the other role (21 = attacker, 22 = goalie).
3. Press the colour of the goal we are attacking (19 = blue, 20 = yellow). Play starts.
4. Pressing any button during play stops the robot and resets it to the idle state. Pick again
   to restart.

Without buttons (on the bench), use the keyboard in the terminal: `1` blue, `2` yellow, `3`
attacker, `4` goalie, `5` camera-only mode.

Camera-only mode (`5`) plays without the lidar: it chases the ball with the camera and pushes
it forward along the heading it had when the mode started. It is a fallback if the lidar dies.

If the IMU faults for a sustained period, the robot stops and stays stopped until the IMU is
healthy again and you pick colour and role again.

## The debug page

The robot serves a web page on port 8080; the console prints the address at startup
(`http://<pi-address>:8080/`).

| Page | What's on it |
|---|---|
| `/` | Live camera with overlays, the top-down field view (pose, ball, enemies, teammate), state and health. In `--hsv` mode, the colour tuner sliders instead |
| `/calib` | The white-line gate: its keep-out settings, how often it has stepped in |
| `/status` | JSON of the robot's state, for scripts |
| `/stream.mjpg` | Just the camera stream |

The HSV tuner and the gate page have a Save button: it prints the values to the terminal for
you to paste into `bot/vision.py` or `bot/motion.py`.

## Debug sessions: watching a match back

Start the robots with `--debug` (or `DEBUG_SESSION=1`, or the switch on the debug page). A
debug session lasts as long as the robot is powered on: switching recording off on the debug
page pauses it, and switching it on again carries on in the same session. For the next match,
power the robot off and on, and it starts a new one. To record every match, add
`Environment=DEBUG_SESSION=1` to the systemd service.

Each session is one folder, `logs/debug_r<id>_<time>/`:

| File | What's in it |
|---|---|
| `session.json` | The robot, the start time, its clock, and how much has been written (updated every few seconds) |
| `ticks.jsonl` | One line per play tick (10 a second while idle): pose, ball, enemies, teammate, state, motor commands. Also one line per message from the teammate, which is how two sessions get lined up |
| `lidar.txt`, `capture.txt`, `motion.txt` | The `--lidarlog`, `--capturelog` and `--motionlog` logs, if you asked for them |

Every file is synced to the card once a second, so pulling the power loses at most the last
second. Recording pauses itself if the card gets below 500 MB free.

Copy the folders off the Pis and, from this folder on any computer with OpenCV and a screen:

```
python simulator.py logs/debug_r1_20260928_140312
python simulator.py logs/debug_r1_20260928_140312 logs/debug_r2_20260928_140305 # both robots
```

With two sessions, the robots are lined up using the Bluetooth messages they exchanged: each
message carries the sender's clock, and each robot logs when it arrived on its own clock, so
comparing the two directions gives the difference between the clocks to within a few hundredths
of a second. If neither session logged any messages (the link was down), they are lined up on
the moment each one started playing instead. The viewer opens at the start of play. Our goal is
always drawn on the left (green), theirs on the right (red), and the field turns round if we
change ends partway through.

On the field you see each robot's pose and heading, its last few seconds of movement, the line
to the ball from its camera, its ball estimate (coloured by source: camera, teammate, Kalman
prediction, memory or pass), and the enemies and teammate it could see. An occluded enemy is
drawn thin. The panel shows, for the tick on screen, the state the robot went in with and came
out with, its pose, IMU, camera ball, ball estimate, possession, mouth camera, loop rates, the
five motor commands, and its last few state changes. The seek bar marks every state change.

| Key | Does |
|---|---|
| space | Pause and play |
| a / d, or left / right | Step one tick back or forward |
| j / l | Jump 5 s back or forward |
| [ / ] | Halve or double the playback speed |
| r | Back to the start of play |
| t | Trails on and off |
| q, Esc | Quit |

Click or drag on the bar at the bottom to seek. The window can be resized. Running it with no
folder prints this help.

## How it works

Everything runs as threads inside `bot/main.py`, sharing one state dictionary (`bot/state.py`)
under a lock. The play loop runs at 50 Hz and is the only thing that commands the drive motors.

### Localisation

The lidar thread (`bot/lidar.py`) reads each revolution, undoes the robot's own motion during
the sweep (deskew, using the IMU), and feeds a particle filter (`bot/localisation.py`) that
matches the scan against the known field walls. The particle filter is the only source of pose.
Between lidar fixes the pose is carried forward with wheel odometry and the IMU
(`bot/odometry.py`, `bot/compass.py`). The same scan is used to find other robots
(`bot/perception.py`), which are then tracked over time (`bot/tracking.py`).

### Vision

The main camera unwraps the fisheye image and finds the orange ball, the goals and enemy
bearings by colour (`bot/vision.py`). Ball positions go through a Kalman filter and a
short-term memory (`bot/tracking.py`), so the robot keeps a good guess for a moment when the
ball is hidden.

### Roles

`bot/controllers.py` has the two roles:

- *Striker*: chases the ball with a curved approach that arrives facing the goal, grabs it with
  the dwibbler, then carries it in. On a long uncontested carry it hides the ball along the
  sideline. It only releases when a shot along its current facing would actually go in, and
  it has a scripted escape for when a defender parks in front of it. Passing is built but
  switched off (`pass_enabled`).
- *Goalie*: holds a line in front of the goal between the ball and the net, charges and clears
  a ball that comes close, spins the roller up early so an incoming ball gets pulled in, and
  escorts the teammate when the ball is far upfield.

### Role swap

With both robots running and paired, one Pi becomes the master and picks the striker every
tick; the other robot keeps goal. A robot out of play hands over first, then the one holding
the ball strikes, then the only one that sees it, then the closer one (by 200 mm), then the one
further upfield. Swaps are at least 2 s apart unless possession changes. If the link drops, the
robot still playing strikes. `bot/network.py`.

### Driving

`bot/motion.py` turns "go this way at this speed" into safe commands: a braking curve so it can
stop in time, keep-outs from walls, enemies and our teammate, a white-line gate that won't let
it cross the boundary, stuck detection and recovery, and a heading controller.
`bot/kinematics.py` converts that into the four wheel speeds.

### Speeds

The speeds are in `bot/drive_config.py`, as fractions of full motor command: normal driving
0.3, rush on a clear lane 0.5 (about the real top speed), pushing in on a held ball 0.4, and
the goalie positions at up to about 0.41.

## Files

- `bot/`: the robot code (run with `python3 -m bot.main`)
- `native/`: optional C++ speed-ups and their parity tests
- `tests/`: pytest suite (no hardware needed), and the motor and ball-shape calibration scripts
- `tools/`: bench tools you run by hand on the robot
- `systemd/`: the boot service
- `start.sh`: boot entry point
- `simulator.py`: the debug session viewer (see [Debug sessions](#debug-sessions-watching-a-match-back))
- `BENCH_TEST_CHECKLIST.md`: checks to do on the real robot

### bot/

| File | What it does |
|---|---|
| `main.py` | Entry point: command-line options, starts every thread, the buttons, the play loop |
| `robot_select.py` | Reads `ROBOT_ID` and loads that robot's config |
| `bot1_config.py`, `bot2_config.py` | Per-robot settings: motor addresses, camera crop and mask, default role |
| `drive_config.py` | Settings shared by both robots: speeds, motor encoder calibration, boot wiggle |
| `state.py` | The shared state dictionary and its lock |
| `controllers.py` | Striker, goalie and camera-only behaviour |
| `motion.py` | Speed model, keep-out guards, white-line gate, stuck recovery, heading control, shot and carry helpers |
| `kinematics.py` | Wheel speeds from a drive direction, speed and spin |
| `lidar.py` | Lidar reading, deskew, and the lidar thread |
| `localisation.py` | The particle filter |
| `perception.py` | Scan matching against the walls and robot detection from the lidar |
| `tracking.py` | Robot tracks, ball velocity, ball memory and the ball Kalman filter |
| `field.py` | Field dimensions and wall geometry |
| `odometry.py` | Wheel odometry and a wheel-slip check |
| `compass.py` | IMU reading, heading between fixes, collision detection |
| `vision.py` | Main camera: fisheye unwrap, ball, goals, enemy bearings |
| `dwibbler.py` | Roller control and "do we have the ball" sensing (roller load plus the mouth camera) |
| `dwibble_cam_calib.py` | Distance from the mouth camera, using the saved calibration points |
| `network.py` | Bluetooth link between the robots, role swap, UDP fallback |
| `debug_server.py` | The debug web page |
| `diagnostics.py` | Health reporting and the wobble check |
| `calibration.py` | Field calibration lap (drives the perimeter and measures the walls) |
| `MotorFuncs_Proto1.py` | Motor setup, calibration, driving and the background write thread |
| `rotom.py` | The motor driver library (I2C protocol) |
| `hardware.py` | Where `Motor` is imported from; kicker stub (no kicker is fitted) |
| `motorcheck.py` | Manual drive test |
| `debug_session.py` | `--debug` sessions: the tick log and teammate clock rows |
| `lidar_debug.py`, `capture_debug.py`, `motion_debug.py`, `logs.py` | The `--lidarlog`, `--capturelog` and `--motionlog` loggers |

### native/

`lidar_native.cpp` (lidar packet parsing), `mcl_native.cpp` (particle filter maths) and
`camera_native.cpp` (a fused colour pass, kept but not used because OpenCV is faster). Build
with `bash native/build.sh`. Each has a `test_*_native.py` that checks it gives the same
answers as the Python version.

### tools/

| Tool | Use |
|---|---|
| `bench_drive_sweep.py` | Measure top speed (`ramp`), stopping distance (`stop`) and the lowest current that holds a speed (`current`) |
| `bench_dwibble_cam.py` | Tune the mouth camera's colours and record its distance calibration points |
| `fisheye_fit.py` | Fit the main camera's fisheye distance model from measured samples |

## Configuration

Settings are plain constants at the top of each module; there are no config files to edit. The
ones you are most likely to touch:

| Setting | File | Default |
|---|---|---|
| Speeds (`base_speed`, `rush_speed`) | `bot/drive_config.py` | 0.3, 0.5 |
| Motor encoder calibration (`MOTOR_CALIB`) | `bot/drive_config.py` | |
| Heading rate source (`imu_gyro_rate_enabled`) | `bot/motion.py` | off (differenced heading); see `BENCH_TEST_CHECKLIST.md` before enabling |
| Ball and goal colours | `bot/vision.py` | tune with `--hsv` |
| Ball shape gate (`ball_min_fill_ratio`) | `bot/vision.py` | tune with `tests/fillratio_calib.py` |
| Camera crop and masks | `bot/bot1_config.py`, `bot/bot2_config.py` | |
| Default role (`DEFAULT_ROLE`) | per-bot config | bot 1 goalie, bot 2 striker |
| Robot link on/off (`bt_team_enabled`) | `bot/network.py` | on |
| Role swap on/off (`bt_dynamic_roles`) | `bot/network.py` | on |
| Passing (`pass_enabled`) | `bot/controllers.py` | off |

Most behaviours also have an `..._enabled` flag next to their settings, so any one of them can
be switched off without touching the code around it.

Files the robot writes at runtime (all ignored by git): `motor_calibration.json` (saved encoder
calibration), `calib_points.json` (mouth camera), and everything in `logs/`.

## Tools and calibration

### Motor encoders

Each motor needs an encoder calibration pair. The defaults are in `MOTOR_CALIB` in
`bot/drive_config.py`. To recalibrate, put the robot on blocks (the wheels spin) and run
`python3 tests/motorcalib.py`. Enter how many drivers and their addresses; it prints
`ELECANGLEOFFSET` and `SINCOSCENTRE` for each, which go into `MOTOR_CALIB` (or `dwibble_calib`
in `bot/dwibbler.py` for the roller). A saved `motor_calibration.json` overrides the defaults
for any motor it lists.

### Colours

Run `ROBOT_ID=1 python3 -m bot.main --hsv`, open the debug page, move the sliders until only
the ball (or goal) is highlighted, click Save, and paste the printed values into
`bot/vision.py`.

### Ball shape gate

`ball_min_fill_ratio` in `bot/vision.py` rejects an orange or red blob that isn't round enough
to be the ball, like a marking on another robot or a red line on the wall. The default is a
guess. Run `ROBOT_ID=1 python3 tests/fillratio_calib.py` on the robot, press `b` with the ball in
view at a few ranges and `x` with a decoy in view, then `t` for a suggested value, and paste it
over the default.

### Mouth camera

`python3 tools/bench_dwibble_cam.py` on the robot: tune its colour, then record the ball at a
few known distances (`c`), and save (`s`).

### Fisheye distance

Measure a set of ball positions (pixel radius and real distance) and run `python3
tools/fisheye_fit.py samples.json`. Re-fit if you change the camera crop.

### Top speed and stopping distance

`python3 tools/bench_drive_sweep.py ramp` and `... stop`, with the robot on a clear floor.

## Tests

The tests need no hardware; the motors and sensors are stubbed.

```
ROBOT_ID=1 python3 -m pytest tests native -q
```

`BENCH_TEST_CHECKLIST.md` lists what can only be checked on the real robot, in the order to do
it.

## Troubleshooting

| Problem | Try |
|---|---|
| "ROBOT_ID env var must be set" | `export ROBOT_ID=1` (or 2) |
| A motor doesn't move, or runs rough | Check its address in the per-bot config, then recalibrate it with `tests/motorcalib.py` |
| Robot drives the wrong way or spins | Check the motor addresses match the wheel positions in the per-bot config |
| Ball not seen, or seen everywhere | Retune colours with `--hsv`; lighting changes between venues |
| Ball locks onto a red marker on another robot, or a red wall line | Retune `ball_min_fill_ratio` (`bot/vision.py`, 0.45) with `tests/fillratio_calib.py`. It rejects blobs that aren't round enough to be the ball |
| Pose jumps around | Run `--lidarlog 60` with the robot parked and read the summary at the end of the log |
| Robots don't swap roles | Check they are paired, and look for `[team] teammate connected` in the console |
| Roller doesn't hold the ball | Check `dwibble_calib` in `bot/dwibbler.py` and the roller's address |
| Robot stopped and won't restart | An IMU fault: wait for `[health] imu: ok`, then pick colour and role again |
