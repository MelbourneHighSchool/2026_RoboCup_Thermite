"""
bot: the robot's control code, every layer live at once.

Entry point: `python3 -m bot.main` (from the repo root).

Module map, bottom of the dependency stack first:
  state.py         the shared shared_state dict + lock and the rebindable globals
  field.py         RCJ field geometry (FieldModel), the shared angle wrap
  compass.py       BNO08x IMU: heading history, fusion helpers, its thread
  hardware.py      Motor, the solenoid kicker, the drive speed model
  dwibbler.py      the ball roller and dwibbler-stall possession sensing
  odometry.py      wheel odometry, the predicted-pose thread, current_pose
  localisation.py  the particle-filter MCL - the only pose estimator
  perception.py    Perception: pose in, tracked-robot detections out
  tracking.py      robot/ball/teammate tracking and memory
  lidar.py         lidar parsing, deskew, the localise-and-detect thread
  vision.py        camera: ball + goal detection, the capture thread
  network.py       Bluetooth team link and the UDP pose fallback
  motion.py        tunables, keep-out guards, recoveries, the movement layer
  controllers.py   StrikerController / GoalieController / CamRunController
  calibration.py   ButtonCalib's perimeter lap and the saved field offsets
  debug_server.py  the debug web UI, overlays and MJPEG stream
  logs.py          --capturelog/--motionlog headers and polling thread
  main.py          CLI, thread startup, the mode buttons, play_loop
"""
