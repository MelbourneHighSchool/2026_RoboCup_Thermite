#!/usr/bin/env python3
# bench_dwibble_cam.py: bench tool for the second camera (dwibbler-mouth
# possession cam, bot/dwibbler.py). Two jobs:
#
# 1. Retune dwibble_cam_lower/upper: live 320x240 feed + orange mask, H/S/V
#    bounds nudged with the keyboard, final values printed for pasting into
#    bot/dwibbler.py. Same workflow as the main camera's HSV tuner page.
#
# 2. Collect calibration points for the piecewise ball-distance regression
#    (see the mounting note in bot/dwibbler.py: the camera rides the
#    dwibbler, so it swings back and tilts up once the ball touches the
#    roller. Pixel position is not a fixed function of distance until this
#    regression exists). Place the ball at a known distance from the roller
#    face, press [c] to record, repeat across the range, press [s] to write
#    calib_points.json. Each record: distance_mm, ball pixel centre, ball
#    pixel radius, ball y-offset from frame centre.
#
# Run on the Pi from public-repo/: python3 tools/bench_dwibble_cam.py (the
# calib_points.json it saves lands in the working directory, where the bot
# looks for it; DWIBBLE_CALIB_PATH overrides).
# Keys: [j]/[l] H lo/hi, [u]/[o] S lo/hi, [n]/[m] V lo/hi (as printed at boot),
#       [c] capture calibration point (asks distance in the terminal),
#       [s] save calib_points.json, [p] print current HSV, [q] quit.

import json
import sys

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError: # not on the Pi
    print("picamera2 not installed; this tool runs on the robot only.")
    sys.exit(1)

# starting point: bot/dwibbler.py's current values
lower = np.array([6, 171, 13], dtype=np.uint8)
upper = np.array([20, 255, 255], dtype=np.uint8)
RES = (320, 240)
MIN_FRAC = 0.35

picam2 = Picamera2(camera_num=1)
picam2.configure(picam2.create_video_configuration(
    main={"size": RES, "format": "RGB888"}))
picam2.start()
print(f"[bench] dwibble cam live at {RES[0]}x{RES[1]}")
print("[bench] keys: j/l H lo-hi, u/o S lo-hi, n/m V lo-hi, c capture point, "
      "s save points, p print HSV, q quit")

points = []


def ball_props(mask):
    """largest orange blob: (cx, cy, radius_px, frac) or None."""
    frac = float(mask.mean()) / 255.0
    if frac < 0.005:
        return None
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(c)
    return cx, cy, r, frac


while True:
    frame = picam2.capture_array("main")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    vis = frame.copy()

    props = ball_props(mask)
    if props:
        cx, cy, r, frac = props
        cv2.circle(vis, (int(cx), int(cy)), int(r) + 2, (0, 255, 80), 2)
        cv2.line(vis, (RES[0] // 2, RES[1] // 2), (int(cx), int(cy)),
                 (0, 210, 255), 1)
        ok = frac >= MIN_FRAC
        cv2.putText(vis, f"frac {frac:.2f} ({'held' if ok else 'open'})",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (80, 220, 80) if ok else (200, 200, 200), 1)
        cv2.putText(vis, f"dx {cx - RES[0] / 2:+.0f} dy {cy - RES[1] / 2:+.0f} "
                         f"r {r:.0f}px",
                    (4, RES[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1)
    else:
        cv2.putText(vis, "no blob", (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (60, 60, 220), 1)

    cv2.imshow("raw", vis)
    cv2.imshow("mask", mask)
    k = cv2.waitKey(30) & 0xFF

    if k == ord('q'):
        break
    elif k == ord('j'):
        lower[0] = max(0, lower[0] - 1)
    elif k == ord('l'):
        lower[0] = min(179, lower[0] + 1)
    elif k == ord('u'):
        lower[1] = max(0, lower[1] - 1)
    elif k == ord('o'):
        lower[1] = min(255, lower[1] + 1)
    elif k == ord('n'):
        lower[2] = max(0, lower[2] - 1)
    elif k == ord('m'):
        lower[2] = min(255, lower[2] + 1)
    elif k == ord('p'):
        print("dwibble_cam_lower = np.array("
              f"{lower.tolist()}, dtype=np.uint8)")
        print("dwibble_cam_upper = np.array("
              f"{upper.tolist()}, dtype=np.uint8)")
    elif k == ord('c') and props:
        cx, cy, r, frac = props
        try:
            d = float(input("ball distance from roller face (mm): "))
        except ValueError:
            print("not a number, skipped")
            continue
        points.append({
            "distance_mm": d,
            "cx": float(cx), "cy": float(cy),
            "dx_px": float(cx - RES[0] / 2),
            "dy_px": float(cy - RES[1] / 2),
            "radius_px": float(r),
            "frac": frac,
        })
        print(f"recorded {len(points)} points, latest at {d:.0f} mm")
    elif k == ord('s'):
        with open("calib_points.json", "w") as f:
            json.dump({"resolution": list(RES), "points": points}, f,
                      indent=2)
        print(f"wrote calib_points.json ({len(points)} points), regress "
              "distance_mm vs dy_px/radius_px piecewise per mounting pose")

cv2.destroyAllWindows()
