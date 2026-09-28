#!/usr/bin/env python3
# fillratio_calib.py: tune ball_min_fill_ratio (bot/vision.py) on the robot's own camera.
#
#     ROBOT_ID=1 python3 tests/fillratio_calib.py
#
# The gate throws out orange or red blobs that aren't round enough to be the ball, such as
# a marker on another robot or a line on the wall. How round the ball looks depends on the
# lens, the crop, the lighting and the range, so the number comes from samples, not a guess.
#
# Show it the ball and press b, show it a decoy and press x. Do both near and far, then press
# t for a suggested value. Every blob is drawn with its fill ratio, green if it passes the
# current threshold and red if not.
#
# Keys:
#     b logs the biggest blob as the ball
#     x logs it as a decoy
#     + and - move the live threshold by 0.01
#     t prints a suggested threshold from the samples
#     u undoes the last sample
#     s saves the samples to fillratio_samples.json
#     q quits
#
# Not a pytest file.

import json
import sys
import time
from pathlib import Path

import cv2

# run directly, not through pytest, so put public-repo/ on the path for bot.*
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bot.robot_select as robot_select
import bot.vision as vision

robot_select.select_and_publish_config()

if vision.Picamera2 is None:
    print("picamera2 not installed; this tool runs on the robot only.")
    sys.exit(1)

DRAW_SCALE = 2 # the unwrap is narrow (n_r rows); shown this many times larger

# the same sensor mode as the camera thread, or the crop and ring geometry don't line up
frame_duration = int(1_000_000 / vision.sensor_fps)
picam2 = vision.Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": vision.sensor_mode, "format": "RGB888"},
    sensor={"output_size": vision.sensor_mode, "bit_depth": 8},
    controls={"FrameDurationLimits": (frame_duration, frame_duration)}))
picam2.start()
if vision.cam_saturation is not None:
    picam2.set_controls({"Saturation": float(vision.cam_saturation)})

(map_x, map_y, r_lut, theta_lut, col_bearing, inner_blank, _notch_region,
 inner_r, outer_r, _ring_r) = vision._ring_geometry()
print(f"[fillratio] ring {inner_r}-{outer_r} px, {len(r_lut)}x{len(theta_lut)} unwrap, "
      f"orange {vision.orange_lower.tolist()} - {vision.orange_upper.tolist()}, "
      f"starting threshold {vision.ball_min_fill_ratio}")
print("[fillratio] keys: b ball, x decoy, +/- nudge threshold, t suggest, u undo, "
      "s save, q quit")

# each sample: {"label": "ball" | "decoy", "area": float, "fill_ratio": float}
samples = []


def _grab_candidates():
    """this frame's (orange mask, HSV, candidates), made the same way the robot makes them."""
    frame = picam2.capture_array("main")
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h_rot, w_rot = frame.shape[:2]
    frame = frame[
        vision.crop_top : h_rot - vision.crop_bottom if vision.crop_bottom else None,
        vision.crop_left : w_rot - vision.crop_right if vision.crop_right else None,
    ]
    orange, hsv, _s_raw = vision._ball_mask(
        frame, vision.orange_lower, vision.orange_upper, map_x, map_y, col_bearing, inner_blank)
    return orange, hsv, vision._ball_candidates(orange, r_lut, len(theta_lut))


def _draw(orange, hsv, candidates):
    vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for _cx, _cy, x, y, w, h, _area, fill_ratio in candidates:
        ok = fill_ratio >= vision.ball_min_fill_ratio
        colour = (80, 220, 80) if ok else (60, 60, 220)
        cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 1)
        cv2.putText(vis, f"{fill_ratio:.2f}", (x, max(10, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
    vis = cv2.resize(vis, (vis.shape[1] * DRAW_SCALE, vis.shape[0] * DRAW_SCALE),
                     interpolation=cv2.INTER_NEAREST)
    balls = sum(s["label"] == "ball" for s in samples)
    decoys = sum(s["label"] == "decoy" for s in samples)
    cv2.putText(vis, f"threshold {vision.ball_min_fill_ratio:.2f}  "
                     f"samples: {balls} ball, {decoys} decoy",
                (6, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
    cv2.imshow("fillratio_calib: unwrap", vis)
    cv2.imshow("fillratio_calib: mask", orange)


while True:
    orange, hsv, candidates = _grab_candidates()
    _draw(orange, hsv, candidates)
    k = cv2.waitKey(30) & 0xFF

    if k == ord("q"):
        break
    elif k in (ord("b"), ord("x")):
        if not candidates:
            print("[fillratio] nothing in view, not logged")
            continue
        *_rest, area, fill_ratio = max(candidates, key=lambda cand: cand[6])
        label = "ball" if k == ord("b") else "decoy"
        samples.append({"label": label, "area": float(area), "fill_ratio": float(fill_ratio)})
        print(f"[fillratio] logged {label}: area {area:.0f} px, fill ratio {fill_ratio:.3f} "
              f"({len(samples)} samples)")
    elif k == ord("u") and samples:
        gone = samples.pop()
        print(f"[fillratio] undid {gone['label']} at {gone['fill_ratio']:.3f}")
    elif k == ord("+"):
        vision.ball_min_fill_ratio = round(min(1.0, vision.ball_min_fill_ratio + 0.01), 3)
    elif k == ord("-"):
        vision.ball_min_fill_ratio = round(max(0.0, vision.ball_min_fill_ratio - 0.01), 3)
    elif k == ord("t"):
        t, correct, total = vision.suggest_fill_ratio_threshold(
            [(s["label"] == "ball", s["fill_ratio"]) for s in samples])
        if t is None:
            print("[fillratio] log at least one ball and one decoy sample first")
        else:
            print(f"[fillratio] suggested ball_min_fill_ratio = {t:.3f} "
                  f"({correct}/{total} samples correctly separated)")
            if correct < total:
                print("[fillratio] not a clean separation: some ball and decoy samples "
                      "overlap in fill ratio. Collect more samples across the ball's real "
                      "range/angle spread before trusting this, or accept the miss rate.")
    elif k == ord("s"):
        with open("fillratio_samples.json", "w") as f:
            json.dump({"orange_lower": vision.orange_lower.tolist(),
                      "orange_upper": vision.orange_upper.tolist(),
                      "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "samples": samples}, f, indent=2)
        print(f"[fillratio] wrote fillratio_samples.json ({len(samples)} samples)")

cv2.destroyAllWindows()
