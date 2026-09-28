"""Debug web UI: camera and field overlay panels, the HSV tuner, the render thread, and the
HTTP server (page, MJPEG stream, HSV and gate-calibration endpoints).
"""

import http.server
import json
import math
import socketserver
import time

import cv2
import numpy as np

from bot.field import FieldModel
from bot.state import _frame_cond as frame_cond
from bot.state import _jpeg_cond as jpeg_cond
from bot.state import _lock as lock
from bot.state import _state as shared_state
from bot.vision import _cam_direction as cam_direction
from bot.vision import _cam_handle_centres as cam_handle_centres
from bot.vision import _sat_boost as sat_boost
from bot.vision import (mouth_notch_enabled, mouth_notch_half_deg,
                        mouth_notch_px, sensor_size)
import bot.vision as vision
import bot.state as state
import bot.motion as motion
import bot.debug_session as debug_session


# Debug / field display
debug_port = 8080

# Per-bot lidar handle wedge for the overlay, set by robot_select at boot.
handle_exclusion_deg = None


default_panel_h = 360 # debug panel height (px)
field_pad = 16
field_scale = (default_panel_h - 2 * field_pad) / FieldModel.field_y
field_w = int(FieldModel.field_x * field_scale) + 2 * field_pad


# Debug rendering

def render_camera_panel(pose, ball, frame, mask, cx_px, cy_px, radius,
                         disp_scale=None, panel_h=None, cam_fps=None):
    """overlay on the raw camera frame: orange mask tint, exclusion rings, handle wedges, the
    detected ball, and pose/fps text.
    """
    ph = panel_h or default_panel_h
    if frame is None:
        return np.zeros((sensor_size[1], sensor_size[0], 3), dtype=np.uint8)

    h_src, w_src = frame.shape[:2]
    vis = frame.copy()

    if mask is not None:
        mask_up = cv2.resize(mask, (w_src, h_src), interpolation=cv2.INTER_NEAREST)
        orange_layer = np.zeros_like(vis)
        orange_layer[:, :] = (0, 100, 255)
        alpha = (mask_up > 0).astype(np.float32) * 0.5
        for c in range(3):
            vis[:, :, c] = (
                vis[:, :, c] * (1 - alpha) + orange_layer[:, :, c] * alpha
            ).astype(np.uint8)

    # Exclusion ring centre: image centre (matches detection)
    cxi = w_src // 2
    cyi = h_src // 2
    max_r = min(w_src, h_src) // 2
    r_inner = int(vision.exclusion_inner_frac * max_r)
    r_outer = int(vision.exclusion_outer_frac * max_r)
    cv2.circle(vis, (cxi, cyi), r_inner, (0, 0, 180), 1)
    cv2.circle(vis, (cxi, cyi), r_outer, (0, 0, 210), 2)
    cv2.line(vis, (cxi - 14, cyi), (cxi + 14, cyi), (80, 80, 80), 1)
    cv2.line(vis, (cxi, cyi - 14), (cxi, cyi + 14), (80, 80, 80), 1)

    # mouth notch: the scanned wedge of the inner circle (green arc and its radial
    # edges), so the overlay shows the region the detector really scans
    if mouth_notch_enabled and mouth_notch_px > 0:
        r_notch = max(1, r_inner - int(round(mouth_notch_px)))
        a0 = -mouth_notch_half_deg
        a1 = +mouth_notch_half_deg
        for a in (a0, a1):
            sdx, sdy = cam_direction(a)
            cv2.line(vis,
                     (int(cxi + r_notch * sdx), int(cyi + r_notch * sdy)),
                     (int(cxi + r_inner * sdx), int(cyi + r_inner * sdy)),
                     (0, 200, 120), 1)
        arc = np.array(
            [[int(cxi + r_notch * cam_direction(a)[0]),
              int(cyi + r_notch * cam_direction(a)[1])]
             for a in np.linspace(a0, a1, 24)], dtype=np.int32)
        cv2.polylines(vis, [arc], False, (0, 200, 120), 1)

    # Handle exclusion wedges, two boundary lines + an arc at r_outer per wedge.
    if handle_exclusion_deg > 0:
        hexcl = handle_exclusion_deg
        excl_colour = (0, 200, 255) # amber
        for centre_a in cam_handle_centres:
            # Boundary lines from r_inner to r_outer
            for edge_a in (centre_a - hexcl, centre_a + hexcl):
                sdx, sdy = cam_direction(edge_a)
                cv2.line(vis,
                         (int(cxi + r_inner * sdx), int(cyi + r_inner * sdy)),
                         (int(cxi + r_outer * sdx), int(cyi + r_outer * sdy)),
                         excl_colour, 1)
            # Arc at r_outer spanning the wedge
            arc_angles = np.linspace(centre_a - hexcl, centre_a + hexcl,
                                     max(3, int(hexcl * 2)))
            arc_pts = np.array(
                [[int(cxi + r_outer * cam_direction(a)[0]),
                  int(cyi + r_outer * cam_direction(a)[1])]
                 for a in arc_angles], dtype=np.int32)
            cv2.polylines(vis, [arc_pts], False, excl_colour, 1)

    if ball is not None and cx_px is not None:
        angle_deg, dist_mm = ball
        bx = int(cx_px)
        by = int(cy_px)
        cv2.line(vis, (cxi, cyi), (bx, by), (0, 210, 255), 1)
        if radius:
            cv2.circle(vis, (bx, by), int(radius), (0, 255, 60), 2)
        else:
            cv2.circle(vis, (bx, by), 5, (0, 255, 60), -1)
        cv2.putText(vis, f"{angle_deg:+.1f} deg  {dist_mm:.0f} mm",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1)
    else:
        cv2.putText(vis, "no ball", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 60, 255), 1)

    if pose is not None:
        x, y, hdg = pose
        cv2.putText(vis, f"({x:.0f}, {y:.0f})  {hdg:.1f} deg",
                    (6, ph - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    if cam_fps is not None:
        fps_label = f"{cam_fps} fps"
        (tw, _), _ = cv2.getTextSize(fps_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(vis, fps_label, (w_src - tw - 6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 255), 1)

    # blob diagnostics: mask size in unwrap cells, the radial cell the centre landed
    # on, and how far the sub-pixel refinement moved it
    blob = shared_state.get("ball_blob")
    if blob is not None:
        bw, bh, brow, bshift = blob
        cv2.putText(vis, f"blob {bw:.0f}x{bh:.0f}  row {brow:.2f}"
                         f"  subpx {bshift:+.2f}",
                    (6, ph - 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (150, 150, 150), 1)

    return vis


def render_field_panel(pose, ball, enemies=None, teammate_pos=None, lidar_hz=None,
                        target_w=None, target_h=None):
    """top-down field panel: robot pose, ball, tracked enemies and the teammate."""
    FX, FY = FieldModel.field_x, FieldModel.field_y
    P = field_pad
    if target_w and target_h:
        S = min((target_w - 2*P) / FX, (target_h - 2*P) / FY)
        W = target_w
        H = target_h
    else:
        S = field_scale
        W = field_w
        H = default_panel_h
    img = np.full((H, W, 3), 22, dtype=np.uint8)

    def fd(fx, fy):
        """field mm (fx, fy) -> panel pixel coords (scaled, padded, y-flipped)."""
        return (int(fx * S) + P, H - P - int(fy * S))

    for (x1, y1), (x2, y2) in FieldModel.segments:
        cv2.line(img, fd(x1, y1), fd(x2, y2), (190, 190, 190), 1)
    cv2.line(img, fd(0, FY / 2), fd(FX, FY / 2), (55, 55, 55), 1)
    cv2.circle(img, fd(FX / 2, FY / 2), int(500 * S), (55, 55, 55), 1)

    # Draw detected enemies (red circles)
    for en in (enemies or []):
        cv2.circle(img, fd(en["x"], en["y"]),
                   max(3, int(105 * S)), (0, 0, 220), 2)

    # Draw teammate (green circle)
    if teammate_pos is not None:
        cv2.circle(img, fd(teammate_pos[0], teammate_pos[1]),
                   max(3, int(105 * S)), (0, 200, 80), 2)

    if pose is not None:
        rx, ry, hdg = pose
        if ball is not None:
            b_angle, b_dist = ball
            b_rad = math.radians(b_angle + hdg)
            bfx = rx + b_dist * math.sin(b_rad)
            bfy = ry + b_dist * math.cos(b_rad)
            cv2.circle(img, fd(bfx, bfy), max(2, int(110 * S / 2)), (0, 100, 255), -1)
        cv2.circle(img, fd(rx, ry), max(2, int(105 * S)), (100, 190, 255), 2)
        hdg_rad = math.radians(hdg)
        ax = rx + 250 * math.sin(hdg_rad)
        ay = ry + 250 * math.cos(hdg_rad)
        cv2.arrowedLine(img, fd(rx, ry), fd(ax, ay), (100, 190, 255), 2, tipLength=0.35)
        # least-defended goal aim: a ray toward the widest open run, green when
        # mostly open, red as it gets blocked
        og = shared_state.get("open_goal")
        if og is not None and og.get("bearing_deg") is not None:
            ar = math.radians(hdg + og["bearing_deg"])
            blk = og.get("blocked_frac", 0.0)
            col = (60, int(220 * (1 - blk)) + 30, int(220 * blk) + 30)
            cv2.line(img, fd(rx, ry),
                     fd(rx + 900 * math.sin(ar), ry + 900 * math.cos(ar)),
                     col, 2)
        cv2.putText(img, f"({rx:.0f}, {ry:.0f})",
                    (4, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        cv2.putText(img, f"{hdg:.1f} deg",
                    (4, H - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    if lidar_hz is not None and lidar_hz > 0:
        hz_label = f"{lidar_hz:.1f} Hz"
        (tw, _), _ = cv2.getTextSize(hz_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(img, hz_label, (W - tw - 4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 220, 255), 1)

    # Status line: solo/team * state * possession, visible at a glance.
    with lock:
        solo = shared_state["solo"]
        st = shared_state["my_state"]
        has_bl = shared_state["drib_has_ball"]
    tag = "solo" if solo else "team"
    tag_col = (90, 170, 90) if solo else (0, 200, 80)
    cv2.putText(img, tag, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, tag_col, 1)
    label = (st or "-") + ("  ball" if has_bl else "")
    cv2.putText(img, label, (44, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 140, 255) if has_bl else (140, 140, 140), 1)
    return img


def render_dwibble_cam_panel():
    """second-camera panel: the raw frame with the ball mask tinted, the ball blob (circle,
    dx/dy from centre, pixel radius), the calibrated distance once calib_points.json loads,
    and the frac readout. A placeholder while the camera thread hasn't published.
    """
    import bot.dwibbler as dw
    with lock:
        frame = shared_state.get("dwibble_cam_frame")
        mask = shared_state.get("dwibble_cam_mask")
        frac = shared_state.get("dwibble_cam_frac")
        seen = shared_state.get("dwibble_cam_seen")
        ball = shared_state.get("dwibble_cam_ball")
    if frame is None:
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(img, "dwibble cam: no frame", (8, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1)
        return img
    h_src, w_src = frame.shape[:2]
    vis = frame.copy()
    if mask is not None:
        mask_up = cv2.resize(mask, (w_src, h_src), interpolation=cv2.INTER_NEAREST)
        orange_layer = np.zeros_like(vis)
        orange_layer[:, :] = (0, 100, 255)
        alpha = (mask_up > 0).astype(np.float32) * 0.5
        for c in range(3):
            vis[:, :, c] = (vis[:, :, c] * (1 - alpha)
                            + orange_layer[:, :, c] * alpha).astype(np.uint8)
    # the ball blob: circle, crosshair at frame centre, dx/dy/radius readout (the same
    # features bench_dwibble_cam.py collects for the calibration)
    if ball is not None:
        bdx, bdy, br, dist_mm = ball
        cx, cy = w_src // 2, h_src // 2
        bx, by = int(cx + bdx), int(cy + bdy)
        cv2.line(vis, (cx - 10, cy), (cx + 10, cy), (200, 200, 200), 1)
        cv2.line(vis, (cx, cy - 10), (cx, cy + 10), (200, 200, 200), 1)
        cv2.circle(vis, (bx, by), max(3, int(br)) + 2, (80, 220, 80), 2)
        geo = f"dx {bdx:+.0f} dy {bdy:+.0f} r {br:.0f}px"
        if dist_mm is not None:
            geo += f"  d {dist_mm:.0f}mm"
        else:
            geo += "  (uncalibrated)"
        cv2.putText(vis, geo, (6, h_src - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (150, 220, 150) if dist_mm is not None
                    else (150, 150, 150), 1)
    # possession indicator: a border that turns green once frac clears
    # dwibble_cam_min_frac (the test is frame-wide)
    ok = frac is not None and frac >= dw.dwibble_cam_min_frac
    col = (80, 220, 80) if ok else (60, 60, 200)
    cv2.rectangle(vis, (0, 0), (w_src - 1, h_src - 1), col, 2)
    txt = f"frac {0.0 if frac is None else frac:.2f} / {dw.dwibble_cam_min_frac:.2f}" \
          + ("  held" if seen else "")
    cv2.putText(vis, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (80, 220, 80) if seen else (200, 200, 200), 1)
    return vis


def render_debug_jpeg():
    """camera panel stacked on the field panel, JPEG-encoded: the debug page's main view."""
    with lock:
        pose = shared_state["pose"]
        ball = shared_state["ball"]
        frame = shared_state["frame"]
        mask = shared_state["mask"]
        cx_px = shared_state["cx_px"]
        cy_px = shared_state["cy_px"]
        radius = shared_state["radius"]
        enemies = shared_state["enemies"]
        teammate_pos = shared_state["teammate_pos_bt"] or shared_state["teammate_pos"]
        cam_fps = shared_state["cam_fps"]
        lidar_hz = shared_state["lidar_hz"]

    cam = render_camera_panel(pose, ball, frame, mask, cx_px, cy_px, radius,
                                 cam_fps=cam_fps)
    cam_w = cam.shape[1]
    # Second camera beside the main one, scaled to the same panel width
    dwcam = render_dwibble_cam_panel()
    dwcam = cv2.resize(dwcam, (cam_w, int(dwcam.shape[0] * cam_w / dwcam.shape[1])))
    # Field below camera, scaled to same width
    field_h = int(cam_w * FieldModel.field_y / FieldModel.field_x)
    field = render_field_panel(pose, ball, enemies, teammate_pos,
                                lidar_hz=lidar_hz,
                                target_w=cam_w, target_h=field_h)
    div = np.full((4, cam_w, 3), 60, dtype=np.uint8)
    comb = np.concatenate([cam, div, dwcam, div, field], axis=0)
    ok, buf = cv2.imencode(".jpg", comb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return bytes(buf) if ok else None


def render_hsv_jpeg():
    """side-by-side raw frame + orange mask for the HSV tuner, JPEG-encoded."""
    with lock:
        frame = shared_state["frame"]
        mask = shared_state["mask"]
        lower = shared_state["hsv_lower"]
        upper = shared_state["hsv_upper"]

    if frame is None:
        return None

    # Re-apply current (possibly just-changed) HSV values
    if lower is not None and upper is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat_boost(hsv)
        mask = cv2.inRange(hsv, lower, upper)

    h, w = frame.shape[:2]
    disp_w = w * 2
    disp_h = h * 2
    left = cv2.resize(frame, (disp_w // 2, disp_h))
    right = cv2.cvtColor(
                  cv2.resize(mask, (disp_w // 2, disp_h)),
                  cv2.COLOR_GRAY2BGR)

    # Tint detected pixels orange on the right panel
    orange_tint = np.zeros_like(right)
    orange_tint[:, :] = (0, 100, 255)
    amask = (right[:, :, 0] > 0).astype(np.float32)[:, :, None]
    right = (right * (1 - amask * 0.6) + orange_tint * amask * 0.6).astype(np.uint8)

    div = np.full((disp_h, 2, 3), 60, dtype=np.uint8)
    comb = np.concatenate([left, div, right], axis=1)
    ok, buf = cv2.imencode(".jpg", comb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return bytes(buf) if ok else None


# HTTP server

debug_html = b"""\
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Robot debug</title>
<style>body{background:#111;color:#ccc;font-family:monospace;padding:10px}
img{display:block;max-width:100%;border:1px solid #444}h2{font-size:14px;margin:4px 0}
#fps{font-size:12px;color:#8f8;margin:4px 0}
#dwcam{font-size:12px;color:#8cf;margin:4px 0;min-height:14px}
#diag{font-size:12px;color:#fc6;margin:4px 0;min-height:14px}
#slip{font-size:12px;color:#f66;margin:4px 0;min-height:14px}
#imu{font-size:12px;color:#ccc;margin:4px 0;min-height:14px}</style>
</head><body>
<h2>thermite &mdash; debug</h2>
<img src="/stream.mjpg">
<div id="fps">loop: - Hz &nbsp; cam: - fps &nbsp; lidar: - Hz</div>
<div id="dwcam">dwibble cam: -</div>
<div id="diag"></div>
<div id="slip"></div>
<div id="imu">imu: -</div>
<div id="rec"><button onclick="toggleRec()" id="recbtn">Debug recording: off</button></div>
<script>
function poll(){
  fetch("/status").then(r=>r.json()).then(d=>{
    const f = v => v === null || v === undefined ? "--" : (+v).toFixed(1);
    document.getElementById("fps").textContent =
      `loop: ${f(d.loop_hz)} Hz   cam: ${f(d.cam_fps)} fps   lidar: ${f(d.lidar_hz)} Hz`;
    const c = d.dwibble_cam || {};
    document.getElementById("dwcam").textContent =
      `dwibble cam: ${c.live ? "live" : "no feed"}   frac: ` +
      `${c.frac === null || c.frac === undefined ? "--" : (+c.frac).toFixed(2)}   ` +
      `${c.seen ? "held" : "open"}`;
    const rb = document.getElementById("recbtn");
    rb.textContent = "Debug recording: " + (d.debug_recording ? "on" : "off");
    rb.style.color = d.debug_recording ? "#f66" : "#ccc";
    rb.title = d.debug_folder || "";
    const diag = document.getElementById("diag");
    diag.textContent = (d.diag_findings && d.diag_age_s !== null && d.diag_age_s < 3)
      ? "[diag] " + d.diag_findings.join("  |  ") : "";
    const slip = document.getElementById("slip");
    slip.textContent = (d.wheel_slip_trust !== null && d.wheel_slip_trust !== undefined
                        && d.wheel_slip_trust < 0.7)
      ? `[slip] trust ${d.wheel_slip_trust.toFixed(2)}  `
        + `(wheel ${f(d.wheel_speed_mms)} mm/s vs lidar ${f(d.lidar_speed_mms)} mm/s)`
      : "";
    document.getElementById("imu").textContent =
      `imu: heading ${f(d.imu_heading)} deg   rate ${f(d.imu_yaw_rate_dps)} deg/s` +
      (d.imu_gyro_rate_enabled ? " (gyro)" : " (differenced)");
  }).catch(()=>{});
}
function toggleRec(){
  const on = document.getElementById("recbtn").textContent.includes("off");
  fetch("/set_debug_recording",{method:"post",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({on:on}))
    .then(()=>poll());
}
setInterval(poll, 500);
poll();
</script>
</body></html>
"""

hsv_html = b"""\
<!DOCTYPE html><html><head><meta charset="utf-8"><title>HSV Tuner</title>
<style>
body{background:#111;color:#ccc;font-family:monospace;padding:10px}
img{display:block;max-width:100%;border:1px solid #444;margin-bottom:10px}
.row{margin:5px 0;display:flex;align-items:center;gap:8px}
input[type=range]{width:260px}
button{background:#333;color:#ccc;border:1px solid #666;padding:5px 14px;
       cursor:pointer;font-family:monospace}
button:hover{background:#444}
#status{color:#8f8;margin-top:6px;font-size:12px}
#fps{font-size:12px;color:#8f8;margin:4px 0}
#diag{font-size:12px;color:#fc6;margin:4px 0;min-height:14px}
#slip{font-size:12px;color:#f66;margin:4px 0;min-height:14px}
</style></head><body>
<h2>HSV Tuner</h2>
<img src="/stream.mjpg" id="preview">
<div id="fps">loop: - Hz &nbsp; cam: - fps &nbsp; lidar: - Hz</div>
<div id="diag"></div>
<div id="slip"></div>
<form id="f">
<div class="row">H min<input type="range" min="0" max="179" value="5" id="hmin" oninput="send()"><span id="v_hmin">5</span></div>
<div class="row">H max<input type="range" min="0" max="179" value="20" id="hmax" oninput="send()"><span id="v_hmax">20</span></div>
<div class="row">S min<input type="range" min="0" max="255" value="150" id="smin" oninput="send()"><span id="v_smin">150</span></div>
<div class="row">S max<input type="range" min="0" max="255" value="255" id="smax" oninput="send()"><span id="v_smax">255</span></div>
<div class="row">V min<input type="range" min="0" max="255" value="100" id="vmin" oninput="send()"><span id="v_vmin">100</span></div>
<div class="row">V max<input type="range" min="0" max="255" value="255" id="vmax" oninput="send()"><span id="v_vmax">255</span></div>
</form>
<button onclick="save()">Save to hsv_settings.json</button>
<div id="status"></div>
<script>
const ids = ["hmin","hmax","smin","smax","vmin","vmax"];
function vals(){
  const o={};
  ids.forEach(k=>{
    const v=parseInt(document.getElementById(k).value);
    document.getElementById("v_"+k).textContent=v;
    o[k]=v;
  });
  return o;
}
function send(){
  fetch("/set_hsv",{method:"post",headers:{"Content-Type":"application/json"},
        body:JSON.stringify(vals())});
}
function save(){
  fetch("/save_hsv",{method:"post",headers:{"Content-Type":"application/json"},
        body:JSON.stringify(vals())})
    .then(r=>r.text()).then(t=>{document.getElementById("status").textContent=t});
}
// init sliders from server on load
fetch("/get_hsv").then(r=>r.json()).then(d=>{
  document.getElementById("hmin").value=d.lower[0];
  document.getElementById("hmax").value=d.upper[0];
  document.getElementById("smin").value=d.lower[1];
  document.getElementById("smax").value=d.upper[1];
  document.getElementById("vmin").value=d.lower[2];
  document.getElementById("vmax").value=d.upper[2];
  ids.forEach(k=>{
    document.getElementById("v_"+k).textContent=
      document.getElementById(k).value;
  });
  send();
});
function poll(){
  fetch("/status").then(r=>r.json()).then(d=>{
    const f = v => v === null || v === undefined ? "--" : (+v).toFixed(1);
    document.getElementById("fps").textContent =
      `loop: ${f(d.loop_hz)} Hz   cam: ${f(d.cam_fps)} fps   lidar: ${f(d.lidar_hz)} Hz`;
    const diag = document.getElementById("diag");
    diag.textContent = (d.diag_findings && d.diag_age_s !== null && d.diag_age_s < 3)
      ? "[diag] " + d.diag_findings.join("  |  ") : "";
    const slip = document.getElementById("slip");
    slip.textContent = (d.wheel_slip_trust !== null && d.wheel_slip_trust !== undefined
                        && d.wheel_slip_trust < 0.7)
      ? `[slip] trust ${d.wheel_slip_trust.toFixed(2)}  `
        + `(wheel ${f(d.wheel_speed_mms)} mm/s vs lidar ${f(d.lidar_speed_mms)} mm/s)`
      : "";
  }).catch(()=>{});
}
setInterval(poll, 500);
poll();
</script></body></html>
"""

# Gate calibration page: tune the white-line gate (motion._white_line_gate) live on the
# field. Same idiom as the HSV tuner: sliders set constants through /set_calib, Save
# prints them for pasting into bot/motion.py, and /status carries the gate's intervention
# counters and the pose-fusion ages.
calib_html = b"""\
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Gate calibration</title>
<style>
body{background:#111;color:#ccc;font-family:monospace;padding:10px}
h2{font-size:14px;margin:8px 0}
.row{margin:5px 0;display:flex;align-items:center;gap:8px}
input[type=range]{width:280px}
button{background:#333;color:#ccc;border:1px solid #666;padding:5px 14px;
       cursor:pointer;font-family:monospace}
button:hover{background:#444}
#status{color:#8f8;margin-top:6px;font-size:12px;min-height:14px}
#telem{font-size:12px;color:#8cf;margin:6px 0;min-height:14px}
.note{font-size:11px;color:#889;margin:6px 0}
</style></head><body>
<h2>White-line gate calibration</h2>
<div class="note">Bench panel for bot/motion.py's _white_line_gate.
Sliders apply live (no restart); Save prints the values for pasting into
bot/motion.py. The robot is still driving, so keep the wheels clear.</div>
<div class="row"><input type="checkbox" id="enabled"> gate enabled</div>
<div class="row">Slide band (mm)<input type="range" min="5" max="300" step="5" value="40" id="slide_mm" oninput="setCalib()"><span id="v_slide_mm">40</span></div>
<div class="row">Max pose age (s)<input type="range" min="0.05" max="2.0" step="0.05" value="0.3" id="max_pose_age_s" oninput="setCalib()"><span id="v_max_pose_age_s">0.3</span></div>
<button onclick="saveCalib()">Save (print to terminal)</button>
<button onclick="resetCounters()">Reset counters</button>
<div id="telem"></div>
<div id="status"></div>
<script>
function setVal(k){document.getElementById("v_"+k).textContent=
  document.getElementById(k).value;}
function body(){
  return {enabled: document.getElementById("enabled").checked,
          slide_mm: parseFloat(document.getElementById("slide_mm").value),
          max_pose_age_s: parseFloat(document.getElementById("max_pose_age_s").value)};
}
function setCalib(){
  setVal("slide_mm"); setVal("max_pose_age_s");
  fetch("/set_calib",{method:"post",headers:{"Content-Type":"application/json"},
        body:JSON.stringify(body())});
}
function saveCalib(){
  fetch("/save_calib",{method:"post",headers:{"Content-Type":"application/json"},
        body:JSON.stringify(body())})
    .then(r=>r.text()).then(t=>{document.getElementById("status").textContent=t});
}
function resetCounters(){
  fetch("/reset_gate_counters",{method:"post"})
    .then(()=>{document.getElementById("status").textContent="counters reset";});
}
function poll(){
  fetch("/status").then(r=>r.json()).then(d=>{
    const f = v => v === null || v === undefined ? "--" : (+v).toFixed(2);
    const g = d.gate || {};
    const t = document.getElementById("telem");
    t.textContent = `events: ${g.trim_events ?? "-"}   last trim: ` +
      `${f(g.last_trim_frac)}   last block: ${g.last_block ? "yes" : "no"}   ` +
      `pose age: ${f(d.pose_age_s)}s   live age: ${f(d.pose_live_age_s)}s   ` +
      `live drift: ${f(d.pose_live_drift_mm)}mm`;
  }).catch(()=>{});
}
setInterval(poll, 500);
poll();
fetch("/get_calib").then(r=>r.json()).then(d=>{
  document.getElementById("enabled").checked = d.enabled;
  document.getElementById("slide_mm").value = d.slide_mm;
  document.getElementById("max_pose_age_s").value = d.max_pose_age_s;
  setVal("slide_mm"); setVal("max_pose_age_s");
});
</script></body></html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    """request handler: the HTML pages (debug, HSV tuner, gate calibration), the MJPEG stream,
    and the HSV and gate-calibration endpoints.
    """

    def do_GET(self):
        """serve "/" (debug or HSV page), "/stream.mjpg" (MJPEG), "/status", "/get_hsv" and the
        calibration page.
        """
        if self.path == "/":
            page = hsv_html if state.mode == "hsv" else debug_html
            self.send_bytes(page, "text/html")

        elif self.path == "/calib":
            self.send_bytes(calib_html, "text/html")

        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with jpeg_cond:
                        jpeg_cond.wait()
                        data = state.jpeg_bytes
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + data + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path == "/get_hsv":
            with lock:
                lo = shared_state["hsv_lower"]
                hi = shared_state["hsv_upper"]
            body = json.dumps({"lower": lo.tolist(), "upper": hi.tolist()}).encode()
            self.send_bytes(body, "application/json")

        elif self.path == "/status":
            # thread rates for the debug page, plus diagnostics' latest motion
            # findings
            with lock:
                diag_findings = shared_state.get("diag_findings")
                diag_last_t = shared_state.get("diag_last_t")
                wheel_slip_trust = shared_state.get("wheel_slip_trust")
                wheel_speed_mms = shared_state.get("wheel_speed_mms")
                lidar_speed_mms = shared_state.get("lidar_speed_mms")
                pose = shared_state.get("pose")
                pose_t = shared_state.get("pose_t")
                pose_live = shared_state.get("pose_live")
                dw_cam_frac = shared_state.get("dwibble_cam_frac")
                dw_cam_seen = shared_state.get("dwibble_cam_seen")
                dw_cam_frame = shared_state.get("dwibble_cam_frame") is not None
                dw_cam_ball = shared_state.get("dwibble_cam_ball")
                imu_heading = shared_state.get("imu_heading")
                imu_yaw_rate_dps = shared_state.get("imu_yaw_rate_dps")
                recording = debug_session.recording
                sess = debug_session._session
                debug_folder = sess.folder if sess is not None else None
            now = time.monotonic()
            body = json.dumps({
                "loop_hz": shared_state.get("loop_hz"),
                "cam_fps": shared_state.get("cam_fps"),
                "lidar_hz": shared_state.get("lidar_hz"),
                # second camera: live orange fraction, confirm state, feed
                # alive, and the blob geometry (dx/dy/radius, plus
                # calibrated distance once loaded)
                "dwibble_cam": {
                    "live": dw_cam_frame,
                    "frac": dw_cam_frac,
                    "seen": dw_cam_seen,
                    "ball": (None if dw_cam_ball is None else {
                        "dx": dw_cam_ball[0], "dy": dw_cam_ball[1],
                        "r": dw_cam_ball[2], "dist_mm": dw_cam_ball[3],
                    }),
                },
                # debug session recording and its folder
                "debug_recording": recording,
                "debug_folder": debug_folder,
                # IMU heading and gyroscope rate, for checking the rate's sign by hand
                "imu_heading": imu_heading,
                "imu_yaw_rate_dps": imu_yaw_rate_dps,
                "imu_gyro_rate_enabled": motion.imu_gyro_rate_enabled,
                "diag_findings": diag_findings,
                "diag_age_s": (None if diag_last_t is None
                                  else now - diag_last_t),
                "wheel_slip_trust": wheel_slip_trust,
                "wheel_speed_mms": wheel_speed_mms,
                "lidar_speed_mms": lidar_speed_mms,
                # pose fusion: the raw fit's age, the extension's age, and
                # how far the live extension sits from the last fit (grows
                # between revolutions, collapses on each fresh fit)
                "pose_age_s": (None if pose is None or pose_t is None
                                   else now - pose_t),
                "pose_live_age_s": (None if pose_live is None
                                    else now - pose_live[3]),
                "pose_live_drift_mm": (None if pose is None or pose_live is None
                                       else math.hypot(pose_live[0] - pose[0],
                                                       pose_live[1] - pose[1])),
                # white-line gate calibration: live constants and
                # intervention counters
                "gate": {
                    "enabled": motion.white_gate_enabled,
                    "slide_mm": motion.white_gate_slide_mm,
                    "max_pose_age_s": motion.white_gate_max_pose_age_s,
                    "trim_events": motion.gate_trim_events,
                    "last_trim_frac": motion.gate_last_trim_frac,
                    "last_block": motion.gate_last_block,
                },
            }).encode()
            self.send_bytes(body, "application/json")

        elif self.path == "/get_calib":
            body = json.dumps({
                "enabled": motion.white_gate_enabled,
                "slide_mm": motion.white_gate_slide_mm,
                "max_pose_age_s": motion.white_gate_max_pose_age_s,
            }).encode()
            self.send_bytes(body, "application/json")

        else:
            self.send_error(404)

    def do_POST(self):
        """handle "/set_hsv" (apply thresholds live), "/save_hsv" (print them for pasting into
        bot/vision.py), "/set_calib" and "/save_calib" (the white-line gate, pasted into
        bot/motion.py), and "/reset_gate_counters".
        """
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/set_hsv":
            lo = np.array([body.get("hmin", 0),
                           body.get("smin", 0),
                           body.get("vmin", 0)], dtype=np.uint8)
            hi = np.array([body.get("hmax", 179),
                           body.get("smax", 255),
                           body.get("vmax", 255)], dtype=np.uint8)
            with lock:
                shared_state["hsv_lower"] = lo
                shared_state["hsv_upper"] = hi
            self.send_bytes(b"ok", "text/plain")

        elif self.path == "/save_hsv":
            with lock:
                lo = shared_state["hsv_lower"]
                hi = shared_state["hsv_upper"]
            msg = (f"orange_lower = np.array({lo.tolist()}, dtype=np.uint8)\n"
                   f"orange_upper = np.array({hi.tolist()}, dtype=np.uint8)")
            print(f"\n[hsv] copy these into bot/vision.py:\n{msg}\n", flush=True)
            self.send_bytes(b"Values printed to the terminal. Copy them into bot/vision.py",
                             "text/plain")

        elif self.path == "/set_calib":
            # live-tune the white-line gate (bench only: nothing persists
            # until /save_calib's output is pasted into bot/motion.py)
            try:
                if "enabled" in body:
                    motion.white_gate_enabled = bool(body["enabled"])
                if "slide_mm" in body:
                    v = float(body["slide_mm"])
                    if not (5.0 <= v <= 300.0):
                        raise ValueError
                    motion.white_gate_slide_mm = v
                if "max_pose_age_s" in body:
                    v = float(body["max_pose_age_s"])
                    if not (0.05 <= v <= 2.0):
                        raise ValueError
                    motion.white_gate_max_pose_age_s = v
                self.send_bytes(b"ok", "text/plain")
            except (TypeError, ValueError):
                self.send_error(400)

        elif self.path == "/save_calib":
            msg = (f"white_gate_enabled = {motion.white_gate_enabled!r}\n"
                   f"white_gate_slide_mm = {motion.white_gate_slide_mm!r}\n"
                   f"white_gate_max_pose_age_s = {motion.white_gate_max_pose_age_s!r}")
            print(f"\n[calib] copy these into bot/motion.py:\n{msg}\n", flush=True)
            self.send_bytes(b"Values printed to the terminal. Copy them into bot/motion.py",
                            "text/plain")

        elif self.path == "/reset_gate_counters":
            motion.reset_gate_calibration_counters()
            self.send_bytes(b"ok", "text/plain")

        elif self.path == "/set_debug_recording":
            # pause or resume the debug session (bot/debug_session.py): {"on": true/false}
            on = bool(body.get("on"))
            debug_session.set_recording(on)
            self.send_bytes(b"ok", "text/plain")

        else:
            self.send_error(404)

    def send_bytes(self, data, content_type):
        """write a full 200 response with the given body and Content-Type."""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        """suppress BaseHTTPRequestHandler's default per-request stderr logging."""
        pass


def http_server_thread():
    """run the debug web server (HTML page + MJPEG stream + HSV endpoints) forever."""
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("0.0.0.0", debug_port), Handler) as srv:
        srv.serve_forever()


def render_thread():
    """render the debug/HSV JPEG for the MJPEG stream, once per new camera frame."""
    while True:
        with frame_cond:
            frame_cond.wait()
        if state.mode == "hsv":
            data = render_hsv_jpeg()
        else:
            data = render_debug_jpeg()
        if data:
            with jpeg_cond:
                state.jpeg_bytes = data
                jpeg_cond.notify_all()
