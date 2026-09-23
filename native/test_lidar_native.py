"""Parity + benchmark for lidar_native against LidarReader.parse_packet/
crc8 (bot/lidar.py). Run from the repo root: `python3 native/test_lidar_native.py`."""
import random
import struct
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "native")
import bot.lidar as m
import lidar_native as ln


def make_packet(speed_dps, start_angle, end_angle, points, good_crc=True):
    buf = bytearray(47)
    buf[0], buf[1] = 0x54, 0x2C
    struct.pack_into("<H", buf, 2, speed_dps)
    struct.pack_into("<H", buf, 4, start_angle)
    for i, (dist, inten) in enumerate(points):
        struct.pack_into("<HB", buf, 6 + i * 3, dist, inten)
    struct.pack_into("<H", buf, 42, end_angle)
    crc = m.LidarReader.crc8(bytes(buf[:-1]))
    buf[46] = crc if good_crc else (crc ^ 0xFF)
    return bytes(buf)


def test_parity(n=20000, seed=0):
    random.seed(seed)
    mismatches = 0
    for _ in range(n):
        speed = random.randint(0, 65000)
        start = random.randint(0, 35999)
        end = random.randint(0, 35999)
        pts = [(random.randint(0, 60000), random.randint(0, 255)) for _ in range(12)]
        good = random.random() < 0.9
        buf = make_packet(speed, start, end, pts, good_crc=good)

        py_pts, py_speed = m.LidarReader.parse_packet(buf)
        cpp_pts, cpp_speed = ln.parse_packet(buf)

        if (py_pts is None) != (cpp_pts is None):
            mismatches += 1
            continue
        if py_pts is None:
            continue
        if py_speed != cpp_speed:
            mismatches += 1
            continue
        for p, (ca, cd, ci) in zip(py_pts, cpp_pts):
            if abs(p.angle_deg - ca) > 1e-9 or p.distance_mm != cd or p.intensity != ci:
                mismatches += 1
                break
    assert mismatches == 0, f"{mismatches}/{n} packets mismatched"
    print(f"parity ok: 0/{n} mismatches")


def bench(n=200000):
    buf = make_packet(3600, 0, 3000, [(1500, 200)] * 12)
    t0 = time.perf_counter()
    for _ in range(n):
        m.LidarReader.parse_packet(buf)
    t1 = time.perf_counter()
    for _ in range(n):
        ln.parse_packet(buf)
    t2 = time.perf_counter()
    py_t, cpp_t = t1 - t0, t2 - t1
    print(f"python: {n/py_t:.0f} pkt/s   cpp: {n/cpp_t:.0f} pkt/s   speedup: {py_t/cpp_t:.1f}x")


if __name__ == "__main__":
    test_parity()
    bench()
