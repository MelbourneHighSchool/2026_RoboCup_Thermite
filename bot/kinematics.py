"""Drivetrain kinematics: the one place the holonomic wheel math lives."""

import math

# set once _native_check() has judged the extension (or its absence)
_native_state = {"checked": False, "ok": False, "solve": None}


def solve_wheel_commands(bearing, speed, rot_r=0.0, rot_theta=0.0,
                         rot_speed=0.0, cap_frac=0.9,
                         turn_radius_mm=94.5):
    """robot-frame (bearing, speed) + rotation -> per-wheel fractional commands."""
    rad = math.radians(bearing)
    f = math.cos(rad) * speed
    s = math.sin(rad) * speed
    trans = {"nw": f + s, "ne": -f + s, "sw": -f + s, "se": f + s}

    # rotation about (rot_r, rot_theta): pure spin + induced translation
    fi = si = 0.0
    if rot_speed and rot_r:
        ind_b = math.radians(rot_theta - math.copysign(90.0, rot_speed))
        ind_c = abs(rot_speed) * rot_r / (math.sqrt(2.0) * turn_radius_mm)
        fi = math.cos(ind_b) * ind_c
        si = math.sin(ind_b) * ind_c

    speeds = {
        "nw": trans["nw"] + fi + si + rot_speed,
        "ne": trans["ne"] - fi + si + rot_speed,
        "sw": trans["sw"] - fi + si - rot_speed,
        "se": trans["se"] + fi + si - rot_speed,
    }

    # per-motor ceiling, scale everything down together
    peak = max(abs(v) for v in speeds.values())
    if peak > cap_frac:
        k = cap_frac / peak
        speeds = {n: v * k for n, v in speeds.items()}

    # se and sw are mounted with reversed polarity, negate the others
    return {n: float(v) if n in ("se", "sw") else float(-v)
            for n, v in speeds.items()}


def _native_check():
    """resolve and (once) sanity-check the optional native solver."""
    if _native_state["checked"]:
        return _native_state["solve"]
    _native_state["checked"] = True
    solve = solve_wheel_commands
    try: # optional compiled extension, same signature
        from bot import _kinematics_native # type: ignore[attr-defined]
        candidate = _kinematics_native.solve_wheel_commands_native
        probe = candidate(45.0, 0.5, 0.0, 0.0, 0.0, 0.9, 110.0)
        python_out = solve_wheel_commands(45.0, 0.5, 0.0, 0.0, 0.0, 0.9, 110.0)
        if (set(probe) == set(python_out)
                and all(abs(probe[k] - python_out[k]) < 1e-9
                        for k in python_out)):
            solve = candidate
            _native_state["ok"] = True
        else:
            print("[kinematics] native solver disagrees with the Python solver, "
                  "staying on Python", flush=True)
    except ImportError:
        pass # not built, the normal case today
    except Exception as exc: # noqa: BLE001 (a broken extension must never take down drive)
        print(f"[kinematics] native solver unusable ({exc!r}), "
              "staying on Python", flush=True)
    _native_state["solve"] = solve
    return solve


def using_native():
    """True when the optional native solver passed its parity check."""
    _native_check()
    return _native_state["ok"]


def solve(bearing, speed, rot_r=0.0, rot_theta=0.0, rot_speed=0.0,
          cap_frac=0.9, turn_radius_mm=94.5):
    """the entry point Motor.drive calls: the native solver once it passes its parity check,
    else Python.
    """
    return _native_check()(bearing, speed, rot_r, rot_theta, rot_speed,
                           cap_frac, turn_radius_mm)


if __name__ == "__main__":
    # bench check: print three cases (pure translation, spin, off-centre rotation) to
    # eyeball.
    cases = [
        # pure translation at 45 deg: nw/se carry it, ne/sw sit at zero; well
        # under the cap
        dict(bearing=45.0, speed=0.5),
        # pure spin: every wheel at |rot_speed|, scaled down to cap_frac
        dict(bearing=0.0, speed=0.0, rot_speed=1.0),
        # off-centre rotation plus translation, enough to engage the cap
        dict(bearing=45.0, speed=0.9, rot_r=200.0, rot_theta=30.0, rot_speed=0.3),
    ]
    for c in cases:
        out = solve_wheel_commands(cap_frac=0.9, **c)
        print(c, "->", {k: round(v, 6) for k, v in out.items()})
    print("native in use:", using_native())
