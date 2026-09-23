#!/usr/bin/env bash
# Build the pybind11 C++ extensions in this directory. Run from anywhere;
# always builds into native/ next to this script.
#
# Deps: g++ (C++17), pybind11 (`pip3 install pybind11`), numpy.
# On the Pi, use `-march=native` too (already in this script) - it
# targets whatever CPU actually builds it, so build ON the Pi, don't
# cross-compile/copy a .so built elsewhere.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY=${PYTHON:-python3}
# Derive everything from $PY itself via sysconfig, not a separate
# python3-config binary - on a machine with more than one Python install
# (seen while developing this), `python3-config` can silently point at a
# DIFFERENT interpreter/ABI than `python3` itself, building a .so this
# exact `python3` can't import. sysconfig always matches the interpreter
# that ran it.
PB_INC=$("$PY" -c "import pybind11; print(pybind11.get_include())")
NP_INC=$("$PY" -c "import numpy; print(numpy.get_include())")
PY_INC=$("$PY" -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
EXT=$("$PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

build() {
    local name="$1"; shift
    echo "building $name$EXT ..."
    g++ -O3 -march=native -ffast-math -fopenmp -Wall -shared -std=c++17 -fPIC \
        $PY_INC -I"$PB_INC" -I"$NP_INC" "$name.cpp" -o "$name$EXT" "$@"
}

build lidar_native
build mcl_native
build camera_native   # kept for the record, see this dir's own README note - not wired into bot/

echo "done: $(ls -- *"$EXT" 2>/dev/null | tr '\n' ' ')"
