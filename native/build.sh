#!/usr/bin/env bash
# Build the pybind11 C++ extensions in this directory. Run from anywhere;
# always builds into native/ next to this script.
#
# Deps: g++ (C++17), pybind11 (`pip3 install pybind11`), numpy.
# The script builds with `-march=native`, which targets whatever CPU runs
# the build, so build on the Pi itself rather than copying a .so over.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY=${PYTHON:-python3}
# Take the paths from $PY's own sysconfig rather than python3-config. With
# more than one Python installed, python3-config can point at a different
# interpreter and build a .so that this python3 can't import.
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
build camera_native # kept but not wired into bot/ (slower than OpenCV)

echo "done: $(ls -- *"$EXT" 2>/dev/null | tr '\n' ' ')"
