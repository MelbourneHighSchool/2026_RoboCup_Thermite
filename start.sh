#!/bin/bash
# start.sh: boot entrypoint. Activates the venv then hands off to
# mainrun.py, no button, plays as soon as the mode/role buttons are
# pressed (sec 1).
#
# Assumes this repo is at ~/Stuff and the venv at ~/env. If either lives
# elsewhere on your Pi, edit STUFF_DIR and VENV_DIR below.
set -e

STUFF_DIR="$HOME/Stuff"
VENV_DIR="$HOME/env"

source "$VENV_DIR/bin/activate"
cd "$STUFF_DIR"
exec python mainrun.py
