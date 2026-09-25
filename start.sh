#!/bin/bash
# start.sh: boot entrypoint. The systemd unit
# (bot/systemd/lidar-robot.service) sets ROBOT_ID, this script activates the
# venv, then hands off to the packaged entrypoint bot/main.py - no button,
# plays as soon as the mode/role buttons are pressed (sec 1).
#
# Assumes this repo is at ~/Stuff and the venv at ~/env. If either lives
# elsewhere on your Pi, edit STUFF_DIR and VENV_DIR below.
set -e

STUFF_DIR="$HOME/Stuff"
VENV_DIR="$HOME/env"

if [ -z "$ROBOT_ID" ]; then
    echo "ROBOT_ID not set (systemd unit should set it) - refusing to guess" >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$STUFF_DIR"
exec python -m bot.main
