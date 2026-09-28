#!/bin/bash
# Boot entry point. The service supplies ROBOT_ID; direct runs must set it,
# for example: ROBOT_ID=1 ./start.sh.
set -euo pipefail

REPO_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/env}"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Virtual environment not found at $VENV_DIR" >&2
    exit 1
fi

if [ "${ROBOT_ID:-}" != "1" ] && [ "${ROBOT_ID:-}" != "2" ]; then
    echo "Set ROBOT_ID to 1 or 2 before starting the robot." >&2
    exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
exec python3 -m bot.main "$@"
