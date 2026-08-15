#!/usr/bin/env bash
set -euo pipefail

EXE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="$EXE_DIR/.."
CONFIG="$PROJECT_DIR/condloc_fm/configs/fm_L1024.json"
DEVICE="${1:-0}"

cd "$PROJECT_DIR/condloc_fm"

python fm_train.py --config "$CONFIG" --device "$DEVICE"
