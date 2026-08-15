#!/usr/bin/env bash
set -euo pipefail

EXE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="$EXE_DIR/.."
CONFIG="$PROJECT_DIR/image_fm/configs/fm_L64_P32.json"
DEVICE="${1:-0}"

cd "$PROJECT_DIR/image_fm"

python fm_train.py --config "$CONFIG" --device "$DEVICE"
