#!/usr/bin/env bash
set -euo pipefail

EXE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$EXE_DIR/../.."
PROJECT_DIR="$EXE_DIR/.."
CONFIG="$PROJECT_DIR/image_fm/configs/fm_L64_P64.json"
CKPT="$REPO_DIR/check_points/image/fm_L64_P64.pt"
DEVICE="${1:-0}"

cd "$PROJECT_DIR/image_fm"

python fm_sample.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --device "$DEVICE"
