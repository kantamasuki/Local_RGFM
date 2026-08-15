#!/usr/bin/env bash
set -euo pipefail

EXE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$EXE_DIR/../.."
PROJECT_DIR="$EXE_DIR/.."
CONFIG="$PROJECT_DIR/condloc_fm/configs/fm_L1024.json"
CKPT="$REPO_DIR/check_points/cond_loc/fm_L1024.pt"
DEVICE="${1:-0}"

cd "$PROJECT_DIR/condloc_fm"

python fm_sample.py \
  --config_L1024 "$CONFIG" \
  --ckpt_L1024 "$CKPT" \
  --device "$DEVICE"
