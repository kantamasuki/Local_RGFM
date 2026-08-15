#!/usr/bin/env bash
set -euo pipefail

EXE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$EXE_DIR/../.."
PROJECT_DIR="$EXE_DIR/.."
CONFIG="$PROJECT_DIR/ising_fm/configs/L1024_xi128.json"
CKPT="$REPO_DIR/check_points/ising/fm_L1024_xi128.pt"
DEVICE="${1:-0}"

cd "$PROJECT_DIR/ising_fm"

python fm_evaluate.py \
  --config_L1024 "$CONFIG" \
  --ckpt_L1024 "$CKPT" \
  --device "$DEVICE"
