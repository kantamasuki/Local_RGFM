#!/usr/bin/env bash
set -euo pipefail

EXE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$EXE_DIR/../.."
PROJECT_DIR="$EXE_DIR/.."
CONFIG="$PROJECT_DIR/image_fm/configs/fm_L64_P32.json"
CKPT="$REPO_DIR/check_points/image/fm_L64_P32.pt"
FID_STATS="$PROJECT_DIR/FID_features/ffhq64_raw_L64_cleanfid.npz"
DEVICE="${1:-0}"

cd "$PROJECT_DIR/image_fm"

python fm_evaluate.py \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --fid_stats "$FID_STATS" \
  --device "$DEVICE"
