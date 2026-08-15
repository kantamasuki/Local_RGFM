#!/usr/bin/env bash
set -euo pipefail

EXE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$EXE_DIR/../.."
PROJECT_DIR="$EXE_DIR/.."
CONFIG_DIR="$PROJECT_DIR/image_rgfm/configs/rg_L64_P32"
CKPT_DIR="$REPO_DIR/check_points/image/rgfm_baseL64_P32"
DEVICE="${1:-0}"

cd "$PROJECT_DIR/image_rgfm"

python rgfm_sample.py \
  --config64 "$PROJECT_DIR/image_rgfm/configs/rg_L64_P64/L64_P64_D16_t0p00_0p63.json" \
  --ckpt64 "$REPO_DIR/check_points/image/rgfm_baseL64_P64/L64.pt" \
  --config32 "$CONFIG_DIR/L32_P32_D16_t0p63_0p72.json" \
  --ckpt32 "$CKPT_DIR/L32.pt" \
  --config16 "$CONFIG_DIR/L16_P16_D16_t0p72_1p00.json" \
  --ckpt16 "$CKPT_DIR/L16.pt" \
  --device "$DEVICE"
