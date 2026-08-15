#!/usr/bin/env bash
set -euo pipefail

EXE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$EXE_DIR/../.."
PROJECT_DIR="$EXE_DIR/.."
CONFIG_DIR="$PROJECT_DIR/image_rgfm/configs/rg_L256_P32"
CKPT_DIR="$REPO_DIR/check_points/image/rgfm_baseL256_P32"
DEVICE="${1:-0}"

cd "$PROJECT_DIR/image_rgfm"

python rgfm_sample_L256.py \
  --config256 "$CONFIG_DIR/L256_P32_D16_t0p00_0p46.json" \
  --ckpt256 "$CKPT_DIR/L256.pt" \
  --config128 "$CONFIG_DIR/L128_P32_D16_t0p46_0p55.json" \
  --ckpt128 "$CKPT_DIR/L128.pt" \
  --config64 "$CONFIG_DIR/L64_P32_D16_t0p55_0p63.json" \
  --ckpt64 "$CKPT_DIR/L64.pt" \
  --config32 "$CONFIG_DIR/L32_P32_D16_t0p63_0p72.json" \
  --ckpt32 "$CKPT_DIR/L32.pt" \
  --config16 "$CONFIG_DIR/L16_P16_D16_t0p72_1p00.json" \
  --ckpt16 "$CKPT_DIR/L16.pt" \
  --device "$DEVICE"
