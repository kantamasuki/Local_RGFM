#!/usr/bin/env bash
set -euo pipefail

EXE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$EXE_DIR/../.."
PROJECT_DIR="$EXE_DIR/.."
CONFIG_DIR="$PROJECT_DIR/condloc_rgfm/configs/condloc_L1024"
CKPT_DIR="$REPO_DIR/check_points/cond_loc/rgfm_baseL1024"
DEVICE="${1:-0}"

cd "$PROJECT_DIR/condloc_rgfm"

python rgfm_sample.py \
  --config_L1024 "$CONFIG_DIR/L1024_t0p00_0p41.json" \
  --ckpt_L1024 "$CKPT_DIR/L1024.pt" \
  --config_L512 "$CONFIG_DIR/L512_t0p41_0p50.json" \
  --ckpt_L512 "$CKPT_DIR/L512.pt" \
  --config_L256 "$CONFIG_DIR/L256_t0p50_0p59.json" \
  --ckpt_L256 "$CKPT_DIR/L256.pt" \
  --config_L128 "$CONFIG_DIR/L128_t0p59_0p69.json" \
  --ckpt_L128 "$CKPT_DIR/L128.pt" \
  --config_L64 "$CONFIG_DIR/L64_t0p69_0p78.json" \
  --ckpt_L64 "$CKPT_DIR/L64.pt" \
  --config_L32 "$CONFIG_DIR/L32_t0p78_0p87.json" \
  --ckpt_L32 "$CKPT_DIR/L32.pt" \
  --config_L16 "$CONFIG_DIR/L16_t0p87_1p00.json" \
  --ckpt_L16 "$CKPT_DIR/L16.pt" \
  --device "$DEVICE"
