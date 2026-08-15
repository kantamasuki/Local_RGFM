#!/usr/bin/env bash
set -euo pipefail

EXE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$EXE_DIR/../.."
PROJECT_DIR="$EXE_DIR/.."
CONFIG_DIR="$PROJECT_DIR/ising_rgfm/configs/ising1d_L1024_xi128"
CKPT_DIR="$REPO_DIR/check_points/ising/rgfm_baseL1024_xi128"
DEVICE="${1:-0}"

cd "$PROJECT_DIR/ising_rgfm"

python rgfm_sample.py \
  --config_L1024 "$CONFIG_DIR/L1024.json" \
  --ckpt_L1024 "$CKPT_DIR/L1024.pt" \
  --config_L512 "$CONFIG_DIR/L512.json" \
  --ckpt_L512 "$CKPT_DIR/L512.pt" \
  --config_L256 "$CONFIG_DIR/L256.json" \
  --ckpt_L256 "$CKPT_DIR/L256.pt" \
  --config_L128 "$CONFIG_DIR/L128.json" \
  --ckpt_L128 "$CKPT_DIR/L128.pt" \
  --config_L64 "$CONFIG_DIR/L64.json" \
  --ckpt_L64 "$CKPT_DIR/L64.pt" \
  --config_L32 "$CONFIG_DIR/L32.json" \
  --ckpt_L32 "$CKPT_DIR/L32.pt" \
  --config_L16 "$CONFIG_DIR/L16.json" \
  --ckpt_L16 "$CKPT_DIR/L16.pt" \
  --device "$DEVICE"
