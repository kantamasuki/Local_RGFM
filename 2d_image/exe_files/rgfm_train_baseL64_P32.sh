#!/usr/bin/env bash
set -euo pipefail

EXE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="$EXE_DIR/.."
CONFIG_DIR="$PROJECT_DIR/image_rgfm/configs/rg_L64_P32"
DEVICE="${1:-0}"

cd "$PROJECT_DIR/image_rgfm"

python rgfm_train.py --config "$CONFIG_DIR/L64_P32_D16_t0p00_0p63.json" --device "$DEVICE"
python rgfm_train.py --config "$CONFIG_DIR/L32_P32_D16_t0p63_0p72.json" --device "$DEVICE"
python rgfm_train.py --config "$CONFIG_DIR/L16_P16_D16_t0p72_1p00.json" --device "$DEVICE"
