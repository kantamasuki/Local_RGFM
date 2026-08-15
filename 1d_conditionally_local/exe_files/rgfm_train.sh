#!/usr/bin/env bash
set -euo pipefail

EXE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="$EXE_DIR/.."
CONFIG_DIR="$PROJECT_DIR/condloc_rgfm/configs/condloc_L1024"
DEVICE="${1:-0}"

cd "$PROJECT_DIR/condloc_rgfm"

python rgfm_train.py --config "$CONFIG_DIR/L1024_t0p00_0p41.json" --device "$DEVICE"
python rgfm_train.py --config "$CONFIG_DIR/L512_t0p41_0p50.json" --device "$DEVICE"
python rgfm_train.py --config "$CONFIG_DIR/L256_t0p50_0p59.json" --device "$DEVICE"
python rgfm_train.py --config "$CONFIG_DIR/L128_t0p59_0p69.json" --device "$DEVICE"
python rgfm_train.py --config "$CONFIG_DIR/L64_t0p69_0p78.json" --device "$DEVICE"
python rgfm_train.py --config "$CONFIG_DIR/L32_t0p78_0p87.json" --device "$DEVICE"
python rgfm_train.py --config "$CONFIG_DIR/L16_t0p87_1p00.json" --device "$DEVICE"
