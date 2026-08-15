#!/usr/bin/env bash
set -euo pipefail

EXE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="$EXE_DIR/.."
CONFIG_DIR="$PROJECT_DIR/ising_rgfm/configs/ising1d_L1024_xi128"
DEVICE="${1:-0}"

cd "$PROJECT_DIR/ising_rgfm"

python rgfm_train.py --config "$CONFIG_DIR/L1024.json" --device "$DEVICE"
python rgfm_train.py --config "$CONFIG_DIR/L512.json" --device "$DEVICE"
python rgfm_train.py --config "$CONFIG_DIR/L256.json" --device "$DEVICE"
python rgfm_train.py --config "$CONFIG_DIR/L128.json" --device "$DEVICE"
python rgfm_train.py --config "$CONFIG_DIR/L64.json" --device "$DEVICE"
python rgfm_train.py --config "$CONFIG_DIR/L32.json" --device "$DEVICE"
python rgfm_train.py --config "$CONFIG_DIR/L16.json" --device "$DEVICE"
