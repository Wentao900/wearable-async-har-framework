#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-safe}"

case "$MODE" in
  safe)
    CONFIG="configs/wisdm-gpu-safe.yaml"
    ;;
  fast)
    CONFIG="configs/wisdm-gpu-fast.yaml"
    ;;
  *)
    echo "Usage: $0 [safe|fast]" >&2
    exit 1
    ;;
esac

echo "Running WISDM with $CONFIG"
python3 scripts/train.py --config "$CONFIG"
