#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-safe}"

case "$MODE" in
  safe)
    CONFIG="configs/pamap2-gpu-safe.yaml"
    ;;
  fast)
    CONFIG="configs/pamap2-gpu-fast.yaml"
    ;;
  *)
    echo "Usage: $0 [safe|fast]" >&2
    exit 1
    ;;
esac

echo "Running PAMAP2 with $CONFIG"
python3 scripts/train.py --config "$CONFIG"
