#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-safe}"
RUN_MODE="${2:-foreground}"

case "$MODE" in
  safe)
    CONFIG="configs/wisdm-gpu-safe.yaml"
    ;;
  fast)
    CONFIG="configs/wisdm-gpu-fast.yaml"
    ;;
  *)
    echo "Usage: $0 [safe|fast] [foreground|background]" >&2
    exit 1
    ;;
esac

case "$RUN_MODE" in
  foreground)
    echo "Running WISDM in foreground with $CONFIG"
    python3 scripts/train.py --config "$CONFIG"
    ;;
  background)
    echo "Starting WISDM background run with $CONFIG"
    bash scripts/run_gpu_background.sh --config "$CONFIG" --run-name "wisdm-${MODE}"
    ;;
  *)
    echo "Usage: $0 [safe|fast] [foreground|background]" >&2
    exit 1
    ;;
esac
