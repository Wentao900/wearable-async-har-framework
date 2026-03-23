#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_gpu_background.sh --config <config.yaml> [--run-name <name>] [--output-dir <dir>]

Examples:
  bash scripts/run_gpu_background.sh --config configs/pamap2-gpu-safe.yaml
  bash scripts/run_gpu_background.sh --config configs/wisdm-gpu-fast.yaml --run-name wisdm-fast-1gpu

What it does:
  - creates a timestamped run directory,
  - launches training with nohup in the background,
  - writes stdout/stderr to <run-dir>/train.log,
  - relies on scripts/train.py to write runtime_info.json and results.json.
EOF
}

CONFIG=""
RUN_NAME=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Missing required --config argument." >&2
  usage >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file not found: $CONFIG" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

if [[ -z "$OUTPUT_DIR" ]]; then
  BASE_OUTPUT_DIR="$(python3 - "$CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
with config_path.open('r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}
print(config.get('logging', {}).get('output_dir', 'outputs/default'))
PY
)"
  if [[ -n "$RUN_NAME" ]]; then
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TIMESTAMP}-${RUN_NAME}"
  else
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TIMESTAMP}"
  fi
fi

mkdir -p "$OUTPUT_DIR"
LOG_PATH="${OUTPUT_DIR}/train.log"
PID_PATH="${OUTPUT_DIR}/run.pid"

nohup python3 scripts/train.py --config "$CONFIG" --output-dir "$OUTPUT_DIR" >"$LOG_PATH" 2>&1 &
PID=$!
echo "$PID" > "$PID_PATH"

cat <<EOF
Started background GPU run.

config:      $CONFIG
output dir:  $OUTPUT_DIR
log file:    $LOG_PATH
pid:         $PID

Useful commands:
  tail -f "$LOG_PATH"
  cat "$OUTPUT_DIR/runtime_info.json"
  cat "$OUTPUT_DIR/results.json"
  ps -p $PID -o pid=,etime=,cmd=
EOF
