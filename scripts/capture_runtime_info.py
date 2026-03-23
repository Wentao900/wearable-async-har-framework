from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.runtime_info import build_runtime_info, write_runtime_info


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture runtime metadata for an experiment run.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", required=True, help="Run output directory.")
    parser.add_argument(
        "--status",
        default="running",
        help="Run status to record, e.g. running, completed, failed.",
    )
    parser.add_argument(
        "--write-path",
        default=None,
        help="Optional explicit JSON output path. Defaults to <output-dir>/runtime_info.json.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    write_path = Path(args.write_path) if args.write_path else output_dir / "runtime_info.json"

    payload = build_runtime_info(
        project_root=PROJECT_ROOT,
        config_path=Path(args.config),
        output_dir=output_dir,
        status=args.status,
        config=config,
    )
    write_runtime_info(write_path, payload)
    print(f"Wrote runtime metadata to: {write_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
