from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DatasetConfigurationError
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.runtime_info import build_runtime_info, write_runtime_info


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the wearable async HAR scaffold.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override. Defaults to logging.output_dir from the config.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(args.config)
    if args.output_dir:
        config.setdefault("logging", {})["output_dir"] = args.output_dir

    dataset_name = str(config.get("data", {}).get("dataset", "synthetic")).lower()
    output_dir = Path(config.get("logging", {}).get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_info_path = output_dir / "runtime_info.json"

    initial_runtime_info = build_runtime_info(
        project_root=PROJECT_ROOT,
        config_path=config_path,
        output_dir=output_dir,
        status="running",
        config=config,
    )
    write_runtime_info(runtime_info_path, initial_runtime_info)

    try:
        trainer = Trainer(config)
        results = trainer.train()
    except FileNotFoundError as exc:
        failed_runtime_info = build_runtime_info(
            project_root=PROJECT_ROOT,
            config_path=config_path,
            output_dir=output_dir,
            status="failed",
            config=config,
            extra={"error": str(exc)},
        )
        write_runtime_info(runtime_info_path, failed_runtime_info)
        if dataset_name == "pamap2":
            print(
                "PAMAP2 dataset files were not found. "
                "Expected raw files like data/PAMAP2/Protocol/subject101.dat. "
                "This repo includes only a starter loader, not bundled PAMAP2 data.",
                file=sys.stderr,
            )
            print(f"Details: {exc}", file=sys.stderr)
            return 1
        if dataset_name == "wisdm":
            print(
                "WISDM dataset file was not found. "
                "Expected a raw file like data/WISDM/WISDM_ar_v1.1_raw.txt. "
                "This repo includes only a starter loader, not bundled WISDM data.",
                file=sys.stderr,
            )
            print(f"Details: {exc}", file=sys.stderr)
            return 1
        raise
    except (DatasetConfigurationError, ValueError) as exc:
        failed_runtime_info = build_runtime_info(
            project_root=PROJECT_ROOT,
            config_path=config_path,
            output_dir=output_dir,
            status="failed",
            config=config,
            extra={"error": str(exc)},
        )
        write_runtime_info(runtime_info_path, failed_runtime_info)
        print(f"Training could not start: {exc}", file=sys.stderr)
        return 1

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    completed_runtime_info = build_runtime_info(
        project_root=PROJECT_ROOT,
        config_path=config_path,
        output_dir=output_dir,
        status="completed",
        config=config,
        extra={"results_path": str(results_path.resolve())},
    )
    write_runtime_info(runtime_info_path, completed_runtime_info)

    print(json.dumps(results, indent=2))
    print(f"\nSaved results to: {results_path}")
    print(f"Saved runtime metadata to: {runtime_info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
