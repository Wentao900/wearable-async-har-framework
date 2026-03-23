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


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the wearable async HAR scaffold.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_name = str(config.get("data", {}).get("dataset", "synthetic")).lower()

    try:
        trainer = Trainer(config)
        results = trainer.train()
    except FileNotFoundError as exc:
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
        print(f"Training could not start: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(config.get("logging", {}).get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nSaved results to: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
