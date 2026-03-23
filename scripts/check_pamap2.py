from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.pamap2 import PAMAP2Dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/PAMAP2")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    ds = PAMAP2Dataset(root=args.root, split=args.split)
    sample = ds[0]
    summary = {
        "num_samples": len(ds),
        "modalities": list(sample["values"].keys()),
        "shapes": {k: list(v.shape) for k, v in sample["values"].items()},
        "label": sample["label"],
        "metadata": sample["metadata"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
