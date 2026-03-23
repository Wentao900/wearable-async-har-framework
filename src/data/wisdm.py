from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional

import csv
import numpy as np

from .dataset_base import BaseWearableDataset, SensorSample


@dataclass
class WISDMWindow:
    values: Dict[str, np.ndarray]
    timestamps: Dict[str, np.ndarray]
    masks: Dict[str, np.ndarray]
    label: int
    metadata: Dict[str, Any]


class WISDMDataset(BaseWearableDataset):
    """Minimal WISDM adapter scaffold.

    Expected layout (starter assumption):
        data/WISDM/WISDM_ar_v1.1_raw.txt

    This loader is intentionally conservative and should be treated as a
    starter path. WISDM parsing conventions vary across mirrors and versions,
    so preprocessing assumptions should be verified before reporting results.
    """

    LABEL_TO_ID = {
        "Walking": 0,
        "Jogging": 1,
        "Upstairs": 2,
        "Downstairs": 3,
        "Sitting": 4,
        "Standing": 5,
    }

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        window_size: int = 256,
        stride: int = 128,
        modalities: Optional[List[str]] = None,
    ) -> None:
        super().__init__(split=split)
        self.root = Path(root)
        self.window_size = window_size
        self.stride = stride
        self.modalities_list = modalities or ["accelerometer"]
        self.samples: List[WISDMWindow] = []

        raw_file = self.root / "WISDM_ar_v1.1_raw.txt"
        if not raw_file.exists():
            raise FileNotFoundError(
                f"WISDM raw file not found: {raw_file}. "
                "Place the dataset under data/WISDM/WISDM_ar_v1.1_raw.txt"
            )

        rows = self._load_rows(raw_file)
        if not rows:
            raise FileNotFoundError("WISDM file was found but no usable rows were parsed.")

        grouped = self._group_by_user(rows)
        split_users = self._select_split(sorted(grouped.keys()), split)
        for user_id in split_users:
            self.samples.extend(self._build_user_windows(user_id, grouped[user_id]))

        if not self.samples:
            raise FileNotFoundError(
                "WISDM starter adapter did not produce any windows. "
                "Check file format, split, and window parameters."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SensorSample:
        item = self.samples[idx]
        return SensorSample(
            values=item.values,
            timestamps=item.timestamps,
            masks=item.masks,
            label=item.label,
            metadata=item.metadata,
        )

    def modalities(self) -> List[str]:
        return self.modalities_list

    def channels_per_modality(self) -> Dict[str, int]:
        return {modality: 3 for modality in self.modalities_list}

    def _load_rows(self, raw_file: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with raw_file.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for record in reader:
                if len(record) < 6:
                    continue
                try:
                    user = int(record[0].strip())
                    label = record[1].strip()
                    timestamp = float(record[2].strip())
                    x = float(record[3].strip())
                    y = float(record[4].strip())
                    z = float(record[5].strip().rstrip(";"))
                except ValueError:
                    continue
                if label not in self.LABEL_TO_ID:
                    continue
                rows.append(
                    {
                        "user": user,
                        "label": label,
                        "timestamp": timestamp,
                        "xyz": np.array([x, y, z], dtype=np.float32),
                    }
                )
        return rows

    def _group_by_user(self, rows: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(int(row["user"]), []).append(row)
        return grouped

    def _select_split(self, users: List[int], split: str) -> List[int]:
        if len(users) < 3:
            return users
        n = len(users)
        train_end = max(1, int(0.7 * n))
        val_end = max(train_end + 1, int(0.85 * n))
        if split == "train":
            return users[:train_end]
        if split == "val":
            return users[train_end:val_end]
        if split == "test":
            return users[val_end:]
        return users

    def _build_user_windows(self, user_id: int, rows: List[Dict[str, Any]]) -> List[WISDMWindow]:
        rows = sorted(rows, key=lambda r: float(r["timestamp"]))
        windows: List[WISDMWindow] = []
        if len(rows) < self.window_size:
            return windows

        for start in range(0, len(rows) - self.window_size + 1, self.stride):
            chunk = rows[start : start + self.window_size]
            labels = [self.LABEL_TO_ID[r["label"]] for r in chunk]
            label = int(np.bincount(labels).argmax())
            xyz = np.stack([r["xyz"] for r in chunk], axis=0)  # (T, C)
            ts = np.array([r["timestamp"] for r in chunk], dtype=np.float32)
            values = {"accelerometer": xyz.T.astype(np.float32)}  # (C, T)
            timestamps = {"accelerometer": ts}
            masks = {"accelerometer": np.ones((self.window_size,), dtype=np.float32)}
            windows.append(
                WISDMWindow(
                    values=values,
                    timestamps=timestamps,
                    masks=masks,
                    label=label,
                    metadata={"user": user_id, "start_index": start, "window_size": self.window_size},
                )
            )
        return windows
