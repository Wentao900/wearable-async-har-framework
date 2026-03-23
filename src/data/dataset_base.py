from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class SensorSample:
    values: Dict[str, Any]
    timestamps: Dict[str, Any]
    masks: Dict[str, Any]
    label: int
    metadata: Dict[str, Any]


class BaseWearableDataset:
    """Base class for asynchronous multimodal wearable datasets.

    A concrete dataset should expose per-modality values, timestamps, and masks.
    This keeps asynchrony explicit instead of hiding it during preprocessing.
    """

    def __init__(self, split: str = "train") -> None:
        self.split = split

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> SensorSample:
        raise NotImplementedError

    def modalities(self) -> List[str]:
        raise NotImplementedError
