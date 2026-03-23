from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from .dataset_base import BaseWearableDataset, SensorSample


@dataclass
class PAMAP2Window:
    values: Dict[str, np.ndarray]
    timestamps: Dict[str, np.ndarray]
    masks: Dict[str, np.ndarray]
    label: int
    metadata: Dict[str, Any]


class PAMAP2Dataset(BaseWearableDataset):
    """Minimal PAMAP2 adapter scaffold.

    This adapter is intentionally conservative:
    - it documents expected file layout,
    - loads basic tabular data if present,
    - exposes modality-specific arrays,
    - but does NOT claim a finalized preprocessing protocol.

    Expected layout:
        data/PAMAP2/
          Protocol/
            subject101.dat
            subject102.dat
            ...

    Notes:
    - PAMAP2 has many columns and several missing values.
    - This scaffold only extracts a small, explicit subset for a first baseline.
    - You should verify column mappings against the official dataset description before using it in a paper.
    - Subject-wise splits in this scaffold are configurable from YAML, but they are starter controls,
      not a claimed paper-standard evaluation protocol.
    """

    DEFAULT_MODALITY_COLUMNS = {
        "accelerometer": [4, 5, 6],
        "gyroscope": [10, 11, 12],
    }

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        window_size: int = 256,
        stride: int = 128,
        modalities: Optional[List[str]] = None,
        train_subjects: Optional[Iterable[int | str]] = None,
        val_subjects: Optional[Iterable[int | str]] = None,
        test_subjects: Optional[Iterable[int | str]] = None,
    ) -> None:
        super().__init__(split=split)
        self.root = Path(root)
        self.window_size = window_size
        self.stride = stride
        self.modalities_list = modalities or ["accelerometer", "gyroscope"]
        self.samples: List[PAMAP2Window] = []
        self.subject_splits = {
            "train": self._normalize_subject_ids(train_subjects),
            "val": self._normalize_subject_ids(val_subjects),
            "test": self._normalize_subject_ids(test_subjects),
        }

        protocol_dir = self.root / "Protocol"
        if not protocol_dir.exists():
            raise FileNotFoundError(
                f"PAMAP2 Protocol directory not found: {protocol_dir}. "
                "Place the dataset under data/PAMAP2/Protocol/."
            )

        files = sorted(protocol_dir.glob("subject*.dat"))
        if not files:
            raise FileNotFoundError(
                f"No PAMAP2 subject files found under {protocol_dir}."
            )

        split_files = self._select_split(files, split)
        for file_path in split_files:
            self.samples.extend(self._load_subject(file_path))

        if not self.samples:
            raise ValueError(
                "PAMAP2 scaffold loaded zero windows. "
                "Check that the raw files exist, the configured subject split is not empty, "
                "and that window_size/stride are compatible with your data. "
                "This adapter is still a conservative starter pipeline, not finalized preprocessing."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        sample = SensorSample(
            values=item.values,
            timestamps=item.timestamps,
            masks=item.masks,
            label=item.label,
            metadata=item.metadata,
        )
        return asdict(sample)

    def modalities(self) -> List[str]:
        return self.modalities_list

    def channels_per_modality(self) -> Dict[str, int]:
        return {
            modality: len(self.DEFAULT_MODALITY_COLUMNS[modality])
            for modality in self.modalities_list
        }

    def _normalize_subject_ids(
        self, subjects: Optional[Iterable[int | str]]
    ) -> Optional[List[int]]:
        if subjects is None:
            return None
        normalized: List[int] = []
        for subject in subjects:
            subject_str = str(subject).strip()
            if not subject_str:
                continue
            if subject_str.startswith("subject"):
                subject_str = subject_str[len("subject") :]
            if subject_str.endswith(".dat"):
                subject_str = subject_str[: -len(".dat")]
            normalized.append(int(subject_str))
        return normalized

    def _subject_id_from_file(self, file_path: Path) -> int:
        name = file_path.stem
        if not name.startswith("subject"):
            raise ValueError(f"Unexpected PAMAP2 filename: {file_path.name}")
        return int(name[len("subject") :])

    def _uses_explicit_subject_splits(self) -> bool:
        return any(subjects is not None for subjects in self.subject_splits.values())

    def _select_split(self, files: List[Path], split: str) -> List[Path]:
        if self._uses_explicit_subject_splits():
            requested_subjects = self.subject_splits.get(split)
            if requested_subjects is None:
                return []

            file_map = {self._subject_id_from_file(path): path for path in files}
            missing_subjects = [subject for subject in requested_subjects if subject not in file_map]
            if missing_subjects:
                raise ValueError(
                    f"Configured PAMAP2 {split}_subjects include missing files for subjects: {missing_subjects}. "
                    f"Available subjects: {sorted(file_map)}"
                )
            return [file_map[subject] for subject in requested_subjects]

        if len(files) < 3:
            return files
        n = len(files)
        train_end = max(1, int(0.7 * n))
        val_end = max(train_end + 1, int(0.85 * n))
        if split == "train":
            return files[:train_end]
        if split == "val":
            return files[train_end:val_end]
        if split == "test":
            return files[val_end:]
        return files

    def _load_subject(self, file_path: Path) -> List[PAMAP2Window]:
        data = np.genfromtxt(file_path)
        if data.ndim != 2:
            return []

        timestamps = data[:, 0]
        labels = data[:, 1].astype(int)

        valid = labels > 0
        data = data[valid]
        timestamps = timestamps[valid]
        labels = labels[valid]
        if len(data) < self.window_size:
            return []

        subject_id = self._subject_id_from_file(file_path)
        windows: List[PAMAP2Window] = []
        for start in range(0, len(data) - self.window_size + 1, self.stride):
            end = start + self.window_size
            chunk = data[start:end]
            chunk_labels = labels[start:end]
            chunk_ts = timestamps[start:end]
            label = int(np.bincount(chunk_labels).argmax())

            values: Dict[str, np.ndarray] = {}
            ts_map: Dict[str, np.ndarray] = {}
            masks: Dict[str, np.ndarray] = {}
            for modality in self.modalities_list:
                cols = self.DEFAULT_MODALITY_COLUMNS.get(modality)
                if cols is None:
                    raise ValueError(f"Unsupported modality for PAMAP2 scaffold: {modality}")
                feats = chunk[:, cols].astype(np.float32).T
                mask = (~np.isnan(feats).all(axis=0)).astype(np.float32)
                feats = np.nan_to_num(feats, nan=0.0)
                values[modality] = feats
                ts_map[modality] = chunk_ts.astype(np.float32)
                masks[modality] = mask

            windows.append(
                PAMAP2Window(
                    values=values,
                    timestamps=ts_map,
                    masks=masks,
                    label=label,
                    metadata={
                        "dataset": "pamap2",
                        "subject_id": subject_id,
                        "subject_file": file_path.name,
                        "split": self.split,
                        "start_index": start,
                        "end_index": end,
                    },
                )
            )
        return windows
