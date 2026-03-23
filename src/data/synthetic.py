from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from .dataset_base import BaseWearableDataset, SensorSample


class SyntheticWearableDataset(BaseWearableDataset, Dataset):
    """Tiny synthetic dataset for smoke-testing the training pipeline.

    This is intentionally *not* a real HAR benchmark. It only creates structured,
    class-conditioned signals so the repo can be executed end-to-end on CPU.
    """

    DEFAULT_CHANNELS = {
        "accelerometer": 3,
        "gyroscope": 3,
    }

    SPLIT_SIZES = {
        "train": 96,
        "val": 32,
        "test": 32,
    }

    def __init__(
        self,
        split: str = "train",
        modalities: List[str] | None = None,
        window_size: int = 128,
        num_classes: int = 4,
        seed: int = 7,
        timestamp_jitter_ms: float = 5.0,
        packet_loss_prob: float = 0.05,
    ) -> None:
        BaseWearableDataset.__init__(self, split=split)
        Dataset.__init__(self)
        self._modalities = modalities or ["accelerometer", "gyroscope"]
        self.window_size = window_size
        self.num_classes = num_classes
        self.seed = seed
        self.timestamp_jitter_ms = timestamp_jitter_ms
        self.packet_loss_prob = packet_loss_prob
        self.size = self.SPLIT_SIZES.get(split, 32)

    def __len__(self) -> int:
        return self.size

    def modalities(self) -> List[str]:
        return self._modalities

    def channels_per_modality(self) -> Dict[str, int]:
        return {modality: self.DEFAULT_CHANNELS.get(modality, 3) for modality in self._modalities}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int | Dict[str, str | int]]:
        generator = torch.Generator().manual_seed(self.seed + idx + self._split_offset())
        label = idx % self.num_classes

        values: Dict[str, torch.Tensor] = {}
        timestamps: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}

        base_time = torch.linspace(0.0, 1.0, self.window_size, dtype=torch.float32)
        base_freq = float(label + 1)

        for modality_idx, modality in enumerate(self._modalities):
            channels = self.DEFAULT_CHANNELS.get(modality, 3)
            modality_scale = 1.0 + 0.2 * modality_idx
            signal_channels = []
            for channel_idx in range(channels):
                phase = 0.35 * channel_idx + 0.15 * modality_idx
                signal = torch.sin(2 * torch.pi * base_freq * modality_scale * base_time + phase)
                signal = signal + 0.5 * torch.cos(torch.pi * (channel_idx + 1) * base_time)
                noise = 0.05 * torch.randn(self.window_size, generator=generator)
                signal_channels.append(signal + noise)
            values[modality] = torch.stack(signal_channels, dim=0)

            jitter = torch.randn(self.window_size, generator=generator) * (self.timestamp_jitter_ms / 1000.0)
            timestamps[modality] = torch.clamp(base_time + jitter, min=0.0)

            mask = (torch.rand(self.window_size, generator=generator) > self.packet_loss_prob).float()
            if mask.sum() == 0:
                mask[0] = 1.0
            masks[modality] = mask
            values[modality] = values[modality] * mask.unsqueeze(0)

        sample = SensorSample(
            values=values,
            timestamps=timestamps,
            masks=masks,
            label=label,
            metadata={"dataset": "synthetic", "split": self.split, "index": idx},
        )
        return asdict(sample)

    def _split_offset(self) -> int:
        return {"train": 0, "val": 10_000, "test": 20_000}.get(self.split, 30_000)


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def collate_sensor_batch(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor] | torch.Tensor | List[Dict]]:
    modalities = batch[0]["values"].keys()
    values = {
        modality: torch.stack([_to_tensor(item["values"][modality]) for item in batch], dim=0)
        for modality in modalities
    }
    timestamps = {
        modality: torch.stack([_to_tensor(item["timestamps"][modality]) for item in batch], dim=0)
        for modality in modalities
    }
    masks = {
        modality: torch.stack([_to_tensor(item["masks"][modality]) for item in batch], dim=0)
        for modality in modalities
    }
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    metadata = [item["metadata"] for item in batch]
    return {
        "values": values,
        "timestamps": timestamps,
        "masks": masks,
        "label": labels,
        "metadata": metadata,
    }


def create_synthetic_dataloaders(config: Dict) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    runtime_cfg = config.get("runtime", {})
    model_cfg = config.get("model", {})

    modalities = data_cfg.get("modalities", ["accelerometer", "gyroscope"])
    common_kwargs = {
        "modalities": modalities,
        "window_size": int(data_cfg.get("window_size", 128)),
        "num_classes": int(model_cfg.get("num_classes", 4)),
        "seed": int(data_cfg.get("seed", 7)),
        "timestamp_jitter_ms": float(data_cfg.get("timestamp_jitter_ms", 5)),
        "packet_loss_prob": float(data_cfg.get("packet_loss_prob", 0.05)),
    }

    train_ds = SyntheticWearableDataset(split="train", **common_kwargs)
    val_ds = SyntheticWearableDataset(split="val", **common_kwargs)
    test_ds = SyntheticWearableDataset(split="test", **common_kwargs)

    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(runtime_cfg.get("num_workers", 0))

    dataloaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_sensor_batch),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_sensor_batch),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_sensor_batch),
    }
    return dataloaders, train_ds.channels_per_modality()
