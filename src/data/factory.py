from __future__ import annotations

from typing import Dict, Tuple

from torch.utils.data import DataLoader

from .pamap2 import PAMAP2Dataset
from .synthetic import SyntheticWearableDataset, collate_sensor_batch, create_synthetic_dataloaders


class DatasetConfigurationError(RuntimeError):
    """Raised when a configured dataset cannot be constructed honestly."""


def create_pamap2_dataloaders(config: Dict) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    runtime_cfg = config.get("runtime", {})

    modalities = data_cfg.get("modalities", ["accelerometer", "gyroscope"])
    common_kwargs = {
        "root": data_cfg.get("root", "data/PAMAP2"),
        "window_size": int(data_cfg.get("window_size", 256)),
        "stride": int(data_cfg.get("stride", 128)),
        "modalities": modalities,
    }

    train_ds = PAMAP2Dataset(split="train", **common_kwargs)
    val_ds = PAMAP2Dataset(split="val", **common_kwargs)
    test_ds = PAMAP2Dataset(split="test", **common_kwargs)

    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(runtime_cfg.get("num_workers", 0))

    dataloaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_sensor_batch),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_sensor_batch),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_sensor_batch),
    }
    return dataloaders, train_ds.channels_per_modality()


def create_dataloaders(config: Dict) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    dataset_name = str(config.get("data", {}).get("dataset", "synthetic")).lower()
    if dataset_name == "synthetic":
        return create_synthetic_dataloaders(config)
    if dataset_name == "pamap2":
        return create_pamap2_dataloaders(config)
    raise DatasetConfigurationError(
        f"Unsupported dataset '{dataset_name}'. Expected one of: synthetic, pamap2."
    )
