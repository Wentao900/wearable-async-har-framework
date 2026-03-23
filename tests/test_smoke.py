import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.data import create_dataloaders
from src.training.trainer import Trainer
from src.utils.config import load_config


def test_repo_has_docs():
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "method-framework.md").exists()
    assert (root / "docs" / "experiment-plan.md").exists()


def test_synthetic_training_smoke():
    root = Path(__file__).resolve().parents[1]
    config = load_config(str(root / "configs" / "base.yaml"))
    config["training"]["epochs"] = 1
    trainer = Trainer(config)
    results = trainer.train()
    assert results["status"] == "completed"
    assert results["dataset"] == "synthetic"
    assert "test" in results


def _write_dummy_pamap2_subject(protocol_dir, subject_idx, steps=40):
    rows = []
    for step in range(steps):
        row = np.zeros(13, dtype=float)
        row[0] = step * 0.01
        row[1] = 1 + (step % 3)
        row[4:7] = np.array([step, step + 1, step + 2], dtype=float)
        row[10:13] = np.array([step + 3, step + 4, step + 5], dtype=float)
        rows.append(row)
    np.savetxt(protocol_dir / f"subject{subject_idx}.dat", np.vstack(rows), fmt="%.6f")



def test_pamap2_dataloader_smoke(tmp_path):
    protocol_dir = tmp_path / "PAMAP2" / "Protocol"
    protocol_dir.mkdir(parents=True)

    for subject_idx in range(101, 104):
        _write_dummy_pamap2_subject(protocol_dir, subject_idx)

    config = {
        "data": {
            "dataset": "pamap2",
            "root": str(tmp_path / "PAMAP2"),
            "window_size": 16,
            "stride": 8,
            "modalities": ["accelerometer", "gyroscope"],
        },
        "training": {"batch_size": 4},
        "runtime": {"num_workers": 0},
    }
    dataloaders, channels = create_dataloaders(config)
    batch = next(iter(dataloaders["train"]))

    assert channels == {"accelerometer": 3, "gyroscope": 3}
    assert batch["values"]["accelerometer"].shape[1:] == (3, 16)
    assert batch["masks"]["accelerometer"].shape[1:] == (16,)
    assert batch["label"].ndim == 1



def test_pamap2_explicit_subject_splits_override_automatic_split(tmp_path):
    protocol_dir = tmp_path / "PAMAP2" / "Protocol"
    protocol_dir.mkdir(parents=True)

    for subject_idx in range(101, 106):
        _write_dummy_pamap2_subject(protocol_dir, subject_idx)

    config = {
        "data": {
            "dataset": "pamap2",
            "root": str(tmp_path / "PAMAP2"),
            "window_size": 16,
            "stride": 8,
            "modalities": ["accelerometer", "gyroscope"],
            "train_subjects": [103, 105],
            "val_subjects": [102],
            "test_subjects": [101, 104],
        },
        "training": {"batch_size": 4},
        "runtime": {"num_workers": 0},
    }
    dataloaders, _ = create_dataloaders(config)

    train_subjects = set(dataloaders["train"].dataset.samples[i].metadata["subject_id"] for i in range(len(dataloaders["train"].dataset)))
    val_subjects = set(dataloaders["val"].dataset.samples[i].metadata["subject_id"] for i in range(len(dataloaders["val"].dataset)))
    test_subjects = set(dataloaders["test"].dataset.samples[i].metadata["subject_id"] for i in range(len(dataloaders["test"].dataset)))

    assert train_subjects == {103, 105}
    assert val_subjects == {102}
    assert test_subjects == {101, 104}



def test_pamap2_automatic_split_still_works_without_explicit_subject_lists(tmp_path):
    protocol_dir = tmp_path / "PAMAP2" / "Protocol"
    protocol_dir.mkdir(parents=True)

    for subject_idx in range(101, 106):
        _write_dummy_pamap2_subject(protocol_dir, subject_idx)

    config = {
        "data": {
            "dataset": "pamap2",
            "root": str(tmp_path / "PAMAP2"),
            "window_size": 16,
            "stride": 8,
            "modalities": ["accelerometer", "gyroscope"],
        },
        "training": {"batch_size": 4},
        "runtime": {"num_workers": 0},
    }
    dataloaders, _ = create_dataloaders(config)

    train_subjects = set(dataloaders["train"].dataset.samples[i].metadata["subject_id"] for i in range(len(dataloaders["train"].dataset)))
    val_subjects = set(dataloaders["val"].dataset.samples[i].metadata["subject_id"] for i in range(len(dataloaders["val"].dataset)))
    test_subjects = set(dataloaders["test"].dataset.samples[i].metadata["subject_id"] for i in range(len(dataloaders["test"].dataset)))

    assert train_subjects == {101, 102, 103}
    assert val_subjects == {104}
    assert test_subjects == {105}


def test_train_script_reports_missing_pamap2_data(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "missing-pamap2.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset: pamap2",
                "  root: missing/PAMAP2",
                "  window_size: 16",
                "  stride: 8",
                "  modalities:",
                "    - accelerometer",
                "    - gyroscope",
                "model:",
                "  hidden_dim: 16",
                "  num_classes: 4",
                "training:",
                "  batch_size: 2",
                "  epochs: 1",
                "runtime:",
                "  device: cpu",
                "  num_workers: 0",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "train.py"), "--config", str(config_path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "PAMAP2 dataset files were not found" in result.stderr


def test_train_script_reports_missing_wisdm_data(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "missing-wisdm.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset: wisdm",
                "  root: missing/WISDM",
                "  window_size: 16",
                "  stride: 8",
                "  modalities:",
                "    - accelerometer",
                "model:",
                "  hidden_dim: 16",
                "  num_classes: 6",
                "training:",
                "  batch_size: 2",
                "  epochs: 1",
                "runtime:",
                "  device: cpu",
                "  num_workers: 0",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "train.py"), "--config", str(config_path)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "WISDM dataset file was not found" in result.stderr


def test_train_script_writes_runtime_info_and_honors_output_override(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "custom-run"

    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "train.py"),
            "--config",
            str(root / "configs" / "base.yaml"),
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    runtime_info = json.loads((output_dir / "runtime_info.json").read_text(encoding="utf-8"))
    results = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))

    assert runtime_info["status"] == "completed"
    assert Path(runtime_info["output_dir"]) == output_dir.resolve()
    assert runtime_info["config_summary"]["dataset"] == "synthetic"
    assert runtime_info["torch"]["version"] is not None
    assert results["status"] == "completed"
