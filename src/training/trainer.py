from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from src.data import create_dataloaders
from src.models import WearableBaselineModel
from src.training.evaluator import Evaluator


class Trainer:
    """Minimal training loop for the runnable wearable HAR scaffold."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.device = self._resolve_device(config)
        self.dataloaders, channels_per_modality = create_dataloaders(config)
        model_cfg = config.get("model", {})
        self.model = WearableBaselineModel(
            channels_per_modality=channels_per_modality,
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_classes=int(model_cfg.get("num_classes", 4)),
            encoder_name=model_cfg.get("encoder", "tcn"),
            fusion_mode=model_cfg.get("fusion", "gated"),
            alignment_strategy=model_cfg.get("alignment_strategy", "nearest-neighbor"),
            reference_timeline_strategy=model_cfg.get("reference_timeline_strategy", "densest"),
        ).to(self.device)
        training_cfg = config.get("training", {})
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(training_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
        )
        self.evaluator = Evaluator(device=self.device)
        self.output_dir = Path(config.get("logging", {}).get("output_dir", "outputs/default"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict[str, Any]:
        training_cfg = self.config.get("training", {})
        epochs = int(training_cfg.get("epochs", 1))
        history = []
        best_state = self._initialize_best_tracking(training_cfg)
        early_stopping_cfg = training_cfg.get("early_stopping", {})
        patience_counter = 0
        stopped_early = False

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_examples = 0

            for batch in self.dataloaders["train"]:
                batch = self.evaluator._move_batch_to_device(batch)
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                labels = batch["label"]
                loss = self.criterion(outputs["logits"], labels)
                loss.backward()
                self.optimizer.step()

                predictions = outputs["logits"].argmax(dim=-1)
                total_loss += loss.item() * labels.size(0)
                total_correct += (predictions == labels).sum().item()
                total_examples += labels.size(0)

            train_metrics = {
                "loss": total_loss / max(total_examples, 1),
                "accuracy": total_correct / max(total_examples, 1),
            }
            val_metrics = self.evaluator.evaluate(self.model, self.dataloaders["val"])
            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}

            if best_state["enabled"]:
                improved, current_value = self._maybe_update_best(epoch, val_metrics, training_cfg, best_state)
                record["best_so_far"] = {
                    "monitor": best_state["monitor"],
                    "value": best_state["best_value"],
                    "epoch": best_state["best_epoch"],
                    "improved": improved,
                }
                if self._early_stopping_enabled(early_stopping_cfg):
                    if improved:
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    record["early_stopping"] = {
                        "enabled": True,
                        "patience": int(early_stopping_cfg.get("patience", 0)),
                        "wait": patience_counter,
                    }
                    if patience_counter > int(early_stopping_cfg.get("patience", 0)):
                        stopped_early = True
                        history.append(record)
                        break
            history.append(record)

        if best_state["enabled"] and best_state["checkpoint_path"] and best_state["checkpoint_path"].exists():
            checkpoint = torch.load(best_state["checkpoint_path"], map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        test_metrics = self.evaluator.evaluate(self.model, self.dataloaders["test"])
        results = {
            "status": "completed",
            "dataset": self.config.get("data", {}).get("dataset", "synthetic"),
            "device": str(self.device),
            "history": history,
            "epochs_completed": len(history),
            "epochs_requested": epochs,
            "stopped_early": stopped_early,
            "test": test_metrics,
        }
        if best_state["enabled"] and best_state["best_epoch"] is not None:
            results["best"] = {
                "monitor": best_state["monitor"],
                "mode": best_state["mode"],
                "epoch": best_state["best_epoch"],
                "value": best_state["best_value"],
                "checkpoint_path": str(best_state["checkpoint_path"]) if best_state["checkpoint_path"] else None,
            }
        return results

    def _resolve_device(self, config: Dict[str, Any]) -> torch.device:
        requested = str(config.get("runtime", {}).get("device", "cpu")).lower()
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _initialize_best_tracking(self, training_cfg: Dict[str, Any]) -> Dict[str, Any]:
        checkpoint_cfg = training_cfg.get("checkpoint", {})
        early_stopping_cfg = training_cfg.get("early_stopping", {})
        enabled = bool(checkpoint_cfg.get("save_best", False) or self._early_stopping_enabled(early_stopping_cfg))
        monitor = str(early_stopping_cfg.get("monitor") or checkpoint_cfg.get("monitor") or "val_accuracy")
        mode = str(early_stopping_cfg.get("mode") or checkpoint_cfg.get("mode") or self._infer_mode_from_monitor(monitor)).lower()
        filename = str(checkpoint_cfg.get("filename", "best_checkpoint.pt"))
        checkpoint_path = self.output_dir / filename if checkpoint_cfg.get("save_best", False) else None
        return {
            "enabled": enabled,
            "monitor": monitor,
            "mode": mode,
            "best_epoch": None,
            "best_value": None,
            "checkpoint_path": checkpoint_path,
        }

    def _early_stopping_enabled(self, early_stopping_cfg: Dict[str, Any]) -> bool:
        return bool(early_stopping_cfg.get("enabled", False))

    def _infer_mode_from_monitor(self, monitor: str) -> str:
        return "min" if "loss" in monitor.lower() else "max"

    def _extract_monitored_value(self, monitor: str, val_metrics: Dict[str, float]) -> float:
        normalized = monitor.lower().replace("val_", "")
        if normalized not in val_metrics:
            raise ValueError(
                f"Unsupported monitor '{monitor}'. Supported validation metrics: {sorted(val_metrics)}"
            )
        return float(val_metrics[normalized])

    def _is_improvement(self, current: float, best: float | None, mode: str, min_delta: float) -> bool:
        if best is None:
            return True
        if mode == "min":
            return current < (best - min_delta)
        return current > (best + min_delta)

    def _maybe_update_best(
        self,
        epoch: int,
        val_metrics: Dict[str, float],
        training_cfg: Dict[str, Any],
        best_state: Dict[str, Any],
    ) -> Tuple[bool, float]:
        current_value = self._extract_monitored_value(best_state["monitor"], val_metrics)
        min_delta = float(training_cfg.get("early_stopping", {}).get("min_delta", 0.0))
        improved = self._is_improvement(current_value, best_state["best_value"], best_state["mode"], min_delta)
        if improved:
            best_state["best_value"] = current_value
            best_state["best_epoch"] = epoch
            if best_state["checkpoint_path"] is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "monitor": best_state["monitor"],
                        "monitor_value": current_value,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.config,
                    },
                    best_state["checkpoint_path"],
                )
        return improved, current_value
