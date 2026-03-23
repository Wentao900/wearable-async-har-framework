from __future__ import annotations

from typing import Any, Dict

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

    def train(self) -> Dict[str, Any]:
        training_cfg = self.config.get("training", {})
        epochs = int(training_cfg.get("epochs", 1))
        history = []

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
            history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        test_metrics = self.evaluator.evaluate(self.model, self.dataloaders["test"])
        return {
            "status": "completed",
            "dataset": self.config.get("data", {}).get("dataset", "synthetic"),
            "device": str(self.device),
            "history": history,
            "test": test_metrics,
        }

    def _resolve_device(self, config: Dict[str, Any]) -> torch.device:
        requested = str(config.get("runtime", {}).get("device", "cpu")).lower()
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
