from __future__ import annotations

from typing import Any, Dict

import torch


class Evaluator:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def evaluate(self, model: Any, dataloader: Any) -> Dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)
                outputs = model(batch)
                labels = batch["label"]
                loss = criterion(outputs["logits"], labels)
                predictions = outputs["logits"].argmax(dim=-1)

                total_loss += loss.item() * labels.size(0)
                total_correct += (predictions == labels).sum().item()
                total_examples += labels.size(0)

        if total_examples == 0:
            return {"loss": 0.0, "accuracy": 0.0}
        return {
            "loss": total_loss / total_examples,
            "accuracy": total_correct / total_examples,
        }

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "values": {k: v.to(self.device) for k, v in batch["values"].items()},
            "timestamps": {k: v.to(self.device) for k, v in batch["timestamps"].items()},
            "masks": {k: v.to(self.device) for k, v in batch["masks"].items()},
            "label": batch["label"].to(self.device),
            "metadata": batch["metadata"],
        }
