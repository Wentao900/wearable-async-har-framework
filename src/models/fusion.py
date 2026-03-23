from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class FusionBlock(nn.Module):
    """Minimal multimodal fusion block for runnable baselines."""

    def __init__(self, hidden_dim: int, num_modalities: int, mode: str = "mean") -> None:
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        if mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * num_modalities, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_modalities),
            )
        else:
            self.gate = None

    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        ordered = [modality_features[name] for name in sorted(modality_features.keys())]
        stacked = torch.stack(ordered, dim=1)
        if self.mode == "gated" and self.gate is not None:
            gate_logits = self.gate(torch.cat(ordered, dim=-1))
            weights = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)
            return (stacked * weights).sum(dim=1)
        return stacked.mean(dim=1)
