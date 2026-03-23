from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


@dataclass
class AlignmentOutput:
    aligned_features: Dict[str, torch.Tensor]
    alignment_metadata: Dict[str, object]


class AsyncAlignmentModule(nn.Module):
    """Very small alignment scaffold.

    This does not solve irregular alignment in a research-grade way. It simply
    rescales values by the observed mask ratio so the rest of the pipeline keeps
    timestamps and missingness explicit without faking advanced alignment.
    """

    def __init__(self, strategy: str = "mask-aware-rescale") -> None:
        super().__init__()
        self.strategy = strategy

    def forward(self, inputs: Dict[str, Dict[str, torch.Tensor]]) -> AlignmentOutput:
        values = inputs["values"]
        masks = inputs["masks"]
        aligned = {}
        observed_ratio = {}
        for modality, tensor in values.items():
            mask = masks[modality].unsqueeze(1)
            ratio = masks[modality].mean(dim=-1, keepdim=True).clamp_min(1e-6)
            aligned[modality] = (tensor * mask) / ratio.unsqueeze(-1)
            observed_ratio[modality] = ratio.squeeze(-1)
        return AlignmentOutput(
            aligned_features=aligned,
            alignment_metadata={"strategy": self.strategy, "observed_ratio": observed_ratio},
        )
