from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .alignment import AsyncAlignmentModule
from .encoders import ModalityEncoderBank
from .fusion import FusionBlock


class WearableBaselineModel(nn.Module):
    def __init__(
        self,
        channels_per_modality: Dict[str, int],
        hidden_dim: int,
        num_classes: int,
        encoder_name: str = "tcn",
        fusion_mode: str = "gated",
    ) -> None:
        super().__init__()
        self.alignment = AsyncAlignmentModule()
        self.encoder_bank = ModalityEncoderBank(
            channels_per_modality=channels_per_modality,
            encoder_name=encoder_name,
            hidden_dim=hidden_dim,
        )
        self.fusion = FusionBlock(hidden_dim=hidden_dim, num_modalities=len(channels_per_modality), mode=fusion_mode)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor] | torch.Tensor]) -> Dict[str, torch.Tensor | Dict[str, object]]:
        alignment_output = self.alignment({"values": batch["values"], "masks": batch["masks"]})
        modality_features = self.encoder_bank(alignment_output.aligned_features)
        fused = self.fusion(modality_features)
        logits = self.classifier(fused)
        return {
            "logits": logits,
            "embeddings": fused,
            "alignment_metadata": alignment_output.alignment_metadata,
        }
