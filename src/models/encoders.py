from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class BaseEncoder(nn.Module):
    def __init__(self, name: str = "base") -> None:
        super().__init__()
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConvEncoder(BaseEncoder):
    def __init__(self, input_channels: int, hidden_dim: int, name: str = "conv") -> None:
        super().__init__(name=name)
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class TCNEncoder(ConvEncoder):
    def __init__(self, input_channels: int, hidden_dim: int) -> None:
        super().__init__(input_channels=input_channels, hidden_dim=hidden_dim, name="tcn")


class CNNEncoder(ConvEncoder):
    def __init__(self, input_channels: int, hidden_dim: int) -> None:
        super().__init__(input_channels=input_channels, hidden_dim=hidden_dim, name="cnn")


class GRUEncoder(BaseEncoder):
    def __init__(self, input_channels: int, hidden_dim: int) -> None:
        super().__init__(name="gru")
        self.gru = nn.GRU(input_size=input_channels, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        _, hidden = self.gru(sequence)
        return hidden[-1]


def build_encoder(encoder_name: str, input_channels: int, hidden_dim: int) -> nn.Module:
    encoder_name = encoder_name.lower()
    if encoder_name == "gru":
        return GRUEncoder(input_channels=input_channels, hidden_dim=hidden_dim)
    if encoder_name == "cnn":
        return CNNEncoder(input_channels=input_channels, hidden_dim=hidden_dim)
    return TCNEncoder(input_channels=input_channels, hidden_dim=hidden_dim)


class ModalityEncoderBank(nn.Module):
    def __init__(self, channels_per_modality: Dict[str, int], encoder_name: str, hidden_dim: int) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                modality: build_encoder(encoder_name=encoder_name, input_channels=channels, hidden_dim=hidden_dim)
                for modality, channels in channels_per_modality.items()
            }
        )

    def forward(self, values: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {modality: encoder(values[modality]) for modality, encoder in self.encoders.items()}
