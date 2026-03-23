from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn


@dataclass
class AlignmentOutput:
    aligned_features: Dict[str, torch.Tensor]
    alignment_metadata: Dict[str, object]


class AsyncAlignmentModule(nn.Module):
    """Simple timestamp-aware baseline alignment.

    This is a deliberately small and honest v1 module. It does *not* claim to
    solve asynchronous wearable fusion in a research-grade way.

    Current behavior:
    - choose a reference modality/timeline,
    - for each reference timestamp, pick the nearest valid source sample from
      each modality,
    - zero-fill positions that cannot be aligned,
    - expose masks and selected source indices as metadata.

    The goal is to keep timestamps and missingness explicit while providing a
    real alignment step that is still CPU-friendly.
    """

    def __init__(self, strategy: str = "nearest-neighbor", reference_mode: str = "densest") -> None:
        super().__init__()
        self.strategy = strategy
        self.reference_mode = reference_mode

    def forward(self, inputs: Dict[str, Dict[str, torch.Tensor]]) -> AlignmentOutput:
        values = inputs["values"]
        timestamps = inputs["timestamps"]
        masks = inputs["masks"]

        modalities = list(values.keys())
        reference_modality = self._select_reference_modality(modalities=modalities, masks=masks)
        reference_timestamps = timestamps[reference_modality]
        reference_mask = masks[reference_modality]

        aligned_features: Dict[str, torch.Tensor] = {}
        aligned_masks: Dict[str, torch.Tensor] = {}
        source_indices: Dict[str, torch.Tensor] = {}
        valid_ratio: Dict[str, torch.Tensor] = {}

        for modality in modalities:
            aligned, aligned_mask, gathered_indices = self._align_single_modality(
                source_values=values[modality],
                source_timestamps=timestamps[modality],
                source_mask=masks[modality],
                reference_timestamps=reference_timestamps,
                reference_mask=reference_mask,
            )
            aligned_features[modality] = aligned
            aligned_masks[modality] = aligned_mask
            source_indices[modality] = gathered_indices
            valid_ratio[modality] = aligned_mask.mean(dim=-1)

        return AlignmentOutput(
            aligned_features=aligned_features,
            alignment_metadata={
                "strategy": self.strategy,
                "reference_mode": self.reference_mode,
                "reference_modality": reference_modality,
                "reference_timestamps": reference_timestamps,
                "reference_mask": reference_mask,
                "aligned_masks": aligned_masks,
                "source_indices": source_indices,
                "valid_ratio": valid_ratio,
            },
        )

    def _select_reference_modality(self, modalities: List[str], masks: Dict[str, torch.Tensor]) -> str:
        if self.reference_mode == "first":
            return modalities[0]
        if self.reference_mode == "densest":
            return max(modalities, key=lambda modality: float(masks[modality].float().mean().item()))
        if self.reference_mode in modalities:
            return self.reference_mode
        raise ValueError(
            f"Unsupported reference_mode: {self.reference_mode}. "
            f"Expected one of: densest, first, or a modality name from {modalities}."
        )

    def _align_single_modality(
        self,
        source_values: torch.Tensor,
        source_timestamps: torch.Tensor,
        source_mask: torch.Tensor,
        reference_timestamps: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, channels, ref_steps = source_values.size(0), source_values.size(1), reference_timestamps.size(1)
        aligned = source_values.new_zeros((batch_size, channels, ref_steps))
        aligned_mask = source_mask.new_zeros((batch_size, ref_steps))
        source_indices = torch.full((batch_size, ref_steps), -1, dtype=torch.long, device=source_values.device)

        for batch_idx in range(batch_size):
            valid_source = source_mask[batch_idx] > 0
            valid_reference = reference_mask[batch_idx] > 0
            if not torch.any(valid_source) or not torch.any(valid_reference):
                continue

            valid_positions = torch.nonzero(valid_source, as_tuple=False).squeeze(-1)
            valid_times = source_timestamps[batch_idx, valid_source]
            valid_values = source_values[batch_idx, :, valid_source]
            target_positions = torch.nonzero(valid_reference, as_tuple=False).squeeze(-1)
            target_times = reference_timestamps[batch_idx, valid_reference]

            distances = torch.abs(target_times.unsqueeze(1) - valid_times.unsqueeze(0))
            nearest = torch.argmin(distances, dim=1)
            picked_positions = valid_positions[nearest]

            aligned[batch_idx, :, target_positions] = valid_values[:, nearest]
            aligned_mask[batch_idx, target_positions] = 1.0
            source_indices[batch_idx, target_positions] = picked_positions

        return aligned, aligned_mask, source_indices
