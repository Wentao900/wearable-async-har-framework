import torch

from src.models.alignment import AsyncAlignmentModule
from src.models.baseline import WearableBaselineModel


def test_async_alignment_nearest_neighbor_uses_reference_timeline():
    module = AsyncAlignmentModule(reference_mode="first")
    values = {
        "accelerometer": torch.tensor([[[1.0, 2.0, 3.0]]]),
        "gyroscope": torch.tensor([[[10.0, 20.0, 30.0]]]),
    }
    timestamps = {
        "accelerometer": torch.tensor([[0.0, 1.0, 2.0]]),
        "gyroscope": torch.tensor([[0.1, 1.6, 2.2]]),
    }
    masks = {
        "accelerometer": torch.tensor([[1.0, 1.0, 1.0]]),
        "gyroscope": torch.tensor([[1.0, 1.0, 1.0]]),
    }

    output = module({"values": values, "timestamps": timestamps, "masks": masks})

    assert output.alignment_metadata["reference_modality"] == "accelerometer"
    assert torch.equal(output.aligned_features["accelerometer"], values["accelerometer"])
    assert torch.equal(output.aligned_features["gyroscope"], torch.tensor([[[10.0, 20.0, 30.0]]]))
    assert torch.equal(output.alignment_metadata["source_indices"]["gyroscope"], torch.tensor([[0, 1, 2]]))


def test_async_alignment_respects_missing_masks_and_zero_fills_invalid_reference_steps():
    module = AsyncAlignmentModule(reference_mode="first")
    values = {
        "accelerometer": torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),
        "gyroscope": torch.tensor([[[10.0, 20.0, 30.0, 40.0]]]),
    }
    timestamps = {
        "accelerometer": torch.tensor([[0.0, 0.5, 1.0, 1.5]]),
        "gyroscope": torch.tensor([[0.1, 0.6, 1.1, 1.6]]),
    }
    masks = {
        "accelerometer": torch.tensor([[1.0, 0.0, 1.0, 1.0]]),
        "gyroscope": torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
    }

    output = module({"values": values, "timestamps": timestamps, "masks": masks})

    expected = torch.tensor([[[10.0, 0.0, 30.0, 40.0]]])
    expected_mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    expected_indices = torch.tensor([[0, -1, 2, 2]])

    assert torch.equal(output.aligned_features["gyroscope"], expected)
    assert torch.equal(output.alignment_metadata["aligned_masks"]["gyroscope"], expected_mask)
    assert torch.equal(output.alignment_metadata["source_indices"]["gyroscope"], expected_indices)


def test_baseline_model_forward_exposes_alignment_metadata():
    model = WearableBaselineModel(
        channels_per_modality={"accelerometer": 3, "gyroscope": 3},
        hidden_dim=8,
        num_classes=4,
    )
    batch = {
        "values": {
            "accelerometer": torch.randn(2, 3, 16),
            "gyroscope": torch.randn(2, 3, 16),
        },
        "timestamps": {
            "accelerometer": torch.linspace(0.0, 1.0, 16).repeat(2, 1),
            "gyroscope": (torch.linspace(0.0, 1.0, 16) + 0.01).repeat(2, 1),
        },
        "masks": {
            "accelerometer": torch.ones(2, 16),
            "gyroscope": torch.ones(2, 16),
        },
    }

    output = model(batch)

    assert output["logits"].shape == (2, 4)
    assert output["alignment_metadata"]["reference_modality"] in {"accelerometer", "gyroscope"}
    assert "aligned_masks" in output["alignment_metadata"]
