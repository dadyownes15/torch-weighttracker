import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from torch_weighttracker import WeightTracker
from torch_weighttracker.calculations import CalcType
from torch_weighttracker.trackers import TrackerType, tracker_class_for_type
from torch_weighttracker.trackers.nvidia_2_4_sparsity import Nvidia24Sparsity


class FakeQuantZeroSmall(nn.Module):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.where(weight.abs() < 0.5, torch.zeros_like(weight), weight)


class TinyTwoLayerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 2, bias=False)
        self.fc2 = nn.Linear(4, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x) + self.fc2(x)


class TinyAttentionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=4, num_heads=2, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        return y


def _linear_with_weight(weight: torch.Tensor) -> nn.Sequential:
    model = nn.Sequential(nn.Linear(weight.shape[1], weight.shape[0], bias=False))
    with torch.no_grad():
        model[0].weight.copy_(weight)
    return model


def _track(model: nn.Module, **kwargs):
    return (
        WeightTracker(model)
        .create_tracker(TrackerType.NVIDIA_2_4_SPARSITY, **kwargs)
        .track()
    )


def test_nvidia_2_4_calculation_counts_linear_exact_two_zero_blocks() -> None:
    model = _linear_with_weight(
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 2.0],
                [3.0, 0.0, 4.0, 0.0],
            ]
        )
    )

    result = WeightTracker(model).get_calculation(CalcType.BLOCK_2_4_SPARSITY)()

    torch.testing.assert_close(result, torch.tensor([[2.0, 2.0, 2.0, 0.0]]))


def test_nvidia_2_4_tracker_distinguishes_eligible_from_strict_blocks() -> None:
    model = _linear_with_weight(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))

    metrics = _track(model)

    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/strict_block_fraction"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/nvidia_eligible_block_fraction"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/strict_layers"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/nvidia_eligible_layers"],
        torch.tensor(1.0),
    )


def test_nvidia_2_4_tracker_counts_invalid_blocks() -> None:
    model = _linear_with_weight(torch.tensor([[0.0, 1.0, 2.0, 3.0]]))

    metrics = _track(model)

    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/strict_block_fraction"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/nvidia_eligible_block_fraction"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/total_layers"],
        torch.tensor(1.0),
    )


def test_nvidia_2_4_tails_are_reported_and_make_layer_non_strict() -> None:
    model = _linear_with_weight(torch.tensor([[0.0, 0.0, 1.0, 2.0, 0.0, 0.0]]))

    metrics = _track(model)

    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/strict_block_fraction"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/nvidia_eligible_block_fraction"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/strict_layers"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/nvidia_eligible_layers"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/tail_elements"],
        torch.tensor(2.0),
    )


def test_nvidia_2_4_conv2d_groups_along_input_channel_axis() -> None:
    model = nn.Sequential(nn.Conv2d(4, 2, kernel_size=2, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[0].weight[0, 0:2, :, :] = 0.0
        model[0].weight[1, :, :, :] = 0.0

    result = WeightTracker(model).get_calculation(CalcType.BLOCK_2_4_SPARSITY)()

    torch.testing.assert_close(result, torch.tensor([[4.0, 8.0, 8.0, 0.0]]))


def test_nvidia_2_4_tracker_is_registered_and_accepts_string_name() -> None:
    model = _linear_with_weight(torch.tensor([[0.0, 0.0, 1.0, 2.0]]))
    tracker = WeightTracker(model)

    created = tracker.create_tracker("nvidia_2_4_sparsity")

    assert CalcType.BLOCK_2_4_SPARSITY.value == "2_4_block_sparsity"
    assert TrackerType.NVIDIA_2_4_SPARSITY.value == "nvidia_2_4_sparsity"
    assert tracker_class_for_type("nvidia_2_4_sparsity") is Nvidia24Sparsity
    assert created.required_calculations == (CalcType.BLOCK_2_4_SPARSITY,)
    torch.testing.assert_close(
        created.track()["nvidia_2_4_sparsity/strict_block_fraction"],
        torch.tensor(1.0),
    )


def test_nvidia_2_4_include_and_ignore_filter_supported_layers() -> None:
    model = TinyTwoLayerModel()
    with torch.no_grad():
        model.fc1.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, 2.0],
                    [3.0, 0.0, 4.0, 0.0],
                ]
            )
        )
        model.fc2.weight.copy_(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    tracker = WeightTracker(model)

    included = tracker.create_tracker(
        TrackerType.NVIDIA_2_4_SPARSITY,
        include=[model.fc1],
        log_layerwise_stats=True,
    ).track()
    ignored = tracker.create_tracker(
        TrackerType.NVIDIA_2_4_SPARSITY,
        ignore=[model.fc2],
        log_layerwise_stats=True,
    ).track()

    for metrics in (included, ignored):
        torch.testing.assert_close(
            metrics["nvidia_2_4_sparsity/strict_block_fraction"],
            torch.tensor(1.0),
        )
        torch.testing.assert_close(
            metrics["nvidia_2_4_sparsity/strict_layers"],
            torch.tensor(1.0),
        )
        assert "nvidia_2_4_sparsity/layers/fc1/strict_block_fraction" in metrics
        assert "nvidia_2_4_sparsity/layers/fc2/strict_block_fraction" not in metrics


def test_nvidia_2_4_layerwise_metrics_are_flat_and_opt_in() -> None:
    model = TinyTwoLayerModel()
    with torch.no_grad():
        model.fc1.weight.copy_(
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, 2.0],
                    [3.0, 0.0, 4.0, 0.0],
                ]
            )
        )
        model.fc2.weight.copy_(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    tracker = WeightTracker(model)

    default_metrics = tracker.create_tracker(TrackerType.NVIDIA_2_4_SPARSITY).track()
    layerwise = tracker.create_tracker(
        TrackerType.NVIDIA_2_4_SPARSITY,
        log_layerwise_stats=True,
    ).track()

    assert set(default_metrics) == {
        "nvidia_2_4_sparsity/strict_block_fraction",
        "nvidia_2_4_sparsity/nvidia_eligible_block_fraction",
        "nvidia_2_4_sparsity/strict_layers",
        "nvidia_2_4_sparsity/nvidia_eligible_layers",
        "nvidia_2_4_sparsity/total_layers",
        "nvidia_2_4_sparsity/tail_elements",
    }
    torch.testing.assert_close(
        layerwise["nvidia_2_4_sparsity/layers/fc1/strict_block_fraction"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        layerwise["nvidia_2_4_sparsity/layers/fc2/strict_block_fraction"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        layerwise["nvidia_2_4_sparsity/layers/fc2/nvidia_eligible_block_fraction"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        layerwise["nvidia_2_4_sparsity/layers/fc2/is_strict_layer"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        layerwise["nvidia_2_4_sparsity/layers/fc2/is_nvidia_eligible_layer"],
        torch.tensor(1.0),
    )


def test_nvidia_2_4_reads_effective_parametrized_weight() -> None:
    model = _linear_with_weight(torch.tensor([[0.1, -0.2, 1.0, -1.0]]))
    parametrize.register_parametrization(model[0], "weight", FakeQuantZeroSmall())

    metrics = _track(model)

    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/strict_block_fraction"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/strict_layers"],
        torch.tensor(1.0),
    )


def test_nvidia_2_4_excludes_unsupported_weighted_modules() -> None:
    model = nn.Sequential(nn.BatchNorm1d(4), nn.Linear(4, 1, bias=False))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[0.0, 0.0, 1.0, 2.0]]))

    metrics = _track(model, log_layerwise_stats=True)

    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/total_layers"],
        torch.tensor(1.0),
    )
    assert "nvidia_2_4_sparsity/layers/0/strict_block_fraction" not in metrics
    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/layers/1/strict_block_fraction"],
        torch.tensor(1.0),
    )


def test_nvidia_2_4_counts_multihead_attention_projection_weights() -> None:
    model = TinyAttentionModel()
    with torch.no_grad():
        model.attn.in_proj_weight.fill_(1.0)
        model.attn.in_proj_weight[:6, :] = 0.0

    metrics = _track(
        model,
        include=[model.attn],
        ignore=[model.attn.out_proj],
    )

    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/strict_block_fraction"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/nvidia_eligible_block_fraction"],
        torch.tensor(0.5),
    )
    torch.testing.assert_close(
        metrics["nvidia_2_4_sparsity/total_layers"],
        torch.tensor(1.0),
    )
