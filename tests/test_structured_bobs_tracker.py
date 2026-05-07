import pytest
import torch
import torch.nn as nn

from torch_structracker import StructureTracker
from torch_structracker.torch_pruning.dependency import DependencyGraph
from torch_structracker.trackers import TrackerType


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=False)
        self.fc2 = nn.Linear(3, 2, bias=False)
        self.fc3 = nn.Linear(2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.fc2(self.fc1(x)))


class ConvOnly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DirectFusedMHA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(4, 2, batch_first=True, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.mha(x, x, x, need_weights=False)
        return output


class DirectSeparateMHA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            4,
            2,
            kdim=6,
            vdim=8,
            batch_first=True,
            bias=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        output, _ = self.mha(query, key, value, need_weights=False)
        return output


class TinyQKVBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(4, 12, bias=False)
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        return self.proj(q + k + v)


def dependency_groups(model, example_inputs, root_module_types):
    graph = DependencyGraph().build_dependency(
        model=model,
        example_inputs=example_inputs,
    )
    return list(graph.get_all_groups(root_module_types=root_module_types))


def mlp_with_sequential_weights() -> TinyMLP:
    model = TinyMLP()
    with torch.no_grad():
        model.fc1.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            )
        )
        model.fc2.weight.copy_(
            torch.tensor(
                [
                    [7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0],
                ]
            )
        )
        model.fc3.weight.copy_(torch.tensor([[13.0, 14.0]]))
    return model


def bobs_metrics(model, example_inputs, root_module_types, **kwargs):
    groups = dependency_groups(model, example_inputs, root_module_types)
    tracker = StructureTracker(model=model, groups=groups, **kwargs).create_tracker(
        TrackerType.BOBS_TRACKER
    )
    return tracker.track()


def test_bobs_tracker_counts_all_active_linear_structures():
    metrics = bobs_metrics(
        mlp_with_sequential_weights(),
        torch.ones(1, 2),
        [nn.Linear],
    )

    assert metrics["bobs"] == pytest.approx(14 * 32 * 32)
    assert metrics["baseline_bobs"] == pytest.approx(14 * 32 * 32)
    assert metrics["bobs_rel"] == pytest.approx(1.0)


def test_bobs_tracker_uses_global_activity_for_removable_units():
    model = mlp_with_sequential_weights()
    with torch.no_grad():
        model.fc1.weight[0].zero_()
        model.fc2.weight[:, 0].zero_()

    metrics = bobs_metrics(model, torch.ones(1, 2), [nn.Linear])

    assert metrics["bobs"] == pytest.approx(10 * 32 * 32)
    assert metrics["baseline_bobs"] == pytest.approx(14 * 32 * 32)
    assert metrics["bobs_rel"] == pytest.approx(10 / 14)


def test_bobs_tracker_keeps_unit_active_if_any_coupled_slice_is_nonzero():
    model = mlp_with_sequential_weights()
    with torch.no_grad():
        model.fc1.weight[0].zero_()

    metrics = bobs_metrics(model, torch.ones(1, 2), [nn.Linear])

    assert metrics["bobs"] == pytest.approx(14 * 32 * 32)


def test_bobs_tracker_reads_layer_bitrates():
    model = mlp_with_sequential_weights()
    model.fc1.weight_bitrate = 2
    model.fc1.activation_bitrate = 4
    model.fc2.bitrate = 1

    metrics = bobs_metrics(model, torch.ones(1, 2), [nn.Linear])

    assert metrics["bobs"] == pytest.approx(6 * 8 + 6 * 1 + 2 * 32 * 32)


def test_bobs_tracker_counts_conv2d_kernel_multiplier():
    model = ConvOnly()
    with torch.no_grad():
        model.conv.weight.fill_(1.0)

    metrics = bobs_metrics(model, torch.ones(1, 3, 8, 8), [nn.Conv2d])

    assert metrics["bobs"] == pytest.approx(3 * 4 * 3 * 3 * 32 * 32)


def test_bobs_tracker_counts_direct_mha_as_qkv_plus_out_projection():
    model = DirectFusedMHA().eval()
    model.mha.bitrate = 1
    model.mha.out_proj.bitrate = 1
    with torch.no_grad():
        model.mha.in_proj_weight.fill_(1.0)
        model.mha.out_proj.weight.fill_(1.0)

    metrics = bobs_metrics(
        model,
        torch.ones(2, 3, 4),
        [nn.MultiheadAttention],
    )

    assert metrics["bobs"] == pytest.approx(4 * 12 + 4 * 4)


def test_bobs_tracker_rejects_direct_mha_with_unequal_qkv_dims():
    model = DirectSeparateMHA().eval()
    groups = dependency_groups(
        model,
        (
            torch.ones(2, 3, 4),
            torch.ones(2, 3, 6),
            torch.ones(2, 3, 8),
        ),
        [nn.MultiheadAttention],
    )

    with pytest.raises(ValueError, match="separate-projection MHA"):
        StructureTracker(model=model, groups=groups).create_tracker(
            TrackerType.BOBS_TRACKER
        )


def test_bobs_tracker_counts_fused_qkv_head_view_with_projection():
    model = TinyQKVBlock().eval()
    model.qkv.bitrate = 1
    model.proj.bitrate = 1
    with torch.no_grad():
        model.qkv.weight.fill_(1.0)
        model.proj.weight.fill_(1.0)

    groups = dependency_groups(model, torch.ones(2, 3, 4), [nn.Linear])
    metrics = StructureTracker(
        model=model,
        groups=groups,
        num_heads={model.qkv: 2},
        prune_num_heads=True,
    ).create_tracker(TrackerType.BOBS_TRACKER).track()

    assert metrics["bobs"] == pytest.approx(4 * 12 + 4 * 4)
