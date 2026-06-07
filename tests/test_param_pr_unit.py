from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from tests.test_calculation_specs import _tracker_from_groups
from torch_weighttracker.calculations import CalcType
from torch_weighttracker.canonical_units import canonicalize_groups
from torch_weighttracker.torch_pruning.pruner.function import (
    prune_batchnorm_out_channels,
    prune_conv_in_channels,
    prune_conv_out_channels,
    prune_linear_in_channels,
    prune_linear_out_channels,
)
from torch_weighttracker.weight_tracker import WeightTracker


class FakeGroup:
    def __init__(self, *items) -> None:
        self.items = list(items)


def _member(module: nn.Module, handler, indices: tuple[int, ...]):
    return SimpleNamespace(
        dep=SimpleNamespace(
            target=SimpleNamespace(module=module),
            handler=handler,
        ),
        root_idxs=indices,
        idxs=indices,
    )


class TinyLinearChain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=False)
        self.fc2 = nn.Linear(3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


def _linear_chain_tracker() -> WeightTracker:
    model = TinyLinearChain()
    with torch.no_grad():
        model.fc1.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                    [2.0, 3.0],
                ]
            )
        )
        model.fc2.weight.copy_(torch.tensor([[4.0, 0.0, 6.0]]))

    hidden_group = FakeGroup(
        _member(model.fc1, prune_linear_out_channels, (0, 1, 2)),
        _member(model.fc2, prune_linear_in_channels, (0, 1, 2)),
    )
    output_group = FakeGroup(
        _member(model.fc2, prune_linear_out_channels, (0,)),
    )
    groups = canonicalize_groups((hidden_group, output_group))
    return _tracker_from_groups(model, groups)


def test_param_pr_unit_linear_chain_uses_cross_group_removed_units() -> None:
    tracker = _linear_chain_tracker()
    baseline = tracker.get_calculation(CalcType.BASELINE_PARAM_PR_UNIT_PR_GROUP)
    change = tracker.get_calculation(CalcType.GROUP_UNIT_PARAM_CHANGE)
    param_pr_unit = tracker.get_calculation(CalcType.PARAM_PR_UNIT)

    l2_norm = tracker.get_calculation(CalcType.L2_NORM_PR_UNIT)()
    active_mask = l2_norm.gt(0).to(dtype=l2_norm.dtype)
    active_pr_group = tracker.get_calculation(CalcType.UNITS_TO_GROUP)(active_mask)
    removed_pr_group = (
        tracker.get_calculation(CalcType.INIT_UNIT_PR_GROUP_COUNT)()
        - active_pr_group
    )

    torch.testing.assert_close(baseline(), torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(removed_pr_group, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(change(removed_pr_group), torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(param_pr_unit(), torch.tensor([3.0, 0.0, 3.0, 2.0]))


class TinyQKVProjectionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(4, 12, bias=False)
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qkv(x)[..., :4] + self.proj(x)


def test_param_pr_unit_qkv_group_keeps_static_opposite_axis_cost() -> None:
    model = TinyQKVProjectionBlock()
    with torch.no_grad():
        model.qkv.weight.fill_(1.0)
        model.proj.weight.fill_(1.0)
        model.qkv.weight[(1, 5, 9), :] = 0.0
        model.proj.weight[:, 1] = 0.0

    groups = canonicalize_groups(
        (
            FakeGroup(
                _member(model.proj, prune_linear_in_channels, (0, 1, 2, 3)),
                _member(model.qkv, prune_linear_out_channels, tuple(range(12))),
            ),
        ),
        num_heads={model.qkv: 2},
        prune_dim=False,
        prune_num_heads=False,
    )
    tracker = _tracker_from_groups(
        model,
        groups,
        num_heads={model.qkv: 2},
        prune_dim=False,
        prune_num_heads=False,
    )

    baseline = tracker.get_calculation(CalcType.BASELINE_PARAM_PR_UNIT_PR_GROUP)
    change = tracker.get_calculation(CalcType.GROUP_UNIT_PARAM_CHANGE)
    param_pr_unit = tracker.get_calculation(CalcType.PARAM_PR_UNIT)

    torch.testing.assert_close(baseline(), torch.tensor([16.0]))
    torch.testing.assert_close(change(torch.tensor([1.0])), torch.tensor([0.0]))
    torch.testing.assert_close(
        param_pr_unit(),
        torch.tensor([16.0, 0.0, 16.0, 16.0]),
    )


class TinyConvChain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


def test_param_pr_unit_conv_chain_applies_kernel_area_change_edges() -> None:
    model = TinyConvChain()
    with torch.no_grad():
        model.conv1.weight.fill_(1.0)
        model.conv2.weight.fill_(1.0)
        model.conv1.weight[1] = 0.0
        model.conv2.weight[:, 1] = 0.0

    hidden_group = FakeGroup(
        _member(model.conv1, prune_conv_out_channels, (0, 1)),
        _member(model.conv2, prune_conv_in_channels, (0, 1)),
    )
    output_group = FakeGroup(
        _member(model.conv2, prune_conv_out_channels, (0, 1, 2, 3)),
    )
    groups = canonicalize_groups((hidden_group, output_group))
    tracker = _tracker_from_groups(model, groups)

    baseline = tracker.get_calculation(CalcType.BASELINE_PARAM_PR_UNIT_PR_GROUP)
    change = tracker.get_calculation(CalcType.GROUP_UNIT_PARAM_CHANGE)
    param_pr_unit = tracker.get_calculation(CalcType.PARAM_PR_UNIT)

    l2_norm = tracker.get_calculation(CalcType.L2_NORM_PR_UNIT)()
    active_mask = l2_norm.gt(0).to(dtype=l2_norm.dtype)
    active_pr_group = tracker.get_calculation(CalcType.UNITS_TO_GROUP)(active_mask)
    removed_pr_group = (
        tracker.get_calculation(CalcType.INIT_UNIT_PR_GROUP_COUNT)()
        - active_pr_group
    )

    torch.testing.assert_close(baseline(), torch.tensor([43.0, 8.0]))
    torch.testing.assert_close(removed_pr_group, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(change(removed_pr_group), torch.tensor([0.0, 4.0]))
    torch.testing.assert_close(
        param_pr_unit(),
        torch.tensor([43.0, 0.0, 4.0, 4.0, 4.0, 4.0]),
    )


def test_param_pr_unit_dedupes_duplicate_members_on_same_module_axis() -> None:
    model = TinyLinearChain()
    with torch.no_grad():
        model.fc1.weight.fill_(1.0)
        model.fc2.weight.fill_(1.0)

    hidden_group = FakeGroup(
        _member(model.fc1, prune_linear_out_channels, (0, 1, 2)),
        _member(model.fc2, prune_linear_in_channels, (0, 1, 2)),
        _member(model.fc2, prune_linear_in_channels, (0, 1, 2)),
    )
    output_group = FakeGroup(
        _member(model.fc2, prune_linear_out_channels, (0,)),
    )
    groups = canonicalize_groups((hidden_group, output_group))
    tracker = _tracker_from_groups(model, groups)

    baseline = tracker.get_calculation(CalcType.BASELINE_PARAM_PR_UNIT_PR_GROUP)
    change = tracker.get_calculation(CalcType.GROUP_UNIT_PARAM_CHANGE)

    torch.testing.assert_close(baseline(), torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(
        change(torch.tensor([0.0, 1.0])),
        torch.tensor([1.0, 0.0]),
    )


class TinyResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if downsample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        return out + identity


class TinyResidualStage(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 4, kernel_size=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(4)
        self.block1 = TinyResidualBlock(4, 4, downsample=False)
        self.block2 = TinyResidualBlock(4, 8, downsample=True)
        self.head = nn.Linear(8, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_bn(self.stem_conv(x))
        x = self.block1(x)
        x = self.block2(x)
        x = x.mean(dim=(2, 3))
        return self.head(x)


def _residual_stage_groups(model: TinyResidualStage):
    downsample_group = FakeGroup(
        _member(model.block2.downsample[0], prune_conv_out_channels, tuple(range(8))),
        _member(
            model.block2.downsample[1],
            prune_batchnorm_out_channels,
            tuple(range(8)),
        ),
        _member(model.block2.bn2, prune_batchnorm_out_channels, tuple(range(8))),
        _member(model.head, prune_linear_in_channels, tuple(range(8))),
        _member(model.block2.conv2, prune_conv_out_channels, tuple(range(8))),
    )
    stem_group = FakeGroup(
        _member(model.stem_conv, prune_conv_out_channels, tuple(range(4))),
        _member(model.stem_bn, prune_batchnorm_out_channels, tuple(range(4))),
        _member(model.block1.conv1, prune_conv_in_channels, tuple(range(4))),
        _member(model.block1.conv2, prune_conv_out_channels, tuple(range(4))),
        _member(model.block1.bn2, prune_batchnorm_out_channels, tuple(range(4))),
        _member(model.block2.downsample[0], prune_conv_in_channels, tuple(range(4))),
        _member(model.block2.conv1, prune_conv_in_channels, tuple(range(4))),
    )
    block1_group = FakeGroup(
        _member(model.block1.conv1, prune_conv_out_channels, tuple(range(4))),
        _member(model.block1.bn1, prune_batchnorm_out_channels, tuple(range(4))),
        _member(model.block1.conv2, prune_conv_in_channels, tuple(range(4))),
    )
    block2_group = FakeGroup(
        _member(model.block2.conv1, prune_conv_out_channels, tuple(range(8))),
        _member(model.block2.bn1, prune_batchnorm_out_channels, tuple(range(8))),
        _member(model.block2.conv2, prune_conv_in_channels, tuple(range(8))),
    )
    return canonicalize_groups(
        (downsample_group, stem_group, block1_group, block2_group)
    )


def _fill_residual_stage(model: TinyResidualStage) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.fill_(1.0)
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.weight.fill_(1.0)


def _zero_residual_group_units(model: TinyResidualStage) -> None:
    with torch.no_grad():
        # downsample group, unit 7
        model.block2.downsample[0].weight[7] = 0.0
        model.block2.downsample[1].weight[7] = 0.0
        model.block2.bn2.weight[7] = 0.0
        model.head.weight[:, 7] = 0.0
        model.block2.conv2.weight[7] = 0.0

        # stem group, unit 2
        model.stem_conv.weight[2] = 0.0
        model.stem_bn.weight[2] = 0.0
        model.block1.conv1.weight[:, 2] = 0.0
        model.block1.conv2.weight[2] = 0.0
        model.block1.bn2.weight[2] = 0.0
        model.block2.downsample[0].weight[:, 2] = 0.0
        model.block2.conv1.weight[:, 2] = 0.0

        # block1.conv1 group, unit 1
        model.block1.conv1.weight[1] = 0.0
        model.block1.bn1.weight[1] = 0.0
        model.block1.conv2.weight[:, 1] = 0.0

        # block2.conv1 group, units 0 and 5
        model.block2.conv1.weight[[0, 5], :, :, :] = 0.0
        model.block2.bn1.weight[[0, 5]] = 0.0
        model.block2.conv2.weight[:, [0, 5], :, :] = 0.0


def test_param_pr_unit_residual_downsample_stage_tracks_cross_group_costs() -> None:
    model = TinyResidualStage()
    _fill_residual_stage(model)
    _zero_residual_group_units(model)
    tracker = _tracker_from_groups(model, _residual_stage_groups(model))

    baseline = tracker.get_calculation(CalcType.BASELINE_PARAM_PR_UNIT_PR_GROUP)
    change = tracker.get_calculation(CalcType.GROUP_UNIT_PARAM_CHANGE)
    param_pr_unit = tracker.get_calculation(CalcType.PARAM_PR_UNIT)

    l2_norm = tracker.get_calculation(CalcType.L2_NORM_PR_UNIT)()
    active_mask = l2_norm.gt(0).to(dtype=l2_norm.dtype)
    active_pr_group = tracker.get_calculation(CalcType.UNITS_TO_GROUP)(active_mask)
    removed_pr_group = (
        tracker.get_calculation(CalcType.INIT_UNIT_PR_GROUP_COUNT)()
        - active_pr_group
    )

    torch.testing.assert_close(
        baseline(),
        torch.tensor([80.0, 181.0, 73.0, 109.0]),
    )
    torch.testing.assert_close(
        removed_pr_group,
        torch.tensor([1.0, 1.0, 1.0, 2.0]),
    )
    torch.testing.assert_close(
        change(removed_pr_group),
        torch.tensor([19.0, 37.0, 18.0, 18.0]),
    )
    torch.testing.assert_close(
        change(torch.tensor([0.0, 2.0, 3.0, 1.0])),
        torch.tensor([11.0, 63.0, 36.0, 18.0]),
    )

    expected = torch.tensor(
        [
            61.0,
            61.0,
            61.0,
            61.0,
            61.0,
            61.0,
            61.0,
            0.0,
            144.0,
            144.0,
            0.0,
            144.0,
            55.0,
            0.0,
            55.0,
            55.0,
            0.0,
            91.0,
            91.0,
            91.0,
            91.0,
            0.0,
            91.0,
            91.0,
        ]
    )
    torch.testing.assert_close(param_pr_unit(), expected)
