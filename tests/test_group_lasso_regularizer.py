from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.test_calculation_specs import _model_and_groups
from torch_structracker.calculations import CalcType
from torch_structracker.canonical_units import canonicalize_groups
from torch_structracker.regularizers import RegularizerType
from torch_structracker.regularizers.group_lasso import GroupLasso
from torch_structracker.structure_tracker import StructureTracker
from torch_structracker.torch_pruning.pruner.function import (
    prune_batchnorm_out_channels,
    prune_conv_in_channels,
    prune_conv_out_channels,
    prune_linear_in_channels,
    prune_linear_out_channels,
)


class TensorCalculation(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value)

    def forward(self, *args) -> torch.Tensor:
        return self.value


class ParameterCalculation(nn.Module):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = nn.Parameter(value)

    def forward(self) -> torch.Tensor:
        return self.value


class UnitsToGroupCalculation(nn.Module):
    def __init__(self, group_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.group_sizes = tuple(int(size) for size in group_sizes)

    def forward(self, unit_values: torch.Tensor) -> torch.Tensor:
        values = []
        start = 0
        for size in self.group_sizes:
            stop = start + size
            values.append(unit_values[start:stop].sum())
            start = stop
        return torch.stack(values)


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


def test_group_lasso_direct_formula_and_l2_gradients() -> None:
    l2_norm_pr_unit = ParameterCalculation(torch.tensor([5.0, 7.0, 11.0, 13.0]))
    regularizer = GroupLasso(
        {
            CalcType.UNIT_ACTIVE_MASK: TensorCalculation(
                torch.tensor([1.0, 0.0, 1.0, 1.0])
            ),
            CalcType.UNITS_TO_GROUP: UnitsToGroupCalculation((3, 1)),
            CalcType.BASELINE_GROUP_SIZES: TensorCalculation(torch.tensor([3.0, 1.0])),
            CalcType.GROUP_CHANGE_EFFECT: TensorCalculation(torch.tensor([10.0, 4.0])),
            CalcType.GROUP_SIZES: TensorCalculation(torch.tensor([3, 1])),
            CalcType.L2_NORM_PR_UNIT: l2_norm_pr_unit,
        }
    )

    loss = regularizer()
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(-230.0))
    torch.testing.assert_close(
        l2_norm_pr_unit.value.grad,
        torch.tensor([-10.0, -10.0, -10.0, 0.0]),
    )


def test_group_lasso_requires_explicit_calculations() -> None:
    with pytest.raises(ValueError, match="missing required calculations"):
        GroupLasso({})


def test_group_lasso_linear_chain_exact_values_and_gradients() -> None:
    model, groups = _model_and_groups()
    tracker = StructureTracker(model, groups=groups)

    expected_l2 = torch.tensor(
        [
            5.0,
            0.0,
            torch.sqrt(torch.tensor(13.0)) + 6.0,
            torch.sqrt(torch.tensor(52.0)),
        ]
    )
    _assert_group_lasso_calculations(
        tracker,
        expected_unit_active_mask=torch.tensor([1.0, 0.0, 1.0, 1.0]),
        expected_active_pr_group=torch.tensor([2.0, 1.0]),
        expected_baseline_group_sizes=torch.tensor([3.0, 1.0]),
        expected_group_change_effect=torch.tensor([3.0, 3.0]),
        expected_group_sizes=torch.tensor([3, 1]),
        expected_l2_norm_pr_unit=expected_l2,
    )

    regularizer = tracker.create_regularizer(RegularizerType.GROUP_LASSO)
    loss = regularizer()
    loss.backward()

    expected_loss = -3.0 * (11.0 + torch.sqrt(torch.tensor(13.0)))
    torch.testing.assert_close(loss.detach(), expected_loss)
    assert torch.isfinite(model.fc1.weight.grad).all()
    assert torch.isfinite(model.fc2.weight.grad).all()
    torch.testing.assert_close(
        model.fc1.weight.grad,
        torch.tensor(
            [
                [-3.0, 0.0],
                [0.0, 0.0],
                [
                    -6.0 / torch.sqrt(torch.tensor(13.0)),
                    -9.0 / torch.sqrt(torch.tensor(13.0)),
                ],
            ]
        ),
    )
    torch.testing.assert_close(
        model.fc2.weight.grad,
        torch.tensor([[-3.0, 0.0, -3.0]]),
    )


class ConvBnChain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.bn1(self.conv1(x)))


def test_group_lasso_conv_bn_chain_exact_values_and_gradients() -> None:
    model = ConvBnChain()
    with torch.no_grad():
        model.conv1.weight.copy_(torch.tensor([[[[2.0]]], [[[0.0]]]]))
        model.bn1.weight.copy_(torch.tensor([4.0, 0.0]))
        model.conv2.weight.copy_(torch.tensor([[[[5.0]], [[0.0]]]]))

    groups = canonicalize_groups(
        (
            FakeGroup(
                _member(model.conv1, prune_conv_out_channels, (0, 1)),
                _member(model.bn1, prune_batchnorm_out_channels, (0, 1)),
                _member(model.conv2, prune_conv_in_channels, (0, 1)),
            ),
        )
    )
    tracker = StructureTracker(model, groups=groups)

    _assert_group_lasso_calculations(
        tracker,
        expected_unit_active_mask=torch.tensor([1.0, 0.0]),
        expected_active_pr_group=torch.tensor([1.0]),
        expected_baseline_group_sizes=torch.tensor([2.0]),
        expected_group_change_effect=torch.tensor([4.0]),
        expected_group_sizes=torch.tensor([2]),
        expected_l2_norm_pr_unit=torch.tensor([11.0, 0.0]),
    )

    loss = tracker.create_regularizer(RegularizerType.GROUP_LASSO)()
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(-44.0))
    assert torch.isfinite(model.conv1.weight.grad).all()
    assert torch.isfinite(model.bn1.weight.grad).all()
    assert torch.isfinite(model.conv2.weight.grad).all()
    torch.testing.assert_close(
        model.conv1.weight.grad,
        torch.tensor([[[[-4.0]]], [[[0.0]]]]),
    )
    torch.testing.assert_close(
        model.bn1.weight.grad,
        torch.tensor([-4.0, 0.0]),
    )
    torch.testing.assert_close(
        model.conv2.weight.grad,
        torch.tensor([[[[-4.0]], [[0.0]]]]),
    )


class TinyQKVProjectionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(4, 12, bias=False)
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qkv(x)[..., :4] + self.proj(x)


@dataclass(frozen=True)
class QKVGroupLassoCase:
    name: str
    prune_dim: bool | None
    prune_num_heads: bool
    zero_embed_indices: tuple[int, ...]
    expected_unit_active_mask: torch.Tensor
    expected_active_pr_group: torch.Tensor
    expected_baseline_group_sizes: torch.Tensor
    expected_group_change_effect: torch.Tensor
    expected_group_sizes: torch.Tensor
    expected_l2_norm_pr_unit: torch.Tensor
    expected_loss: torch.Tensor
    expected_qkv_grad_active_rows: tuple[int, ...]
    expected_proj_grad_active_cols: tuple[int, ...]
    expected_grad_value: float


QKV_GROUP_LASSO_CASES = (
    QKVGroupLassoCase(
        name="channel",
        prune_dim=False,
        prune_num_heads=False,
        zero_embed_indices=(1,),
        expected_unit_active_mask=torch.tensor([1.0, 0.0, 1.0, 1.0]),
        expected_active_pr_group=torch.tensor([3.0]),
        expected_baseline_group_sizes=torch.tensor([4.0]),
        expected_group_change_effect=torch.tensor([16.0]),
        expected_group_sizes=torch.tensor([4]),
        expected_l2_norm_pr_unit=torch.tensor([8.0, 0.0, 8.0, 8.0]),
        expected_loss=torch.tensor(-384.0),
        expected_qkv_grad_active_rows=(0, 2, 3, 4, 6, 7, 8, 10, 11),
        expected_proj_grad_active_cols=(0, 2, 3),
        expected_grad_value=-8.0,
    ),
    QKVGroupLassoCase(
        name="head",
        prune_dim=False,
        prune_num_heads=True,
        zero_embed_indices=(2, 3),
        expected_unit_active_mask=torch.tensor([1.0, 0.0]),
        expected_active_pr_group=torch.tensor([1.0]),
        expected_baseline_group_sizes=torch.tensor([2.0]),
        expected_group_change_effect=torch.tensor([32.0]),
        expected_group_sizes=torch.tensor([2]),
        expected_l2_norm_pr_unit=torch.tensor([16.0, 0.0]),
        expected_loss=torch.tensor(-512.0),
        expected_qkv_grad_active_rows=(0, 1, 4, 5, 8, 9),
        expected_proj_grad_active_cols=(0, 1),
        expected_grad_value=-16.0,
    ),
    QKVGroupLassoCase(
        name="head_dim",
        prune_dim=True,
        prune_num_heads=False,
        zero_embed_indices=(1, 3),
        expected_unit_active_mask=torch.tensor([1.0, 0.0]),
        expected_active_pr_group=torch.tensor([1.0]),
        expected_baseline_group_sizes=torch.tensor([2.0]),
        expected_group_change_effect=torch.tensor([32.0]),
        expected_group_sizes=torch.tensor([2]),
        expected_l2_norm_pr_unit=torch.tensor([16.0, 0.0]),
        expected_loss=torch.tensor(-512.0),
        expected_qkv_grad_active_rows=(0, 2, 4, 6, 8, 10),
        expected_proj_grad_active_cols=(0, 2),
        expected_grad_value=-16.0,
    ),
)


@pytest.mark.parametrize(
    "case",
    QKV_GROUP_LASSO_CASES,
    ids=[case.name for case in QKV_GROUP_LASSO_CASES],
)
def test_group_lasso_qkv_projection_exact_values_and_gradients(
    case: QKVGroupLassoCase,
) -> None:
    model = TinyQKVProjectionBlock()
    with torch.no_grad():
        model.qkv.weight.fill_(1.0)
        model.proj.weight.fill_(1.0)
        _zero_qkv_embed_indices(model.qkv, case.zero_embed_indices)
        model.proj.weight[:, case.zero_embed_indices] = 0.0

    groups = canonicalize_groups(
        (
            FakeGroup(
                _member(model.proj, prune_linear_in_channels, (0, 1, 2, 3)),
                _member(model.qkv, prune_linear_out_channels, tuple(range(12))),
            ),
        ),
        num_heads={model.qkv: 2},
        prune_dim=case.prune_dim,
        prune_num_heads=case.prune_num_heads,
    )
    assert tuple(group.length for group in groups) == tuple(
        int(size) for size in case.expected_group_sizes
    )
    tracker = StructureTracker(
        model,
        groups=groups,
        num_heads={model.qkv: 2},
        prune_dim=case.prune_dim,
        prune_num_heads=case.prune_num_heads,
    )

    _assert_group_lasso_calculations(
        tracker,
        expected_unit_active_mask=case.expected_unit_active_mask,
        expected_active_pr_group=case.expected_active_pr_group,
        expected_baseline_group_sizes=case.expected_baseline_group_sizes,
        expected_group_change_effect=case.expected_group_change_effect,
        expected_group_sizes=case.expected_group_sizes,
        expected_l2_norm_pr_unit=case.expected_l2_norm_pr_unit,
    )

    loss = tracker.create_regularizer(RegularizerType.GROUP_LASSO)()
    loss.backward()

    torch.testing.assert_close(loss.detach(), case.expected_loss)
    assert torch.isfinite(model.qkv.weight.grad).all()
    assert torch.isfinite(model.proj.weight.grad).all()
    torch.testing.assert_close(
        model.qkv.weight.grad,
        _expected_qkv_grad(
            active_rows=case.expected_qkv_grad_active_rows,
            value=case.expected_grad_value,
        ),
    )
    torch.testing.assert_close(
        model.proj.weight.grad,
        _expected_proj_grad(
            active_cols=case.expected_proj_grad_active_cols,
            value=case.expected_grad_value,
        ),
    )


def _assert_group_lasso_calculations(
    tracker: StructureTracker,
    *,
    expected_unit_active_mask: torch.Tensor,
    expected_active_pr_group: torch.Tensor,
    expected_baseline_group_sizes: torch.Tensor,
    expected_group_change_effect: torch.Tensor,
    expected_group_sizes: torch.Tensor,
    expected_l2_norm_pr_unit: torch.Tensor,
) -> None:
    unit_active_mask = tracker.get_calculation(CalcType.UNIT_ACTIVE_MASK)()
    active_pr_group = tracker.get_calculation(CalcType.UNITS_TO_GROUP)(
        unit_active_mask
    )
    baseline_group_sizes = tracker.get_calculation(CalcType.BASELINE_GROUP_SIZES)()
    group_change_effect = tracker.get_calculation(CalcType.GROUP_CHANGE_EFFECT)()
    group_sizes = tracker.get_calculation(CalcType.GROUP_SIZES)()
    l2_norm_pr_unit = tracker.get_calculation(CalcType.L2_NORM_PR_UNIT)()

    torch.testing.assert_close(unit_active_mask, expected_unit_active_mask)
    torch.testing.assert_close(active_pr_group, expected_active_pr_group)
    torch.testing.assert_close(baseline_group_sizes, expected_baseline_group_sizes)
    torch.testing.assert_close(group_change_effect, expected_group_change_effect)
    torch.testing.assert_close(group_sizes, expected_group_sizes)
    torch.testing.assert_close(l2_norm_pr_unit, expected_l2_norm_pr_unit)


def _zero_qkv_embed_indices(module: nn.Linear, embed_indices: tuple[int, ...]) -> None:
    embed_dim = module.out_features // 3
    row_indices = []
    for base in (0, embed_dim, 2 * embed_dim):
        row_indices.extend(base + index for index in embed_indices)
    module.weight[row_indices, :] = 0.0


def _expected_qkv_grad(
    *,
    active_rows: tuple[int, ...],
    value: float,
) -> torch.Tensor:
    grad = torch.zeros(12, 4)
    grad[list(active_rows), :] = value
    return grad


def _expected_proj_grad(
    *,
    active_cols: tuple[int, ...],
    value: float,
) -> torch.Tensor:
    grad = torch.zeros(4, 4)
    grad[:, list(active_cols)] = value
    return grad
