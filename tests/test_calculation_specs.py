from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import torch_weighttracker.calculations.calculations as calc_impl
from torch_weighttracker.calculations import CalcType
from torch_weighttracker.calculations.cached_calc import CachedCalculation
from torch_weighttracker.canonical_units import canonicalize_groups
from torch_weighttracker.torch_pruning.pruner.function import (
    prune_batchnorm_out_channels,
    prune_linear_in_channels,
    prune_linear_out_channels,
)
from torch_weighttracker.weight_tracker import WeightTracker


class TinyLinearChain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=False)
        self.fc2 = nn.Linear(3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class TinyQKVProjectionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(4, 12, bias=False)
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qkv(x)[..., :4] + self.proj(x)


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


def _model_and_groups():
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
    return model, canonicalize_groups((hidden_group, output_group))


def test_weight_tracker_builds_public_calculations_from_specs() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)

    calculations = tracker.ensure_calculations(
        (
            CalcType.ACTIVE_UNITS,
            CalcType.UNIT_ACTIVE_MASK,
            CalcType.UNITS_TO_GROUP,
            CalcType.INIT_UNIT_PR_GROUP_COUNT,
            CalcType.GROUP_CHANGE_EFFECT,
            CalcType.GROUP_SIZES,
            CalcType.L2_NORM_PR_UNIT,
            CalcType.STRUCTURED_UNIT_SUM,
        )
    )

    assert tracker.get_calculation(CalcType.UNIT_ACTIVE_MASK) is calculations[
        CalcType.UNIT_ACTIVE_MASK
    ]
    assert isinstance(
        calculations[CalcType.INIT_UNIT_PR_GROUP_COUNT],
        CachedCalculation,
    )
    assert isinstance(calculations[CalcType.GROUP_CHANGE_EFFECT], CachedCalculation)
    assert isinstance(calculations[CalcType.GROUP_SIZES], CachedCalculation)
    assert not isinstance(calculations[CalcType.ACTIVE_UNITS], CachedCalculation)

    torch.testing.assert_close(
        calculations[CalcType.ACTIVE_UNITS](),
        torch.tensor([2.0, 0.0, 2.0, 1.0]),
    )
    torch.testing.assert_close(
        calculations[CalcType.UNIT_ACTIVE_MASK](),
        torch.tensor([1.0, 0.0, 1.0, 1.0]),
    )
    torch.testing.assert_close(
        calculations[CalcType.UNITS_TO_GROUP](torch.tensor([1.0, 2.0, 3.0, 4.0])),
        torch.tensor([6.0, 4.0]),
    )
    torch.testing.assert_close(
        calculations[CalcType.INIT_UNIT_PR_GROUP_COUNT](),
        torch.tensor([3.0, 1.0]),
    )
    torch.testing.assert_close(
        calculations[CalcType.GROUP_CHANGE_EFFECT](),
        torch.tensor([3.0, 3.0]),
    )
    torch.testing.assert_close(
        calculations[CalcType.GROUP_SIZES](),
        torch.tensor([3, 1], dtype=torch.long),
    )
    torch.testing.assert_close(
        calculations[CalcType.L2_NORM_PR_UNIT](),
        torch.tensor(
            [
                torch.sqrt(torch.tensor(17.0)),
                0.0,
                7.0,
                torch.sqrt(torch.tensor(52.0)),
            ]
        ),
    )
    torch.testing.assert_close(
        calculations[CalcType.STRUCTURED_UNIT_SUM](),
        torch.tensor([5.0, 0.0, 11.0, 10.0]),
    )


def test_module_axis_and_bitrates_share_weighted_module_order() -> None:
    model, groups = _model_and_groups()
    model.fc1.activation_bitrate = 8
    model.fc1.weight_bitrate = 2
    model.fc2.bitrate = 4
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

    calculations = tracker.ensure_calculations(
        (
            CalcType.UNIT_ACTIVE_MASK,
            CalcType.UNITS_TO_MODULE_AXIS,
            CalcType.ACTIVE_MACS_PR_MODULE,
            CalcType.BITRATE_PR_MODULE,
        )
    )

    unit_active_mask = calculations[CalcType.UNIT_ACTIVE_MASK]()
    module_axis = calculations[CalcType.UNITS_TO_MODULE_AXIS](unit_active_mask)
    active_macs = calculations[CalcType.ACTIVE_MACS_PR_MODULE]()
    bitrates = calculations[CalcType.BITRATE_PR_MODULE]()
    module_bops = active_macs * bitrates.view(-1, 2).prod(dim=1)

    torch.testing.assert_close(module_axis, torch.tensor([0.0, 2.0, 2.0, 1.0]))
    torch.testing.assert_close(active_macs, torch.tensor([4.0, 2.0]))
    torch.testing.assert_close(bitrates, torch.tensor([8.0, 2.0, 4.0, 4.0]))
    torch.testing.assert_close(module_bops, torch.tensor([64.0, 32.0]))


def test_structured_bops_supports_qkv_projection_head_dim_group() -> None:
    model = TinyQKVProjectionBlock()
    model.qkv.bitrate = 1
    model.proj.bitrate = 1
    with torch.no_grad():
        model.qkv.weight.fill_(1)
        model.proj.weight.fill_(1)
        model.qkv.weight[(1, 3, 5, 7, 9, 11), :] = 0
        model.proj.weight[:, (1, 3)] = 0

    qkv_proj_group = FakeGroup(
        _member(model.proj, prune_linear_in_channels, (0, 1, 2, 3)),
        _member(model.qkv, prune_linear_out_channels, tuple(range(12))),
    )
    groups = canonicalize_groups(
        (qkv_proj_group,),
        num_heads={model.qkv: 2},
        prune_dim=True,
    )
    tracker = WeightTracker(
        model,
        example_inputs=torch.randn(1, 4),
        groups=groups,
        num_heads={model.qkv: 2},
        prune_dim=True,
    )

    calculations = tracker.ensure_calculations(
        (
            CalcType.UNIT_ACTIVE_MASK,
            CalcType.UNIT_DELTA_TO_MODULE_AXIS,
            CalcType.BASELINE_MODULE_AXES,
            CalcType.ACTIVE_MACS_PR_MODULE,
            CalcType.BITRATE_PR_MODULE,
        )
    )

    active_units = calculations[CalcType.UNIT_ACTIVE_MASK]()
    axis_delta = calculations[CalcType.UNIT_DELTA_TO_MODULE_AXIS](active_units)
    baseline_axes = calculations[CalcType.BASELINE_MODULE_AXES]()
    active_macs = calculations[CalcType.ACTIVE_MACS_PR_MODULE]()
    bitrates = calculations[CalcType.BITRATE_PR_MODULE]()
    module_bops = active_macs * bitrates.view(-1, 2).prod(dim=1)

    torch.testing.assert_close(active_units, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(axis_delta, torch.tensor([0.0, -6.0, -2.0, 0.0]))
    torch.testing.assert_close(
        baseline_axes,
        torch.tensor([4.0, 12.0, 4.0, 4.0]),
    )
    torch.testing.assert_close(active_macs, torch.tensor([24.0, 8.0]))
    torch.testing.assert_close(module_bops, torch.tensor([24.0, 8.0]))

    structured_bops = tracker.create_tracker("structured_bops", log_total_bops=True)
    metrics = structured_bops.track()
    assert metrics["structured_bops_pr_module"].keys() == {"qkv", "proj"}
    torch.testing.assert_close(
        metrics["structured_bops_pr_module"]["qkv"],
        torch.tensor(24.0),
    )
    torch.testing.assert_close(
        metrics["structured_bops_pr_module"]["proj"],
        torch.tensor(8.0),
    )
    torch.testing.assert_close(
        metrics["structured_bops_compression_rate_pr_module"]["qkv"],
        torch.tensor(1.0 - 24.0 / (48.0 * 32.0 * 32.0)),
    )
    torch.testing.assert_close(
        metrics["structured_bops_compression_rate_pr_module"]["proj"],
        torch.tensor(1.0 - 8.0 / (16.0 * 32.0 * 32.0)),
    )
    torch.testing.assert_close(
        metrics["structured_bops_compression"],
        torch.tensor(1.0 - 32.0 / (64.0 * 32.0 * 32.0)),
    )
    torch.testing.assert_close(
        torch.stack(tuple(metrics["structured_bops_pr_module"].values())),
        torch.tensor([24.0, 8.0]),
    )
    torch.testing.assert_close(metrics["structured_bops"], torch.tensor(32.0))


def test_feature_only_module_axis_plan_uses_output_cost_axis() -> None:
    model = nn.Sequential(nn.BatchNorm2d(4))
    batchnorm = model[0]
    with torch.no_grad():
        batchnorm.weight.copy_(torch.tensor([1.0, 0.0, 1.0, 0.0]))

    groups = canonicalize_groups(
        (
            FakeGroup(
                _member(batchnorm, prune_batchnorm_out_channels, (0, 1, 2, 3)),
            ),
        )
    )
    tracker = WeightTracker(model, groups=groups)
    calculations = tracker.ensure_calculations(
        (
            CalcType.UNIT_ACTIVE_MASK,
            CalcType.UNITS_TO_MODULE_AXIS,
            CalcType.UNIT_DELTA_TO_MODULE_AXIS,
            CalcType.BASELINE_MODULE_AXES,
        )
    )

    active_units = calculations[CalcType.UNIT_ACTIVE_MASK]()
    module_axis = calculations[CalcType.UNITS_TO_MODULE_AXIS](active_units)
    axis_delta = calculations[CalcType.UNIT_DELTA_TO_MODULE_AXIS](active_units)
    baseline_axes = calculations[CalcType.BASELINE_MODULE_AXES]()

    torch.testing.assert_close(active_units, torch.tensor([1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(baseline_axes, torch.tensor([-1.0, 4.0]))
    torch.testing.assert_close(module_axis, torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(axis_delta, torch.tensor([0.0, -2.0]))


def test_active_macs_uses_module_axis_cost_indices_calculation() -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, example_inputs=torch.randn(1, 2), groups=groups)

    active_macs = tracker.get_calculation(CalcType.ACTIVE_MACS_PR_MODULE)
    cost_indices = tracker.get_calculation(CalcType.MODULE_AXIS_COST_INDICES)

    assert active_macs.calc(CalcType.MODULE_AXIS_COST_INDICES) is cost_indices
    assert not hasattr(active_macs, "cost_axis_indices")
    assert not hasattr(active_macs, "cost_axis_module_indices")


def test_missing_groups_fail_for_group_required_calculations() -> None:
    with pytest.raises(ValueError, match="requires dependency groups"):
        WeightTracker(TinyLinearChain()).get_calculation(CalcType.UNIT_ACTIVE_MASK)


def test_circular_calculation_dependencies_fail_clearly(monkeypatch) -> None:
    model, groups = _model_and_groups()
    tracker = WeightTracker(model, groups=groups)

    monkeypatch.setitem(
        calc_impl.CALCULATION_SPECS,
        CalcType.ACTIVE_UNITS,
        calc_impl.CalculationSpec(
            calculation_type=CalcType.ACTIVE_UNITS,
            required_calculations=(CalcType.UNIT_ACTIVE_MASK,),
            create=lambda ctx, deps: nn.Identity(),
        ),
    )
    monkeypatch.setitem(
        calc_impl.CALCULATION_SPECS,
        CalcType.UNIT_ACTIVE_MASK,
        calc_impl.CalculationSpec(
            calculation_type=CalcType.UNIT_ACTIVE_MASK,
            required_calculations=(CalcType.ACTIVE_UNITS,),
            create=lambda ctx, deps: nn.Identity(),
        ),
    )

    with pytest.raises(ValueError, match="Circular calculation dependency"):
        tracker.get_calculation(CalcType.ACTIVE_UNITS)
