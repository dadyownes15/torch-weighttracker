from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import torch_structracker.calculations.calculations as calc_impl
from torch_structracker.calculations import CalcType
from torch_structracker.calculations.cached_calc import CachedCalculation
from torch_structracker.canonical_units import canonicalize_groups
from torch_structracker.structure_tracker import StructureTracker
from torch_structracker.torch_pruning.pruner.function import (
    prune_linear_in_channels,
    prune_linear_out_channels,
)


class TinyLinearChain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 3, bias=False)
        self.fc2 = nn.Linear(3, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


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


def test_structure_tracker_builds_public_calculations_from_specs() -> None:
    model, groups = _model_and_groups()
    tracker = StructureTracker(model, groups=groups)

    calculations = tracker.ensure_calculations(
        (
            CalcType.ACTIVE_UNITS,
            CalcType.UNIT_ACTIVE_MASK,
            CalcType.UNITS_TO_GROUP,
            CalcType.BASELINE_GROUP_SIZES,
            CalcType.GROUP_CHANGE_EFFECT,
            CalcType.GROUP_SIZES,
            CalcType.L2_NORM_PR_UNIT,
            CalcType.STRUCTURED_UNIT_SUM,
        )
    )

    assert tracker.get_calculation(CalcType.UNIT_ACTIVE_MASK) is calculations[
        CalcType.UNIT_ACTIVE_MASK
    ]
    assert isinstance(calculations[CalcType.BASELINE_GROUP_SIZES], CachedCalculation)
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
        calculations[CalcType.BASELINE_GROUP_SIZES](),
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
                5.0,
                0.0,
                torch.sqrt(torch.tensor(13.0)) + 6.0,
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
    tracker = StructureTracker(model, groups=groups)

    calculations = tracker.ensure_calculations(
        (
            CalcType.UNIT_ACTIVE_MASK,
            CalcType.UNITS_TO_MODULE_AXIS,
            CalcType.BITRATE_PR_MODULE,
        )
    )

    unit_active_mask = calculations[CalcType.UNIT_ACTIVE_MASK]()
    module_axis = calculations[CalcType.UNITS_TO_MODULE_AXIS](unit_active_mask)
    bitrates = calculations[CalcType.BITRATE_PR_MODULE]()
    module_bops = (module_axis * bitrates).view(-1, 2).prod(dim=1)

    torch.testing.assert_close(module_axis, torch.tensor([0.0, 2.0, 2.0, 1.0]))
    torch.testing.assert_close(bitrates, torch.tensor([8.0, 2.0, 4.0, 4.0]))
    torch.testing.assert_close(module_bops, torch.tensor([0.0, 32.0]))


def test_missing_groups_fail_for_group_required_calculations() -> None:
    with pytest.raises(ValueError, match="requires dependency groups"):
        StructureTracker(TinyLinearChain()).get_calculation(CalcType.UNIT_ACTIVE_MASK)


def test_circular_calculation_dependencies_fail_clearly(monkeypatch) -> None:
    model, groups = _model_and_groups()
    tracker = StructureTracker(model, groups=groups)

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
