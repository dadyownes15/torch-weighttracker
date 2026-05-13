import pytest
import torch
import torch.nn as nn

from torch_structracker.calculations import (
    BaseCalculation,
    CalcType,
    create_calculation,
    create_pipeline_calculation,
)
from torch_structracker.calculations.active_params_pr_unit import ActiveParamsPrUnit
from torch_structracker.calculations.reduction_calc import MappedReductionCalculation
from torch_structracker.canonical_units import CanonicalUnitGroup, UnitKind
from torch_structracker.extractors.extractor import TensorSpec, ValueTensorRef
from torch_structracker.plans.mapping_plan import create_unit_to_group_acc
from torch_structracker.reductions.builder import (
    FullSelection,
    IndexSelection,
    ReductionMapping,
    ReductionPlanBuilder,
    ReductionRecord,
    SegmentSelection,
)
from torch_structracker.reductions.ops import IdentityTensorReduction
from torch_structracker.regularizers.group_lasso import GroupLasso
from torch_structracker.trackers.base import BaseTracker
from torch_structracker.trackers.structured_bops import StructuredBOPs


class TensorOp(nn.Module):
    def __init__(self, values: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values))

    @property
    def output_spec(self) -> TensorSpec:
        return TensorSpec(
            shape=self.values.shape,
            dtype=self.values.dtype,
            device=self.values.device,
        )

    @property
    def output_length(self) -> int:
        return self.values.numel()

    def forward(self) -> torch.Tensor:
        return self.values


class ParameterOp(nn.Module):
    def __init__(self, values: tuple[float, ...]) -> None:
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values))

    @property
    def output_spec(self) -> TensorSpec:
        return TensorSpec(
            shape=self.values.shape,
            dtype=self.values.dtype,
            device=self.values.device,
        )

    @property
    def output_length(self) -> int:
        return self.values.numel()

    def forward(self) -> torch.Tensor:
        return self.values


class StaticCalculation(BaseCalculation):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = value

    def forward(self) -> torch.Tensor:
        return self.value


class ParameterCalculation(BaseCalculation):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.value = nn.Parameter(value.clone())

    def forward(self) -> torch.Tensor:
        return self.value


def _record(op, source, target) -> ReductionRecord:
    return ReductionRecord(
        op=op,
        mapping=ReductionMapping(source=source, target=target),
    )


def _unit_groups() -> tuple[CanonicalUnitGroup, ...]:
    return (
        CanonicalUnitGroup(
            group_id=0,
            offset=0,
            length=2,
            unit_kind=UnitKind.CHANNEL,
            members=(),
            raw_group=object(),
        ),
        CanonicalUnitGroup(
            group_id=1,
            offset=2,
            length=2,
            unit_kind=UnitKind.CHANNEL,
            members=(),
            raw_group=object(),
        ),
    )


def _unit_to_group_plan(input_value: torch.Tensor):
    input_spec = TensorSpec(
        shape=input_value.shape,
        dtype=input_value.dtype,
        device=input_value.device,
    )
    return create_unit_to_group_acc(
        _unit_groups(),
        input_tensor_ref=ValueTensorRef(value=input_value, spec=input_spec),
        reduction_mapper=lambda _: IdentityTensorReduction(),
    )


def test_mapped_calculation_allocates_fresh_outputs_and_keeps_index_buffers() -> None:
    builder = ReductionPlanBuilder()
    builder.add(_record(TensorOp((1.0, 2.0)), FullSelection(), IndexSelection((0, 1))))
    calculation = MappedReductionCalculation(builder.finalize())

    index_ptrs = tuple(index.data_ptr() for index in calculation.destination_indices)
    first = calculation()
    second = calculation()

    assert not hasattr(calculation, "accumulator")
    assert first.data_ptr() != second.data_ptr()
    assert index_ptrs == tuple(index.data_ptr() for index in calculation.destination_indices)
    torch.testing.assert_close(first, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(second, torch.tensor([1.0, 2.0]))


def test_pipeline_calculation_allocates_fresh_outputs_and_keeps_index_buffers() -> None:
    input_value = torch.tensor([1.0, 2.0, 3.0, 4.0])
    calculation = create_pipeline_calculation(_unit_to_group_plan(input_value))
    buffer_ptrs = {
        name: buffer.data_ptr()
        for name, buffer in calculation.named_buffers()
        if name != "output_anchor"
    }

    first = calculation(input_value)
    second = calculation(input_value)

    assert not hasattr(calculation, "accumulator")
    assert first.data_ptr() != second.data_ptr()
    assert buffer_ptrs == {
        name: buffer.data_ptr()
        for name, buffer in calculation.named_buffers()
        if name != "output_anchor"
    }
    torch.testing.assert_close(first, torch.tensor([3.0, 7.0]))
    torch.testing.assert_close(second, torch.tensor([3.0, 7.0]))


def test_mapped_calculation_propagates_gradients_through_index_add() -> None:
    op = ParameterOp((1.0, 2.0, 3.0))
    builder = ReductionPlanBuilder()
    builder.add(_record(op, FullSelection(), IndexSelection((0, 0, 1))))
    calculation = MappedReductionCalculation(builder.finalize())

    loss = calculation().square().sum()
    loss.backward()

    torch.testing.assert_close(op.values.grad, torch.tensor([6.0, 6.0, 6.0]))


def test_pipeline_calculation_propagates_gradients_through_gather_index_add() -> None:
    input_value = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    calculation = create_pipeline_calculation(_unit_to_group_plan(input_value))

    loss = calculation(input_value).square().sum()
    loss.backward()

    torch.testing.assert_close(input_value.grad, torch.tensor([6.0, 6.0, 14.0, 14.0]))


def test_active_params_pr_unit_is_grad_compatible_when_input_is_grad_tensor() -> None:
    input_value = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    calculation = ActiveParamsPrUnit(
        unit_to_group_acc=create_pipeline_calculation(_unit_to_group_plan(input_value)),
        unit_active_mask=StaticCalculation(input_value),
        baseline_group_size=torch.tensor([2.0, 2.0]),
        group_change_effect=torch.tensor([10.0, 4.0]),
        group_lengths=torch.tensor([2, 2]),
    )

    calculation().sum().backward()

    torch.testing.assert_close(input_value.grad, torch.tensor([20.0, 20.0, 8.0, 8.0]))


class GradModeTracker(BaseTracker):
    def __init__(self) -> None:
        super().__init__()
        self.compute_grad_enabled = None
        self.metric_grad_enabled = None

    def compute(self):
        self.compute_grad_enabled = torch.is_grad_enabled()
        return torch.tensor(1.0)

    def toMetric(self, result):
        self.metric_grad_enabled = torch.is_grad_enabled()
        return {"value": result}


def test_tracker_track_runs_compute_and_metric_under_no_grad() -> None:
    tracker = GradModeTracker()

    with torch.enable_grad():
        metric = tracker.track()

    assert metric["value"].item() == 1.0
    assert tracker.compute_grad_enabled is False
    assert tracker.metric_grad_enabled is False


def test_create_calculation_constructs_l2_norm_pr_unit() -> None:
    builder = ReductionPlanBuilder()
    builder.add(_record(TensorOp((1.0, 2.0)), FullSelection(), SegmentSelection(0, 2)))

    calculation = create_calculation(
        CalcType.L2_NORM_PR_UNIT,
        builder.finalize(),
    )

    assert isinstance(calculation, MappedReductionCalculation)
    assert calculation.calculation_type == CalcType.L2_NORM_PR_UNIT


def test_group_lasso_multiplies_active_params_and_l2_norms() -> None:
    unit_active_mask = StaticCalculation(torch.tensor([1.0, 0.0, 1.0, 1.0]))
    unit_to_group = create_pipeline_calculation(
        _unit_to_group_plan(torch.tensor([1.0, 0.0, 1.0, 1.0]))
    )
    l2_norm_pr_unit = ParameterCalculation(torch.tensor([5.0, 7.0, 11.0, 13.0]))
    regularizer = GroupLasso(
        {
            CalcType.L2_NORM_PR_UNIT: l2_norm_pr_unit,
            CalcType.UNITS_TO_GROUP: unit_to_group,
            CalcType.UNIT_ACTIVE_MASK: unit_active_mask,
            CalcType.BASELINE_GROUP_SIZES: StaticCalculation(torch.tensor([2.0, 2.0])),
            CalcType.GROUP_CHANGE_EFFECT: StaticCalculation(torch.tensor([10.0, 4.0])),
            CalcType.GROUP_SIZES: StaticCalculation(torch.tensor([2, 2])),
        }
    )

    loss = regularizer()
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(-120.0))
    torch.testing.assert_close(
        l2_norm_pr_unit.value.grad,
        torch.tensor([-10.0, -10.0, 0.0, 0.0]),
    )


def test_group_lasso_requires_explicit_calculations() -> None:
    with pytest.raises(ValueError, match="missing required calculations"):
        GroupLasso({})


def test_structured_bops_requires_explicit_calculations() -> None:
    with pytest.raises(ValueError, match="missing required calculations"):
        StructuredBOPs({})
