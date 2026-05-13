import pytest
import torch
import torch.nn as nn

from torch_structracker.calculations import BaseCalculation
from torch_structracker.calculations.active_params_pr_unit import ActiveParamsPrUnit
from torch_structracker.calculations.reduction_calc import MappedReductionCalculation
from torch_structracker.calculations.pipeline_calc import PipelineCalc
from torch_structracker.canonical_units import CanonicalUnitGroup, UnitKind
from torch_structracker.extractors.extractor import TensorSpec, ValueTensorRef
from torch_structracker.plans.mapping_plan import create_unit_to_group_acc
from torch_structracker.reductions.builder import (
    FullSelection,
    IndexedEntry,
    IndexedGatherEntry,
    IndexSelection,
    ReductionMapping,
    ReductionPlanBuilder,
    ReductionRecord,
    SegmentEntry,
    SegmentSelection,
)
from torch_structracker.reductions.ops import IdentityTensorReduction


class MetadataOnlyOp:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        output_length: int | None = None,
    ) -> None:
        self._output_spec = TensorSpec(
            shape=torch.Size(shape),
            dtype=dtype,
            device=torch.device(device),
        )
        self._output_length = _numel(shape) if output_length is None else output_length
        self.call_count = 0

    @property
    def output_spec(self) -> TensorSpec:
        return self._output_spec

    @property
    def output_length(self) -> int:
        return self._output_length

    def __call__(self) -> torch.Tensor:
        self.call_count += 1
        raise AssertionError("Reduction op must not execute during finalize.")


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


class StaticCalculation(BaseCalculation):
    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value, persistent=False)
        self.call_count = 0

    def forward(self) -> torch.Tensor:
        self.call_count += 1
        return self.value


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for size in shape:
        total *= int(size)
    return total


def _record(
    op: MetadataOnlyOp,
    source,
    target,
) -> ReductionRecord:
    return ReductionRecord(
        op=op,
        mapping=ReductionMapping(source=source, target=target),
    )


def test_finalize_derives_output_spec_from_segment_entries() -> None:
    builder = ReductionPlanBuilder()
    op = MetadataOnlyOp((3,), dtype=torch.float64)

    builder.add(_record(op, FullSelection(), builder.reserve_segment(3)))

    plan = builder.finalize()

    assert plan.output_spec == TensorSpec(
        shape=torch.Size([3]),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    assert plan.segment_entries == (SegmentEntry(op=op, start=0, length=3),)


def test_finalize_derives_output_spec_from_indexed_entries() -> None:
    builder = ReductionPlanBuilder()
    op = MetadataOnlyOp((2,), dtype=torch.float64)

    builder.add(_record(op, FullSelection(), IndexSelection((1, 4))))

    plan = builder.finalize()

    assert plan.output_length == 5
    assert plan.output_spec == TensorSpec(
        shape=torch.Size([5]),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    assert plan.indexed_entries == (
        IndexedEntry(op=op, destination_indices=(1, 4)),
    )


def test_full_source_to_full_target_lowers_to_output_segment() -> None:
    builder = ReductionPlanBuilder()
    op = MetadataOnlyOp((3,))

    builder.add(_record(op, FullSelection(), FullSelection()))
    plan = builder.finalize()

    assert plan.segment_entries == (SegmentEntry(op=op, start=0, length=3),)


def test_segment_source_to_index_target_lowers_to_indexed_gather() -> None:
    builder = ReductionPlanBuilder()
    op = MetadataOnlyOp((5,))

    builder.add(
        _record(
            op,
            SegmentSelection(start=1, length=2),
            IndexSelection((0, 2)),
        )
    )
    plan = builder.finalize()

    assert plan.indexed_entries == ()
    assert plan.indexed_gather_entries == (
        IndexedGatherEntry(
            op=op,
            source_indices=(1, 2),
            destination_indices=(0, 2),
        ),
    )


def test_segment_source_to_single_index_target_repeats_destination() -> None:
    builder = ReductionPlanBuilder()
    op = MetadataOnlyOp((5,))

    builder.add(
        _record(
            op,
            SegmentSelection(start=1, length=3),
            IndexSelection((4,)),
        )
    )
    plan = builder.finalize()

    assert plan.indexed_gather_entries == (
        IndexedGatherEntry(
            op=op,
            source_indices=(1, 2, 3),
            destination_indices=(4, 4, 4),
        ),
    )


def test_full_source_to_single_index_target_repeats_destination() -> None:
    builder = ReductionPlanBuilder()
    op = MetadataOnlyOp((3,))

    builder.add(_record(op, FullSelection(), IndexSelection((2,))))
    plan = builder.finalize()

    assert plan.indexed_entries == (
        IndexedEntry(op=op, destination_indices=(2, 2, 2)),
    )


def test_index_source_to_index_target_lowers_to_indexed_gather() -> None:
    builder = ReductionPlanBuilder()
    op = MetadataOnlyOp((5,))

    builder.add(
        _record(
            op,
            IndexSelection((1, 4)),
            IndexSelection((0, 3)),
        )
    )
    plan = builder.finalize()

    assert plan.indexed_gather_entries == (
        IndexedGatherEntry(
            op=op,
            source_indices=(1, 4),
            destination_indices=(0, 3),
        ),
    )


def test_singleton_index_selection_lowers_to_single_element_gather() -> None:
    builder = ReductionPlanBuilder()
    op = MetadataOnlyOp((3,))

    builder.add(
        _record(
            op,
            IndexSelection((2,)),
            IndexSelection((5,)),
        )
    )
    plan = builder.finalize()

    assert plan.indexed_gather_entries == (
        IndexedGatherEntry(
            op=op,
            source_indices=(2,),
            destination_indices=(5,),
        ),
    )


def test_index_source_to_segment_target_lowers_to_indexed_gather() -> None:
    builder = ReductionPlanBuilder()
    op = MetadataOnlyOp((4,))

    builder.add(
        _record(
            op,
            IndexSelection((0, 3)),
            SegmentSelection(start=5, length=2),
        )
    )
    plan = builder.finalize()

    assert plan.indexed_gather_entries == (
        IndexedGatherEntry(
            op=op,
            source_indices=(0, 3),
            destination_indices=(5, 6),
        ),
    )


def test_segment_source_to_segment_target_lowers_to_indexed_gather() -> None:
    builder = ReductionPlanBuilder()
    op = MetadataOnlyOp((6,))

    builder.add(
        _record(
            op,
            SegmentSelection(start=2, length=3),
            SegmentSelection(start=0, length=3),
        )
    )
    plan = builder.finalize()

    assert plan.indexed_gather_entries == (
        IndexedGatherEntry(
            op=op,
            source_indices=(2, 3, 4),
            destination_indices=(0, 1, 2),
        ),
    )


def test_finalize_rejects_empty_builder() -> None:
    with pytest.raises(ValueError, match="empty"):
        ReductionPlanBuilder(output_length=3).finalize()


def test_finalize_rejects_negative_output_length() -> None:
    builder = ReductionPlanBuilder(output_length=-1)
    builder.indexed_entries.append(
        IndexedEntry(op=MetadataOnlyOp((0,)), destination_indices=())
    )

    with pytest.raises(ValueError, match="output length"):
        builder.finalize()


def test_finalize_rejects_mixed_dtype_entries() -> None:
    builder = ReductionPlanBuilder()
    builder.add(
        _record(
            MetadataOnlyOp((1,), dtype=torch.float32),
            FullSelection(),
            SegmentSelection(0, 1),
        )
    )
    builder.add(
        _record(
            MetadataOnlyOp((1,), dtype=torch.float64),
            FullSelection(),
            SegmentSelection(1, 1),
        )
    )

    with pytest.raises(ValueError, match="dtype"):
        builder.finalize()


def test_finalize_rejects_mixed_device_entries() -> None:
    builder = ReductionPlanBuilder()
    builder.add(
        _record(
            MetadataOnlyOp((1,), device="cpu"),
            FullSelection(),
            SegmentSelection(0, 1),
        )
    )
    builder.add(
        _record(
            MetadataOnlyOp((1,), device="meta"),
            FullSelection(),
            SegmentSelection(1, 1),
        )
    )

    with pytest.raises(ValueError, match="device"):
        builder.finalize()


def test_finalize_rejects_non_flat_op_output_spec() -> None:
    builder = ReductionPlanBuilder()
    builder.add(
        _record(
            MetadataOnlyOp((2, 2)),
            FullSelection(),
            SegmentSelection(0, 4),
        )
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        builder.finalize()


@pytest.mark.parametrize(
    ("target", "match"),
    (
        (SegmentSelection(-1, 1), "start"),
        (SegmentSelection(0, -1), "length"),
    ),
)
def test_finalize_rejects_invalid_segment_targets(
    target: SegmentSelection,
    match: str,
) -> None:
    builder = ReductionPlanBuilder()

    with pytest.raises(ValueError, match=match):
        builder.add(_record(MetadataOnlyOp((1,)), FullSelection(), target))


def test_finalize_rejects_segment_bounds_outside_output_length() -> None:
    builder = ReductionPlanBuilder(output_length=2)
    builder.segment_entries.append(
        SegmentEntry(op=MetadataOnlyOp((2,)), start=1, length=2)
    )

    with pytest.raises(ValueError, match="exceeds"):
        builder.finalize()


def test_finalize_rejects_segment_length_mismatch() -> None:
    builder = ReductionPlanBuilder(output_length=2)

    with pytest.raises(ValueError, match="lengths"):
        builder.add(
            _record(
                MetadataOnlyOp((2,)),
                FullSelection(),
                SegmentSelection(0, 1),
            )
        )


def test_finalize_rejects_indexed_destination_count_mismatch() -> None:
    builder = ReductionPlanBuilder()

    with pytest.raises(ValueError, match="lengths"):
        builder.add(
            _record(
                MetadataOnlyOp((2,)),
                FullSelection(),
                IndexSelection((0, 1, 2)),
            )
        )


def test_finalize_rejects_negative_indexed_destinations() -> None:
    builder = ReductionPlanBuilder()

    with pytest.raises(ValueError, match="non-negative"):
        builder.add(
            _record(
                MetadataOnlyOp((1,)),
                FullSelection(),
                IndexSelection((-1,)),
            )
        )
    assert builder.output_length == 0


def test_finalize_rejects_indexed_destinations_outside_output_length() -> None:
    builder = ReductionPlanBuilder(output_length=2)
    builder.indexed_entries.append(
        IndexedEntry(op=MetadataOnlyOp((1,)), destination_indices=(2,))
    )

    with pytest.raises(ValueError, match="exceeds"):
        builder.finalize()


def test_add_rejects_indexed_destinations_outside_fixed_output_length() -> None:
    builder = ReductionPlanBuilder(output_length=2)

    with pytest.raises(ValueError, match="fixed output length"):
        builder.add(
            _record(
                MetadataOnlyOp((1,)),
                FullSelection(),
                IndexSelection((2,)),
            )
        )


def test_finalize_rejects_gather_source_target_count_mismatch() -> None:
    builder = ReductionPlanBuilder()

    with pytest.raises(ValueError, match="lengths"):
        builder.add(
            _record(
                MetadataOnlyOp((3,)),
                IndexSelection((1, 2)),
                IndexSelection((0, 1, 2)),
            )
        )


def test_finalize_rejects_negative_source_selection() -> None:
    builder = ReductionPlanBuilder()

    with pytest.raises(ValueError, match="non-negative"):
        builder.add(
            _record(
                MetadataOnlyOp((2,)),
                IndexSelection((-1,)),
                IndexSelection((0,)),
            )
        )


def test_add_rejects_negative_segment_source_selection() -> None:
    builder = ReductionPlanBuilder()

    with pytest.raises(ValueError, match="non-negative"):
        builder.add(
            _record(
                MetadataOnlyOp((2,)),
                SegmentSelection(-1, 1),
                IndexSelection((0,)),
            )
        )


def test_finalize_rejects_source_selection_outside_op_output() -> None:
    builder = ReductionPlanBuilder()

    with pytest.raises(ValueError, match="exceeds"):
        builder.add(
            _record(
                MetadataOnlyOp((2,)),
                SegmentSelection(start=2, length=1),
                IndexSelection((0,)),
            )
        )


def test_finalize_does_not_execute_reduction_ops() -> None:
    builder = ReductionPlanBuilder()
    op = MetadataOnlyOp((2,))

    builder.add(_record(op, FullSelection(), SegmentSelection(0, 2)))
    builder.finalize()

    assert op.call_count == 0


def test_mapped_reduction_calculation_executes_all_runtime_entry_types() -> None:
    builder = ReductionPlanBuilder()
    builder.add(
        _record(
            TensorOp((1.0, 2.0)),
            FullSelection(),
            SegmentSelection(0, 2),
        )
    )
    builder.add(
        _record(
            TensorOp((10.0, 20.0)),
            FullSelection(),
            IndexSelection((2, 4)),
        )
    )
    builder.add(
        _record(
            TensorOp((100.0, 200.0, 300.0)),
            IndexSelection((0, 2)),
            IndexSelection((1, 3)),
        )
    )

    calculation = MappedReductionCalculation(builder.finalize())

    torch.testing.assert_close(
        calculation(),
        torch.tensor([1.0, 102.0, 10.0, 300.0, 20.0]),
    )


def test_unit_to_group_mapping_plan_compiles_pipeline_plan() -> None:
    input_value = torch.tensor([1.0, 2.0, 3.0])
    input_spec = TensorSpec(
        shape=input_value.shape,
        dtype=input_value.dtype,
        device=input_value.device,
    )
    groups = (
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
            length=1,
            unit_kind=UnitKind.CHANNEL,
            members=(),
            raw_group=object(),
        ),
    )

    plan = create_unit_to_group_acc(
        groups,
        input_tensor_ref=ValueTensorRef(value=input_value, spec=input_spec),
        reduction_mapper=lambda _: IdentityTensorReduction(),
    )

    assert plan.kind() == "pipeline_plan"
    assert plan.input_spec == input_spec
    assert plan.output_length == 2
    assert tuple(
        (entry.source_indices, entry.destination_indices)
        for entry in plan.indexed_gather_entries
    ) == (
        ((0, 1), (0, 0)),
        ((2,), (1,)),
    )


def test_pipeline_calculation_accumulates_unit_values_to_compact_groups() -> None:
    input_value = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    input_spec = TensorSpec(
        shape=input_value.shape,
        dtype=input_value.dtype,
        device=input_value.device,
    )
    groups = (
        CanonicalUnitGroup(
            group_id=0,
            offset=0,
            length=3,
            unit_kind=UnitKind.CHANNEL,
            members=(),
            raw_group=object(),
        ),
        CanonicalUnitGroup(
            group_id=1,
            offset=3,
            length=2,
            unit_kind=UnitKind.CHANNEL,
            members=(),
            raw_group=object(),
        ),
    )
    plan = create_unit_to_group_acc(
        groups,
        input_tensor_ref=ValueTensorRef(value=input_value, spec=input_spec),
        reduction_mapper=lambda _: IdentityTensorReduction(),
    )

    calculation = PipelineCalc(plan)

    torch.testing.assert_close(
        calculation(input_value),
        torch.tensor([2.0, 1.0]),
    )


def test_active_params_pr_unit_broadcasts_group_effect_to_units() -> None:
    input_value = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    input_spec = TensorSpec(
        shape=input_value.shape,
        dtype=input_value.dtype,
        device=input_value.device,
    )
    groups = (
        CanonicalUnitGroup(
            group_id=0,
            offset=0,
            length=3,
            unit_kind=UnitKind.CHANNEL,
            members=(),
            raw_group=object(),
        ),
        CanonicalUnitGroup(
            group_id=1,
            offset=3,
            length=2,
            unit_kind=UnitKind.CHANNEL,
            members=(),
            raw_group=object(),
        ),
    )
    unit_to_group_acc = PipelineCalc(
        create_unit_to_group_acc(
            groups,
            input_tensor_ref=ValueTensorRef(value=input_value, spec=input_spec),
            reduction_mapper=lambda _: IdentityTensorReduction(),
        )
    )
    unit_active_mask = StaticCalculation(input_value)
    calculation = ActiveParamsPrUnit(
        unit_to_group_acc=unit_to_group_acc,
        unit_active_mask=unit_active_mask,
        baseline_group_size=torch.tensor([3.0, 2.0]),
        group_change_effect=torch.tensor([10.0, 4.0]),
        group_lengths=torch.tensor([3, 2]),
    )

    torch.testing.assert_close(
        calculation(),
        torch.tensor([-10.0, -10.0, -10.0, -4.0, -4.0]),
    )
    assert unit_active_mask.call_count == 1


def test_active_params_pr_unit_handles_transformer_unit_kinds() -> None:
    input_value = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
    input_spec = TensorSpec(
        shape=input_value.shape,
        dtype=input_value.dtype,
        device=input_value.device,
    )
    groups = (
        CanonicalUnitGroup(
            group_id=0,
            offset=0,
            length=2,
            unit_kind=UnitKind.HEAD,
            members=(),
            raw_group=object(),
        ),
        CanonicalUnitGroup(
            group_id=1,
            offset=2,
            length=3,
            unit_kind=UnitKind.HEAD_DIM,
            members=(),
            raw_group=object(),
        ),
    )
    unit_to_group_acc = PipelineCalc(
        create_unit_to_group_acc(
            groups,
            input_tensor_ref=ValueTensorRef(value=input_value, spec=input_spec),
            reduction_mapper=lambda _: IdentityTensorReduction(),
        )
    )
    calculation = ActiveParamsPrUnit(
        unit_to_group_acc=unit_to_group_acc,
        unit_active_mask=StaticCalculation(input_value),
        baseline_group_size=torch.tensor([2.0, 3.0]),
        group_change_effect=torch.tensor([64.0, 8.0]),
        group_lengths=torch.tensor([2, 3]),
    )

    torch.testing.assert_close(
        calculation(),
        torch.tensor([-64.0, -64.0, -8.0, -8.0, -8.0]),
    )
