from __future__ import annotations

from typing import Iterable, cast

import torch

from torch_weighttracker.canonical_units import (
    CanonicalMember,
    CanonicalUnitGroup,
    UnitAxis,
)
from torch_weighttracker.extractors.extractor import TensorSpec, ValueTensorRef
from torch_weighttracker.reductions.builder import (
    IndexSelection,
    PipelinePlan,
    ReductionMapping,
    ReductionPlanBuilder,
    SegmentSelection,
)
from torch_weighttracker.reductions.compiler import (
    GenericReductionPlanner,
    MappingStrategy,
)
from torch_weighttracker.reductions.helpers import (
    PipelineReductionRule,
    TensorReductionMapper,
)
from torch_weighttracker.reductions.ops import ReductionOp


class FromStructureUnitToGroupUnitMapper(MappingStrategy[CanonicalUnitGroup]):
    def map(
        self,
        element: CanonicalUnitGroup,
        op: ReductionOp,
        builder: ReductionPlanBuilder,
    ) -> ReductionMapping:
        source = SegmentSelection(
            start=element.offset,
            length=element.length,
        )
        target = IndexSelection((element.group_id,))
        return ReductionMapping(source=source, target=target)


def create_unit_to_group_acc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    input_tensor_ref: ValueTensorRef,
    reduction_mapper: TensorReductionMapper[CanonicalUnitGroup],
) -> PipelinePlan:
    canonical_groups = tuple(groups)
    planner = GenericReductionPlanner[CanonicalUnitGroup](elements=canonical_groups)

    compiled = planner.compile(
        PipelineReductionRule[CanonicalUnitGroup](
            input=input_tensor_ref,
            reduction_mapper=reduction_mapper,
            mapping_strategy=FromStructureUnitToGroupUnitMapper(),
        ),
        input_spec=input_tensor_ref.source_spec(),
    )
    return cast(PipelinePlan, compiled)


def create_unit_input_ref(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> ValueTensorRef:
    dtype = torch.float32 if dtype is None else dtype
    device = torch.device("cpu") if device is None else torch.device(device)
    value = torch.empty(total_units(groups), dtype=dtype, device=device)
    return ValueTensorRef(
        value=value,
        spec=TensorSpec(
            shape=value.shape,
            dtype=value.dtype,
            device=value.device,
        ),
    )


def total_units(groups: Iterable[CanonicalUnitGroup]) -> int:
    return sum(int(group.length) for group in groups)


def module_axis_for_member(member: CanonicalMember) -> int:
    return 0 if member.unit_axis == UnitAxis.IN_CHANNEL else 1
