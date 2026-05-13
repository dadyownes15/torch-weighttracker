from __future__ import annotations

from typing import Iterable, cast

from torch_structracker.canonical_units import CanonicalUnitGroup
from torch_structracker.extractors.extractor import ValueTensorRef
from torch_structracker.reductions.builder import (
    IndexSelection,
    PipelinePlan,
    ReductionMapping,
    ReductionPlanBuilder,
    SegmentSelection,
)
from torch_structracker.reductions.compiler import (
    GenericReductionPlanner,
    MappingStrategy,
)
from torch_structracker.reductions.helpers import (
    PipelineReductionRule,
    TensorReductionMapper,
)
from torch_structracker.reductions.ops import ReductionOp


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


