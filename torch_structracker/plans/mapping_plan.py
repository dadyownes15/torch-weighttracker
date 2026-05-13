from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable, cast

import torch
import torch.nn as nn

from torch_structracker.canonical_units import (
    CanonicalUnitGroup,
    UnitAxis,
    canonical_members,
)
from torch_structracker.extractors.extractor import TensorSpec, ValueTensorRef
from torch_structracker.reductions.builder import (
    IndexSelection,
    PipelinePlan,
    ReductionMapping,
    ReductionPlanBuilder,
    ReductionRecord,
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
from torch_structracker.reductions.ops import IdentityTensorReduction, ReductionOp


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


def create_units_to_group_plan(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PipelinePlan:
    canonical_groups = tuple(groups)
    input_ref = _unit_input_ref(canonical_groups, device=device, dtype=dtype)
    return create_unit_to_group_acc(
        canonical_groups,
        input_tensor_ref=input_ref,
        reduction_mapper=lambda _: IdentityTensorReduction(),
    )


def create_units_to_module_axis_plan(
    groups: Iterable[CanonicalUnitGroup],
    *,
    weighted_module_index: Mapping[nn.Module, int],
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PipelinePlan:
    canonical_groups = tuple(groups)
    input_ref = _unit_input_ref(canonical_groups, device=device, dtype=dtype)
    builder = ReductionPlanBuilder(output_length=len(weighted_module_index) * 2)
    seen: set[tuple[int, int, int]] = set()
    record_count = 0

    for member in canonical_members(canonical_groups):
        module_index = weighted_module_index.get(member.module)
        if module_index is None:
            continue

        axis = 0 if member.unit_axis == UnitAxis.IN_CHANNEL else 1
        source_indices: list[int] = []
        for unit_index in member.unit_indices:
            key = (module_index, axis, int(unit_index))
            if key in seen:
                continue
            seen.add(key)
            source_indices.append(int(unit_index))

        if not source_indices:
            continue

        op = ReductionOp(input_ref, IdentityTensorReduction())
        target = module_index * 2 + axis
        builder.add(
            ReductionRecord(
                op=op,
                mapping=ReductionMapping(
                    source=IndexSelection(tuple(source_indices)),
                    target=IndexSelection((target,) * len(source_indices)),
                ),
            )
        )
        record_count += 1

    if record_count == 0:
        op = ReductionOp(input_ref, IdentityTensorReduction())
        builder.add(
            ReductionRecord(
                op=op,
                mapping=ReductionMapping(
                    source=IndexSelection(()),
                    target=IndexSelection(()),
                ),
            )
        )

    return cast(PipelinePlan, builder.finalize(input_ref.source_spec()))


def _unit_input_ref(
    groups: tuple[CanonicalUnitGroup, ...],
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> ValueTensorRef:
    dtype = torch.float32 if dtype is None else dtype
    device = torch.device("cpu") if device is None else torch.device(device)
    value = torch.empty(_total_units(groups), dtype=dtype, device=device)
    return ValueTensorRef(
        value=value,
        spec=TensorSpec(
            shape=value.shape,
            dtype=value.dtype,
            device=value.device,
        ),
    )


def _total_units(groups: Iterable[CanonicalUnitGroup]) -> int:
    return sum(int(group.length) for group in groups)
