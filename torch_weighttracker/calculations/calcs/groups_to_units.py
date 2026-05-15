from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import group_input_spec
from torch_weighttracker.calculations.pipeline_calc import PipelineCalc
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.canonical_units import CanonicalUnitGroup
from torch_weighttracker.extractors.extractor import TensorSpec, ValueTensorRef
from torch_weighttracker.reductions.builder import (
    IndexSelection,
    PipelinePlan,
    ReductionMapping,
    ReductionPlanBuilder,
    ReductionRecord,
)
from torch_weighttracker.reductions.ops import IdentityTensorReduction, ReductionOp


def create_groups_to_units_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    input_spec: TensorSpec,
) -> PipelineCalc:
    return PipelineCalc(
        create_groups_to_units_plan(groups, input_spec=input_spec),
        calculation_type=CalcType.GROUPS_TO_UNITS,
    )


def create_groups_to_units_plan(
    groups: Iterable[CanonicalUnitGroup],
    *,
    input_spec: TensorSpec,
) -> PipelinePlan:
    canonical_groups = tuple(groups)
    input_ref = ValueTensorRef(
        value=torch.empty(
            input_spec.shape,
            dtype=input_spec.dtype,
            device=input_spec.device,
        ),
        spec=input_spec,
    )
    builder = ReductionPlanBuilder(
        output_length=sum(int(group.length) for group in canonical_groups)
    )

    source_indices: list[int] = []
    target_indices: list[int] = []
    for group in canonical_groups:
        source_indices.extend([int(group.group_id)] * int(group.length))
        target_indices.extend(range(int(group.offset), int(group.offset + group.length)))

    op = ReductionOp(input_ref, IdentityTensorReduction())
    builder.add(
        ReductionRecord(
            op=op,
            mapping=ReductionMapping(
                source=IndexSelection(tuple(source_indices)),
                target=IndexSelection(tuple(target_indices)),
            ),
        )
    )

    return cast(PipelinePlan, builder.finalize(input_ref.source_spec()))


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.GROUPS_TO_UNITS,
    create=lambda ctx, deps: create_groups_to_units_calc(
        ctx.canonical_groups,
        input_spec=group_input_spec(ctx),
    ),
)
