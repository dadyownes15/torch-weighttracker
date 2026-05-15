from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import calculation_device, calculation_dtype
from torch_weighttracker.calculations.pipeline_calc import PipelineCalc
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.canonical_units import CanonicalUnitGroup, canonical_members
from torch_weighttracker.plans.mapping_plan import (
    create_unit_input_ref,
    module_axis_for_member,
)
from torch_weighttracker.reductions.builder import (
    IndexSelection,
    PipelinePlan,
    ReductionMapping,
    ReductionPlanBuilder,
    ReductionRecord,
)
from torch_weighttracker.reductions.ops import IdentityTensorReduction, ReductionOp


def create_units_to_module_axis_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    weighted_module_index: Mapping[nn.Module, int],
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PipelineCalc:
    """
    Returns one value per weighted-module axis by summing input unit values.

    Output: 1D tensor with length `2 * len(weighted_modules)`, ordered as
    `(input_axis, output_axis)` for each weighted module.
    Input: 1D tensor with length equal to the total canonical unit count. Element
    `i` is the value for canonical unit `i`.
    """
    plan = _build_units_to_module_axis_plan(
        groups,
        weighted_module_index=weighted_module_index,
        device=device,
        dtype=dtype,
    )
    return PipelineCalc(plan, calculation_type=CalcType.UNITS_TO_MODULE_AXIS)


def _build_units_to_module_axis_plan(
    groups: Iterable[CanonicalUnitGroup],
    *,
    weighted_module_index: Mapping[nn.Module, int],
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PipelinePlan:
    canonical_groups = tuple(groups)
    input_ref = create_unit_input_ref(canonical_groups, device=device, dtype=dtype)
    builder = ReductionPlanBuilder(output_length=len(weighted_module_index) * 2)
    seen: set[tuple[int, int, int]] = set()
    record_count = 0

    for member in canonical_members(canonical_groups):
        module_index = weighted_module_index.get(member.module)
        if module_index is None:
            continue

        axis = module_axis_for_member(member)
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


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.UNITS_TO_MODULE_AXIS,
    create=lambda ctx, deps: create_units_to_module_axis_calc(
        ctx.canonical_groups,
        weighted_module_index=ctx.weighted_module_index,
        device=calculation_device(ctx),
        dtype=calculation_dtype(ctx),
    ),
)
