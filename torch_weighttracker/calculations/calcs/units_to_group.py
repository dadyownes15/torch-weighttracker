from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import calculation_device, calculation_dtype
from torch_weighttracker.calculations.pipeline_calc import PipelineCalc
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.canonical_units import CanonicalUnitGroup
from torch_weighttracker.plans.mapping_plan import (
    create_unit_input_ref,
    create_unit_to_group_acc,
)
from torch_weighttracker.reductions.builder import PipelinePlan
from torch_weighttracker.reductions.ops import IdentityTensorReduction


def create_units_to_group_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PipelineCalc:
    """
    Returns one value per canonical group by summing the input values for the
    canonical units that belong to that group.

    Output: 1D tensor with length `len(canonical_groups)`.
    Input: 1D tensor with length equal to the total canonical unit count. Element
    `i` is the value for canonical unit `i`.
    """
    return PipelineCalc(
        _build_units_to_group_plan(groups, device=device, dtype=dtype),
        calculation_type=CalcType.UNITS_TO_GROUP,
    )


def _build_units_to_group_plan(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PipelinePlan:
    canonical_groups = tuple(groups)
    input_ref = create_unit_input_ref(canonical_groups, device=device, dtype=dtype)
    plan = create_unit_to_group_acc(
        canonical_groups,
        input_tensor_ref=input_ref,
        reduction_mapper=lambda _: IdentityTensorReduction(),
    )
    return cast(PipelinePlan, plan)


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.UNITS_TO_GROUP,
    create=lambda ctx, deps: create_units_to_group_calc(
        ctx.canonical_groups,
        device=calculation_device(ctx),
        dtype=calculation_dtype(ctx),
    ),
)
