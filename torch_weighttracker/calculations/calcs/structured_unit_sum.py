from __future__ import annotations

from collections.abc import Iterable

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.reduction_calc import ReductionCalc
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.canonical_units import CanonicalUnitGroup
from torch_weighttracker.operations import WeightOperationType
from torch_weighttracker.plans.unit_weight_operation_plan import create_group_member_plan


def create_structured_unit_sum_calc(
    groups: Iterable[CanonicalUnitGroup],
) -> ReductionCalc:
    """
    Returns the sum of live weights for each canonical unit.

    Output: 1D tensor with length equal to the total canonical unit count.
    Input: none.
    """
    return ReductionCalc(
        create_group_member_plan(groups, WeightOperationType.SUM),
        calculation_type=CalcType.STRUCTURED_UNIT_SUM,
    )


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.STRUCTURED_UNIT_SUM,
    create=lambda ctx, deps: create_structured_unit_sum_calc(ctx.canonical_groups),
)
