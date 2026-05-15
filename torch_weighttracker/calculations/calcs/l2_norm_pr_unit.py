from __future__ import annotations

from collections.abc import Iterable

import torch

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.reduction_calc import ReductionCalc
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.canonical_units import CanonicalUnitGroup
from torch_weighttracker.operations import WeightOperationType
from torch_weighttracker.plans.unit_weight_operation_plan import create_group_member_plan


class L2NormPrUnit(ReductionCalc):
    calculation_type = CalcType.L2_NORM_PR_UNIT

    def forward(self) -> torch.Tensor:
        squared_sum = super().forward()
        active = squared_sum.gt(0)
        safe_squared_sum = torch.where(
            active,
            squared_sum,
            torch.ones_like(squared_sum),
        )
        return torch.where(active, safe_squared_sum.sqrt(), squared_sum)


def create_l2_norm_pr_unit_calc(
    groups: Iterable[CanonicalUnitGroup],
) -> L2NormPrUnit:
    """
    Returns the coupled L2 norm of live weights for each canonical unit.

    Output: 1D tensor with length equal to the total canonical unit count.
    Input: none.
    """
    return L2NormPrUnit(
        create_group_member_plan(groups, WeightOperationType.SQUARED_SUM),
        calculation_type=CalcType.L2_NORM_PR_UNIT,
    )


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.L2_NORM_PR_UNIT,
    create=lambda ctx, deps: create_l2_norm_pr_unit_calc(ctx.canonical_groups),
)
