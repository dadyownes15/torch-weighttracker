from __future__ import annotations

from collections.abc import Iterable

from torch_structracker.calculations.base import CalcType
from torch_structracker.calculations.reduction_calc import ReductionCalc
from torch_structracker.canonical_units import CanonicalUnitGroup
from torch_structracker.plans.unit_weight_operation_plan import create_group_change_effect_plan


def create_group_change_effect_calc(
    groups: Iterable[CanonicalUnitGroup],
) -> ReductionCalc:
    return ReductionCalc(
        create_group_change_effect_plan(groups),
        calculation_type=CalcType.GROUP_CHANGE_EFFECT,
    )
