from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from torch_structracker.calculations.base import CalcType, Calculation
from torch_structracker.calculations.static_calc import StaticCalc
from torch_structracker.canonical_units import CanonicalUnitGroup
from torch_structracker.plans.unit_weight_operation_plan import count_group_units


class BaselineGroupSizesCalc(StaticCalc):
    calculation_type = CalcType.BASELINE_GROUP_SIZES
    required_calculations = (CalcType.UNITS_TO_GROUP,)

    def __init__(
        self,
        baseline_group_sizes: torch.Tensor,
    ) -> None:
        super().__init__(baseline_group_sizes)


def create_baseline_group_sizes_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    units_to_group: nn.Module,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> BaselineGroupSizesCalc:
    canonical_groups = tuple(groups)
    dtype = torch.float32 if dtype is None else dtype
    unit_ones = torch.ones(
        count_group_units(canonical_groups),
        dtype=dtype,
        device=torch.device("cpu") if device is None else torch.device(device),
    )
    with torch.no_grad():
        baseline_group_sizes = units_to_group(unit_ones)
    return BaselineGroupSizesCalc(baseline_group_sizes)
