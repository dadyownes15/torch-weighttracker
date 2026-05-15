from __future__ import annotations

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType, Calculation
from torch_weighttracker.calculations.spec import CalculationSpec


class UnitActiveMaskCalc(Calculation):
    """
    Returns a binary active mask for each canonical unit.

    Output: 1D tensor with length equal to the total canonical unit count.
    Input: none.
    """

    calculation_type = CalcType.UNIT_ACTIVE_MASK

    def __init__(self, active_units: nn.Module) -> None:
        super().__init__({CalcType.ACTIVE_UNITS: active_units})

    def forward(self) -> torch.Tensor:
        raw = self.compute(CalcType.ACTIVE_UNITS)
        return raw.gt(0).to(dtype=raw.dtype)


def create_unit_active_mask_calc(active_units: nn.Module) -> UnitActiveMaskCalc:
    return UnitActiveMaskCalc(active_units=active_units)


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.UNIT_ACTIVE_MASK,
    required_calculations=(CalcType.ACTIVE_UNITS,),
    create=lambda ctx, deps: create_unit_active_mask_calc(
        deps[CalcType.ACTIVE_UNITS]
    ),
)
