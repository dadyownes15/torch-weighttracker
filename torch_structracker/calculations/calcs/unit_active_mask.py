from __future__ import annotations

import torch
import torch.nn as nn

from torch_structracker.calculations.base import CalcType, Calculation


class UnitActiveMaskCalc(Calculation):
    calculation_type = CalcType.UNIT_ACTIVE_MASK
    required_calculations = (CalcType.ACTIVE_UNITS,)

    def __init__(self, active_units: nn.Module) -> None:
        super().__init__({CalcType.ACTIVE_UNITS: active_units})

    def forward(self) -> torch.Tensor:
        raw = self.compute(CalcType.ACTIVE_UNITS)
        return raw.gt(0).to(dtype=raw.dtype)


def create_unit_active_mask_calc(active_units: nn.Module) -> UnitActiveMaskCalc:
    return UnitActiveMaskCalc(active_units=active_units)
