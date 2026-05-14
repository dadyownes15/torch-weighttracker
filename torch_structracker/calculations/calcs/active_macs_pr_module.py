from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from torch_structracker.calculations.base import CalcType, Calculation
from torch_structracker.calculations.context import CalculationContext


class ActiveMacsPrModuleCalc(Calculation):
    calculation_type = CalcType.ACTIVE_MACS_PR_MODULE
    required_calculations = (
        CalcType.UNIT_ACTIVE_MASK,
        CalcType.UNIT_DELTA_TO_MODULE_AXIS,
        CalcType.BASELINE_MACS_PR_MODULE,
        CalcType.BASELINE_MODULE_AXES,
    )

    def __init__(self, dependencies: Mapping[CalcType, nn.Module]) -> None:
        super().__init__(dependencies)

    def forward(self) -> torch.Tensor:
        active_units = self.compute(CalcType.UNIT_ACTIVE_MASK)
        baseline_axes = self.compute(CalcType.BASELINE_MODULE_AXES)
        baseline_macs = self.compute(CalcType.BASELINE_MACS_PR_MODULE)
        axis_delta = self.compute(
            CalcType.UNIT_DELTA_TO_MODULE_AXIS,
            active_units,
        ).view_as(baseline_axes)
        active_axes = baseline_axes + axis_delta
        return baseline_macs * (active_axes / baseline_axes).prod(dim=1)


def create_active_macs_pr_module_calc(
    ctx: CalculationContext,
    *,
    dependencies: Mapping[CalcType, nn.Module],
) -> ActiveMacsPrModuleCalc:
    return ActiveMacsPrModuleCalc(dependencies)
