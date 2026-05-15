from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType, Calculation
from torch_weighttracker.calculations.context import CalculationContext
from torch_weighttracker.calculations.spec import CalculationSpec


class ActiveMacsPrModuleCalc(Calculation):
    """
    Returns the active runtime MAC count for each weighted module.

    Output: 1D tensor with length `len(weighted_modules)`.
    Input: none.
    """

    calculation_type = CalcType.ACTIVE_MACS_PR_MODULE

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


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.ACTIVE_MACS_PR_MODULE,
    required_calculations=(
        CalcType.UNIT_ACTIVE_MASK,
        CalcType.UNIT_DELTA_TO_MODULE_AXIS,
        CalcType.BASELINE_MACS_PR_MODULE,
        CalcType.BASELINE_MODULE_AXES,
    ),
    create=lambda ctx, deps: create_active_macs_pr_module_calc(
        ctx,
        dependencies=deps,
    ),
)
