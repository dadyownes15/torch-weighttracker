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

    def forward(self) -> torch.Tensor:
        active_units = self.compute(CalcType.UNIT_ACTIVE_MASK)
        baseline_axes = self.compute(CalcType.BASELINE_MODULE_AXES)
        baseline_macs = self.compute(CalcType.BASELINE_MACS_PR_MODULE)
        cost_axis_indices = self.compute(CalcType.MODULE_AXIS_COST_INDICES)
        axis_delta = self.compute(
            CalcType.UNIT_DELTA_TO_MODULE_AXIS,
            active_units,
        )
        active_axes = baseline_axes + axis_delta
        active_cost_axes = active_axes.index_select(0, cost_axis_indices)
        baseline_cost_axes = baseline_axes.index_select(0, cost_axis_indices)
        ratios = active_cost_axes / baseline_cost_axes
        module_scale = _product_by_module(
            ratios,
            cost_axis_indices // 2,
            num_modules=baseline_macs.numel(),
        )
        return baseline_macs * module_scale


def create_active_macs_pr_module_calc(
    ctx: CalculationContext,
    *,
    dependencies: Mapping[CalcType, nn.Module],
) -> ActiveMacsPrModuleCalc:
    return ActiveMacsPrModuleCalc(dependencies)


def _product_by_module(
    values: torch.Tensor,
    module_indices: torch.Tensor,
    *,
    num_modules: int,
) -> torch.Tensor:
    result = values.new_ones(num_modules)
    if values.numel() == 0:
        return result
    return result.scatter_reduce_(
        0,
        module_indices,
        values,
        reduce="prod",
        include_self=True,
    )


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.ACTIVE_MACS_PR_MODULE,
    required_calculations=(
        CalcType.UNIT_ACTIVE_MASK,
        CalcType.UNIT_DELTA_TO_MODULE_AXIS,
        CalcType.BASELINE_MACS_PR_MODULE,
        CalcType.BASELINE_MODULE_AXES,
        CalcType.MODULE_AXIS_COST_INDICES,
    ),
    create=lambda ctx, deps: create_active_macs_pr_module_calc(
        ctx,
        dependencies=deps,
    ),
)
