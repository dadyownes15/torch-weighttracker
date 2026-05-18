from __future__ import annotations

import torch

from torch_weighttracker.calculations.base import CalcType, Calculation
from torch_weighttracker.calculations.spec import CalculationSpec


class ModuleAxisCostIndicesCalc(Calculation):
    """
    Returns flat module-axis indices that represent real MAC cost axes.

    Output: 1D long tensor of flat axis indices into BASELINE_MODULE_AXES.
    Input: none.
    """

    calculation_type = CalcType.MODULE_AXIS_COST_INDICES

    def forward(self) -> torch.Tensor:
        baseline_axes = self.compute(CalcType.BASELINE_MODULE_AXES)
        return baseline_axes.gt(0).nonzero(as_tuple=False).flatten()


def create_module_axis_cost_indices_calc(
    dependencies,
) -> ModuleAxisCostIndicesCalc:
    return ModuleAxisCostIndicesCalc(dependencies)


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.MODULE_AXIS_COST_INDICES,
    required_calculations=(CalcType.BASELINE_MODULE_AXES,),
    requires_groups=False,
    cache_constant=True,
    create=lambda ctx, deps: create_module_axis_cost_indices_calc(deps),
)
