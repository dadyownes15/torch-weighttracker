from __future__ import annotations

import torch

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import (
    CalculationContext,
    calculation_device,
    calculation_dtype,
)
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.calculations.static_calc import StaticCalc
from torch_weighttracker.plans.module_axis_plan import create_module_axis_plan


class BaselineModuleAxesCalc(StaticCalc):
    """
    Returns the initial input-axis and output-axis size for each weighted module.

    Output: 1D tensor with length `2 * len(weighted_modules)`, ordered as
    `(input_axis, output_axis)` for each weighted module.
    Input: none.
    """

    calculation_type = CalcType.BASELINE_MODULE_AXES

    def __init__(self, baseline_axes: torch.Tensor) -> None:
        super().__init__(baseline_axes)


def create_baseline_module_axes_calc(
    ctx: CalculationContext,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> BaselineModuleAxesCalc:
    axis_plan = create_module_axis_plan(ctx)
    return BaselineModuleAxesCalc(
        axis_plan.baseline_axes.to(device=device, dtype=dtype)
    )


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.BASELINE_MODULE_AXES,
    requires_groups=False,
    cache_constant=True,
    create=lambda ctx, deps: create_baseline_module_axes_calc(
        ctx,
        device=calculation_device(ctx),
        dtype=calculation_dtype(ctx),
    ),
)
