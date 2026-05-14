from __future__ import annotations

import torch

from torch_structracker.calculations.base import CalcType
from torch_structracker.calculations.calcs.active_macs_pr_module import (
    ActiveMacsPrModuleCalc,
    create_active_macs_pr_module_calc,
)
from torch_structracker.calculations.calcs.active_units import create_active_units_calc
from torch_structracker.calculations.calcs.baseline_macs_pr_module import (
    create_baseline_macs_pr_module_calc,
)
from torch_structracker.calculations.calcs.baseline_module_axes import (
    create_baseline_module_axes_calc,
)
from torch_structracker.calculations.calcs.baseline_group_sizes import create_baseline_group_sizes_calc
from torch_structracker.calculations.calcs.bitrate_pr_module import create_bitrate_pr_module_calc
from torch_structracker.calculations.calcs.group_change_effect import create_group_change_effect_calc
from torch_structracker.calculations.calcs.group_sizes import create_group_sizes_calc
from torch_structracker.calculations.calcs.l2_norm_pr_unit import create_l2_norm_pr_unit_calc
from torch_structracker.calculations.calcs.structured_unit_sum import create_structured_unit_sum_calc
from torch_structracker.calculations.calcs.unit_active_mask import create_unit_active_mask_calc
from torch_structracker.calculations.calcs.unit_delta_to_module_axis import (
    create_unit_delta_to_module_axis_calc,
)
from torch_structracker.calculations.calcs.units_to_group import create_units_to_group_calc
from torch_structracker.calculations.calcs.units_to_module_axis import create_units_to_module_axis_calc
from torch_structracker.calculations.context import CalculationContext
from torch_structracker.calculations.spec import CalculationSpec


def _calculation_dtype(ctx: CalculationContext) -> torch.dtype:
    if ctx.dtype is not None:
        return ctx.dtype

    for parameter in ctx.model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype

    return torch.float32


def _calculation_device(ctx: CalculationContext) -> torch.device | str | None:
    if ctx.device is not None:
        return ctx.device

    for parameter in ctx.model.parameters():
        return parameter.device

    return torch.device("cpu")


CALCULATION_SPECS: dict[CalcType, CalculationSpec] = {
    CalcType.ACTIVE_UNITS: CalculationSpec(
        calculation_type=CalcType.ACTIVE_UNITS,
        create=lambda ctx, deps: create_active_units_calc(ctx.canonical_groups),
    ),
    CalcType.UNIT_ACTIVE_MASK: CalculationSpec(
        calculation_type=CalcType.UNIT_ACTIVE_MASK,
        required_calculations=(CalcType.ACTIVE_UNITS,),
        create=lambda ctx, deps: create_unit_active_mask_calc(
            deps[CalcType.ACTIVE_UNITS]
        ),
    ),
    CalcType.UNITS_TO_GROUP: CalculationSpec(
        calculation_type=CalcType.UNITS_TO_GROUP,
        create=lambda ctx, deps: create_units_to_group_calc(
            ctx.canonical_groups,
            device=_calculation_device(ctx),
            dtype=_calculation_dtype(ctx),
        ),
    ),
    CalcType.BASELINE_GROUP_SIZES: CalculationSpec(
        calculation_type=CalcType.BASELINE_GROUP_SIZES,
        required_calculations=(CalcType.UNITS_TO_GROUP,),
        cache_constant=True,
        create=lambda ctx, deps: create_baseline_group_sizes_calc(
            ctx.canonical_groups,
            units_to_group=deps[CalcType.UNITS_TO_GROUP],
            device=_calculation_device(ctx),
            dtype=_calculation_dtype(ctx),
        ),
    ),
    CalcType.GROUP_CHANGE_EFFECT: CalculationSpec(
        calculation_type=CalcType.GROUP_CHANGE_EFFECT,
        cache_constant=True,
        create=lambda ctx, deps: create_group_change_effect_calc(
            ctx.canonical_groups
        ),
    ),
    CalcType.GROUP_SIZES: CalculationSpec(
        calculation_type=CalcType.GROUP_SIZES,
        cache_constant=True,
        create=lambda ctx, deps: create_group_sizes_calc(
            ctx.canonical_groups,
            device=_calculation_device(ctx),
        ),
    ),
    CalcType.UNITS_TO_MODULE_AXIS: CalculationSpec(
        calculation_type=CalcType.UNITS_TO_MODULE_AXIS,
        create=lambda ctx, deps: create_units_to_module_axis_calc(
            ctx.canonical_groups,
            weighted_module_index=ctx.weighted_module_index,
            device=_calculation_device(ctx),
            dtype=_calculation_dtype(ctx),
        ),
    ),
    CalcType.UNIT_DELTA_TO_MODULE_AXIS: CalculationSpec(
        calculation_type=CalcType.UNIT_DELTA_TO_MODULE_AXIS,
        create=lambda ctx, deps: create_unit_delta_to_module_axis_calc(
            ctx.canonical_groups,
            weighted_module_index=ctx.weighted_module_index,
            device=_calculation_device(ctx),
            dtype=_calculation_dtype(ctx),
        ),
    ),
    CalcType.BASELINE_MODULE_AXES: CalculationSpec(
        calculation_type=CalcType.BASELINE_MODULE_AXES,
        requires_groups=False,
        cache_constant=True,
        create=lambda ctx, deps: create_baseline_module_axes_calc(
            ctx.weighted_modules,
            device=_calculation_device(ctx),
            dtype=_calculation_dtype(ctx),
        ),
    ),
    CalcType.BASELINE_MACS_PR_MODULE: CalculationSpec(
        calculation_type=CalcType.BASELINE_MACS_PR_MODULE,
        requires_groups=False,
        cache_constant=True,
        create=lambda ctx, deps: create_baseline_macs_pr_module_calc(
            ctx,
            device=_calculation_device(ctx),
            dtype=_calculation_dtype(ctx),
        ),
    ),
    CalcType.ACTIVE_MACS_PR_MODULE: CalculationSpec(
        calculation_type=CalcType.ACTIVE_MACS_PR_MODULE,
        required_calculations=ActiveMacsPrModuleCalc.required_calculations,
        create=lambda ctx, deps: create_active_macs_pr_module_calc(
            ctx,
            dependencies=deps,
        ),
    ),
    CalcType.BITRATE_PR_MODULE: CalculationSpec(
        calculation_type=CalcType.BITRATE_PR_MODULE,
        requires_groups=False,
        create=lambda ctx, deps: create_bitrate_pr_module_calc(
            ctx.weighted_modules,
            device=_calculation_device(ctx),
            dtype=_calculation_dtype(ctx),
        ),
    ),
    CalcType.L2_NORM_PR_UNIT: CalculationSpec(
        calculation_type=CalcType.L2_NORM_PR_UNIT,
        create=lambda ctx, deps: create_l2_norm_pr_unit_calc(ctx.canonical_groups),
    ),
    CalcType.STRUCTURED_UNIT_SUM: CalculationSpec(
        calculation_type=CalcType.STRUCTURED_UNIT_SUM,
        create=lambda ctx, deps: create_structured_unit_sum_calc(ctx.canonical_groups),
    ),
}
