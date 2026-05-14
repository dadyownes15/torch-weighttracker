from __future__ import annotations

from torch_structracker.calculations.base import CalcType, Calculation
from torch_structracker.calculations.calcs import (
    ActiveMacsPrModuleCalc,
    BaselineGroupSizesCalc,
    UnitPrGroup,
    UnitActiveMaskCalc,
    create_active_macs_pr_module_calc,
    create_active_units_calc,
    create_baseline_group_sizes_calc,
    create_bitrate_pr_module_calc,
    create_group_change_effect_calc,
    create_group_sizes_calc,
    create_unit_active_mask_calc,
    create_units_to_group_calc,
    create_units_to_module_axis_calc,
)
from torch_structracker.calculations.context import CalculationContext
from torch_structracker.calculations.pipeline_calc import PipelineCalc
from torch_structracker.calculations.reduction_calc import ReductionCalc
from torch_structracker.calculations.registry import CALCULATION_SPECS
from torch_structracker.calculations.spec import CalculationSpec
from torch_structracker.reductions.builder import MappedReductionPlan, PipelinePlan


def create_calculation(
    calculation_type: CalcType | str,
    plan: MappedReductionPlan,
) -> ReductionCalc:
    return ReductionCalc(plan, calculation_type=calculation_type)


def create_pipeline_calculation(
    plan: PipelinePlan,
    *,
    calculation_type: CalcType | str | None = None,
) -> PipelineCalc:
    return PipelineCalc(plan, calculation_type=calculation_type)


__all__ = [
    "ActiveMacsPrModuleCalc",
    "BaselineGroupSizesCalc",
    "CALCULATION_SPECS",
    "CalcType",
    "Calculation",
    "CalculationContext",
    "CalculationSpec",
    "UnitPrGroup",
    "UnitActiveMaskCalc",
    "create_active_macs_pr_module_calc",
    "create_active_units_calc",
    "create_baseline_group_sizes_calc",
    "create_bitrate_pr_module_calc",
    "create_calculation",
    "create_group_change_effect_calc",
    "create_group_sizes_calc",
    "create_pipeline_calculation",
    "create_unit_active_mask_calc",
    "create_units_to_group_calc",
    "create_units_to_module_axis_calc",
]
