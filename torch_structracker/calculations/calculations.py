from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_structracker.calculations.base import CalcType, Calculation
from torch_structracker.calculations.cached_calc import CachedCalculation
from torch_structracker.calculations.pipeline_calc import PipelineCalc
from torch_structracker.calculations.reduction_calc import ReductionCalc
from torch_structracker.canonical_units import CanonicalUnitGroup, UnitAxis
from torch_structracker.extractors.codeq_bitrate_extractor import ModuleBitrateExtractor
from torch_structracker.plans.bitrate_plan import create_codeq_bitrates
from torch_structracker.plans.mapping_plan import (
    create_units_to_group_plan,
    create_units_to_module_axis_plan,
)
from torch_structracker.plans.unit_weight_operation_plan import (
    count_group_units,
    create_active_units_plan,
    create_group_change_effect_plan,
    create_l2_norm_pr_unit_plan,
    create_structured_unit_sum_plan,
)
from torch_structracker.reductions.builder import (
    MappedReductionPlan,
    PipelinePlan,
)


@dataclass(frozen=True)
class CalculationSpec:
    calculation_type: CalcType
    create: Callable[[CalculationContext, Mapping[CalcType, nn.Module]], nn.Module]
    required_calculations: tuple[CalcType, ...] = ()
    requires_groups: bool = True
    cache_constant: bool = False


@dataclass(frozen=True)
class CalculationContext:
    model: nn.Module
    canonical_groups: tuple[CanonicalUnitGroup, ...]
    device: torch.device | str | None
    dtype: torch.dtype | None
    weighted_modules: tuple[nn.Module, ...]
    weighted_module_index: Mapping[nn.Module, int]
    example_inputs: object | None = None


class UnitActiveMaskCalc(Calculation):
    calculation_type = CalcType.UNIT_ACTIVE_MASK
    required_calculations = (CalcType.ACTIVE_UNITS,)

    def __init__(self, active_units: nn.Module) -> None:
        super().__init__({CalcType.ACTIVE_UNITS: active_units})

    def forward(self) -> torch.Tensor:
        raw = self.compute(CalcType.ACTIVE_UNITS)
        return raw.gt(0).to(dtype=raw.dtype)


class BaselineGroupSizesCalc(Calculation):
    calculation_type = CalcType.BASELINE_GROUP_SIZES
    required_calculations = (CalcType.UNITS_TO_GROUP,)
    cache_constant = True

    def __init__(
        self,
        *,
        units_to_group: nn.Module,
        unit_ones: torch.Tensor,
    ) -> None:
        super().__init__({CalcType.UNITS_TO_GROUP: units_to_group})
        self.register_buffer("unit_ones", unit_ones, persistent=False)

    def forward(self) -> torch.Tensor:
        return self.compute(CalcType.UNITS_TO_GROUP, self.unit_ones)


class GroupSizesCalc(Calculation):
    calculation_type = CalcType.GROUP_SIZES
    cache_constant = True

    def __init__(self, group_sizes: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("group_sizes", group_sizes, persistent=False)

    def forward(self) -> torch.Tensor:
        return self.group_sizes


class ActiveMacsPrModuleCalc(Calculation):
    calculation_type = CalcType.ACTIVE_MACS_PR_MODULE
    required_calculations = (
        CalcType.UNIT_ACTIVE_MASK,
        CalcType.UNITS_TO_MODULE_AXIS,
        CalcType.BASELINE_MACS_PR_MODULE,
        CalcType.BASELINE_MACS_PR_MODULE
    )
    
    def forward(self) -> torch.Tensor:
        active_units = self.compute(CalcType.UNIT_ACTIVE_MASK)
        active_axes = self.compute(CalcType.UNITS_TO_MODULE_AXIS, active_units).view(-1, 2)
        ratios = torch.where(
            self.baseline_axes.gt(0),
            active_axes / self.baseline_axes.clamp_min(1),
            torch.ones_like(active_axes),
        )
        return self.baseline_macs * ratios.prod(dim=1)


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


def create_active_units_calc(
    groups: Iterable[CanonicalUnitGroup],
) -> ReductionCalc:
    return ReductionCalc(
        create_active_units_plan(groups),
        calculation_type=CalcType.ACTIVE_UNITS,
    )


def create_units_to_group_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PipelineCalc:
    return PipelineCalc(
        create_units_to_group_plan(groups, device=device, dtype=dtype),
        calculation_type=CalcType.UNITS_TO_GROUP,
    )


def create_unit_active_mask_calc(active_units: nn.Module) -> UnitActiveMaskCalc:
    return UnitActiveMaskCalc(active_units=active_units)


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
    return BaselineGroupSizesCalc(
        units_to_group=units_to_group,
        unit_ones=unit_ones,
    )


def create_group_sizes_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device | str | None = None,
) -> GroupSizesCalc:
    group_sizes = torch.tensor(
        [group.length for group in groups],
        dtype=torch.long,
        device=torch.device("cpu") if device is None else torch.device(device),
    )
    return GroupSizesCalc(group_sizes)


def create_group_change_effect_calc(
    groups: Iterable[CanonicalUnitGroup],
) -> ReductionCalc:
    return ReductionCalc(
        create_group_change_effect_plan(groups),
        calculation_type=CalcType.GROUP_CHANGE_EFFECT,
    )


def create_units_to_module_axis_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    weighted_module_index: Mapping[nn.Module, int],
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PipelineCalc:
    plan = create_units_to_module_axis_plan(
        groups,
        weighted_module_index=weighted_module_index,
        device=device,
        dtype=dtype,
    )
    return PipelineCalc(plan, calculation_type=CalcType.UNITS_TO_MODULE_AXIS)


def create_active_macs_pr_module_calc(
    ctx: CalculationContext,
    *,
    unit_active_mask: Calculation,
    units_to_module_axis: Calculation,
    baseline_macs: CachedCalculation,
    baseline_axes: CachedCalculation,
    represented_axes: CachedCalculation,
) -> ActiveMacsPrModuleCalc:
    device = torch.device("cpu") if ctx.device is None else torch.device(ctx.device)
    dtype = _calculation_dtype(ctx)
    baseline_macs = _baseline_macs_by_weighted_module(ctx, device=device, dtype=dtype)
    baseline_axes = torch.tensor(
        [_module_axis_sizes(module) for module in ctx.weighted_modules],
        dtype=dtype,
        device=device,
    )
    represented_axes = torch.zeros(
        (len(ctx.weighted_modules), 2),
        dtype=torch.bool,
        device=device,
    )
    for group in ctx.canonical_groups:
        for member in group.members:
            module_index = ctx.weighted_module_index.get(member.module)
            if module_index is None:
                continue
            axis = 0 if member.unit_axis == UnitAxis.IN_CHANNEL else 1
            represented_axes[module_index, axis] = True

    return ActiveMacsPrModuleCalc(
        unit_active_mask=unit_active_mask,
        units_to_module_axis=units_to_module_axis,
        baseline_macs=baseline_macs,
        baseline_axes=baseline_axes,
        represented_axes=represented_axes,
    )


def create_bitrate_pr_module_calc(
    modules: Iterable[nn.Module],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> ReductionCalc:
    extractor = ModuleBitrateExtractor(device=device, dtype=dtype)
    return ReductionCalc(
        create_codeq_bitrates(modules, extractor=extractor),
        calculation_type=CalcType.BITRATE_PR_MODULE,
    )


def _baseline_macs_by_weighted_module(
    ctx: CalculationContext,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if ctx.example_inputs is None:
        values = [_dense_weight_macs(module) for module in ctx.weighted_modules]
    else:
        names_by_module = {module: name for name, module in ctx.model.named_modules()}
        analysis = fvcore.nn.FlopCountAnalysis(ctx.model, ctx.example_inputs)
        by_module = analysis.by_module()
        values = [
            float(by_module.get(names_by_module.get(module, ""), 0.0))
            for module in ctx.weighted_modules
        ]

    return torch.tensor(values, dtype=dtype, device=device)


def _module_axis_sizes(module: nn.Module) -> tuple[float, float]:
    if isinstance(module, nn.Conv2d):
        return float(module.in_channels), float(module.out_channels)
    if isinstance(module, nn.Linear):
        return float(module.in_features), float(module.out_features)
    if isinstance(module, nn.MultiheadAttention):
        return float(module.embed_dim), float(module.embed_dim)

    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        if weight.ndim >= 2:
            return float(weight.shape[1]), float(weight.shape[0])
        if weight.ndim == 1:
            return float(weight.shape[0]), float(weight.shape[0])

    return 1.0, 1.0


def _dense_weight_macs(module: nn.Module) -> float:
    if isinstance(module, nn.Conv2d):
        return float(module.weight.numel())
    if isinstance(module, nn.Linear):
        return float(module.weight.numel())
    if isinstance(module, nn.MultiheadAttention):
        total = 0
        for name in (
            "in_proj_weight",
            "q_proj_weight",
            "k_proj_weight",
            "v_proj_weight",
        ):
            value = getattr(module, name, None)
            if isinstance(value, torch.Tensor):
                total += int(value.numel())
        out_proj = getattr(module, "out_proj", None)
        if out_proj is not None and isinstance(
            getattr(out_proj, "weight", None),
            torch.Tensor,
        ):
            total += int(out_proj.weight.numel())
        return float(total)

    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return float(weight.numel())
    return 0.0


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
    CalcType.ACTIVE_MACS_PR_MODULE: CalculationSpec(
        calculation_type=CalcType.ACTIVE_MACS_PR_MODULE,
        required_calculations=ActiveMacsPrModuleCalc.required_calculations,
        create=lambda ctx, deps: ActiveMacsPrModuleCalc(
            deps,
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
        create=lambda ctx, deps: ReductionCalc(
            create_l2_norm_pr_unit_plan(ctx.canonical_groups),
            calculation_type=CalcType.L2_NORM_PR_UNIT,
        ),
    ),
    CalcType.STRUCTURED_UNIT_SUM: CalculationSpec(
        calculation_type=CalcType.STRUCTURED_UNIT_SUM,
        create=lambda ctx, deps: ReductionCalc(
            create_structured_unit_sum_plan(ctx.canonical_groups),
            calculation_type=CalcType.STRUCTURED_UNIT_SUM,
        ),
    ),
}
