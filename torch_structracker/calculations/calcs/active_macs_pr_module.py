from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from torch_structracker.calculations.base import CalcType, Calculation
from torch_structracker.calculations.cached_calc import CachedCalculation
from torch_structracker.calculations.context import CalculationContext
from torch_structracker.canonical_units import UnitAxis


class ActiveMacsPrModuleCalc(Calculation):
    calculation_type = CalcType.ACTIVE_MACS_PR_MODULE
    required_calculations = (
        CalcType.UNIT_ACTIVE_MASK,
        CalcType.UNITS_TO_MODULE_AXIS,
        CalcType.BASELINE_MACS_PR_MODULE,
    )

    def forward(self) -> torch.Tensor:
        active_units = self.compute(CalcType.UNIT_ACTIVE_MASK)
        active_axes = self.compute(CalcType.UNITS_TO_MODULE_AXIS, active_units).view(-1, 2)
        ratios = torch.where(
            self.baseline_axes.gt(0),
            active_axes / self.compute(CalcType.BASELINE_MACS_PR_MODULE).clamp_min(1),
            torch.ones_like(active_axes),
        )
        return self.baseline_macs * ratios.prod(dim=1)


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
