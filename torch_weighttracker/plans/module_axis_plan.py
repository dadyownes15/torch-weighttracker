from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_weighttracker.calculations.context import (
    CalculationContext,
    calculation_device,
    calculation_dtype,
)
from torch_weighttracker.canonical_units import CanonicalMember, UnitAxis


@dataclass(frozen=True)
class ModuleAxisPlan:
    baseline_axes: torch.Tensor


def create_module_axis_plan(ctx: CalculationContext) -> ModuleAxisPlan:
    dtype = calculation_dtype(ctx)
    device = torch.device(calculation_device(ctx))
    baseline_values: list[float] = []

    for module in ctx.weighted_modules:
        axis_sizes = module_axis_sizes(module)
        for value in axis_sizes:
            baseline_values.append(float(value))

    return ModuleAxisPlan(
        baseline_axes=torch.tensor(baseline_values, dtype=dtype, device=device),
    )


def module_axis_sizes(module: nn.Module) -> tuple[float, float]:
    if isinstance(module, nn.Conv2d):
        return _conv_axis_sizes(module)

    if isinstance(module, nn.Linear):
        return float(module.in_features), float(module.out_features)

    if isinstance(module, nn.MultiheadAttention):
        raise ValueError(
            "ACTIVE_MACS_PR_MODULE does not support nn.MultiheadAttention parent "
            "modules in V1. Use projection Linear modules or add explicit MHA "
            "operation terms."
        )

    feature_size = _feature_axis_size(module)
    if feature_size is not None:
        return -1.0, float(feature_size)

    raise ValueError(
        "Baseline module-axis sizes are not implemented for "
        f"{module.__class__.__name__}."
    )


def module_axis_for_member(member: CanonicalMember) -> int:
    return 0 if member.unit_axis == UnitAxis.IN_CHANNEL else 1


def _conv_axis_sizes(module: nn.Conv2d) -> tuple[float, float]:
    if module.groups != 1:
        raise ValueError(
            "ACTIVE_MACS_PR_MODULE currently supports only Conv2d(groups=1)."
        )
    return float(module.in_channels), float(module.out_channels)


def _feature_axis_size(module: nn.Module) -> int | None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return int(module.num_features)

    if isinstance(module, nn.LayerNorm):
        return _layernorm_feature_dim(module)

    if _is_rmsnorm_module(module):
        return _rmsnorm_feature_dim(module)

    if isinstance(module, nn.GroupNorm):
        return int(module.num_channels)

    if isinstance(module, nn.modules.instancenorm._InstanceNorm):
        return int(module.num_features)

    if isinstance(module, nn.Embedding):
        return int(module.embedding_dim)

    return None


def _layernorm_feature_dim(module: nn.LayerNorm) -> int:
    normalized_shape = tuple(module.normalized_shape)
    if len(normalized_shape) == 0:
        raise ValueError("LayerNorm normalized_shape must not be empty.")
    return int(normalized_shape[-1])


def _is_rmsnorm_module(module: nn.Module) -> bool:
    return module.__class__.__name__.endswith("RMSNorm") and isinstance(
        getattr(module, "weight", None),
        torch.Tensor,
    )


def _rmsnorm_feature_dim(module: nn.Module) -> int:
    weight = module.weight
    if weight.ndim == 0:
        raise ValueError("RMSNorm weight must not be scalar.")
    return int(weight.shape[-1])
