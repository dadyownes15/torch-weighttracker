from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import calculation_device, calculation_dtype
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.calculations.static_calc import StaticCalc


class BaselineModuleAxesCalc(StaticCalc):
    """
    Returns the initial input-axis and output-axis size for each weighted module.

    Output: 2D tensor with shape `(len(weighted_modules), 2)`.
    Input: none.
    """

    calculation_type = CalcType.BASELINE_MODULE_AXES

    def __init__(self, baseline_axes: torch.Tensor) -> None:
        super().__init__(baseline_axes)


def create_baseline_module_axes_calc(
    modules: Iterable[nn.Module],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> BaselineModuleAxesCalc:
    baseline_axes = torch.tensor(
        [_module_axis_sizes(module) for module in modules],
        dtype=dtype,
        device=device,
    )
    return BaselineModuleAxesCalc(baseline_axes)


def _module_axis_sizes(module: nn.Module) -> tuple[float, float]:
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

    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return float(module.num_features), float(module.num_features)

    if isinstance(module, nn.LayerNorm):
        feature_dim = _layernorm_feature_dim(module)
        return float(feature_dim), float(feature_dim)

    if isinstance(module, nn.GroupNorm):
        return float(module.num_channels), float(module.num_channels)

    if isinstance(module, nn.modules.instancenorm._InstanceNorm):
        return float(module.num_features), float(module.num_features)

    if isinstance(module, nn.Embedding):
        return float(module.embedding_dim), float(module.embedding_dim)

    raise ValueError(
        "Baseline module-axis sizes are not implemented for "
        f"{module.__class__.__name__}."
    )


def _conv_axis_sizes(module: nn.Conv2d) -> tuple[float, float]:
    if module.groups != 1:
        raise ValueError(
            "ACTIVE_MACS_PR_MODULE currently supports only Conv2d(groups=1)."
        )
    return float(module.in_channels), float(module.out_channels)


def _layernorm_feature_dim(module: nn.LayerNorm) -> int:
    normalized_shape = tuple(module.normalized_shape)
    if len(normalized_shape) == 0:
        raise ValueError("LayerNorm normalized_shape must not be empty.")
    return int(normalized_shape[-1])


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.BASELINE_MODULE_AXES,
    requires_groups=False,
    cache_constant=True,
    create=lambda ctx, deps: create_baseline_module_axes_calc(
        ctx.weighted_modules,
        device=calculation_device(ctx),
        dtype=calculation_dtype(ctx),
    ),
)
