from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch
import torch.nn as nn

from torch_weighttracker.canonical_units import CanonicalUnitGroup


@dataclass(frozen=True)
class CalculationContext:
    model: nn.Module
    canonical_groups: tuple[CanonicalUnitGroup, ...]
    device: torch.device | str | None
    dtype: torch.dtype | None
    weighted_modules: tuple[nn.Module, ...]
    weighted_module_index: Mapping[nn.Module, int]
    example_inputs: object | None = None


def calculation_dtype(ctx: CalculationContext) -> torch.dtype:
    if ctx.dtype is not None:
        return ctx.dtype

    for parameter in ctx.model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype

    return torch.float32


def calculation_device(ctx: CalculationContext) -> torch.device | str:
    if ctx.device is not None:
        return ctx.device

    for parameter in ctx.model.parameters():
        return parameter.device

    return torch.device("cpu")


def group_input_spec(ctx: CalculationContext) -> "TensorSpec":
    from torch_weighttracker.extractors.extractor import TensorSpec

    return TensorSpec(
        shape=torch.Size([len(ctx.canonical_groups)]),
        dtype=calculation_dtype(ctx),
        device=torch.device(calculation_device(ctx)),
    )
