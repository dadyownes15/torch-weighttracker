from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import (
    calculation_device,
    calculation_dtype,
)
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.calculations.static_calc import StaticCalc
from torch_weighttracker.canonical_units import CanonicalUnitGroup, UnitAxis


class PrunableBiasParamPrUnitPrGroup(StaticCalc):
    calculation_type = CalcType.PRUNABLE_BIAS_PARAM_PR_UNIT_PR_GROUP


def create_prunable_bias_param_pr_unit_pr_group_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> PrunableBiasParamPrUnitPrGroup:
    canonical_groups = tuple(groups)
    values = torch.zeros(len(canonical_groups), device=device, dtype=dtype)

    for group in canonical_groups:
        seen_modules: set[int] = set()
        for member in group.members:
            module = member.module
            if not isinstance(module, nn.Linear):
                continue
            if member.unit_axis == UnitAxis.IN_CHANNEL:
                continue
            if id(module) in seen_modules:
                continue

            bias = getattr(module, "bias", None)
            if not isinstance(bias, torch.Tensor):
                continue
            if int(bias.numel()) % int(group.length) != 0:
                raise ValueError(
                    "Prunable Linear bias size must be divisible by the canonical "
                    "group length."
                )
            seen_modules.add(id(module))
            values[int(group.group_id)] += float(bias.numel()) / float(group.length)

    return PrunableBiasParamPrUnitPrGroup(values)


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.PRUNABLE_BIAS_PARAM_PR_UNIT_PR_GROUP,
    cache_constant=True,
    create=lambda ctx, deps: create_prunable_bias_param_pr_unit_pr_group_calc(
        ctx.canonical_groups,
        device=calculation_device(ctx),
        dtype=calculation_dtype(ctx),
    ),
)
