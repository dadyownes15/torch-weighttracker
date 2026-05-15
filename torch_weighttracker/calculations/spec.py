from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import CalculationContext


@dataclass(frozen=True)
class CalculationSpec:
    calculation_type: CalcType
    create: Callable[[CalculationContext, Mapping[CalcType, nn.Module]], nn.Module]
    required_calculations: tuple[CalcType, ...] = ()
    requires_groups: bool = True
    cache_constant: bool = False
