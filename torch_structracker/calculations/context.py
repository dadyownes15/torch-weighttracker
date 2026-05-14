from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

import torch
import torch.nn as nn

from torch_structracker.canonical_units import CanonicalUnitGroup


@dataclass(frozen=True)
class CalculationContext:
    model: nn.Module
    canonical_groups: tuple[CanonicalUnitGroup, ...]
    device: torch.device | str | None
    dtype: torch.dtype | None
    weighted_modules: tuple[nn.Module, ...]
    weighted_module_index: Mapping[nn.Module, int]
    example_inputs: object | None = None
