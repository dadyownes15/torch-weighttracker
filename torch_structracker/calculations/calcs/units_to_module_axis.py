from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn

from torch_structracker.calculations.base import CalcType
from torch_structracker.calculations.pipeline_calc import PipelineCalc
from torch_structracker.canonical_units import CanonicalUnitGroup
from torch_structracker.plans.mapping_plan import create_units_to_module_axis_plan


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
