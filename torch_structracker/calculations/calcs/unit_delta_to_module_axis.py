from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
import torch.nn as nn

from torch_structracker.calculations.base import CalcType
from torch_structracker.calculations.pipeline_calc import PipelineCalc
from torch_structracker.canonical_units import CanonicalUnitGroup
from torch_structracker.plans.mapping_plan import create_unit_delta_to_module_axis_plan


def create_unit_delta_to_module_axis_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    weighted_module_index: Mapping[nn.Module, int],
    device: torch.device | str,
    dtype: torch.dtype,
) -> PipelineCalc:
    plan = create_unit_delta_to_module_axis_plan(
        groups,
        weighted_module_index=weighted_module_index,
        device=device,
        dtype=dtype,
    )
    return PipelineCalc(plan, calculation_type=CalcType.UNIT_DELTA_TO_MODULE_AXIS)
