from __future__ import annotations

from collections.abc import Iterable

import torch

from torch_structracker.calculations.base import CalcType
from torch_structracker.calculations.pipeline_calc import PipelineCalc
from torch_structracker.canonical_units import CanonicalUnitGroup
from torch_structracker.plans.mapping_plan import create_units_to_group_plan


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
