from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from torch_structracker.calculations.base import CalcType
from torch_structracker.calculations.reduction_calc import ReductionCalc
from torch_structracker.extractors.codeq_bitrate_extractor import ModuleBitrateExtractor
from torch_structracker.plans.bitrate_plan import create_codeq_bitrates


def create_bitrate_pr_module_calc(
    modules: Iterable[nn.Module],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> ReductionCalc:
    extractor = ModuleBitrateExtractor(device=device, dtype=dtype)
    return ReductionCalc(
        create_codeq_bitrates(modules, extractor=extractor),
        calculation_type=CalcType.BITRATE_PR_MODULE,
    )
