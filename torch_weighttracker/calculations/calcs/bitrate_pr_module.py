from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import calculation_device, calculation_dtype
from torch_weighttracker.calculations.reduction_calc import ReductionCalc
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.extractors.codeq_bitrate_extractor import ModuleBitrateExtractor
from torch_weighttracker.reductions.builder import (
    FullSelection,
    ReductionMapping,
    ReductionPlan,
    ReductionPlanBuilder,
    ReductionRecord,
)
from torch_weighttracker.reductions.compiler import create_module_plan
from torch_weighttracker.reductions.ops import IdentityTensorReduction, ReductionOp


class CodeQBitrateRule:
    def __init__(self, extractor: ModuleBitrateExtractor | None = None) -> None:
        self.extractor = ModuleBitrateExtractor() if extractor is None else extractor
        self.reduction = IdentityTensorReduction()

    def emit(
        self,
        element: nn.Module,
        builder: ReductionPlanBuilder,
    ) -> Iterable[ReductionRecord]:
        bitrate_tensor = self.extractor.bind(element)
        if bitrate_tensor is None:
            return ()

        op = ReductionOp(
            source_ref=bitrate_tensor,
            reduction=self.reduction,
        )
        target = builder.reserve_segment(op.output_length)

        return (
            ReductionRecord(
                op=op,
                mapping=ReductionMapping(
                    source=FullSelection(),
                    target=target,
                ),
            ),
        )


def create_bitrate_pr_module_calc(
    modules: Iterable[nn.Module],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> ReductionCalc:
    """
    Returns the activation and weight bitrate for each weighted module.

    Output: 1D tensor with length `2 * len(weighted_modules)`, ordered as
    `(activation_bitrate, weight_bitrate)` for each module.
    Input: none.
    """
    extractor = ModuleBitrateExtractor(device=device, dtype=dtype)
    return ReductionCalc(
        _build_codeq_bitrates_plan(modules, extractor=extractor),
        calculation_type=CalcType.BITRATE_PR_MODULE,
    )


def _build_codeq_bitrates_plan(
    modules: Iterable[nn.Module],
    *,
    extractor: ModuleBitrateExtractor | None = None,
) -> ReductionPlan:
    return create_module_plan(list(modules), CodeQBitrateRule(extractor))


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.BITRATE_PR_MODULE,
    requires_groups=False,
    create=lambda ctx, deps: create_bitrate_pr_module_calc(
        ctx.weighted_modules,
        device=calculation_device(ctx),
        dtype=calculation_dtype(ctx),
    ),
)
