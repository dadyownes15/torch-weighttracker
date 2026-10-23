from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import (
    calculation_device,
    calculation_dtype,
)
from torch_weighttracker.calculations.reduction_calc import ReductionCalc
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.extractors.bitwidth_extractor import (
    BitwidthFn,
    ModuleBitwidthExtractor,
)
from torch_weighttracker.reductions.builder import (
    FullSelection,
    ReductionMapping,
    ReductionPlan,
    ReductionPlanBuilder,
    ReductionRecord,
)
from torch_weighttracker.reductions.compiler import create_module_plan
from torch_weighttracker.reductions.ops import IdentityTensorReduction, ReductionOp


class ModuleBitwidthRule:
    def __init__(self, extractor: ModuleBitwidthExtractor | None = None) -> None:
        self.extractor = ModuleBitwidthExtractor() if extractor is None else extractor
        self.reduction = IdentityTensorReduction()

    def emit(
        self,
        element: nn.Module,
        builder: ReductionPlanBuilder,
    ) -> Iterable[ReductionRecord]:
        bitwidth_ref = self.extractor.bind(element)
        if bitwidth_ref is None:
            return ()

        op = ReductionOp(
            source_ref=bitwidth_ref,
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
    bitwidth_fn: BitwidthFn | None = None,
    module_names: Iterable[str] = (),
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> ReductionCalc:
    """
    Returns the current activation and weight bitwidth for each weighted module.

    Bitwidths come from `bitwidth_fn(name, module)` and are re-read on every
    call; without it every module is `(32, 32)`.

    Output: 1D tensor with length `2 * len(weighted_modules)`, ordered as
    `(activation_bits, weight_bits)` for each module.
    Input: none.
    """
    modules = tuple(modules)
    module_names = tuple(module_names)
    extractor = ModuleBitwidthExtractor(
        bitwidth_fn=bitwidth_fn,
        module_names=(
            dict(zip(modules, module_names, strict=True)) if module_names else None
        ),
        device=device,
        dtype=dtype,
    )
    return ReductionCalc(
        _build_bitwidths_plan(modules, extractor=extractor),
        calculation_type=CalcType.BITRATE_PR_MODULE,
    )


def _build_bitwidths_plan(
    modules: Iterable[nn.Module],
    *,
    extractor: ModuleBitwidthExtractor | None = None,
) -> ReductionPlan:
    return create_module_plan(list(modules), ModuleBitwidthRule(extractor))


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.BITRATE_PR_MODULE,
    requires_groups=False,
    create=lambda ctx, deps: create_bitrate_pr_module_calc(
        ctx.weighted_modules,
        bitwidth_fn=ctx.bitwidth_fn,
        module_names=ctx.weighted_module_names,
        device=calculation_device(ctx),
        dtype=calculation_dtype(ctx),
    ),
)
