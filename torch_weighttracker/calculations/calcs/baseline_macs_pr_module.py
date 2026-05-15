from __future__ import annotations

import torch

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import (
    CalculationContext,
    calculation_device,
    calculation_dtype,
)
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.calculations.static_calc import StaticCalc


class BaselineMacsPrModuleCalc(StaticCalc):
    """
    Returns the initial runtime MAC count for each weighted module.

    Output: 1D tensor with length `len(weighted_modules)`.
    Input: none.
    """

    calculation_type = CalcType.BASELINE_MACS_PR_MODULE

    def __init__(self, baseline_macs: torch.Tensor) -> None:
        super().__init__(baseline_macs)


def create_baseline_macs_pr_module_calc(
    ctx: CalculationContext,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> BaselineMacsPrModuleCalc:
    if ctx.example_inputs is None:
        raise ValueError(
            "BASELINE_MACS_PR_MODULE requires example_inputs so fvcore can "
            "compute runtime MACs."
        )

    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as error:
        raise RuntimeError(
            "BASELINE_MACS_PR_MODULE requires fvcore. Install fvcore or disable "
            "structured BOPs MAC accounting."
        ) from error

    values = _fvcore_macs_by_weighted_module(ctx, FlopCountAnalysis)
    return BaselineMacsPrModuleCalc(
        torch.tensor(values, dtype=dtype, device=device)
    )


def _fvcore_macs_by_weighted_module(
    ctx: CalculationContext,
    flop_count_analysis,
) -> list[float]:
    names_by_module = {module: name for name, module in ctx.model.named_modules()}
    analysis = flop_count_analysis(ctx.model, ctx.example_inputs)
    if hasattr(analysis, "unsupported_ops_warnings"):
        analysis = analysis.unsupported_ops_warnings(False)
    if hasattr(analysis, "uncalled_modules_warnings"):
        analysis = analysis.uncalled_modules_warnings(False)

    by_module = analysis.by_module()
    uncalled = (
        set(analysis.uncalled_modules())
        if hasattr(analysis, "uncalled_modules")
        else set()
    )

    values: list[float] = []
    missing: list[str] = []
    for module in ctx.weighted_modules:
        name = names_by_module.get(module)
        if name is None or name not in by_module or name in uncalled:
            missing.append("<unnamed>" if name is None else name)
            continue
        values.append(float(by_module[name]))

    if missing:
        raise ValueError(
            "fvcore did not report MACs for weighted modules: "
            + ", ".join(missing)
        )

    return values


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.BASELINE_MACS_PR_MODULE,
    requires_groups=False,
    cache_constant=True,
    create=lambda ctx, deps: create_baseline_macs_pr_module_calc(
        ctx,
        device=calculation_device(ctx),
        dtype=calculation_dtype(ctx),
    ),
)
