from __future__ import annotations

import torch

from torch_structracker.calculations.base import CalcType
from torch_structracker.calculations.context import CalculationContext
from torch_structracker.calculations.static_calc import StaticCalc


class BaselineMacsPrModuleCalc(StaticCalc):
    required_calculations = ()
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
