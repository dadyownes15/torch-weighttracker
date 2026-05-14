from torch_structracker.calculations.base import CalcType
from torch_structracker.calculations.static_calc import StaticCalc

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


class BaselineMacsPrModule(StaticCalc):
    required_calculations = ()
    calculation_type = CalcType.BASELINE_MACS_PR_MODULE

    def __init__(self, baseline_pr_module: torch.Tensor) -> None:
        super().__init__(
            baseline_pr_module)


def create_baseline_macs_pr_module(
    tracked_modules: list[nn.Module],
    example_input: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype = torch.int,
) -> BaselineMacsPrModule:
    baseline_pr_module = _baseline_macs_by_weighted_module(
        tracked_modules,
        model,
        example_input,
        device,
        dtype,
    )
    return BaselineMacsPrModule(baseline_pr_module)


def _baseline_macs_by_weighted_module(
    tracked_modules: list[nn.Module],
    model: nn.Module,
    example_input: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.int,
) -> torch.Tensor:
    analysis = FlopCountAnalysis(model, example_input)
    by_module = analysis.by_module()

    # Only store the ones we track
    values = [
        float(by_module.get(module, 0.0))
        for module in tracked_modules
    ]

    return torch.tensor(values, dtype=dtype, device=device)
