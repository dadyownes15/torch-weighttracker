from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import BaseCalculation, CalcType
from torch_weighttracker.calculations.context import (
    calculation_device,
    calculation_dtype,
)
from torch_weighttracker.calculations.spec import CalculationSpec


class UnstructuredSparsityPrModuleCalc(BaseCalculation):
    calculation_type = CalcType.UNSTRUCTURED_SPARSITY_PR_MODULE

    def __init__(
        self,
        modules: Iterable[nn.Module],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.weighted_modules = tuple(modules)
        self.device = None if device is None else torch.device(device)
        self.dtype = dtype

    def forward(self) -> torch.Tensor:
        rows = [
            _module_zero_and_total(module, device=self.device, dtype=self.dtype)
            for module in self.weighted_modules
        ]
        if len(rows) == 0:
            return torch.empty((0, 2), device=self.device, dtype=self.dtype)

        return torch.stack(rows)


def create_unstructured_sparsity_pr_module_calc(
    modules: Iterable[nn.Module],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> UnstructuredSparsityPrModuleCalc:
    """
    Returns zero and total effective weight counts for each weighted module.

    Output: 2D tensor with shape `(len(weighted_modules), 2)`, ordered as
    `(zero_weight_elements, weight_elements)` for each module.
    Input: none.
    """
    return UnstructuredSparsityPrModuleCalc(
        modules,
        device=device,
        dtype=torch.float32 if dtype is None else dtype,
    )


def _module_zero_and_total(
    module: nn.Module,
    *,
    device: torch.device | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = _effective_weight_tensors(module)
    zero_count = sum(int(weight.detach().eq(0).sum().item()) for weight in weights)
    total_count = sum(int(weight.numel()) for weight in weights)
    return torch.tensor((zero_count, total_count), device=device, dtype=dtype)


def _effective_weight_tensors(module: nn.Module) -> tuple[torch.Tensor, ...]:
    if isinstance(module, nn.MultiheadAttention):
        in_proj_weight = getattr(module, "in_proj_weight", None)
        if isinstance(in_proj_weight, torch.Tensor):
            return (in_proj_weight,)

        weights = tuple(
            weight
            for weight in (
                getattr(module, "q_proj_weight", None),
                getattr(module, "k_proj_weight", None),
                getattr(module, "v_proj_weight", None),
            )
            if isinstance(weight, torch.Tensor)
        )
        if weights:
            return weights

    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return (weight,)

    return ()


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.UNSTRUCTURED_SPARSITY_PR_MODULE,
    requires_groups=False,
    create=lambda ctx, deps: create_unstructured_sparsity_pr_module_calc(
        ctx.weighted_modules,
        device=calculation_device(ctx),
        dtype=calculation_dtype(ctx),
    ),
)
