from __future__ import annotations

from collections.abc import Iterable
from math import prod

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import BaseCalculation, CalcType
from torch_weighttracker.calculations.context import (
    calculation_device,
    calculation_dtype,
)
from torch_weighttracker.calculations.spec import CalculationSpec


class Block24SparsityCalc(BaseCalculation):
    calculation_type = CalcType.BLOCK_2_4_SPARSITY

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
            _module_2_4_block_sparsity_stats(
                module,
                device=self.device,
                dtype=self.dtype,
            )
            for module in self.weighted_modules
        ]
        if len(rows) == 0:
            return torch.empty((0, 4), device=self.device, dtype=self.dtype)

        return torch.stack(rows)


def create_block_2_4_sparsity_calc(
    modules: Iterable[nn.Module],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Block24SparsityCalc:
    """
    Returns strict, eligible, total, and tail 2:4 counts per weighted module.

    Output: 2D tensor with shape `(len(weighted_modules), 4)`, ordered as
    `(strict_blocks, eligible_blocks, total_blocks, tail_elements)` per module.
    Input: none.
    """
    return Block24SparsityCalc(
        modules,
        device=device,
        dtype=torch.float32 if dtype is None else dtype,
    )


def supports_2_4_block_sparsity(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.Linear,
            nn.MultiheadAttention,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
        ),
    )


def _module_2_4_block_sparsity_stats(
    module: nn.Module,
    *,
    device: torch.device | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    stats = [
        _tensor_2_4_block_sparsity_stats(
            weight,
            reduction_dim,
            device=device,
            dtype=dtype,
        )
        for weight, reduction_dim in _effective_weight_tensors(module)
    ]
    if len(stats) == 0:
        return torch.zeros((4,), device=device, dtype=dtype)

    return torch.stack(stats).sum(dim=0)


def _effective_weight_tensors(
    module: nn.Module,
) -> tuple[tuple[torch.Tensor, int], ...]:
    if isinstance(module, nn.MultiheadAttention):
        in_proj_weight = getattr(module, "in_proj_weight", None)
        if isinstance(in_proj_weight, torch.Tensor):
            return ((in_proj_weight, 1),)

        weights = tuple(
            (weight, 1)
            for weight in (
                getattr(module, "q_proj_weight", None),
                getattr(module, "k_proj_weight", None),
                getattr(module, "v_proj_weight", None),
            )
            if isinstance(weight, torch.Tensor)
        )
        return weights

    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return ()

    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return ((weight, 1),)

    return ()


def _tensor_2_4_block_sparsity_stats(
    weight: torch.Tensor,
    reduction_dim: int,
    *,
    device: torch.device | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    weight = weight.detach()
    reduction_dim = reduction_dim % weight.ndim
    reduction_size = int(weight.shape[reduction_dim])
    complete_size = reduction_size - (reduction_size % 4)
    outer_elements = prod(
        int(size) for dim, size in enumerate(weight.shape) if dim != reduction_dim
    )
    total_blocks = (complete_size // 4) * outer_elements
    tail_elements = (reduction_size - complete_size) * outer_elements

    if total_blocks == 0:
        return torch.tensor(
            (0, 0, total_blocks, tail_elements),
            device=device,
            dtype=dtype,
        )

    grouped = weight.movedim(reduction_dim, -1)[..., :complete_size]
    blocks = grouped.reshape(*grouped.shape[:-1], complete_size // 4, 4)
    zero_counts = blocks.eq(0).sum(dim=-1)
    strict_blocks = int(zero_counts.eq(2).sum().item())
    eligible_blocks = int(zero_counts.ge(2).sum().item())

    return torch.tensor(
        (strict_blocks, eligible_blocks, total_blocks, tail_elements),
        device=device,
        dtype=dtype,
    )


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.BLOCK_2_4_SPARSITY,
    requires_groups=False,
    create=lambda ctx, deps: create_block_2_4_sparsity_calc(
        ctx.weighted_modules,
        device=calculation_device(ctx),
        dtype=calculation_dtype(ctx),
    ),
)
