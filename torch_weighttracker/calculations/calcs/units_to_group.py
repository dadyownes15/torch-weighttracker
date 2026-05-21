from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import calculation_device, calculation_dtype
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.canonical_units import CanonicalUnitGroup


class UnitsToGroup(nn.Module):
    calculation_type = CalcType.UNITS_TO_GROUP

    def __init__(
        self,
        groups: Iterable[CanonicalUnitGroup],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        canonical_groups = tuple(groups)
        device = torch.device("cpu") if device is None else torch.device(device)
        dtype = torch.float32 if dtype is None else dtype

        group_indices = [
            int(group.group_id)
            for group in canonical_groups
            for _ in range(int(group.length))
        ]
        self.output_length = (
            0 if len(canonical_groups) == 0 else max(group_indices, default=-1) + 1
        )
        self.output_shape = torch.Size([self.output_length])

        self.register_buffer(
            "group_indices",
            torch.tensor(group_indices, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "output_anchor",
            torch.empty((), dtype=dtype, device=device),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.output_anchor.new_zeros(self.output_shape)
        values = x.reshape(-1)
        if self.group_indices.numel() == 0:
            return out
        return out.index_add_(0, self.group_indices, values)


def create_units_to_group_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> UnitsToGroup:
    """
    Returns one value per canonical group by summing the input values for the
    canonical units that belong to that group.

    Output: 1D tensor with length `len(canonical_groups)`.
    Input: 1D tensor with length equal to the total canonical unit count. Element
    `i` is the value for canonical unit `i`.
    """
    return UnitsToGroup(groups, device=device, dtype=dtype)


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.UNITS_TO_GROUP,
    create=lambda ctx, deps: create_units_to_group_calc(
        ctx.canonical_groups,
        device=calculation_device(ctx),
        dtype=calculation_dtype(ctx),
    ),
)
