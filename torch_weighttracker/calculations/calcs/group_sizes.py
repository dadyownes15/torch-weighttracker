from __future__ import annotations

from collections.abc import Iterable

import torch

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import calculation_device
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.calculations.static_calc import StaticCalc
from torch_weighttracker.canonical_units import CanonicalUnitGroup


class UnitPrGroup(StaticCalc):
    """
    Returns the number of canonical units in each canonical group.

    Output: 1D integer tensor with length `len(canonical_groups)`.
    Input: none.
    """

    calculation_type = CalcType.GROUP_SIZES

    def __init__(self, group_sizes: torch.Tensor) -> None:
        super().__init__(group_sizes)


def create_group_sizes_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.long,
) -> UnitPrGroup:
    group_sizes = torch.tensor(
        [group.length for group in groups],
        dtype=dtype,
        device=device,
    )
    return UnitPrGroup(group_sizes)


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.GROUP_SIZES,
    cache_constant=True,
    create=lambda ctx, deps: create_group_sizes_calc(
        ctx.canonical_groups,
        device=calculation_device(ctx),
    ),
)
