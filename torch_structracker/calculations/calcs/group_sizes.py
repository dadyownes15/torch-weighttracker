from __future__ import annotations

from collections.abc import Iterable

import torch

from torch_structracker.calculations.base import CalcType
from torch_structracker.calculations.static_calc import StaticCalc
from torch_structracker.canonical_units import CanonicalUnitGroup


class UnitPrGroup(StaticCalc):
    calculation_type = CalcType.GROUP_SIZES

    def __init__(self, group_sizes: torch.Tensor) -> None:
        super().__init__(group_sizes)


def create_group_sizes_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.long
) -> UnitPrGroup:
    group_sizes = torch.tensor(
        [group.length for group in groups],
        dtype=dtype,
        device=device
    )
    return UnitPrGroup(group_sizes)
