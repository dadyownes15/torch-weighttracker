from __future__ import annotations

from collections.abc import Iterable

import torch

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import calculation_device, calculation_dtype
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.calculations.static_calc import StaticCalc
from torch_weighttracker.canonical_units import CanonicalUnitGroup


class InitialUnitCountPrGroup(StaticCalc):
    """
    Returns the initial count of canonical units for each canonical group.

    Output: 1D tensor with length `len(canonical_groups)`.
    Input: none.
    """

    calculation_type = CalcType.INIT_UNIT_PR_GROUP_COUNT

    def __init__(
        self,
        initial_unit_count_pr_group: torch.Tensor,
    ) -> None:
        super().__init__(initial_unit_count_pr_group)


def create_baseline_group_sizes_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> InitialUnitCountPrGroup:
    group_lengths = [group.length for group in groups]
    return InitialUnitCountPrGroup(
        torch.tensor(group_lengths, device=device, dtype=dtype)
    )


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.INIT_UNIT_PR_GROUP_COUNT,
    cache_constant=True,
    create=lambda ctx, deps: create_baseline_group_sizes_calc(
        ctx.canonical_groups,
        device=calculation_device(ctx),
        dtype=calculation_dtype(ctx),
    ),
)
