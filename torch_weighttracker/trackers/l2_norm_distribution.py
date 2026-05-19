from __future__ import annotations

from collections.abc import Iterable

import torch

from torch_weighttracker.calculations import CalcType, CalculationContext
from torch_weighttracker.consumer_ignore import (
    ConsumerFilter,
    FilterItem,
    filter_canonical_members,
)
from torch_weighttracker.trackers.base import BaseTracker
from torch_weighttracker.trackers.group_names import group_names


class L2NormDistribution(BaseTracker):
    required_calculations = (CalcType.L2_NORM_PR_UNIT,)

    def __init__(
        self,
        calculations=None,
        *,
        _group_names: Iterable[str] = (),
        _group_slices: Iterable[tuple[int, int]] = (),
    ) -> None:
        super().__init__(calculations=calculations)
        self.group_names = tuple(_group_names)
        self.group_slices = tuple(_group_slices)

    @classmethod
    def calculation_context(
        cls,
        owner,
        *,
        include: Iterable[FilterItem] = (),
        ignore: Iterable[FilterItem] = (),
        **kwargs,
    ) -> CalculationContext | None:
        filters = ConsumerFilter(include=include, ignore=ignore)
        if not filters:
            return None

        return owner._calculation_context(
            canonical_groups=filter_canonical_members(
                owner.canonical_groups,
                filters,
            ),
        )

    @classmethod
    def constructor_kwargs(
        cls,
        owner,
        *,
        context: CalculationContext | None = None,
        **kwargs,
    ) -> dict:
        groups = owner.canonical_groups if context is None else context.canonical_groups
        return {
            **kwargs,
            "_group_names": group_names(owner, owner.canonical_groups),
            "_group_slices": tuple(
                (int(group.offset), int(group.length)) for group in groups
            ),
        }

    def compute(self) -> torch.Tensor:
        return self.calc(CalcType.L2_NORM_PR_UNIT)()

    def toMetric(self, result: torch.Tensor):
        return {
            f"l2_norm_distribution/{name}": result.narrow(0, start, length)
            for name, (start, length) in zip(
                self.group_names,
                self.group_slices,
                strict=True,
            )
        }
