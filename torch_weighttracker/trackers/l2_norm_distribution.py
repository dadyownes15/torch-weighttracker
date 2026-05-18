from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import torch

from torch_weighttracker.calculations import CalcType, CalculationContext
from torch_weighttracker.canonical_units import CanonicalUnitGroup, UnitKind
from torch_weighttracker.consumer_ignore import (
    ConsumerFilter,
    FilterItem,
    filter_canonical_members,
)
from torch_weighttracker.trackers.base import BaseTracker


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
            "_group_names": _group_names(owner, owner.canonical_groups),
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


def _group_names(owner, groups: Iterable[CanonicalUnitGroup]) -> tuple[str, ...]:
    groups = tuple(groups)
    base_names = tuple(_base_group_name(owner, group) for group in groups)
    counts = Counter(base_names)

    return tuple(
        (
            base_name
            if counts[base_name] == 1
            else f"{base_name}#{group.group_id}"
        )
        for base_name, group in zip(base_names, groups, strict=True)
    )


def _base_group_name(owner, group: CanonicalUnitGroup) -> str:
    if len(group.members) == 0:
        return f"group_{group.group_id}"

    root = group.members[0]
    module_name = owner._module_names_for_modules((root.module,))[0]
    name = f"{module_name}:{_handler_name(root.handler)}"

    if group.unit_kind != UnitKind.CHANNEL:
        name = f"{name}:{group.unit_kind.value}"

    return name


def _handler_name(handler) -> str:
    name = getattr(handler, "__name__", str(handler))
    if name.endswith("_out_channels"):
        return "prune_out_channels"
    if name.endswith("_in_channels"):
        return "prune_in_channels"
    return name
