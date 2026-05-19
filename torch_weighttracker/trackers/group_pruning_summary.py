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


class GroupPruningSummary(BaseTracker):
    required_calculations = (
        CalcType.UNIT_ACTIVE_MASK,
        CalcType.UNITS_TO_GROUP,
        CalcType.INIT_UNIT_PR_GROUP_COUNT,
        CalcType.BASELINE_PARAM_PR_UNIT_PR_GROUP,
        CalcType.GROUP_UNIT_PARAM_CHANGE,
    )

    def __init__(
        self,
        calculations=None,
        *,
        _group_names: Iterable[str] = (),
    ) -> None:
        super().__init__(calculations=calculations)
        self.group_names = tuple(_group_names)

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
        name_groups = owner.canonical_groups
        if len(name_groups) != len(groups):
            name_groups = groups
        return {
            **kwargs,
            "_group_names": group_names(owner, name_groups),
        }

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        active_mask = self.compute_calculation(CalcType.UNIT_ACTIVE_MASK)
        active_units = self.compute_calculation(CalcType.UNITS_TO_GROUP, active_mask)
        baseline_units = self.compute_calculation(CalcType.INIT_UNIT_PR_GROUP_COUNT)
        pruned_units = baseline_units - active_units

        baseline_param_pr_unit = self.compute_calculation(
            CalcType.BASELINE_PARAM_PR_UNIT_PR_GROUP
        )
        param_change_pr_unit = self.compute_calculation(
            CalcType.GROUP_UNIT_PARAM_CHANGE,
            pruned_units,
        )
        active_param_pr_unit = baseline_param_pr_unit - param_change_pr_unit
        pruned_params = (
            baseline_units * baseline_param_pr_unit
            - active_units * active_param_pr_unit
        )

        return pruned_units, pruned_params

    def toMetric(self, result: tuple[torch.Tensor, torch.Tensor]):
        pruned_units, pruned_params = result
        metrics = {
            "group_pruning/pruned_units": pruned_units.sum(),
            "group_pruning/pruned_params": pruned_params.sum(),
        }

        for name, units, params in zip(
            self.group_names,
            pruned_units,
            pruned_params,
            strict=True,
        ):
            metrics[f"group_pruning/groups/{name}/pruned_units"] = units
            metrics[f"group_pruning/groups/{name}/pruned_params"] = params

        return metrics
