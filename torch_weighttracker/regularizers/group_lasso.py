from collections.abc import Iterable

import torch

from torch_weighttracker.calculations import CalcType, CalculationContext
from torch_weighttracker.consumer_ignore import (
    ConsumerFilter,
    FilterItem,
    filter_canonical_members,
)
from torch_weighttracker.regularizers.base import BaseRegularizer, RegularizerType


class GroupLasso(BaseRegularizer):
    regularizer_type = RegularizerType.GROUP_LASSO
    required_calculations = (
        CalcType.PARAM_PR_UNIT,
        CalcType.L2_NORM_PR_UNIT,
    )

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

    def forward(self) -> torch.Tensor:
        l2_norm_pr_unit = self.compute(CalcType.L2_NORM_PR_UNIT)
        param_calc = self.calc(CalcType.PARAM_PR_UNIT)
        forward_from_l2_norm_pr_unit = getattr(
            param_calc,
            "forward_from_l2_norm_pr_unit",
            None,
        )
        if forward_from_l2_norm_pr_unit is None:
            param_pr_unit = param_calc()
        else:
            param_pr_unit = forward_from_l2_norm_pr_unit(l2_norm_pr_unit)
        return (param_pr_unit.sqrt() * l2_norm_pr_unit).sum()
