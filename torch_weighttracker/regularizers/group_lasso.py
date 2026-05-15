from collections.abc import Iterable

import torch

from torch_weighttracker.calculations import CalcType, CalculationContext
from torch_weighttracker.consumer_ignore import (
    IgnoreItem,
    ModuleIgnore,
    without_ignored_canonical_members,
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
        ignore: Iterable[IgnoreItem] = (),
        **kwargs,
    ) -> CalculationContext | None:
        ignored = ModuleIgnore(ignore)
        if not ignored:
            return None

        return owner._calculation_context(
            canonical_groups=without_ignored_canonical_members(
                owner.canonical_groups,
                ignored,
            ),
        )

    def forward(self) -> torch.Tensor:
        param_pr_unit = self.compute(CalcType.PARAM_PR_UNIT)
        l2_norm_pr_unit = self.compute(CalcType.L2_NORM_PR_UNIT)
        return (param_pr_unit.sqrt() * l2_norm_pr_unit).sum()
