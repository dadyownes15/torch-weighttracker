from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType, Calculation
from torch_weighttracker.calculations.spec import CalculationSpec


class ParamPrUnit(Calculation):
    calculation_type = CalcType.PARAM_PR_UNIT

    def __init__(self, dependencies: Mapping[CalcType, nn.Module]) -> None:
        super().__init__(dependencies)

    def forward(self) -> torch.Tensor:
        l2_norm_pr_unit = self.compute(CalcType.L2_NORM_PR_UNIT)
        active_unit_mask = l2_norm_pr_unit.gt(0).to(dtype=l2_norm_pr_unit.dtype)

        active_units_pr_group = self.compute(
            CalcType.UNITS_TO_GROUP,
            active_unit_mask,
        )
        baseline_units_pr_group = self.compute(CalcType.INIT_UNIT_PR_GROUP_COUNT)
        removed_units_pr_group = baseline_units_pr_group - active_units_pr_group

        param_change_pr_unit_pr_group = self.compute(
            CalcType.GROUP_UNIT_PARAM_CHANGE,
            removed_units_pr_group,
        )
        baseline_param_pr_unit_pr_group = self.compute(
            CalcType.BASELINE_PARAM_PR_UNIT_PR_GROUP
        )
        current_param_pr_unit_pr_group = (
            baseline_param_pr_unit_pr_group - param_change_pr_unit_pr_group
        )

        param_pr_unit = self.compute(
            CalcType.GROUPS_TO_UNITS,
            current_param_pr_unit_pr_group,
        )
        return param_pr_unit * active_unit_mask


def create_param_pr_unit_calc(
    dependencies: Mapping[CalcType, nn.Module],
) -> ParamPrUnit:
    return ParamPrUnit(dependencies)


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.PARAM_PR_UNIT,
    required_calculations=(
        CalcType.L2_NORM_PR_UNIT,
        CalcType.UNITS_TO_GROUP,
        CalcType.GROUPS_TO_UNITS,
        CalcType.INIT_UNIT_PR_GROUP_COUNT,
        CalcType.BASELINE_PARAM_PR_UNIT_PR_GROUP,
        CalcType.GROUP_UNIT_PARAM_CHANGE,
    ),
    create=lambda ctx, deps: create_param_pr_unit_calc(deps),
)
