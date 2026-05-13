import torch

from torch_structracker.calculations import CalcType
from torch_structracker.regularizers.base import BaseRegularizer, RegularizerType


class GroupLasso(BaseRegularizer):
    regularizer_type = RegularizerType.GROUP_LASSO
    required_calculations = (
        CalcType.L2_NORM_PR_UNIT,
        CalcType.UNITS_TO_GROUP,
        CalcType.UNIT_ACTIVE_MASK,
        CalcType.BASELINE_GROUP_SIZES,
        CalcType.GROUP_CHANGE_EFFECT,
        CalcType.GROUP_SIZES,
    )

    def forward(self) -> torch.Tensor:
        unit_active_mask = self.compute(CalcType.UNIT_ACTIVE_MASK)
        active_pr_group = self.compute(
            CalcType.UNITS_TO_GROUP,
            unit_active_mask,
        )
        baseline_group_size = self.compute(CalcType.BASELINE_GROUP_SIZES)
        group_change_effect = self.compute(CalcType.GROUP_CHANGE_EFFECT)
        group_sizes = self.compute(CalcType.GROUP_SIZES)
        l2_norm_pr_unit = self.compute(CalcType.L2_NORM_PR_UNIT)

        active_params_pr_group = (
            active_pr_group - baseline_group_size
        ) * group_change_effect
        active_params_pr_unit = torch.repeat_interleave(
            active_params_pr_group,
            group_sizes,
        )
        return (active_params_pr_unit * l2_norm_pr_unit).sum()
