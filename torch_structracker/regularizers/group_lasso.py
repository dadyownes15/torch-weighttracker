from typing import Mapping, override
import torch
import torch.nn as nn

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

    def __init__(
        self,
        calcs: Mapping[CalcType, nn.Module],
    ) -> None:
        super().__init__(calcs)
        
            
    @override
    def forward(self) -> torch.Tensor:
        unit_active_mask = self.compute(CalcType.UNIT_ACTIVE_MASK)
    
        active_pr_group = self.compute(
            CalcType.UNITS_TO_GROUP,
            unit_active_mask,
        )
    
        baseline_group_size = self.compute(CalcType.BASELINE_GROUP_SIZES)
        group_change_effect = self.compute(CalcType.GROUP_CHANGE_EFFECT)
        group_sizes = self.compute(CalcType.GROUP_SIZES)
    
        group_effect = (
            active_pr_group - baseline_group_size
        ) * group_change_effect
    
        return torch.repeat_interleave(group_effect, group_sizes)