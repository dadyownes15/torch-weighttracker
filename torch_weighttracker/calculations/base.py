from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

import torch
from torch import nn


class CalcType(str, Enum):
    STRUCTURED_UNIT_SUM = "structured_unit_sum"
    ACTIVE_UNITS = "active_units"
    L2_NORM_PR_UNIT = "l2_norm_pr_unit"
    BITRATE_PR_MODULE = "bitrate_pr_module"
    UNSTRUCTURED_SPARSITY_PR_MODULE = "unstructured_sparsity_pr_module"
    UNITS_TO_MODULE_AXIS = "units_to_module_axis"
    UNIT_DELTA_TO_MODULE_AXIS = "unit_delta_to_module_axis"
    ACTIVE_MACS_PR_MODULE = "active_macs_pr_module"
    BASELINE_MACS_PR_MODULE = "baseline_macs_pr_module"
    BASELINE_MODULE_AXES = "baseline_module_axes"
    MODULE_AXIS_COST_INDICES = "module_axis_cost_indices"
    UNITS_TO_GROUP = "units_to_group"
    GROUPS_TO_UNITS = "groups_to_units"
    UNIT_ACTIVE_MASK = "unit_active_mask"
    GROUP_CHANGE_EFFECT = "group_change_effect"
    GROUP_UNIT_PARAM_CHANGE = "group_unit_param_change"
    BASELINE_PARAM_PR_UNIT_PR_GROUP = "baseline_param_pr_unit_pr_group"
    PARAM_PR_UNIT = "param_pr_unit"
    GROUP_SIZES = "group_sizes"
    INIT_UNIT_PR_GROUP_COUNT = "baseline_group_sizes"


class BaseCalculation(nn.Module):
    calculation_type: CalcType | None = None

    def __init__(
        self,
        dependencies: Mapping[CalcType, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.dependencies = nn.ModuleDict(
            {}
            if dependencies is None
            else {calc_type.name: module for calc_type, module in dependencies.items()}
        )
        self.register_buffer("output_anchor", torch.empty(()), persistent=False)

    def calc(self, calc_type: CalcType | str) -> nn.Module:
        calc_type = CalcType(calc_type)
        return self.dependencies[calc_type.name]

    def compute(self, calc_type: CalcType | str, *args, **kwargs) -> torch.Tensor:
        return self.calc(calc_type)(*args, **kwargs)


Calculation = BaseCalculation
