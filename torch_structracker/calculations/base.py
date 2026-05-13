from enum import Enum

import torch
import torch.nn as nn

from torch_structracker.plans.bitrate_plan import CodeQBitrateRule

    
class CalcType(str, Enum):
    STRUCTURED_UNIT_SUM = "structured_unit_sum"
    ACTIVE_UNITS = "active_units"
    L2_NORM_PR_UNIT = "l2_norm_pr_unit"
    BITRATE_PR_MODULE = "bitrate_pr_module"
    UNITS_TO_MODULE_AXIS = "units_to_module"
    UNITS_TO_GROUP = "units_to_group"
    UNIT_ACTIVE_MASK = "unit_active_mask",
    GROUP_CHANGE_EFFECT = "group_change_effect"
    GROUP_SIZES = "group_sizes"
    BASELINE_GROUP_SIZES = "group_sizes"
        


