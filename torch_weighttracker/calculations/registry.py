from __future__ import annotations

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.calcs.active_macs_pr_module import (
    CALCULATION_SPEC as ACTIVE_MACS_PR_MODULE_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.active_units import (
    CALCULATION_SPEC as ACTIVE_UNITS_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.baseline_group_sizes import (
    CALCULATION_SPEC as INIT_UNIT_PR_GROUP_COUNT_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.baseline_macs_pr_module import (
    CALCULATION_SPEC as BASELINE_MACS_PR_MODULE_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.baseline_module_axes import (
    CALCULATION_SPEC as BASELINE_MODULE_AXES_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.baseline_param_pr_unit_pr_group import (
    CALCULATION_SPEC as BASELINE_PARAM_PR_UNIT_PR_GROUP_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.bitrate_pr_module import (
    CALCULATION_SPEC as BITRATE_PR_MODULE_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.block_2_4_sparsity import (
    CALCULATION_SPEC as BLOCK_2_4_SPARSITY_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.group_change_effect import (
    CALCULATION_SPEC as GROUP_CHANGE_EFFECT_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.group_sizes import (
    CALCULATION_SPEC as GROUP_SIZES_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.group_unit_param_change import (
    CALCULATION_SPEC as GROUP_UNIT_PARAM_CHANGE_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.groups_to_units import (
    CALCULATION_SPEC as GROUPS_TO_UNITS_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.l2_norm_pr_unit import (
    CALCULATION_SPEC as L2_NORM_PR_UNIT_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.module_axis_cost_indices import (
    CALCULATION_SPEC as MODULE_AXIS_COST_INDICES_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.param_pr_unit import (
    CALCULATION_SPEC as PARAM_PR_UNIT_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.structured_unit_sum import (
    CALCULATION_SPEC as STRUCTURED_UNIT_SUM_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.unit_active_mask import (
    CALCULATION_SPEC as UNIT_ACTIVE_MASK_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.unit_delta_to_module_axis import (
    CALCULATION_SPEC as UNIT_DELTA_TO_MODULE_AXIS_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.units_to_group import (
    CALCULATION_SPEC as UNITS_TO_GROUP_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.units_to_module_axis import (
    CALCULATION_SPEC as UNITS_TO_MODULE_AXIS_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.unstructured_sparsity_pr_module import (
    CALCULATION_SPEC as UNSTRUCTURED_SPARSITY_PR_MODULE_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.spec import CalculationSpec

_CALCULATION_SPEC_LIST = (
    ACTIVE_UNITS_CALCULATION_SPEC,
    UNIT_ACTIVE_MASK_CALCULATION_SPEC,
    UNITS_TO_GROUP_CALCULATION_SPEC,
    GROUPS_TO_UNITS_CALCULATION_SPEC,
    INIT_UNIT_PR_GROUP_COUNT_CALCULATION_SPEC,
    BASELINE_PARAM_PR_UNIT_PR_GROUP_CALCULATION_SPEC,
    GROUP_UNIT_PARAM_CHANGE_CALCULATION_SPEC,
    PARAM_PR_UNIT_CALCULATION_SPEC,
    GROUP_CHANGE_EFFECT_CALCULATION_SPEC,
    GROUP_SIZES_CALCULATION_SPEC,
    UNITS_TO_MODULE_AXIS_CALCULATION_SPEC,
    UNIT_DELTA_TO_MODULE_AXIS_CALCULATION_SPEC,
    BASELINE_MODULE_AXES_CALCULATION_SPEC,
    MODULE_AXIS_COST_INDICES_CALCULATION_SPEC,
    BASELINE_MACS_PR_MODULE_CALCULATION_SPEC,
    ACTIVE_MACS_PR_MODULE_CALCULATION_SPEC,
    BITRATE_PR_MODULE_CALCULATION_SPEC,
    UNSTRUCTURED_SPARSITY_PR_MODULE_CALCULATION_SPEC,
    BLOCK_2_4_SPARSITY_CALCULATION_SPEC,
    L2_NORM_PR_UNIT_CALCULATION_SPEC,
    STRUCTURED_UNIT_SUM_CALCULATION_SPEC,
)

CALCULATION_SPECS: dict[CalcType, CalculationSpec] = {
    spec.calculation_type: spec for spec in _CALCULATION_SPEC_LIST
}
