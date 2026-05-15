# GroupLasso unit-size calculation spec

## Problem

`GROUP_SIZES` is a static structural value: it stores `group.length` for each
canonical group. It should not drive the `GroupLasso` weighting term directly.

`GroupLasso` should not hand-roll group-to-unit layout expansion with
`torch.repeat_interleave(...)`. The mapping belongs in a pipeline calculation,
using the same reduction mapping machinery as the other unit/group mappings.

## Target shape

`GroupLasso` should consume the final per-unit size calculation:

```python
unit_size_pr_unit = UNIT_SIZE_PR_UNIT()
l2_norm_pr_unit = L2_NORM_PR_UNIT()
loss = (unit_size_pr_unit * l2_norm_pr_unit).sum()
```

`UNIT_SIZE_PR_UNIT` owns the composition:

```text
L2_NORM_PR_UNIT
  -> l2-derived active mask
  -> UNITS_TO_GROUP(active_mask)
  -> GROUP_CHANGE_EFFECT(active_units_pr_group)
  -> GROUPS_TO_UNITS(unit_size_pr_group)
  -> zero inactive units
```

There should be no separate active-count calc class. The active group count is
the existing `UNITS_TO_GROUP` pipeline applied to the L2-derived active mask.

## Calc types

Add two calc types:

```python
class CalcType(str, Enum):
    STRUCTURED_UNIT_SUM = "structured_unit_sum"
    ACTIVE_UNITS = "active_units"
    L2_NORM_PR_UNIT = "l2_norm_pr_unit"
    UNIT_SIZE_PR_UNIT = "unit_size_pr_unit"
    BITRATE_PR_MODULE = "bitrate_pr_module"
    UNITS_TO_MODULE_AXIS = "units_to_module_axis"
    UNIT_DELTA_TO_MODULE_AXIS = "unit_delta_to_module_axis"
    ACTIVE_MACS_PR_MODULE = "active_macs_pr_module"
    BASELINE_MACS_PR_MODULE = "baseline_macs_pr_module"
    BASELINE_MODULE_AXES = "baseline_module_axes"
    UNITS_TO_GROUP = "units_to_group"
    GROUPS_TO_UNITS = "groups_to_units"
    UNIT_ACTIVE_MASK = "unit_active_mask"
    GROUP_CHANGE_EFFECT = "group_change_effect"
    GROUP_SIZES = "group_sizes"
    INIT_UNIT_PR_GROUP_COUNT = "baseline_group_sizes"
```

Do not add `ACTIVE_UNIT_COUNT_PR_GROUP`.

## Pipeline input specs

Pipeline calc factories should own their input-spec dtype defaults. They should
receive the normalized context device from `WeightTracker`, then build their
own `TensorSpec` from factory arguments.

Do not add a generic dtype to `CalculationContext`, and do not route pipeline
dtype through `calculation_dtype(ctx)`.

```python
def create_groups_to_units_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> PipelineCalc:
    canonical_groups = tuple(groups)
    input_spec = TensorSpec(
        shape=torch.Size([len(canonical_groups)]),
        dtype=dtype,
        device=device,
    )
    ...
```

## `GROUPS_TO_UNITS`

Add `torch_weighttracker/calculations/calcs/groups_to_units.py`.

This is a `PipelineCalc` over a `PipelinePlan`. It is not a hand-written tensor
calculation. The factory receives the normalized device and owns the dtype
default used to build its input spec.

```python
from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import calculation_device
from torch_weighttracker.calculations.pipeline_calc import PipelineCalc
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.canonical_units import CanonicalUnitGroup
from torch_weighttracker.extractors.extractor import TensorSpec, ValueTensorRef
from torch_weighttracker.reductions.builder import (
    IndexSelection,
    PipelinePlan,
    ReductionMapping,
    ReductionRecord,
    ReductionPlanBuilder,
    SegmentSelection,
)
from torch_weighttracker.reductions.ops import IdentityTensorReduction, ReductionOp


def create_groups_to_units_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> PipelineCalc:
    """
    Maps one value per canonical group to one value per canonical unit.

    Output: 1D tensor with length equal to the total canonical unit count.
    Input: 1D tensor with length `len(canonical_groups)`. Element `g` is the
    value for canonical group `g`.
    """
    canonical_groups = tuple(groups)
    input_spec = TensorSpec(
        shape=torch.Size([len(canonical_groups)]),
        dtype=dtype,
        device=device,
    )
    return PipelineCalc(
        _create_groups_to_units_plan(canonical_groups, input_spec=input_spec),
        calculation_type=CalcType.GROUPS_TO_UNITS,
    )


def _create_groups_to_units_plan(
    groups: Iterable[CanonicalUnitGroup],
    *,
    input_spec: TensorSpec,
) -> PipelinePlan:
    canonical_groups = tuple(groups)
    input_ref = ValueTensorRef(
        value=torch.empty(
            input_spec.shape,
            dtype=input_spec.dtype,
            device=input_spec.device,
        ),
        spec=input_spec,
    )
    op = ReductionOp(input_ref, IdentityTensorReduction())
    builder = ReductionPlanBuilder(output_length=_total_units(canonical_groups))

    for group in canonical_groups:
        builder.add(
            ReductionRecord(
                op=op,
                mapping=ReductionMapping(
                    source=IndexSelection((group.group_id,) * int(group.length)),
                    target=SegmentSelection(
                        start=int(group.offset),
                        length=int(group.length),
                    ),
                ),
            )
        )

    return cast(PipelinePlan, builder.finalize(input_ref.source_spec()))


def _total_units(groups: Iterable[CanonicalUnitGroup]) -> int:
    return sum(int(group.length) for group in groups)


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.GROUPS_TO_UNITS,
    create=lambda ctx, deps: create_groups_to_units_calc(
        ctx.canonical_groups,
        device=calculation_device(ctx),
    ),
)
```

## `GROUP_CHANGE_EFFECT`

Replace the current zero-input static vector calculation with a pipeline
calculation from active group counts to current unit size per group:

```text
GROUP_CHANGE_EFFECT(active_units_pr_group) -> unit_size_pr_group
```

The structural relation is static, but the output is current because it is
evaluated with current active group counts.

For a linear weight `[out, in]`:

```text
out group unit size = active input group count, or static in_features
in group unit size = active output group count, or static out_features
```

For a conv weight `[out, in, kh, kw]`:

```text
out group unit size = active input group count * kh * kw
in group unit size = active output group count * kh * kw
```

Add `torch_weighttracker/calculations/calcs/group_change_effect.py` in this
shape. This is still a `PipelineCalc` over a `PipelinePlan`, not a custom tensor
calculation.

```python
from __future__ import annotations

from collections.abc import Iterable
from typing import Hashable, cast

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import calculation_device
from torch_weighttracker.calculations.pipeline_calc import PipelineCalc
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.canonical_units import (
    CanonicalMember,
    CanonicalUnitGroup,
    UnitAxis,
    canonical_members,
)
from torch_weighttracker.extractors.extractor import SourceSpec, TensorSpec, ValueTensorRef
from torch_weighttracker.plans.mapping_plan import module_axis_for_member
from torch_weighttracker.reductions.builder import (
    IndexSelection,
    PipelinePlan,
    ReductionMapping,
    ReductionRecord,
    ReductionPlanBuilder,
)
from torch_weighttracker.reductions.ops import ReductionOp, TensorValue


def create_group_change_effect_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> PipelineCalc:
    """
    Maps active canonical-group counts to current unit size per group.

    Output: 1D tensor with length `len(canonical_groups)`.
    Input: 1D tensor with length `len(canonical_groups)`. Element `g` is the
    active unit count for canonical group `g`.
    """
    canonical_groups = tuple(groups)
    input_spec = TensorSpec(
        shape=torch.Size([len(canonical_groups)]),
        dtype=dtype,
        device=device,
    )
    return PipelineCalc(
        _create_group_change_effect_plan(canonical_groups, input_spec=input_spec),
        calculation_type=CalcType.GROUP_CHANGE_EFFECT,
    )


def _create_group_change_effect_plan(
    groups: Iterable[CanonicalUnitGroup],
    *,
    input_spec: TensorSpec,
) -> PipelinePlan:
    canonical_groups = tuple(groups)
    input_ref = ValueTensorRef(
        value=torch.empty(
            input_spec.shape,
            dtype=input_spec.dtype,
            device=input_spec.device,
        ),
        spec=input_spec,
    )
    builder = ReductionPlanBuilder(output_length=len(canonical_groups))
    axis_group_index = _module_axis_group_index(canonical_groups)

    for member in canonical_members(canonical_groups):
        target_group_id = int(member.group_id)
        target_axis = module_axis_for_member(member)
        source_axis = 1 - target_axis
        scale = _member_axis_scale(member)
        source_group_id = axis_group_index.get((member.module, source_axis))

        if source_group_id is None:
            builder.add(
                ReductionRecord(
                    op=ReductionOp(
                        input_ref,
                        ConstantTensorReduction(
                            _static_axis_size(member.module, source_axis) * scale
                        ),
                    ),
                    mapping=ReductionMapping(
                        source=IndexSelection((0,)),
                        target=IndexSelection((target_group_id,)),
                    ),
                )
            )
            continue

        builder.add(
            ReductionRecord(
                op=ReductionOp(input_ref, ScaleTensorReduction(scale)),
                mapping=ReductionMapping(
                    source=IndexSelection((source_group_id,)),
                    target=IndexSelection((target_group_id,)),
                ),
            )
        )

    return cast(PipelinePlan, builder.finalize(input_ref.source_spec()))


class ScaleTensorReduction:
    def __init__(self, scale: float) -> None:
        self.scale = float(scale)

    def __call__(self, value: TensorValue) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("ScaleTensorReduction expects one tensor.")
        return value.reshape(-1) * self.scale

    def output_spec(self, source_spec: SourceSpec) -> TensorSpec:
        if not isinstance(source_spec, TensorSpec):
            raise TypeError("ScaleTensorReduction expects one source tensor.")
        return source_spec

    def identity_key(self) -> Hashable:
        return ("scale", self.scale)


class ConstantTensorReduction:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __call__(self, value: TensorValue) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("ConstantTensorReduction expects one tensor.")
        return value.new_tensor([self.value])

    def output_spec(self, source_spec: SourceSpec) -> TensorSpec:
        if not isinstance(source_spec, TensorSpec):
            raise TypeError("ConstantTensorReduction expects one source tensor.")
        return TensorSpec(
            shape=torch.Size([1]),
            dtype=source_spec.dtype,
            device=source_spec.device,
        )

    def identity_key(self) -> Hashable:
        return ("constant", self.value)


def _module_axis_group_index(
    groups: Iterable[CanonicalUnitGroup],
) -> dict[tuple[nn.Module, int], int]:
    result: dict[tuple[nn.Module, int], int] = {}
    for member in canonical_members(groups):
        axis = module_axis_for_member(member)
        result.setdefault((member.module, axis), int(member.group_id))
    return result


def _member_axis_scale(member: CanonicalMember) -> float:
    module = member.module
    if isinstance(module, nn.Conv2d):
        kernel_h, kernel_w = module.kernel_size
        return float(kernel_h * kernel_w)
    return 1.0


def _static_axis_size(module: nn.Module, axis: int) -> float:
    if isinstance(module, nn.Linear):
        return float(module.in_features if axis == 0 else module.out_features)
    if isinstance(module, nn.Conv2d):
        if module.groups != 1:
            raise ValueError("GROUP_CHANGE_EFFECT supports only Conv2d(groups=1).")
        return float(module.in_channels if axis == 0 else module.out_channels)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return 1.0
    if isinstance(module, nn.LayerNorm):
        return 1.0
    raise ValueError(
        "GROUP_CHANGE_EFFECT is not implemented for "
        f"{module.__class__.__name__}."
    )


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.GROUP_CHANGE_EFFECT,
    create=lambda ctx, deps: create_group_change_effect_calc(
        ctx.canonical_groups,
        device=calculation_device(ctx),
    ),
)
```

This intentionally changes `GROUP_CHANGE_EFFECT` from a cached zero-input
`ReductionCalc` into a pipeline calculation. It should not be marked
`cache_constant=True`, because the runtime output depends on the input active
group counts.

## `UNIT_SIZE_PR_UNIT`

Add `torch_weighttracker/calculations/calcs/unit_size_pr_unit.py`.

This is a composed calculation. It does not own tensor layout. It delegates
unit-to-group and group-to-unit mapping to pipeline calculations.

This assumes `GROUP_CHANGE_EFFECT` is the group relation calculation that accepts
`active_units_pr_group` and returns the current unit size per group. The
relation/plan can be static; the output is current because it is evaluated with
the current active group counts.

```python
from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType, Calculation
from torch_weighttracker.calculations.spec import CalculationSpec


class UnitSizePrUnit(Calculation):
    """
    Returns current unit size for each canonical unit.

    Output: 1D tensor with length equal to the total canonical unit count.
    Input: none.
    """

    calculation_type = CalcType.UNIT_SIZE_PR_UNIT
    required_calculations = (
        CalcType.L2_NORM_PR_UNIT,
        CalcType.UNITS_TO_GROUP,
        CalcType.GROUP_CHANGE_EFFECT,
        CalcType.GROUPS_TO_UNITS,
    )

    def __init__(self, dependencies: Mapping[CalcType, nn.Module]) -> None:
        super().__init__(dependencies)

    def forward(self) -> torch.Tensor:
        l2_norm_pr_unit = self.compute(CalcType.L2_NORM_PR_UNIT)
        unit_is_active = l2_norm_pr_unit.gt(0).to(dtype=l2_norm_pr_unit.dtype)
        active_units_pr_group = self.compute(
            CalcType.UNITS_TO_GROUP,
            unit_is_active,
        )
        unit_size_pr_group = self.compute(
            CalcType.GROUP_CHANGE_EFFECT,
            active_units_pr_group,
        )
        unit_size_pr_unit = self.compute(
            CalcType.GROUPS_TO_UNITS,
            unit_size_pr_group,
        )
        return unit_size_pr_unit * unit_is_active


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.UNIT_SIZE_PR_UNIT,
    required_calculations=UnitSizePrUnit.required_calculations,
    create=lambda ctx, deps: UnitSizePrUnit(deps),
)
```

`UNIT_SIZE_PR_UNIT` is where inactive units become zero. `GroupLasso` should not
multiply by `UNIT_ACTIVE_MASK` or by a local L2-derived mask.

## GroupLasso

Replace the regularizer formula with:

```python
class GroupLasso(BaseRegularizer):
    regularizer_type = RegularizerType.GROUP_LASSO
    required_calculations = (
        CalcType.UNIT_SIZE_PR_UNIT,
        CalcType.L2_NORM_PR_UNIT,
    )

    def forward(self) -> torch.Tensor:
        unit_size_pr_unit = self.compute(CalcType.UNIT_SIZE_PR_UNIT)
        l2_norm_pr_unit = self.compute(CalcType.L2_NORM_PR_UNIT)
        return (unit_size_pr_unit * l2_norm_pr_unit).sum()
```

`GroupLasso` must not call `UNITS_TO_GROUP`, `GROUPS_TO_UNITS`,
`UNIT_ACTIVE_MASK`, or `torch.repeat_interleave(...)` directly.

## Export and registry wiring

Export `create_groups_to_units_calc`, `GROUPS_TO_UNITS_CALCULATION_SPEC`,
`UnitSizePrUnit`, and `UNIT_SIZE_PR_UNIT_CALCULATION_SPEC` through the current
calculation aggregation modules.

In `registry.py`, import module-level specs and add those objects. Do not
duplicate spec bodies in the registry.

```python
from torch_weighttracker.calculations.calcs.groups_to_units import (
    CALCULATION_SPEC as GROUPS_TO_UNITS_CALCULATION_SPEC,
)
from torch_weighttracker.calculations.calcs.unit_size_pr_unit import (
    CALCULATION_SPEC as UNIT_SIZE_PR_UNIT_CALCULATION_SPEC,
)

CALCULATION_SPECS = {
    # ...
    CalcType.GROUPS_TO_UNITS: GROUPS_TO_UNITS_CALCULATION_SPEC,
    CalcType.UNIT_SIZE_PR_UNIT: UNIT_SIZE_PR_UNIT_CALCULATION_SPEC,
}
```

If the registry has already been converted to `_CALCULATION_SPEC_LIST`, add both
spec objects to that list instead.

## Tests

Add or update focused tests for:

- `GROUPS_TO_UNITS` maps `[10.0, 4.0]` to `[10.0, 10.0, 10.0, 4.0]` for group lengths `[3, 1]`.
- `UNIT_SIZE_PR_UNIT` derives activity from `L2_NORM_PR_UNIT`.
- `UNIT_SIZE_PR_UNIT` returns zero for units with zero L2 norm.
- `UNIT_SIZE_PR_UNIT` uses `UNITS_TO_GROUP` for current active group counts.
- `GroupLasso` only multiplies `UNIT_SIZE_PR_UNIT` by `L2_NORM_PR_UNIT`.

## Non-goals

- Do not rename or remove `GROUP_SIZES` in this change.
- Do not compact tensors to active-only layout.
- Do not add a custom calculation class for active group counts.
- Do not hand-roll group-to-unit layout expansion in `GroupLasso`.
