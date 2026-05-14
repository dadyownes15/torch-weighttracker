# Calculation Plan Ownership Refactor Spec

## Goal

Clean up calculation-specific plan ownership so the repo has one clear split:

- `torch_structracker/plans/` contains reusable plan-building primitives.
- `torch_structracker/calculations/calcs/` owns calculation-specific plan choices.
- `torch_structracker/reductions/` owns the generic reduction plan data model,
  builder, compiler, and runtime lowering concepts.

This pass should delete thin aliases such as `create_active_units_plan(...)` and
generic wrappers such as `create_calculation(...)`. It should also remove the
double naming where `MappedReductionPlan` means the same thing as a reduction
plan.

Testing is intentionally out of scope for this spec pass.

## Target Ownership Rules

### Reusable plan code stays in `plans/`

Keep code in `plans/` only when it is broad enough to be reused by more than one
calculation, or when it is a generic builder around canonical groups.

Reusable examples:

- Canonical member tensor extraction.
- Canonical member weight operation mapping.
- Mapping canonical member operation outputs into the canonical unit vector.
- Generic unit-to-group pipeline aggregation.
- Generic mapping strategies such as `FromStructureUnitToGroupUnitMapper`.

### Calculation-specific plan choices move into calcs

A helper should not stay in `plans/` if it only chooses a single operation for a
single calculation.

Move or delete these current helpers:

| Current helper | Target |
| --- | --- |
| `create_active_units_plan(...)` | Delete. `create_active_units_calc(...)` calls `create_group_member_plan(groups, WeightOperationType.ACTIVE)` directly. |
| `create_l2_norm_pr_unit_plan(...)` | Delete. `create_l2_norm_pr_unit_calc(...)` calls `create_group_member_plan(groups, WeightOperationType.L2)` directly. |
| `create_structured_unit_sum_plan(...)` | Delete. `create_structured_unit_sum_calc(...)` calls `create_group_member_plan(groups, WeightOperationType.SUM)` directly. |
| `create_group_change_effect_plan(...)` | Move into `calcs/group_change_effect.py` as a private helper because it is specific to `CalcType.GROUP_CHANGE_EFFECT`. |
| `create_codeq_bitrates(...)` | Move into `calcs/bitrate_pr_module.py` as a private helper because it exists only to build `CalcType.BITRATE_PR_MODULE`. |
| `create_units_to_group_plan(...)` | Move into `calcs/units_to_group.py` as a private helper because it is the default identity-input wrapper for one calc. |
| `create_units_to_module_axis_plan(...)` | Move into `calcs/units_to_module_axis.py` as a private helper because it is specific to `CalcType.UNITS_TO_MODULE_AXIS`. |

The calc factory remains the public entry point for each calculation. The
private helper may still build a plan internally, but that operation choice
lives next to the calculation it creates.

## Target Package Layout

Replace the current flat plan files with folders that separate reusable plan
families:

```text
torch_structracker/plans/
  __init__.py
  member_weights/
    __init__.py
    group_member.py
  pipelines/
    __init__.py
    unit_to_group.py
```

### `plans/member_weights/group_member.py`

Owns reusable canonical member weight planning:

- `CanonicalMemberTensorExtractor`
- `MemberWeightExtractor`
- `UnitWeightReductionMapper`
- `MemberUnitMapper`
- `create_group_member_plan(...)`
- `count_group_units(...)`
- `source_indices_for_member(...)`
- any small private helpers needed by the above

This module must not expose calculation-specific aliases such as
`create_active_units_plan(...)`.

### `plans/pipelines/unit_to_group.py`

Owns reusable pipeline aggregation from canonical unit vectors to canonical
group vectors:

- `FromStructureUnitToGroupUnitMapper`
- `create_unit_to_group_acc(...)`
- shared input-ref helpers only if they are needed by more than one calc

This module must not expose `create_units_to_group_plan(...)` if that function
only builds the identity-input wrapper for `CalcType.UNITS_TO_GROUP`.

### Deleted or emptied current files

After imports are migrated, remove these old flat modules if they no longer
have unique content:

- `torch_structracker/plans/unit_weight_operation_plan.py`
- `torch_structracker/plans/mapping_plan.py`
- `torch_structracker/plans/bitrate_plan.py`

Do not keep compatibility re-export files for these names unless a current
package import is impossible to repair without one. The intended result is a
clean break.

## Calculation Factory Changes

### `calcs/active_units.py`

Replace the current import of `create_active_units_plan` with the reusable
member plan builder and the explicit operation:

```python
from torch_structracker.operations import WeightOperationType
from torch_structracker.plans.member_weights import create_group_member_plan


def create_active_units_calc(groups):
    return ReductionCalc(
        create_group_member_plan(groups, WeightOperationType.ACTIVE),
        calculation_type=CalcType.ACTIVE_UNITS,
    )
```

`ACTIVE_UNITS` must use `WeightOperationType.ACTIVE`. Docs that still describe
this calculation as `COUNT` should be updated in the same refactor.

### `calcs/l2_norm_pr_unit.py`

Replace `create_l2_norm_pr_unit_plan(...)` with:

```python
create_group_member_plan(groups, WeightOperationType.L2)
```

Keep the calculation type as `CalcType.L2_NORM_PR_UNIT`.

### `calcs/structured_unit_sum.py`

Replace `create_structured_unit_sum_plan(...)` with:

```python
create_group_member_plan(groups, WeightOperationType.SUM)
```

Keep the calculation type as `CalcType.STRUCTURED_UNIT_SUM`.

### `calcs/group_change_effect.py`

Move the current representative-unit-to-group implementation here:

- `_create_group_change_effect_plan(...)`
- `_RepresentativeUnitToGroupRule`
- `_representative_source_indices(...)`

This plan remains calculation-specific because it targets group slots and uses
one representative unit per canonical group. It should still reuse generic
member helpers:

- `CanonicalMemberTensorExtractor`
- `UnitWeightReductionMapper`
- `source_indices_for_member`

The operation remains `WeightOperationType.COUNT`, because this calc measures
structural parameter slots removed by one representative unit, not live active
weights.

### `calcs/bitrate_pr_module.py`

Move the current bitrate plan builder here:

- `CodeQBitrateRule`
- `_create_codeq_bitrates(...)`

The public factory remains `create_bitrate_pr_module_calc(...)`. It should
instantiate `ModuleBitrateExtractor(device=device, dtype=dtype)` and pass it to
the private plan helper.

No generic bitrate plan module should remain under `plans/` unless another calc
also starts reusing the same builder.

### `calcs/units_to_group.py`

Move the identity input wrapper here:

- `_create_units_to_group_plan(...)`
- `_unit_input_ref(...)`
- `_total_units(...)`

The reusable part remains `create_unit_to_group_acc(...)` in
`plans/pipelines`.

`create_units_to_group_calc(...)` should call the private wrapper and return
`PipelineCalc(..., calculation_type=CalcType.UNITS_TO_GROUP)`.

### `calcs/units_to_module_axis.py`

Move the current module-axis builder here:

- `_create_units_to_module_axis_plan(...)`
- any local input-ref helper it needs

This builder is specific to `CalcType.UNITS_TO_MODULE_AXIS` because it encodes
the `[num_weighted_modules, 2]` axis layout and `weighted_module_index`
contract. It should not stay in generic `plans/`.

## Calculation Wrapper Deletion

Delete these generic helper functions from
`torch_structracker/calculations/calculations.py`:

```python
def create_calculation(
    calculation_type: CalcType | str,
    plan: MappedReductionPlan,
) -> ReductionCalc:
    return ReductionCalc(plan, calculation_type=calculation_type)


def create_pipeline_calculation(
    plan: PipelinePlan,
    *,
    calculation_type: CalcType | str | None = None,
) -> PipelineCalc:
    return PipelineCalc(plan, calculation_type=calculation_type)
```

Update public exports so these names disappear from:

- `torch_structracker/calculations/calculations.py::__all__`
- `torch_structracker/calculations/__init__.py`

Callers should instantiate the runtime calc classes directly:

```python
ReductionCalc(plan, calculation_type=CalcType.L2_NORM_PR_UNIT)
PipelineCalc(plan, calculation_type=CalcType.UNITS_TO_GROUP)
```

The calc-specific factory functions remain public because they encode real
calculation semantics. The deleted helpers are only pass-through wrappers.

## Reduction Plan Naming

Rename the zero-input reduction plan type from `MappedReductionPlan` to
`ReductionPlan`.

### Exact data model changes

In `torch_structracker/reductions/builder.py`:

- Rename class `MappedReductionPlan` to `ReductionPlan`.
- Keep `PipelinePlan` as the name for input-dependent plans.
- Keep `ComputationPlan` as the shared abstract base if it remains useful.
- Change `ReductionPlan.kind()` from `"mapped_reduction_plan"` to
  `"reduction_plan"`.
- Change `ReductionPlanBuilder.finalize(input_spec=None)` so the no-input path
  returns `ReductionPlan`.

Do not keep a `MappedReductionPlan = ReductionPlan` alias unless the refactor
cannot be completed without it. The goal is to remove the double name, not hide
it behind another export.

### Exact import and type changes

Update all current imports and annotations:

- `from torch_structracker.reductions.builder import MappedReductionPlan`
  becomes `ReductionPlan`.
- `ReductionCalc.__init__(plan: MappedReductionPlan, ...)` becomes
  `ReductionCalc.__init__(plan: ReductionPlan, ...)`.
- Runtime type checks in `ReductionCalc` should validate `ReductionPlan`.
- Error messages should say `ReductionCalc requires a ReductionPlan`.
- Docs should use `ReductionPlan` for zero-input reductions and `PipelinePlan`
  for input-dependent reductions.

Do not rename `ReductionPlanBuilder`. That name already matches the generic
builder concept.

## Public API After Refactor

The expected public calculation surface is:

```text
torch_structracker.calculations
  CalcType
  Calculation
  CalculationContext
  CalculationSpec
  CALCULATION_SPECS
  ReductionCalc
  PipelineCalc
  CachedCalculation
  calc-specific classes and factories
```

The expected public reusable plan surface is:

```text
torch_structracker.plans.member_weights
  create_group_member_plan
  count_group_units
  CanonicalMemberTensorExtractor
  UnitWeightReductionMapper
  MemberUnitMapper

torch_structracker.plans.pipelines
  create_unit_to_group_acc
  FromStructureUnitToGroupUnitMapper
```

The expected public reduction surface is:

```text
torch_structracker.reductions.builder
  ReductionPlan
  PipelinePlan
  ReductionPlanBuilder
  ReductionRecord
  ReductionMapping
  FullSelection
  SegmentSelection
  IndexSelection
```

## Import Migration Checklist

Update imports in these known current files:

- `calcs/active_units.py`
- `calcs/l2_norm_pr_unit.py`
- `calcs/structured_unit_sum.py`
- `calcs/group_change_effect.py`
- `calcs/bitrate_pr_module.py`
- `calcs/baseline_group_sizes.py`
- `calcs/units_to_group.py`
- `calcs/units_to_module_axis.py`
- `calculations/calculations.py`
- `calculations/__init__.py`
- `calculations/reduction_calc.py`
- `calculations/pipeline_calc.py` only if plan imports move
- `reductions/compiler.py`
- docs that mention `MappedReductionPlan` or deleted helper names

Update import users outside package code only after package imports are clean.

## Documentation Cleanup

Update docs in the same refactor so they describe the new ownership model:

- Replace `MappedReductionPlan` with `ReductionPlan`.
- Remove examples that call `create_active_units_plan(...)`.
- Replace calculation-specific plan examples with calc-specific factory examples
  or direct `create_group_member_plan(..., WeightOperationType.X)` calls.
- Correct active-unit docs to use `WeightOperationType.ACTIVE`.
- Keep `create_unit_to_group_acc(...)` examples because that remains a reusable
  pipeline primitive.

## Implementation Order

1. Rename `MappedReductionPlan` to `ReductionPlan` in the reduction layer and
   update `ReductionCalc`.
2. Create the new plan folders and move reusable member-weight and pipeline
   primitives into them.
3. Update calc factories to import the reusable primitives from the new folders.
4. Move calculation-specific plan builders into their owning calc modules.
5. Delete obsolete plan helpers and old flat plan files once no imports remain.
6. Delete `create_calculation(...)` and `create_pipeline_calculation(...)`.
7. Clean package exports and docs so stale names are gone.

The implementation should avoid compatibility shims, broad rewrites, or
unrelated cleanup.
