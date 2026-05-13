# Core Group Calculation Spec

## Scope

Implement factory functions for these calculation types:

- `CalcType.UNITS_TO_GROUP`
- `CalcType.UNIT_ACTIVE_MASK`
- `CalcType.UNITS_TO_MODULE_AXIS`
- `CalcType.BITRATE_PR_MODULE`
- `CalcType.BASELINE_GROUP_SIZES`
- `CalcType.GROUP_CHANGE_EFFECT`
- `CalcType.GROUP_SIZES`

These cover the current `GroupLasso` dependencies and the structured-BOPs
dependencies:

```python
required_calculations = (
    CalcType.UNIT_ACTIVE_MASK,
    CalcType.UNITS_TO_MODULE_AXIS,
    CalcType.BITRATE_PR_MODULE,
)
```

The functions should live in `torch_structracker/calculations/calculations.py`.
They should build on the existing plan/calc primitives instead of duplicating
runtime reduction logic:

- `ReductionCalc` for zero-argument mapped reductions over model state.
- `PipelineCalc` for reductions over an input unit tensor.
- `CachedCalculation` for values that only change when canonical groups change.

Tests are out of scope for this pass.

## Coordinate System

All calculations consume `CanonicalUnitGroup` objects. The global unit vector is
the concatenation of each group range:

```text
group g covers units [group.offset, group.offset + group.length)
total_units = sum(group.length for group in groups)
num_groups = len(groups)
```

All per-unit outputs have shape `[total_units]`.
All per-group outputs have shape `[num_groups]`.
All per-module-axis outputs are flat tensors with shape
`[num_weighted_modules * 2]` and a logical view of
`[num_weighted_modules, 2]`.

Every calculation must use this canonical coordinate system so that
`L2_NORM_PR_UNIT`, active masks, unit-to-group aggregation, and group constants
refer to the same unit positions.

Weighted module ordering must be shared by all module-axis calculations. Use
`ModuleBitrateExtractor.weighted_modules(model)` as the canonical module order
for structured-BOPs calculations.

## Factory Contracts

### `create_units_to_group_calc`

Purpose: aggregate any per-unit tensor into a per-group tensor.

Return type: `PipelineCalc`

Runtime signature:

```python
forward(x: torch.Tensor) -> torch.Tensor
```

Input contract:

- `x` is a flat tensor with shape `[total_units]`.
- `x.dtype` and `x.device` define the output dtype/device.

Output:

- Tensor with shape `[num_groups]`.
- `out[g] = x[group.offset : group.offset + group.length].sum()`.

Implementation:

- Create a dummy `ValueTensorRef` with shape `[total_units]` only for plan
  metadata.
- Use `create_unit_to_group_acc(...)`.
- Use `IdentityTensorReduction` as the pipeline reduction mapper.
- Wrap the resulting `PipelinePlan` in `PipelineCalc`.

This calculation is not cached because it depends on the runtime input tensor.

### `create_unit_active_mask_calc`

Purpose: compute whether each canonical unit is currently active.

Return type: zero-argument `nn.Module`

Runtime signature:

```python
forward() -> torch.Tensor
```

Output:

- Tensor with shape `[total_units]`.
- Values are numeric mask values: `0` for inactive, `1` for active.
- Dtype/device follow the reduction plan output.

Implementation:

- Build a mapped reduction plan over canonical members with
  `WeightOperationType.ACTIVE`.
- Execute it with `ReductionCalc`.
- Because multiple members can contribute to one unit, normalize the accumulated
  output with `raw.gt(0).to(dtype=raw.dtype)`.

This calculation is dynamic and must not be cached because it depends on the
current parameter values.

### `create_units_to_module_axis_calc`

Purpose: aggregate a per-unit tensor into per-module input/output axis counts.

Return type: `PipelineCalc`

Runtime signature:

```python
forward(x: torch.Tensor) -> torch.Tensor
```

Input contract:

- `x` is a flat tensor with shape `[total_units]`.
- For structured-BOPs, `x` is usually `UNIT_ACTIVE_MASK()`.

Output:

- Flat tensor with shape `[num_weighted_modules * 2]`.
- Logical view is `[num_weighted_modules, 2]`.
- Column `0` is the module input axis.
- Column `1` is the module output axis.
- Module order is the same order used by `BITRATE_PR_MODULE`.

Implementation:

- Build a `PipelinePlan` over canonical members.
- Use a dummy `ValueTensorRef` with shape `[total_units]` only for plan
  metadata.
- Use `IdentityTensorReduction` against the input unit tensor.
- Map each canonical unit to the module-axis slot implied by the member:
  - `UnitAxis.IN_CHANNEL` contributes to the input-axis slot.
  - `UnitAxis.OUT_CHANNEL` contributes to the output-axis slot.
  - `UnitAxis.FEATURE`, `UnitAxis.QKV_CHANNEL`, `UnitAxis.QKV_HEAD`, and
    `UnitAxis.QKV_HEAD_DIM` contribute to the output-axis slot unless the
    structured-BOPs implementation later defines a more specific convention.
- De-duplicate `(module, axis, unit_index)` mappings during plan creation so a
  unit is counted once per module axis even if multiple members expose the same
  unit for that axis.
- Wrap the resulting `PipelinePlan` in `PipelineCalc`.

This calculation is not cached because it depends on the runtime input tensor.
The mapping plan is structural and can be compiled once.

### `create_bitrate_pr_module_calc`

Purpose: expose activation and weight bitrates for each weighted module.

Return type: zero-argument `nn.Module`

Runtime signature:

```python
forward() -> torch.Tensor
```

Output:

- Flat tensor with shape `[num_weighted_modules * 2]`.
- Logical view is `[num_weighted_modules, 2]`.
- Column `0` is activation bitrate.
- Column `1` is weight bitrate.
- Module order is the same order used by `UNITS_TO_MODULE_AXIS`.

Implementation:

- Build a mapped reduction plan with `create_codeq_bitrates(...)`.
- The module list must come from `ModuleBitrateExtractor.weighted_modules(model)`
  and must be reused as the shared structured-BOPs module ordering.
- Execute the plan with `ReductionCalc`.

This calculation should not be wrapped in `CachedCalculation` by default.
Bitrates may be provided by quantizer methods or mutable module attributes. If a
caller knows bitrates are fixed, a higher-level caller can cache it explicitly.

Structured-BOPs should combine the two module-axis calculations by viewing both
outputs as `[num_weighted_modules, 2]`. The intended default metric shape is:

```text
axis_counts = UNITS_TO_MODULE_AXIS(UNIT_ACTIVE_MASK()).view(num_modules, 2)
bitrates = BITRATE_PR_MODULE().view(num_modules, 2)
module_bops = axis_counts.prod(dim=1) * bitrates.prod(dim=1)
```

The tracker can then report per-module values and/or their sum.

### `create_group_sizes_calc`

Purpose: expose canonical group lengths for APIs such as
`torch.repeat_interleave`.

Return type: cached zero-argument `nn.Module`

Output:

- Tensor with shape `[num_groups]`.
- Values are `group.length`.
- Dtype must be `torch.long`.

Implementation:

- Build a tensor from canonical group lengths.
- Wrap it as a cached/static calculation.

This value is constant unless the canonical groups are rebuilt.

### `create_baseline_group_sizes_calc`

Purpose: expose the full baseline group size used in group-level arithmetic.

Return type: cached zero-argument `nn.Module`

Output:

- Tensor with shape `[num_groups]`.
- Values are the full canonical group lengths at calculation creation time.
- Dtype should be floating point and match the arithmetic dtype used by the
  active mask pipeline.

Implementation:

- Prefer building this through the same mapping path used at runtime:
  create a per-unit all-ones tensor with shape `[total_units]`, then apply
  `UNITS_TO_GROUP`.
- Cache the resulting tensor.

This value is constant unless the canonical groups are rebuilt.

### `create_group_change_effect_calc`

Purpose: compute a per-group coefficient describing how much the group-level
effect changes when one unit in that group becomes inactive.

Return type: cached zero-argument `nn.Module`

Output:

- Tensor with shape `[num_groups]`.
- Floating point dtype.

Initial implementation formula:

```text
unit_param_count = ReductionCalc(create_group_member_plan(groups, COUNT))()
group_param_count = units_to_group(unit_param_count)
group_change_effect = group_param_count / baseline_group_sizes.clamp_min(1)
```

This gives the average static parameter contribution per canonical unit in each
group. It is built from the mapped-reduction machinery, stays consistent with
the unit coordinate system, and is constant for a fixed group structure.

If a future regularizer needs a different semantic coefficient, change this
factory while keeping the output contract unchanged.

## Cache Semantics

All constants must be wrapped in `CachedCalculation` or an equivalent cached
module:

- `BASELINE_GROUP_SIZES`
- `GROUP_CHANGE_EFFECT`
- `GROUP_SIZES`

`CachedCalculation` must materialize the wrapped calculation result into an
owned buffer. Returning `calculation.output_anchor` is not sufficient for the
current `ReductionCalc` and `PipelineCalc` implementations because they allocate
fresh output tensors in `forward`.

Required behavior:

```python
cached = CachedCalculation(calc)
cached.refresh_cache(*args)
cached()  # returns the materialized cached tensor
```

The cache should refresh during factory construction so callers can use the
constant calculation immediately.

## StructureTracker Integration

`StructureTracker` is responsible for creating these calculations and reusing
shared structural work. Calculation factories should stay small and receive
already-normalized inputs.

### Stored Shared Context

During initialization, `StructureTracker` should keep:

- `self.canonical_groups`: the canonical unit groups.
- `self.calculations`: the existing `CalcType -> nn.Module` memoization dict.
- `self._weighted_module_entries`: lazily initialized tuple of
  `(name, module)` from `ModuleBitrateExtractor.weighted_modules(self.model)`.
- `self._weighted_modules`: lazily initialized tuple of modules derived from
  `_weighted_module_entries`.
- `self._weighted_module_index`: lazily initialized `dict[nn.Module, int]`
  keyed by module identity/order for unit-to-module-axis mapping.

The weighted module helpers must be the only source of module order for both
`UNITS_TO_MODULE_AXIS` and `BITRATE_PR_MODULE`.

### Memoized Dispatch

`get_calculation` should remain the only public calculation entry point. It
must memoize by `CalcType`, so a calculation requested by multiple trackers or
regularizers is constructed once:

```python
def get_calculation(self, calculation_type: CalcType):
    calculation_type = CalcType(calculation_type)
    if calculation_type not in self.calculations:
        self.calculations[calculation_type] = self._create_calculation(
            calculation_type
        )
    return self.calculations[calculation_type]
```

`_create_calculation` should dispatch like this:

```text
UNITS_TO_GROUP         -> create_units_to_group_calc(canonical_groups, ...)
UNIT_ACTIVE_MASK      -> create_unit_active_mask_calc(canonical_groups)
UNITS_TO_MODULE_AXIS  -> create_units_to_module_axis_calc(
                           canonical_groups,
                           weighted_module_index=self._weighted_module_index,
                           ...)
BITRATE_PR_MODULE     -> create_bitrate_pr_module_calc(self._weighted_modules, ...)
BASELINE_GROUP_SIZES  -> create_baseline_group_sizes_calc(
                           canonical_groups,
                           units_to_group=self.get_calculation(UNITS_TO_GROUP),
                           ...)
GROUP_CHANGE_EFFECT   -> create_group_change_effect_calc(
                           canonical_groups,
                           units_to_group=self.get_calculation(UNITS_TO_GROUP),
                           baseline_group_sizes=self.get_calculation(
                               BASELINE_GROUP_SIZES
                           ),
                           ...)
GROUP_SIZES           -> create_group_sizes_calc(canonical_groups, ...)
L2_NORM_PR_UNIT       -> existing group-member mapped reduction path
STRUCTURED_UNIT_SUM   -> existing group-member mapped reduction path
```

All group-based calculations must call `_require_groups`. `BITRATE_PR_MODULE`
does not require groups, but it does require `self.model`.

### Reuse Rules

Reuse these calculations instead of compiling equivalent plans again:

- `BASELINE_GROUP_SIZES` should reuse `UNITS_TO_GROUP` over a per-unit ones
  tensor.
- `GROUP_CHANGE_EFFECT` should reuse `UNITS_TO_GROUP` and
  `BASELINE_GROUP_SIZES`.
- `GroupLasso` and `StructuredBOPs` should receive calculations through
  `ensure_calculations(...)`; they should not construct calculations directly.
- `StructuredBOPs` should reuse `UNIT_ACTIVE_MASK` and pass its output into
  `UNITS_TO_MODULE_AXIS`.
- `UNITS_TO_MODULE_AXIS` and `BITRATE_PR_MODULE` must reuse the same weighted
  module order from the tracker.

Do not cache dynamic calculations:

- `UNIT_ACTIVE_MASK`
- `UNITS_TO_GROUP`
- `UNITS_TO_MODULE_AXIS`
- `BITRATE_PR_MODULE`

Cache structural constants:

- `BASELINE_GROUP_SIZES`
- `GROUP_CHANGE_EFFECT`
- `GROUP_SIZES`

### Dtype And Device

`UNITS_TO_GROUP` needs an arithmetic dtype/device. Prefer tracker `dtype/device`
when provided; otherwise infer them from the first available plan/source tensor
or fall back to CPU `float32` for metadata-only construction. The same
dtype/device rule applies to `UNITS_TO_MODULE_AXIS`.

Constant calculations should use the same arithmetic dtype/device except
`GROUP_SIZES`, which must remain `torch.long` for `torch.repeat_interleave`.

### Required Tracker/Regularizer Usage

`GroupLasso.required_calculations` should stay:

```python
required_calculations = (
    CalcType.L2_NORM_PR_UNIT,
    CalcType.UNITS_TO_GROUP,
    CalcType.UNIT_ACTIVE_MASK,
    CalcType.BASELINE_GROUP_SIZES,
    CalcType.GROUP_CHANGE_EFFECT,
    CalcType.GROUP_SIZES,
)
```

`StructuredBOPs.required_calculations` should be:

```python
required_calculations = (
    CalcType.UNIT_ACTIVE_MASK,
    CalcType.UNITS_TO_MODULE_AXIS,
    CalcType.BITRATE_PR_MODULE,
)
```

`StructureTracker.ensure_calculations(...)` should be the only place these
tuples are expanded into concrete modules.

## Supporting Cleanup

The enum values must be unique and string-valued:

```python
UNITS_TO_MODULE_AXIS = "units_to_module_axis"
UNIT_ACTIVE_MASK = "unit_active_mask"
BITRATE_PR_MODULE = "bitrate_pr_module"
BASELINE_GROUP_SIZES = "baseline_group_sizes"
GROUP_SIZES = "group_sizes"
```

Do not leave a trailing comma on enum values.

The calculations package exports should match the implemented class names. If
the runtime class is named `ReductionCalc`, either export that name directly or
provide an intentional compatibility alias such as:

```python
MappedReductionCalculation = ReductionCalc
```

## Non-Goals

- Do not add tests in this pass.
- Do not change canonicalization rules.
- Do not rewrite regularizer behavior beyond making these calculations
  available with the contracts above.
- Do not make `PipelineCalc` read from `ValueTensorRef` at runtime; it should
  continue to use its input argument.
