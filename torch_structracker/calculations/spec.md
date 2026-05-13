# Core Group Calculation Spec

## Scope

Implement factory functions for these calculation types:

- `CalcType.UNITS_TO_GROUP`
- `CalcType.ACTIVE_UNITS`
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

The main design goal is composability: if one calculation needs another
calculation, that dependency should be declared in a calculation spec and
resolved by `StructureTracker`. Factories should not secretly call back into
`StructureTracker`, and composite calculations should not hide reusable
sub-calculations inside their constructors.

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

### `create_active_units_calc`

Purpose: compute raw active member contribution per canonical unit.

Return type: `ReductionCalc`

Runtime signature:

```python
forward() -> torch.Tensor
```

Output:

- Tensor with shape `[total_units]`.
- Values are accumulated active contributions from canonical members. Values
  may be greater than `1` when multiple members contribute to the same unit.

Implementation:

- Build a mapped reduction plan over canonical members with
  `WeightOperationType.ACTIVE`.
- Execute it with `ReductionCalc`.

This calculation is dynamic and must not be cached because it depends on the
current parameter values. `UNIT_ACTIVE_MASK` composes from this calculation.

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

- Receive `ACTIVE_UNITS` as a declared dependency.
- Because multiple members can contribute to one unit, normalize the accumulated
  output with `raw.gt(0).to(dtype=raw.dtype)`.

This can be implemented as a small first-class calculation:

```python
class UnitActiveMaskCalc(BaseCalculation):
    calculation_type = CalcType.UNIT_ACTIVE_MASK
    required_calculations = (CalcType.ACTIVE_UNITS,)

    def forward(self) -> torch.Tensor:
        raw = self.compute(CalcType.ACTIVE_UNITS)
        return raw.gt(0).to(dtype=raw.dtype)
```

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

Purpose: compute how many structural parameter slots are removed when one unit
from each group is removed.

Return type: `ReductionCalc`, wrapped in `CachedCalculation` by the tracker

Output:

- Tensor with shape `[num_groups]`.
- Values are the structural parameter slots represented by one representative
  unit from each group.
- Dtype follows the `COUNT` reduction output.

Implementation:

- Build a mapped reduction plan over canonical members with
  `WeightOperationType.COUNT`.
- The plan output length is `num_groups`, not `total_units`.
- For every canonical group, select one representative unit. Use
  `group.offset`, the first unit in the group.
- For each canonical member that contributes to that representative unit, map
  only the representative unit's source count into `group.group_id`.
- Accumulate all member contributions into that group slot.
- Execute the plan with `ReductionCalc`.

This calculation must not compute total group params and divide by group size.
The plan should directly count the effect of a single representative unit.

This relies on the structural assumption that every unit inside a canonical
group has the same parameter-removal effect. Under that assumption, the first
unit is representative of all units in the group.

This value is structural. Zeros in a live weight tensor do not change it because
it uses `WeightOperationType.COUNT`, not `ACTIVE`.

If a future regularizer needs a different semantic coefficient, change this
factory while keeping the output contract unchanged.

Plan factory shape:

```python
def create_group_change_effect_plan(
    groups: Iterable[CanonicalUnitGroup],
) -> MappedReductionPlan:
    canonical_groups = tuple(groups)
    members = canonical_members(canonical_groups)

    planner = GenericReductionPlanner[CanonicalMember](
        elements=members,
        output_length=len(canonical_groups),
    )

    compiled = planner.compile(
        ElementReductionRule[CanonicalMember](
            extractor=CanonicalMemberTensorExtractor(),
            reduction_mapper=UnitWeightReductionMapper(WeightOperationType.COUNT),
            mapping_strategy=RepresentativeUnitToGroupMapper(),
        )
    )
    return cast(MappedReductionPlan, compiled)
```

`RepresentativeUnitToGroupMapper` should emit an indexed gather from the member
reduction output to the group slot. It must find source positions whose
canonical destination equals `member.group_offset`.

```python
class RepresentativeUnitToGroupMapper(MappingStrategy[CanonicalMember]):
    def map(
        self,
        member: CanonicalMember,
        op: ReductionOp,
        builder: ReductionPlanBuilder,
    ) -> ReductionMapping:
        source_indices = representative_source_indices(member, op)
        return ReductionMapping(
            source=IndexSelection(source_indices),
            target=IndexSelection((member.group_id,) * len(source_indices)),
        )
```

If a member does not contribute to the representative unit, the rule should emit
no record for that member.

### Composite Calculation Classes

Not every calculation is directly a `ReductionCalc` or `PipelineCalc`.
Composite calculations are still first-class calculations when they appear in
the calculation registry. If a composite calculation is structural, the tracker
wraps it in `CachedCalculation`.

Use small modules for:

- `UnitActiveMaskCalc`: wraps active-unit mapped reduction and thresholds it.
- `BaselineGroupSizesCalc`: composes `UNITS_TO_GROUP` with a per-unit ones
  buffer to produce original group sizes.
- `GroupSizesCalc`: exposes group lengths for `torch.repeat_interleave`.

These classes should share the same calculation interface as plan-backed
calculations:

```python
class BaseCalculation(nn.Module):
    calculation_type: CalcType | None = None
    required_calculations: tuple[CalcType, ...] = ()
    cache_constant: bool = False

    def __init__(
        self,
        dependencies: Mapping[CalcType, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.dependencies = nn.ModuleDict(
            {} if dependencies is None else {
                calc_type.name: module
                for calc_type, module in dependencies.items()
            }
        )

    def calc(self, calc_type: CalcType) -> nn.Module:
        return self.dependencies[calc_type.name]

    def compute(self, calc_type: CalcType, *args, **kwargs) -> torch.Tensor:
        return self.calc(calc_type)(*args, **kwargs)
```

`ReductionCalc` and `PipelineCalc` may either subclass this base or expose
compatible metadata through their calculation specs. Do not force awkward
inheritance if a `CalculationSpec` can describe them cleanly.

## Calculation Specs

Define calculation specs in `torch_structracker/calculations/calculations.py`.
The spec registry is the dependency graph for calculations.

```python
@dataclass(frozen=True)
class CalculationSpec:
    calculation_type: CalcType
    create: Callable[
        [CalculationContext, Mapping[CalcType, nn.Module]],
        nn.Module,
    ]
    required_calculations: tuple[CalcType, ...] = ()
    requires_groups: bool = True
    cache_constant: bool = False
```

The context carries shared tracker state:

```python
@dataclass(frozen=True)
class CalculationContext:
    model: nn.Module
    canonical_groups: tuple[CanonicalUnitGroup, ...]
    device: torch.device | None
    dtype: torch.dtype | None
    weighted_modules: tuple[nn.Module, ...]
    weighted_module_index: Mapping[nn.Module, int]
```

Examples:

```python
CalcType.ACTIVE_UNITS: CalculationSpec(
    calculation_type=CalcType.ACTIVE_UNITS,
    create=lambda ctx, deps: create_active_units_calc(ctx.canonical_groups),
)

CalcType.UNIT_ACTIVE_MASK: CalculationSpec(
    calculation_type=CalcType.UNIT_ACTIVE_MASK,
    required_calculations=(CalcType.ACTIVE_UNITS,),
    create=lambda ctx, deps: create_unit_active_mask_calc(
        active_units=deps[CalcType.ACTIVE_UNITS],
    ),
)

CalcType.BASELINE_GROUP_SIZES: CalculationSpec(
    calculation_type=CalcType.BASELINE_GROUP_SIZES,
    required_calculations=(CalcType.UNITS_TO_GROUP,),
    cache_constant=True,
    create=lambda ctx, deps: create_baseline_group_sizes_calc(
        ctx.canonical_groups,
        units_to_group=deps[CalcType.UNITS_TO_GROUP],
        device=ctx.device,
        dtype=ctx.dtype,
    ),
)

CalcType.GROUP_CHANGE_EFFECT: CalculationSpec(
    calculation_type=CalcType.GROUP_CHANGE_EFFECT,
    cache_constant=True,
    create=lambda ctx, deps: create_group_change_effect_calc(ctx.canonical_groups),
)
```

This lets calculations compose like trackers and regularizers: dependencies are
declared next to the calculation spec, and the tracker resolves them once.

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
ACTIVE_UNITS          -> create_active_units_calc(canonical_groups)
UNIT_ACTIVE_MASK      -> create_unit_active_mask_calc(
                           active_units=self.get_calculation(ACTIVE_UNITS)
                         )
UNITS_TO_MODULE_AXIS  -> create_units_to_module_axis_calc(
                           canonical_groups,
                           weighted_module_index=self._weighted_module_index,
                           ...)
BITRATE_PR_MODULE     -> create_bitrate_pr_module_calc(self._weighted_modules, ...)
BASELINE_GROUP_SIZES  -> create_baseline_group_sizes_calc(
                           canonical_groups,
                           units_to_group=self.get_calculation(UNITS_TO_GROUP),
                           ...)
GROUP_CHANGE_EFFECT   -> create_group_change_effect_calc(canonical_groups)
GROUP_SIZES           -> create_group_sizes_calc(canonical_groups, ...)
L2_NORM_PR_UNIT       -> existing group-member mapped reduction path
STRUCTURED_UNIT_SUM   -> existing group-member mapped reduction path
```

All group-based calculations must call `_require_groups`. `BITRATE_PR_MODULE`
does not require groups, but it does require `self.model`.

### Reuse Rules

Reuse these calculations instead of compiling equivalent plans again:

- `UNIT_ACTIVE_MASK` should reuse `ACTIVE_UNITS`; it should not compile its own
  hidden `ACTIVE` plan.
- `BASELINE_GROUP_SIZES` should reuse `UNITS_TO_GROUP` over a per-unit ones
  tensor.
- `GROUP_CHANGE_EFFECT` should use a direct representative-unit-to-group
  mapped `COUNT` plan; it should not compute total group params and divide by
  group size.
- `GroupLasso` and `StructuredBOPs` should receive calculations through
  `ensure_calculations(...)`; they should not construct calculations directly.
- `StructuredBOPs` should reuse `UNIT_ACTIVE_MASK` and pass its output into
  `UNITS_TO_MODULE_AXIS`.
- `UNITS_TO_MODULE_AXIS` and `BITRATE_PR_MODULE` must reuse the same weighted
  module order from the tracker.

Do not cache dynamic calculations:

- `ACTIVE_UNITS`
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
ACTIVE_UNITS = "active_units"
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

## Test Plan

Before implementation, add failing tests for every public calculation and the
unified `StructureTracker` build path. The workflow should be:

```text
1. Add tests that encode the intended behavior.
2. Confirm the tests fail against the current implementation.
3. Implement the calculations/spec registry/tracker wiring.
4. Confirm the tests pass.
```

Do not rely on large pretrained models. Use small deterministic models where
expected tensors can be solved manually in the test.

### Required Coverage

Test each public calculation:

- `ACTIVE_UNITS`
- `UNIT_ACTIVE_MASK`
- `UNITS_TO_GROUP`
- `BASELINE_GROUP_SIZES`
- `GROUP_CHANGE_EFFECT`
- `GROUP_SIZES`
- `UNITS_TO_MODULE_AXIS`
- `BITRATE_PR_MODULE`
- `L2_NORM_PR_UNIT`
- `STRUCTURED_UNIT_SUM`

Test the unified build behavior:

- `StructureTracker.ensure_calculations(...)` recursively builds required
  calculation dependencies from `CalculationSpec`.
- Calculations are memoized: requesting the same `CalcType` twice returns the
  same module object.
- Cached calculations refresh at construction and return the cached tensor.
- Dynamic calculations are not wrapped in `CachedCalculation`.
- Shared weighted module order is reused by `UNITS_TO_MODULE_AXIS` and
  `BITRATE_PR_MODULE`.
- Circular calculation dependencies raise a clear error.
- Missing groups raise a clear error for group-required calculations.

### Model Coverage

Use small models with hand-computed expected values:

- Linear chain:
  - Two or three `nn.Linear` layers without bias.
  - Manually set weights with simple values and zeros.
  - Verify per-unit sums, L2 norms, active masks, group sizes, and group change
    effect.

- Conv/resnet-ish block:
  - Tiny `Conv2d -> BatchNorm2d -> Conv2d` residual-style block.
  - Keep channel counts small, such as 2 or 3.
  - Manually compute expected channel counts and representative-unit parameter
    effects.
  - Include at least one skipped/residual connection so canonical groups cover
    more than a straight chain.

- Transformer-ish block:
  - Tiny attention module using either `nn.MultiheadAttention` or a fused QKV
    `nn.Linear(embed_dim, 3 * embed_dim)`.
  - Use small dimensions, such as `embed_dim=4`, `num_heads=2`.
  - Cover channel mode and, where supported by canonicalization, head or
    head-dim mode.
  - Manually compute expected group lengths, active masks, and group change
    effects.

- Mixed weighted-module BOPs model:
  - Tiny model with at least two weighted modules.
  - Set module `activation_bitrate`, `weight_bitrate`, or shared `bitrate`
    attributes manually.
  - Verify `BITRATE_PR_MODULE` flat layout:
    `[module0_activation, module0_weight, module1_activation, module1_weight, ...]`.
  - Verify `UNITS_TO_MODULE_AXIS` uses the same module order.
  - Verify structured BOPs can be computed as:

    ```python
    module_bops = (
        units_to_module_axis(unit_active_mask)
        * bitrate_pr_module
    ).view(-1, 2).prod(dim=1)
    ```

### Manual Expected Values

Each test should include explicit expected tensors. Avoid assertions that only
check shape or compare one implementation against another implementation.

Good:

```python
torch.testing.assert_close(
    calculation(),
    torch.tensor([2.0, 3.0, 3.0]),
)
```

Avoid:

```python
expected = naive_version_using_the_same_plan(...)
```

For `GROUP_CHANGE_EFFECT`, expected values must be the representative-unit
structural count per group, not total group parameter count divided in the
assertion. The test may mention the manual reasoning, for example:

```text
group 0 representative unit contributes:
  fc1 row: 2 params
  fc2 column: 1 param
expected GROUP_CHANGE_EFFECT[0] = 3
```

For `UNIT_ACTIVE_MASK`, include weights with zeros inside a unit to verify that
zeros do not affect structural constants, while fully inactive units affect the
dynamic active mask.

## Non-Goals

- Do not change canonicalization rules.
- Do not rewrite regularizer behavior beyond making these calculations
  available with the contracts above.
- Do not make `PipelineCalc` read from `ValueTensorRef` at runtime; it should
  continue to use its input argument.
