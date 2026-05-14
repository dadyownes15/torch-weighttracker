# Active MACs Tensor Calculation Spec

## Goal

`CalcType.ACTIVE_MACS_PR_MODULE` returns one active-MAC scalar per weighted
module, in the same module order as `BITRATE_PR_MODULE`.

The calculation must be tensorized. Runtime should not branch over modules or
canonical groups. All structural information is compiled once into buffers.

## Core Idea

Represent each weighted module by full module axes:

```text
[input_axis_size, output_axis_size]
```

`baseline_axes` contains the actual dense module axes. `active_axes` must have
the same semantics. Non-canonical axes therefore remain at their baseline size;
only axes represented by canonical members are reduced when units become
inactive.

The runtime formula is:

```python
active_macs = baseline_macs * (active_axes / baseline_axes).prod(dim=1)
```

This is exact for weighted modules whose MAC count is separable over the two
module axes, such as `Linear` and dense `Conv2d(groups=1)` with fixed spatial
shape. For attention blocks represented as `qkv` and `proj` linear modules, this
accurately accounts for those projection modules. Attention score/apply terms
are separate operation terms and are not represented by a weighted module unless
we add explicit pseudo-terms later.

## V1 Support Matrix

Supported weighted modules:

- `nn.Linear`
- `nn.Conv2d` with `groups == 1`
- fused-QKV attention projections represented as `nn.Linear(embed_dim, 3 * embed_dim)`
- attention output projections represented as `nn.Linear(embed_dim, embed_dim)`

Rejected for V1:

- `nn.Conv2d` with `groups != 1`, including depthwise and grouped convs
- `nn.MultiheadAttention` parent modules
- attention score/apply operation terms
- fallback MAC estimates from weight element counts
- fallback axis sizes from arbitrary weight shapes

The rejection cases should fail with explicit errors. Silent fallback would mix
different semantics and make structured BOPs appear more accurate than they are.

## Calculation Set

The implementation should use four calculations.

`CalcType` must include:

```python
UNIT_DELTA_TO_MODULE_AXIS = "unit_delta_to_module_axis"
```

`BASELINE_MACS_PR_MODULE` and `BASELINE_MODULE_AXES` already exist as named
calculation types and should be wired as real registry entries.

### `BASELINE_MODULE_AXES`

Purpose: expose the actual dense module axes for each weighted module.

Return type: cached/static zero-argument calculation.

Output:

```text
[num_weighted_modules, 2]
```

Column `0` is the input axis. Column `1` is the output axis.

This calc is structural. It does not depend on current parameter values.

Definition:

```python
class BaselineModuleAxesCalc(StaticCalc):
    calculation_type = CalcType.BASELINE_MODULE_AXES

    def __init__(self, baseline_axes: torch.Tensor) -> None:
        super().__init__(baseline_axes)
```

Factory:

```python
def create_baseline_module_axes_calc(
    modules: Iterable[nn.Module],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> BaselineModuleAxesCalc:
    weighted_modules = tuple(modules)
    dtype = torch.float32 if dtype is None else dtype
    device = torch.device("cpu") if device is None else torch.device(device)
    baseline_axes = torch.tensor(
        [_module_axis_sizes(module) for module in weighted_modules],
        dtype=dtype,
        device=device,
    )
    return BaselineModuleAxesCalc(baseline_axes)
```

Helper:

```python
def _module_axis_sizes(module: nn.Module) -> tuple[float, float]:
    if isinstance(module, nn.Conv2d):
        return _conv_axis_sizes(module)
    if isinstance(module, nn.Linear):
        return float(module.in_features), float(module.out_features)
    if isinstance(module, nn.MultiheadAttention):
        raise ValueError(
            "ACTIVE_MACS_PR_MODULE does not support nn.MultiheadAttention parent "
            "modules in V1. Use projection Linear modules or add explicit MHA "
            "operation terms."
        )

    raise ValueError(
        f"Baseline module-axis sizes are not implemented for "
        f"{module.__class__.__name__}."
    )


def _conv_axis_sizes(module: nn.Conv2d) -> tuple[float, float]:
    if module.groups != 1:
        raise ValueError(
            "ACTIVE_MACS_PR_MODULE currently supports only Conv2d(groups=1)."
        )
    return float(module.in_channels), float(module.out_channels)
```

This deliberately copies only the dense Linear/Conv channel semantics V1
supports instead of calling `PrunerBox` directly, and it still avoids a
weight-shape fallback. If a weighted module cannot be interpreted through these
explicit rules, active MACs should fail rather than silently inventing axes.

### `BASELINE_MACS_PR_MODULE`

Purpose: expose dense baseline MACs for each weighted module.

Return type: cached/static zero-argument calculation.

Output:

```text
[num_weighted_modules]
```

This calc requires `ctx.example_inputs` and fvcore. If either is unavailable,
raise an error. Do not fall back to weight element counts; that would mix
parameter counts with runtime MACs.

fvcore is not assumed to be universally correct. Treat it as the baseline MAC
source only after validating that every weighted module has a named fvcore
entry. If fvcore cannot account for a weighted module used by structured BOPs,
fail loudly and add module-specific support or tests.

Definition:

```python
class BaselineMacsPrModuleCalc(StaticCalc):
    calculation_type = CalcType.BASELINE_MACS_PR_MODULE

    def __init__(self, baseline_macs: torch.Tensor) -> None:
        super().__init__(baseline_macs)
```

Factory:

```python
def create_baseline_macs_pr_module_calc(
    ctx: CalculationContext,
) -> BaselineMacsPrModuleCalc:
    if ctx.example_inputs is None:
        raise ValueError(
            "BASELINE_MACS_PR_MODULE requires example_inputs so fvcore can "
            "compute runtime MACs."
        )

    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as error:
        raise RuntimeError(
            "BASELINE_MACS_PR_MODULE requires fvcore. Install fvcore or disable "
            "structured BOPs MAC accounting."
        ) from error

    values = _fvcore_macs_by_weighted_module(ctx, FlopCountAnalysis)

    return BaselineMacsPrModuleCalc(
        torch.tensor(
            values,
            dtype=_calculation_dtype(ctx),
            device=_calculation_device(ctx),
        )
    )
```

Helpers:

```python
def _fvcore_macs_by_weighted_module(
    ctx: CalculationContext,
    flop_count_analysis,
) -> list[float]:
    names_by_module = {module: name for name, module in ctx.model.named_modules()}
    analysis = flop_count_analysis(ctx.model, ctx.example_inputs)
    by_module = analysis.by_module()

    if hasattr(analysis, "uncalled_modules"):
        uncalled = set(analysis.uncalled_modules())
    else:
        uncalled = set()

    values: list[float] = []
    missing: list[str] = []
    for module in ctx.weighted_modules:
        name = names_by_module.get(module)
        if name is None or name not in by_module or name in uncalled:
            missing.append("<unnamed>" if name is None else name)
            continue
        values.append(float(by_module[name]))

    if missing:
        raise ValueError(
            "fvcore did not report MACs for weighted modules: "
            + ", ".join(missing)
        )

    return values
```

Do not fail just because fvcore reports unsupported operations elsewhere in the
model. The active-MAC baseline only requires reliable attribution for the
weighted modules included in structured BOPs. If an unsupported op corresponds
to a weighted module, it should already be caught by the named-module and
uncalled-module checks above, or by a focused parity test.

Verification requirement: add focused tests that physically prune representative
modules and compare the predicted active MACs against fvcore on the pruned model
for supported cases. Existing fvcore parity tests should be extended for the
new active-MAC path before relying on it for structured BOPs.

### `UNIT_DELTA_TO_MODULE_AXIS`

Purpose: map active unit masks to full module-axis deltas.

Return type: `PipelineCalc`.

Runtime signature:

```python
forward(active_units: torch.Tensor) -> torch.Tensor
```

Output:

```text
[num_weighted_modules * 2]
```

The input is expected to be:

```python
UNIT_ACTIVE_MASK()
```

The pipeline reduction turns active-mask values into axis deltas with:

```python
(active_units - 1) * axis_multiplier
```

So active units contribute `0`, and inactive units contribute a negative
module-axis delta.

Factory:

```python
def create_unit_delta_to_module_axis_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    weighted_module_index: Mapping[nn.Module, int],
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PipelineCalc:
    plan = create_unit_delta_to_module_axis_plan(
        groups,
        weighted_module_index=weighted_module_index,
        device=device,
        dtype=dtype,
    )
    return create_pipeline_calculation(
        plan,
        calculation_type=CalcType.UNIT_DELTA_TO_MODULE_AXIS,
    )
```

Registry:

```python
CalcType.UNIT_DELTA_TO_MODULE_AXIS: CalculationSpec(
    calculation_type=CalcType.UNIT_DELTA_TO_MODULE_AXIS,
    create=lambda ctx, deps: create_unit_delta_to_module_axis_calc(
        ctx.canonical_groups,
        weighted_module_index=ctx.weighted_module_index,
        device=_calculation_device(ctx),
        dtype=_calculation_dtype(ctx),
    ),
)
```

No separate `UNIT_DELTA` calculation is needed. The active-mask-to-delta
conversion is a local torch operation inside the `UNIT_DELTA_TO_MODULE_AXIS`
pipeline reduction.

Suggested file shape:

```text
torch_structracker/calculations/calcs/unit_delta_to_module_axis.py
  create_unit_delta_to_module_axis_calc(...)

torch_structracker/plans/mapping_plan.py
  create_unit_delta_to_module_axis_plan(...)
```

### `ACTIVE_MACS_PR_MODULE`

Purpose: compute active MACs from baseline MACs and active module axes.

Return type: dynamic zero-argument calculation.

Output:

```text
[num_weighted_modules]
```

Required calculations:

```text
UNIT_ACTIVE_MASK
UNIT_DELTA_TO_MODULE_AXIS
BASELINE_MACS_PR_MODULE
BASELINE_MODULE_AXES
```

Runtime:

```python
class ActiveMacsPrModuleCalc(Calculation):
    required_calculations = (
        CalcType.UNIT_ACTIVE_MASK,
        CalcType.UNIT_DELTA_TO_MODULE_AXIS,
        CalcType.BASELINE_MACS_PR_MODULE,
        CalcType.BASELINE_MODULE_AXES,
    )

    def __init__(self, dependencies: Mapping[CalcType, nn.Module]) -> None:
        super().__init__(dependencies)

    def forward(self) -> torch.Tensor:
        active_units = self.compute(CalcType.UNIT_ACTIVE_MASK)
        baseline_axes = self.compute(CalcType.BASELINE_MODULE_AXES)
        baseline_macs = self.compute(CalcType.BASELINE_MACS_PR_MODULE)

        axis_delta = self.compute(
            CalcType.UNIT_DELTA_TO_MODULE_AXIS,
            active_units,
        ).view_as(baseline_axes)
        active_axes = baseline_axes + axis_delta

        return baseline_macs * (active_axes / baseline_axes).prod(dim=1)
```

Do not expose record tensors directly from `ActiveMacsPrModuleCalc`. The
source/target index handling belongs inside `UNIT_DELTA_TO_MODULE_AXIS`, like
the other mapped calculations.

## Pipeline Plan

`UNIT_DELTA_TO_MODULE_AXIS` uses the existing `PipelineCalc` factory. The
runtime input is the active unit mask supplied by its consumer.

The pipeline plan records:

```text
source_unit_index -> target_module_axis_index with axis_multiplier
```

where:

```text
target_module_axis_index = module_index * 2 + axis
axis 0 = input axis
axis 1 = output axis
```

No custom calculation class is needed:

```python
create_pipeline_calculation(
    plan,
    calculation_type=CalcType.UNIT_DELTA_TO_MODULE_AXIS,
)
```

Plan creation:

```python
class ActiveUnitAxisDeltaReduction:
    def __init__(self, multiplier: float) -> None:
        self.multiplier = float(multiplier)

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return (value.reshape(-1) - 1) * self.multiplier

    def output_spec(self, source_spec: TensorSpec) -> TensorSpec:
        return TensorSpec(
            shape=torch.Size([numel(source_spec.shape)]),
            dtype=source_spec.dtype,
            device=source_spec.device,
        )

    def identity_key(self):
        return ("active_unit_axis_delta", self.multiplier)
```

```python
def create_unit_delta_to_module_axis_plan(
    groups: Iterable[CanonicalUnitGroup],
    *,
    weighted_module_index: Mapping[nn.Module, int],
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> PipelinePlan:
    canonical_groups = tuple(groups)
    input_ref = _unit_input_ref(canonical_groups, device=device, dtype=dtype)
    mapper = UnitDeltaToModuleAxisMapper(
        weighted_module_index=weighted_module_index
    )
    planner = GenericReductionPlanner[CanonicalMember](
        elements=canonical_members(canonical_groups),
        output_length=len(weighted_module_index) * 2,
    )
    compiled = planner.compile(
        PipelineReductionRule[CanonicalMember](
            input=input_ref,
            reduction_mapper=lambda member: ActiveUnitAxisDeltaReduction(
                axis_multiplier_for_member(
                    member,
                    axis=module_axis_for_member(member),
                )
            ),
            mapping_strategy=mapper,
        ),
        input_spec=input_ref.source_spec(),
    )
    return cast(PipelinePlan, compiled)
```

```python
class UnitDeltaToModuleAxisMapper(MappingStrategy[CanonicalMember]):
    def __init__(
        self,
        *,
        weighted_module_index: Mapping[nn.Module, int],
    ) -> None:
        self.weighted_module_index = weighted_module_index
        self.seen: set[tuple[int, int, int]] = set()

    def map(
        self,
        element: CanonicalMember,
        op: ReductionOp,
        builder: ReductionPlanBuilder,
    ) -> ReductionMapping:
        module_index = self.weighted_module_index.get(element.module)
        if module_index is None:
            return ReductionMapping(
                source=IndexSelection(()),
                target=IndexSelection(()),
            )

        axis = module_axis_for_member(element)
        target = module_index * 2 + axis
        source_indices: list[int] = []

        for unit_index in element.unit_indices:
            key = (module_index, axis, int(unit_index))
            if key in self.seen:
                continue
            self.seen.add(key)
            source_indices.append(int(unit_index))

        return ReductionMapping(
            source=IndexSelection(tuple(source_indices)),
            target=IndexSelection((target,) * len(source_indices)),
        )
```

`axis_multiplier_for_member(...)` is structural and runs only at factory time.
It uses `member.unit_axis`, `member.source_layout`, `member.num_heads`, and
`member.head_dim`.

For V1, attention multipliers apply only to projection modules represented as
`nn.Linear`, such as a timm-style `qkv: Linear(E, 3E)` and `proj: Linear(E, E)`.
Do not apply the fused-QKV multiplier to an `nn.MultiheadAttention` parent
module; that parent is rejected by `BASELINE_MODULE_AXES`.

Axis selection and multiplier selection are separate:

```python
def module_axis_for_member(member: CanonicalMember) -> int:
    return 0 if member.unit_axis == UnitAxis.IN_CHANNEL else 1
```

This keeps plain linked members such as `proj:prune_in_channels` on the input
axis, while QKV source members stay on the output axis.

The multiplier is based on the semantic unit represented by the canonical group:

```python
def units_to_embed_multiplier_for_member(member: CanonicalMember) -> float:
    if member.unit_axis == UnitAxis.QKV_CHANNEL:
        return 1.0
    if member.unit_axis == UnitAxis.QKV_HEAD:
        return float(member.head_dim)
    if member.unit_axis == UnitAxis.QKV_HEAD_DIM:
        return float(member.num_heads)

    if member.num_heads is not None and member.head_dim is not None:
        if member.group_length == member.num_heads:
            return float(member.head_dim)
        if member.group_length == member.head_dim:
            return float(member.num_heads)
        if member.embed_dim is not None and member.group_length == member.embed_dim:
            return 1.0

    return 1.0
```

The `group_length` checks are what make linked plain members work. For example,
in a `proj:prune_in_channels:head_dim` group, the `proj` member has
`UnitAxis.IN_CHANNEL`, but the group length is `head_dim`, so one canonical unit
still contributes `num_heads` concrete channels to the `proj` input axis.

Then:

```python
def axis_multiplier_for_member(member: CanonicalMember, *, axis: int) -> float:
    multiplier = units_to_embed_multiplier_for_member(member)

    if member.source_layout in {SourceLayout.FUSED_QKV, SourceLayout.SEPARATE_QKV}:
        if axis == 1:
            return 3.0 * multiplier

    return multiplier
```

Concrete `qkv`/`proj` delta example:

```text
H = 4
D = 8
E = 32
qkv:  Linear(32 -> 96)
proj: Linear(32 -> 32)
```

For a `head_dim` group with one inactive canonical unit:

```text
UNIT_ACTIVE_MASK()[u] = 0
```

The compiled records contribute:

```text
qkv output axis:  (0 - 1) * (3 * H) = -12
proj input axis:  (0 - 1) * H       = -4
```

So `UNIT_DELTA_TO_MODULE_AXIS(UNIT_ACTIVE_MASK())` computes a flat module-axis
delta where the `qkv` output slot is reduced by 12 concrete Q/K/V channels and
the `proj` input slot is reduced by 4 concrete embedding channels.
`ACTIVE_MACS` then adds this delta to `BASELINE_MODULE_AXES` before applying
the axis-ratio MAC formula.

## Runtime

`forward` stays a direct composition of calculations:

```python
active_units = self.compute(CalcType.UNIT_ACTIVE_MASK)
baseline_axes = self.compute(CalcType.BASELINE_MODULE_AXES)
baseline_macs = self.compute(CalcType.BASELINE_MACS_PR_MODULE)

axis_delta = self.compute(CalcType.UNIT_DELTA_TO_MODULE_AXIS, active_units).view_as(
    baseline_axes
)
active_axes = baseline_axes + axis_delta

return baseline_macs * (active_axes / baseline_axes).prod(dim=1)
```

This avoids a represented-axis mask. If an axis has no canonical members, the
reduction map emits zero delta and the active axis remains equal to its baseline
size. If all represented units are active, their deltas are zero. If a unit is
inactive, its contribution is subtracted from the axis.

## Baseline Axis Sizes

Default module axis sizes:

```text
Linear:              [in_features, out_features]
Conv2d(groups=1):    [in_channels, out_channels]
```

Other weighted module types must be explicitly added before they can participate
in active MACs.

For fused QKV linear layers, the module itself is still just a `Linear`:

```text
qkv: Linear(E -> 3E)  baseline_axes = [E, 3E]
proj: Linear(E -> E)  baseline_axes = [E, E]
```

For `nn.MultiheadAttention`, V1 should raise instead of assigning
`[embed_dim, embed_dim]`. The parent module mixes Q/K/V projection, output
projection, and attention score/apply work in a way that needs a separate
formula and a clear parent/child attribution policy.

## Axis Multipliers

Plain channel members:

```text
IN_CHANNEL   -> axis 0, multiplier 1
OUT_CHANNEL  -> axis 1, multiplier 1
FEATURE      -> axis 1, multiplier 1
```

Attention semantic members use the canonical member metadata:

```text
QKV_CHANNEL:
  units_to_embed_multiplier = 1

QKV_HEAD:
  units_to_embed_multiplier = head_dim

QKV_HEAD_DIM:
  units_to_embed_multiplier = num_heads
```

For normal attention internal axes, one active semantic unit contributes:

```text
axis_multiplier = units_to_embed_multiplier
```

For fused-QKV output axes, one active semantic unit contributes to Q, K, and V,
so:

```text
axis_multiplier = 3 * units_to_embed_multiplier
```

Example for a timm-style attention block:

```text
qkv:  Linear(E -> 3E)
proj: Linear(E -> E)
H = num_heads
D = head_dim
```

For a `QKV_HEAD_DIM` group connecting `qkv` output and `proj` input:

```text
qkv output multiplier  = 3 * H
proj input multiplier  = H
```

With `active_D` active head-dim units:

```text
qkv active output axis = 3 * H * active_D
proj active input axis = H * active_D
```

## Deduplication

Compile at most one record per:

```text
(module_index, axis, canonical_unit_index)
```

This prevents duplicated members from double-counting the same unit on the same
module axis. For fused-QKV members, the multiplier accounts for Q/K/V expansion;
do not add three separate records for the same semantic unit unless the tensor
source is explicitly split into separate Q, K, and V modules.

## Simple Example

For:

```text
fc1 = Linear(2 -> 3)
fc2 = Linear(3 -> 1)
```

Baseline:

```text
baseline_axes = [[2, 3], [3, 1]]
baseline_macs = [6, 3]
```

If the hidden canonical units are `[active, inactive, active]`, the compiled
records subtract one output unit from `fc1` and one input unit from `fc2`:

```text
active_axes = [[2, 2], [2, 1]]
active_macs = [6 * (2/2) * (2/3), 3 * (2/3) * (1/1)]
            = [4, 2]
```

## Factory Shape

Registry entries should have this shape:

```text
BASELINE_MODULE_AXES:
  required_calculations = ()
  cache_constant = True
  create = create_baseline_module_axes_calc(...)

BASELINE_MACS_PR_MODULE:
  required_calculations = ()
  cache_constant = True
  create = create_baseline_macs_pr_module_calc(...)

UNIT_DELTA_TO_MODULE_AXIS:
  required_calculations = ()
  cache_constant = False
  create = create_unit_delta_to_module_axis_calc(...)

ACTIVE_MACS_PR_MODULE:
  required_calculations = ActiveMacsPrModuleCalc.required_calculations
  cache_constant = False
  create = create_active_macs_pr_module_calc(ctx, dependencies=deps)
```

`ACTIVE_MACS_PR_MODULE` should be created through its factory:

```python
create_active_macs_pr_module_calc(ctx, dependencies=deps)
```

Required calculations:

```text
UNIT_ACTIVE_MASK
UNIT_DELTA_TO_MODULE_AXIS
BASELINE_MACS_PR_MODULE
BASELINE_MODULE_AXES
```

`UNITS_TO_MODULE_AXIS` is not required by this implementation because active
axes are computed directly with weighted axis records. It can remain available
for inspection and other consumers.
