# Structured BOBS Calculator

## Problem Description

The goal is to add an input-free structured BOBS calculator to
`torch-structracker`.

Here, BOBS means a structural bit-operation score:

```text
active structural weight count * weight bitrate * activation bitrate
```

This is not a runtime FLOP/BOP measurement. The calculator should not need:

- an example input
- a batch size
- activation spatial shapes from a forward pass
- cloning and pruning the model
- a FLOP counter

The calculator should answer: given the current model weights, dependency
groups, and bitrates, how many structural weighted connections are still active?

The difficult part is not the final multiplication. The difficult part is:

1. deciding which dependency-group units are globally active,
2. mapping those global units back to each counted layer's input and output axes,
3. doing this without large CPU-bound Python loops in the hot path,
4. keeping GPU memory reasonable for large models.

## Current Background

The repo already has the core abstraction needed for the first part:

```text
dependency groups -> ReducerPlan -> global structured unit tensor
```

`StructuredUnitSum` currently follows this pattern:

```python
plan = compile_reducer_plan_from_groups(
    groups,
    operation_type=WeightOperationType.SUM,
)
```

and in `forward`:

```python
acc.zero_()

for reducer, dst in zip(self.reducers, self.destination_indices):
    acc.index_add_(0, dst, reducer().reshape(-1))

return acc
```

This means the dependency graph has already solved the important coupling
problem. If a hidden unit participates in several connected layers or branches,
the group maps those local slices into the same global structured unit.

For BOBS, we should reuse this idea, but with `L1` instead of raw `SUM`.

## Why L1 Activity

We discussed several ways to determine whether a structured unit is active.

### Boolean AND zero masks

One option was:

```text
local_zero per module-axis
global_zero = AND(local_zero for all coupled slices)
```

This has the right semantics. A global unit is removable only if every coupled
slice is zero.

The downside is that it introduces a separate boolean reduction system and needs
care around unseen units.

### Raw sum plus isclose

Another option was to reuse `StructuredUnitSum` directly and check:

```python
torch.isclose(unit_sum, 0)
```

This is unsafe because weights can cancel:

```text
[1, -1].sum() == 0
```

That slice is not structurally zero.

### Selected approach: L1 activity

Use L1 reducers:

```text
unit_activity = accumulated abs-sum activity across coupled slices
unit active   = unit_activity > eps
unit pruned   = unit_activity <= eps
```

Because L1 values are non-negative, there is no cancellation. A global unit has
zero activity only if every coupled module-axis slice has zero activity.

This is the clean replacement for the earlier norm/count idea.

## Relevant Module Reductions

The existing operation resolver already maps pruning handlers to reduction
dimensions. For BOBS activity, the operation type should be `L1`.

For `nn.Linear`:

```text
prune_linear_out_channels -> L1 over dim=1  # rows/output units
prune_linear_in_channels  -> L1 over dim=0  # columns/input units
```

For `nn.Conv2d` with `groups == 1`:

```text
prune_conv_out_channels -> L1 over dim=(1, 2, 3)
prune_conv_in_channels  -> L1 over dim=(0, 2, 3)
```

For BatchNorm and 1D LayerNorm:

```text
out/in pruning handlers -> L1 over dim=()
```

BatchNorm and LayerNorm should contribute activity, because their nonzero
parameters can keep a global unit from being removable. They should not
contribute BOBS counted layers in v1.

## Counted Layers vs Activity Layers

There are two roles:

```text
activity layer: contributes evidence that a global unit is active
counted layer: receives a structural BOBS count
```

For v1:

```text
Linear:              activity + BOBS
Conv2d groups == 1:  activity + BOBS
BatchNorm:           activity only
LayerNorm 1D:        activity only
Skip/concat/etc:     mapping only through dependency groups
```

Unsupported or deferred:

```text
grouped/depthwise conv
embeddings
MHA unless the current reducer path is explicitly extended for the intended
head/channel semantics
```

Skip connections do not need special BOBS logic. They are represented by the
dependency groups. If a unit appears in multiple branches, all contributing
slices accumulate into the same global unit activity.

## Why Layer Input and Output Counts Are Needed

BOBS for a layer depends on active units on both axes.

For Linear:

```python
weight.shape == [out_features, in_features]
structural_count = active_out * active_in
```

For Conv2d:

```python
weight.shape == [out_channels, in_channels, kernel_h, kernel_w]
structural_count = active_out * active_in * kernel_h * kernel_w
```

Therefore, after computing global unit activity, we must know:

```text
which global units map to this layer's input axis
which global units map to this layer's output axis
```

Without axis information, a layer having "N active units" is not enough to
compute its active structural weight count.

## Pipeline Solution

The preferred calculation pipeline is:

```text
StructuredUnitActivity
  -> LayerActiveCounts
  -> StructuredBobs

LayerBitrates
  -> StructuredBobs
```

### 1. StructuredUnitActivity

This is the current `StructuredUnitSum` pattern, but with `L1` reducers.

Output:

```python
unit_activity: Tensor[num_global_units]
```

Conceptually:

```python
plan = compile_reducer_plan_from_groups(
    groups,
    operation_type=WeightOperationType.L1,
)
```

This calculation determines whether each global dependency unit is active or
pruned/removable.

### 2. LayerActiveCounts

Input:

```python
unit_activity: Tensor[num_global_units]
```

Output:

```python
layer_active_counts: Tensor[num_layers, 2]
```

Column convention:

```text
[:, 0] = active input units
[:, 1] = active output units
```

The calculation first derives active units:

```python
active = unit_activity > eps
```

Then it counts active units per layer axis.

### 3. LayerBitrates

Output:

```python
layer_bitrates: Tensor[num_layers, 2]
```

Column convention:

```text
[:, 0] = weight bitrate
[:, 1] = activation bitrate
```

For v1, direct module attributes are enough:

```text
module.weight_bitrate
module.activation_bitrate
module.bitrate
fallback: 32
```

CodeQ-specific dynamic bit extraction can be added later behind the same
calculation contract.

### 4. StructuredBobs

Inputs:

```python
layer_active_counts
layer_bitrates
```

Output:

```python
bobs_by_layer: Tensor[num_layers]
```

For each counted layer:

```python
structure_count = active_in * active_out * multiplier
bobs = structure_count * weight_bits * activation_bits
```

where:

```text
Linear multiplier = 1
Conv2d multiplier = kernel_h * kernel_w
```

## Tracker Shape

The tracker should be a thin pipeline over reusable calculations, not a large
procedural analyzer.

Conceptually:

```python
class BobsTracker(BaseTracker):
    tracker_type = TrackerType.BOBS_TRACKER

    required_calculations = (
        CalculationType.STRUCTURED_UNIT_ACTIVITY,
        CalculationType.LAYER_ACTIVE_COUNTS,
        CalculationType.LAYER_BITRATES,
        CalculationType.STRUCTURED_BOBS,
    )

    def _compute(self, calculations):
        activity = calculations[CalculationType.STRUCTURED_UNIT_ACTIVITY]()
        counts = calculations[CalculationType.LAYER_ACTIVE_COUNTS](activity)
        bitrates = calculations[CalculationType.LAYER_BITRATES]()

        return calculations[CalculationType.STRUCTURED_BOBS](
            counts,
            bitrates,
        )

    def toMetric(self, bobs_by_layer):
        return {
            "structured_bobs": bobs_by_layer.sum().item(),
            "structured_bobs_by_layer": bobs_by_layer,
        }
```

This keeps calculations reusable while keeping the tracker easy to understand.

## Non-Pipeline Alternative

We also discussed a single large BOBS calculator:

```text
groups/model/bitrates -> total BOBS report
```

This would internally:

1. compute unit activity,
2. compute active counts,
3. extract bitrates,
4. compute BOBS,
5. format a report.

Benefits:

- simpler first script
- fewer public calculation types
- easier to keep all scratch buffers in one place

Problems:

- active unit logic cannot be reused by structured sparsity trackers
- bitrate extraction cannot be reused
- tests become less focused
- it does not fit the current tracker/calculation pattern
- it risks becoming another standalone analyzer instead of part of
  `torch-structracker`

The chosen design is a pipeline, but not an overly fine-grained one. We do not
need separate public calculations for every tiny multiplication. The useful
reuse boundaries are:

```text
unit activity
layer active counts
layer bitrates
structured bobs
```

## Memory and Performance

The first obvious implementation stores explicit destination index tensors, as
`StructuredUnitSum` currently does:

```python
acc.index_add_(0, dst, reducer().reshape(-1))
```

This is simple and GPU-friendly, but can use a lot of GPU memory for large
models because `dst` can contain one entry per mapped unit.

We discussed a more memory-efficient slice approach.

### Slice mappings for unit activity

Instead of storing:

```text
destination_indices = [10, 11, 12, 13, ...]
```

store contiguous mappings:

```text
source_start/source_end -> destination_start/destination_end
```

Runtime:

```python
unit_activity[dst_start:dst_end].add_(
    values[src_start:src_end]
)
```

Tradeoff:

```text
explicit indices:
  more memory
  fewer/larger tensor ops

slice mappings:
  much less memory
  Python loop over slices
```

For large transformer-like models, most mappings are expected to be contiguous,
so slice mappings should reduce memory substantially while keeping the loop count
manageable.

### Prefix-sum layer counts

For mapping active global units back to layer axes, avoid storing per-unit
layer-index tensors.

Store layer-axis slices:

```text
slice_start
slice_end
slice_target = layer_idx * 2 + axis_id
```

Then:

```python
active = unit_activity > eps
prefix[0] = 0
prefix[1:] = cumsum(active)
slice_counts = prefix[slice_end] - prefix[slice_start]
counts_flat.index_add_(0, slice_targets, slice_counts)
layer_counts = counts_flat.view(num_layers, 2)
```

This keeps memory closer to:

```text
O(num_units + num_slices + num_layers)
```

instead of:

```text
O(num_axis_links)
```

### GPU execution

The hot path should run on GPU if buffers are registered on the same device as
the model:

```text
abs/sum reducers
slice add
gt
cumsum
index_add_
where
multiplication
sum
```

Plan construction can use Python and CPU because it happens once.

Bitrate extraction may be a small Python loop in v1 if bitrates live as module
attributes. If dynamic CodeQ bitrates become hot, they should be gathered into a
single tensor representation.

## Caching Discussion

Regularizers and losses should compute fresh because they need valid autograd
graphs.

Trackers can optionally cache no-grad outputs, but the most important cache is
the compiled calculation state:

```text
groups -> reducer plans
groups -> layer-axis plans
model -> counted layer order
```

If persistent tracking cache is added later, the clean API is:

```python
struct_tracker = StructTracker(..., cache_tracking=True)
struct_tracker.attach_optimizer(optimizer)
```

The optimizer post-step hook can mark tracking values dirty after
`optimizer.step()`.

This is separate from the BOBS calculation design and should not be required for
the first correct implementation.

## Recommended Implementation Sequence

1. Add `StructuredUnitActivity` using the existing `ReducerPlan` with `L1`.
2. Add a layer-axis plan builder for counted layers.
3. Add `LayerActiveCounts` using slice/prefix counting.
4. Add `LayerBitrates` with direct attribute fallback.
5. Add `StructuredBobs`.
6. Update `BobsTracker` to pipe the calculations.
7. Add tests for MLP, partial zero coupling, Conv2d, BatchNorm/LayerNorm
   activity-only behavior, and skip/residual dependency groups.
8. After correctness, consider slice mappings inside `ReducerPlan` to reduce
   unit-activity destination-index memory.

## Key Invariant

The central invariant is:

```text
dependency groups define the global structured unit basis
L1 activity determines if each global unit is active
layer-axis mappings convert active global units into layer structural counts
bitrates convert structural counts into BOBS
```

