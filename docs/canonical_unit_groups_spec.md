# Canonical Unit Groups Spec

## Purpose

Structured calculations should share one unit coordinate system.

Today, raw torch-pruning groups expose enough information to build weight
reductions, but each downstream plan has to interpret group offsets, root
indices, local member indices, handlers, and transformer-specific QKV layout on
its own. That makes it easy for unit weight reductions, active-unit counts, and
unit-to-module maps to disagree about what a unit index means.

This spec introduces a compact canonical view over raw dependency groups:

```text
raw torch-pruning groups
    -> canonical unit groups
    -> reduction plans, active-unit maps, unit-to-module maps, regularizers
```

The canonical view is not a copied pruning graph. It is a small, mostly lazy
metadata layer that references the original group members and records the
normalized unit positions needed for computation.

## Design Goals

- Keep raw torch-pruning groups immutable during tracking plan construction.
- Make every calculation use the same global unit positions.
- Move transformer/QKV interpretation out of individual reducers.
- Put group interpretation in the StructureAnalyzer/StructureTracker layer,
  where dependency graph, pruning configuration, and model-level context are
  available.
- Keep `UnitWeightReductionMapper` small: it should map canonical layout and
  axis to a tensor reduction, not rediscover model semantics.
- Avoid memory blowups on large models by storing compact views and using
  segment/range mappings whenever possible.

## Ownership

Canonicalization belongs to the StructureAnalyzer/StructureTracker boundary.

That layer already owns the inputs needed to interpret raw groups:

- the `DependencyGraph`;
- raw torch-pruning groups;
- `root_module_types`;
- ignored layers and ignored parameters;
- `num_heads`;
- `prune_dim` and `prune_num_heads`;
- future model-level interpretation rules.

Plans and reducers should consume canonical groups. They should not inspect the
dependency graph or repeat transformer-specific group interpretation.

The implementation should keep the canonicalization logic in a small testable
module, but the StructureAnalyzer/StructureTracker should call it and store the
result:

```python
tracker.groups              # raw torch-pruning groups, if still exposed
tracker.canonical_groups    # shared normalized unit coordinate system
```

Downstream plan construction should prefer `canonical_groups`:

```python
create_group_member_plan(tracker.canonical_groups, ...)
create_active_units_plan(tracker.canonical_groups)
create_unit_to_module_plan(tracker.canonical_groups)
```

Calculations should not create plans from groups. The StructureAnalyzer or
StructureTracker is responsible for compiling the plan first, then initializing
the calculation:

```python
plan = create_group_member_plan(tracker.canonical_groups, ...)
calculation = MappedReductionCalculation(plan)
```

## Canonical Model

### Unit Groups

A canonical unit group represents one contiguous range in the global unit output
vector.

```python
@dataclass(frozen=True)
class CanonicalUnitGroup:
    group_id: int
    offset: int
    length: int
    unit_kind: UnitKind
    members: Iterable[CanonicalMember]
```

`offset` and `length` define the global coordinate range:

```text
[offset, offset + length)
```

Every plan derived from canonical groups must use this same coordinate system.

### Members

A canonical member is a reduction-oriented view of one source contribution.

```python
class SourceLayout(str, Enum):
    PLAIN = "plain"
    FUSED_QKV = "fused_qkv"
    SEPARATE_QKV = "separate_qkv"


class UnitAxis(str, Enum):
    OUT_CHANNEL = "out_channel"
    IN_CHANNEL = "in_channel"
    FEATURE = "feature"
    QKV_CHANNEL = "qkv_channel"
    QKV_HEAD = "qkv_head"
    QKV_HEAD_DIM = "qkv_head_dim"


@dataclass(frozen=True)
class CanonicalMember:
    group_id: int
    group_offset: int
    group_length: int

    member: object
    module: nn.Module
    handler: object

    source_layout: SourceLayout
    unit_axis: UnitAxis

    destination: SegmentSelection | IndexSelection
    source_indices: tuple[int, ...] | None = None

    embed_dim: int | None = None
    num_heads: int | None = None
    head_dim: int | None = None
```

`destination` says where this member contributes in the global unit vector.
`source_indices` is an optional gather from the member reduction output before
writing to that destination selection.

For full contiguous group writes, use `SegmentSelection` instead of
materializing a tuple of indices.

## Canonicalization Rules

### Plain Layers

For non-transformer members, canonicalization is mostly a handler-to-axis
translation:

```text
Linear out handler      -> source_layout=plain, unit_axis=out_channel
Linear in handler       -> source_layout=plain, unit_axis=in_channel
Conv2d out handler      -> source_layout=plain, unit_axis=out_channel
Conv2d in handler       -> source_layout=plain, unit_axis=in_channel
BatchNorm in/out        -> source_layout=plain, unit_axis=feature
LayerNorm in/out        -> source_layout=plain, unit_axis=feature
```

The canonicalizer maps `member.root_idxs` through the root group's position map:

```text
root_idx -> group_offset + root_position
```

This avoids assuming that root indices are dense or zero-based in the final
output coordinate system.

### Transformer And QKV Groups

Transformer semantics should be normalized before reduction planning.

The canonicalizer is responsible for:

- detecting direct `nn.MultiheadAttention` members;
- detecting fused QKV `Linear(embed_dim, 3 * embed_dim)` members when configured
  with `num_heads`;
- detecting separate Q/K/V layouts;
- choosing the semantic unit axis:
  - `qkv_channel` for embed-channel reductions;
  - `qkv_head` for head reductions;
  - `qkv_head_dim` for head-dimension reductions;
- computing `head_dim = embed_dim // num_heads`;
- emitting one semantic QKV member when possible.

This follows the same architectural idea used by torch-pruning's
`BasePruner`: attention groups are interpreted before final pruning execution,
not inside every downstream operation.

The mapper should not ask whether a `Linear` is secretly a QKV projection. The
canonical member should already say:

```text
source_layout=fused_qkv
unit_axis=qkv_head
embed_dim=...
num_heads=...
head_dim=...
```

### Group Root Rewrites

Some attention graphs are easier to compute if a downstream node becomes the
semantic root, similar to `_downstream_node_as_root_if_attention` in
`torch_pruning/pruner/algorithms/base_pruner.py`.

Canonicalization may re-root the canonical view, but must not mutate the raw
torch-pruning group. The output is simply a different canonical interpretation
of the same raw dependency information.

## Consumers

The same canonical groups should feed multiple plans.

### Unit Weight Reductions

`UnitWeightReductionMapper` consumes `source_layout + unit_axis`:

```text
plain + out_channel -> reduce rows
plain + in_channel  -> reduce columns
plain + feature     -> per-feature reduction
fused_qkv + qkv_*   -> QKV semantic reduction
separate_qkv + qkv_* -> tuple QKV semantic reduction
```

The extractor binds the source layout:

```text
plain        -> module.weight
fused_qkv    -> module.in_proj_weight or qkv.weight
separate_qkv -> q_proj_weight, k_proj_weight, v_proj_weight
```

The target mapper uses canonical destinations directly.

### Active Unit Maps

Active-unit plans use the same canonical members as weight reductions, but with
a `COUNT` operation instead of `SUM`, `L1`, or `L2`. This computes the active
parameter contribution per canonical unit in the current model structure.

For example, if unit `0` is represented by one output row with two weights and
one downstream input column with one weight, the active-unit value for unit `0`
is `3`.

The calculation still receives a compiled `MappedReductionPlan`; it does not
derive shape from an ad hoc tensor. The plan's `output_spec` defines the output
shape, dtype, and device.

### Unit-To-Module Maps

Unit-to-module maps use `CanonicalMember.destination_indices` to connect global
unit positions to owning modules.

This supports multiple map types from the same coordinate system:

```text
unit -> module
unit -> source module
unit -> target module
unit -> qkv projection
unit -> attention head
unit -> parameter source
```

The important invariant is that all maps agree on the same global unit index.

## Runtime And Memory Constraints

Canonicalization must remain compact for large networks.

Do:

- stream canonical members where possible;
- store references to original members/modules;
- use `SegmentSelection` for full contiguous group writes;
- store transformed index tuples only when required;
- emit one semantic QKV member instead of expanding Q/K/V rows.

Do not:

- copy the full dependency graph;
- mutate raw groups in place;
- expand every channel into a separate object;
- materialize full destination tuples for contiguous ranges;
- make every calculation redo transformer detection.

## Invariants

- Raw torch-pruning groups are unchanged by canonicalization.
- Every canonical group has a stable `offset` and `length`.
- Every canonical destination is inside its group's global range unless it is an
  intentional cross-group map.
- If `source_indices` is present, it has the same length as the destination
  selection.
- QKV canonical members always include enough metadata to compute their semantic
  reduction without inspecting analyzer state.
- All downstream plans derived from the same canonical groups share identical
  unit positions.

## Suggested Implementation Path

1. Add canonical dataclasses/enums in a small module owned by the
   StructureAnalyzer/StructureTracker layer.
2. Add an iterator that converts raw groups into canonical members without
   mutating the raw groups.
3. Have StructureAnalyzer/StructureTracker compute and store
   `canonical_groups` after raw group creation and ignored-layer filtering.
4. Update `create_group_member_plan` to consume canonical members.
5. Add indexed-gather support to the mapped reduction builder.
6. Add unit-to-module and active-unit plans on top of the same canonical groups.
7. Add transformer canonicalization tests before expanding mapper behavior.
