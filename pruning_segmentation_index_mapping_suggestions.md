# Segmentation Index to Pruning Group Mapping Suggestions

## Problem

`SegmentationIndex` / canonical unit IDs are currently useful for measurement and
calculation outputs, but `Torch-Pruning` needs indices in the root pruning
module's source coordinate system:

```python
group = dependency_graph.get_pruning_group(
    module=canonical_member.module,
    pruning_fn=canonical_member.handler,
    idxs=source_prune_ids,
)
group.prune()
```

Those `idxs` should be created from the canonical/root member, not from the
global canonical segment offset. The current direction of adding
`source_indices` is right, but it should not use `group_offset`; that offset is
only the global WeightTracker unit offset.

For the pruning adapter, `source_indices` must mean the exact indices understood
by the root Torch-Pruning handler. If Torch-Pruning sees an attention projection
as a plain `Linear` with `prune_linear_out_channels`, then the source indices are
linear out-channel indices. Torch-Pruning will not know that WeightTracker is
treating those channels as MHA heads or head dimensions, so WeightTracker must
expand semantic attention units into the linear-out IDs before calling
`get_pruning_group`.

## Suggested Model

Keep three IDs distinct:

- `unit_id`: index inside a `CanonicalUnitGroup`, from `0..group.length - 1`.
- `canonical_id`: global calculation index, equal to `group.offset + unit_id`.
- `source_id`: root pruning index passed to
  `DependencyGraph.get_pruning_group(..., idxs=source_ids)`.

Add an explicit source mapping to `CanonicalMember`, preferably on the canonical
root member:

```python
source_indices_by_unit: tuple[tuple[int, ...], ...]
```

For plain channel groups this is usually:

```python
((0,), (1,), (2,), ...)
```

For attention head pruning on a root `Linear(out_features=8)` with
`embed_dim=8`, `num_heads=2`, `head_dim=4`:

```python
((0, 1, 2, 3), (4, 5, 6, 7))
```

For attention head-dim pruning:

```python
((0, 4), (1, 5), (2, 6), (3, 7))
```

For QKV-channel pruning on the same linear-out coordinate space:

```python
((0,), (1,), ..., (embed_dim - 1,))
```

If the root module is a fused QKV `Linear(out_features=3 * embed_dim)` and the
actual pruning handler is `prune_linear_out_channels`, then the mapping must
expand into fused linear row IDs:

```python
# embed_dim=8, num_heads=2, head_dim=4
# head 0 across Q, K, V
(0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19)

# head-dim 0 across all heads and Q, K, V
(0, 4, 8, 12, 16, 20)
```

This is the important contract: `source_indices_by_unit` is not a semantic MHA
index. It is the physical root-handler index list to pass to Torch-Pruning.

## Canonical Member Changes

I would update `CanonicalMember` along these lines:

```python
@dataclass(frozen=True)
class CanonicalMember:
    ...
    is_canonical_root: bool
    source_id: str
    source_indices_by_unit: tuple[tuple[int, ...], ...]
```

`source_id` should identify the coordinate space, not just the module. Examples:

- `fc1:prune_out_channels`
- `fc2:prune_in_channels`
- `qkv:prune_linear_out_channels:head`
- `qkv:prune_linear_out_channels:head_dim`
- `qkv:prune_linear_out_channels:fused_qkv_head`
- `qkv:prune_linear_out_channels:fused_qkv_head_dim`

The canonical group should expose one selected pruning root:

```python
@dataclass(frozen=True)
class CanonicalUnitGroup:
    ...
    canonical_member: CanonicalMember
```

This avoids assuming `group.members[0]` is always the correct physical pruning
root. For raw Torch-Pruning groups that is often true, but attention views and
filtered members make the assumption fragile.

## Mapping Construction

During `canonicalize_groups`:

1. Build `root_to_position` from `items[0].root_idxs`, as now.
2. Select a `canonical_member` from the raw group. Initially this can be the
   first measurable/root member, but it should be explicit.
3. Build `source_indices_by_unit` from the canonical member's root handler
   coordinate system and the selected unit axis.
4. Keep `destination` for calculations exactly as it is now.
5. Do not derive physical prune IDs from `group_offset`.

Suggested helper:

```python
def source_indices_by_unit(
    member,
    *,
    group_length: int,
    attention: AttentionUnitConfig | None,
) -> tuple[tuple[int, ...], ...]:
    ...
```

Plain case:

```python
return tuple((idx,) for idx in member_root_indices(member))
```

Attention cases for a non-fused linear-out coordinate space:

```python
if unit_axis == UnitAxis.QKV_HEAD:
    return tuple(
        tuple(range(head * head_dim, (head + 1) * head_dim))
        for head in range(num_heads)
    )

if unit_axis == UnitAxis.QKV_HEAD_DIM:
    return tuple(
        tuple(dim + head * head_dim for head in range(num_heads))
        for dim in range(head_dim)
    )
```

Attention cases for a fused QKV linear-out coordinate space:

```python
def qkv_offsets(embed_dim: int) -> tuple[int, int, int]:
    return (0, embed_dim, 2 * embed_dim)

if unit_axis == UnitAxis.QKV_HEAD:
    return tuple(
        tuple(
            qkv_offset + channel
            for qkv_offset in qkv_offsets(embed_dim)
            for channel in range(head * head_dim, (head + 1) * head_dim)
        )
        for head in range(num_heads)
    )

if unit_axis == UnitAxis.QKV_HEAD_DIM:
    return tuple(
        tuple(
            qkv_offset + head * head_dim + dim
            for qkv_offset in qkv_offsets(embed_dim)
            for head in range(num_heads)
        )
        for dim in range(head_dim)
    )
```

The helper probably needs a `source_layout` or `source_coordinate` argument so
it can choose between plain `embed_dim` linear-out IDs and fused
`3 * embed_dim` linear-out IDs.

## Pruning Flow

Then `get_prune_unit` becomes a small adapter:

```python
def get_prune_unit(self, group_id: int, unit_id: int):
    group = self.canonical_groups[group_id]
    member = group.canonical_member
    source_ids = member.source_indices_by_unit[unit_id]

    return self.dependency_graph.get_pruning_group(
        module=member.module,
        pruning_fn=member.handler,
        idxs=source_ids,
    )
```

Execution should happen separately:

```python
pruning_group = tracker.get_prune_unit(group_id, unit_id)
pruning_group.prune()
```

That keeps inspection, validation, and execution separate.

For many units:

```python
source_ids = sorted(
    {
        source_id
        for unit_id in unit_ids
        for source_id in member.source_indices_by_unit[unit_id]
    }
)
```

Then call `get_pruning_group(..., idxs=source_ids)` once per canonical group.

## How This Connects to `ParamPrUnit`

`ParamPrUnit` already operates over the canonical unit space:

- `UNIT_ACTIVE_MASK` / `ACTIVE_UNITS`
- `UNITS_TO_GROUP`
- `GROUP_UNIT_PARAM_CHANGE`
- `GROUPS_TO_UNITS`

Physical pruning should reuse that same canonical-unit selection, then bridge
from canonical units to source prune IDs only at the final step:

```text
active mask / user-selected canonical IDs
-> group_id + unit_id
-> canonical_group.canonical_member
-> source_indices_by_unit[unit_id]
-> dependency_graph.get_pruning_group(...)
-> group.prune()
```

So the new source mapping should be modeled as static group metadata, not as a
calculation dependency. Calculations stay in canonical space; pruning adapters
translate canonical selections to source IDs.

## Tests I Would Add First

1. Plain linear/conv group:
   - canonical unit `1` maps to source ID `(1,)`.
   - `get_prune_unit(group_id, 1)` builds a Torch-Pruning group rooted at the
     canonical member.

2. Attention head group:
   - `embed_dim=8`, `num_heads=2`, `head_dim=4`.
   - non-fused linear-out head unit `0` maps to `(0, 1, 2, 3)`.
   - non-fused linear-out head unit `1` maps to `(4, 5, 6, 7)`.
   - fused QKV linear-out head unit `0` maps to
     `(0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19)`.

3. Attention head-dim group:
   - non-fused linear-out dim unit `0` maps to `(0, 4)`.
   - non-fused linear-out dim unit `3` maps to `(3, 7)`.
   - fused QKV linear-out dim unit `0` maps to `(0, 4, 8, 12, 16, 20)`.

4. Regression for global offsets:
   - second canonical group with `group.offset > 0`.
   - source IDs still start from local/root source coordinates, not
     `group.offset`.

5. Filtered/ignored members:
   - filtering does not change the selected `canonical_member` unless the root
     member is removed.
   - if the root member is removed, fail explicitly or select a new compatible
     member and recompute source mapping in that member's coordinate space.

## Main Recommendation

Implement `source_id` plus `source_indices_by_unit` as canonical metadata, and
make `CanonicalUnitGroup` explicitly point to its pruning root
`canonical_member`. Then `SegmentationIndex` can remain a calculation/reporting
index, while pruning uses source IDs derived from the canonical member's root
coordinate system.
