# Plan: `view_zero_units` and `prune_zero_units`

## Goal

Add two `WeightTracker` methods that use the existing canonical-unit
calculation pipeline:

- `view_zero_units()`: inspect currently zero canonical units without mutating
  the model.
- `prune_zero_units()`: physically prune those zero canonical units through
  Torch-Pruning.

The implementation should not create a separate sparsity analysis path. It
should reuse the same calculations and plans already used by trackers and
regularizers.

## Existing Building Blocks

The relevant current calculations are:

- `CalcType.ACTIVE_UNITS`: counts active/nonzero weight contribution per
  canonical unit.
- `CalcType.UNIT_ACTIVE_MASK`: converts `ACTIVE_UNITS` into a binary active
  mask.
- `CalcType.UNITS_TO_GROUP`: reduces flat canonical-unit values into per-group
  values.
- `CalcType.PARAM_PR_UNIT`: estimates current parameter footprint per canonical
  unit.

The relevant pruning bridge is:

```python
pruning_group, pruning_idxs = tracker.get_prune_unit(group_id, unit_id)
```

where `unit_id` is local to the canonical group and `pruning_idxs` are already
mapped into the root Torch-Pruning handler coordinate system.

## Proposed Public API

```python
def view_zero_units(self) -> ZeroUnitView:
    ...

def prune_zero_units(self, *, dry_run: bool = False) -> PruneZeroUnitsResult:
    ...
```

`dry_run=True` should run the same selection logic as pruning but skip physical
mutation. This is useful for verifying exactly which pruning groups and indices
would be executed.

## Result Shapes

Keep the first implementation simple and explicit.

```python
@dataclass(frozen=True)
class ZeroUnit:
    group_id: int
    unit_id: int
    canonical_id: int
    pruning_idxs: tuple[int, ...]


@dataclass(frozen=True)
class ZeroUnitGroup:
    group_id: int
    offset: int
    length: int
    zero_units: tuple[ZeroUnit, ...]


@dataclass(frozen=True)
class ZeroUnitView:
    groups: tuple[ZeroUnitGroup, ...]
    total_zero_units: int


@dataclass(frozen=True)
class PruneZeroUnitsResult:
    view: ZeroUnitView
    pruned_units: int
    dry_run: bool
```

The important part is that the view preserves all three coordinate systems:

- `canonical_id`: flat calculation index.
- `group_id` + `unit_id`: canonical group-local index.
- `pruning_idxs`: physical root-handler indices passed to Torch-Pruning.

## `view_zero_units` Flow

1. Require canonical groups.
2. Compute the existing active mask:

   ```python
   active_mask = self.get_calculation(CalcType.UNIT_ACTIVE_MASK)()
   ```

3. For each `CanonicalUnitGroup`, slice its range:

   ```python
   start = group.offset
   stop = group.offset + group.length
   group_mask = active_mask[start:stop]
   ```

4. A unit is zero when `group_mask[unit_id] == 0`.
5. For each zero unit, call `get_prune_unit(group_id, unit_id)` only to resolve
   `pruning_idxs`; do not prune.
6. Return `ZeroUnitView`.

This keeps zero detection tied to the same reduction plans used by
`ACTIVE_UNITS` and `L2_NORM_PR_UNIT`.

## `prune_zero_units` Flow

1. Build `view = self.view_zero_units()`.
2. If `dry_run`, return a `PruneZeroUnitsResult` without mutation.
3. For each group in the view, merge zero units into one physical prune call
   where possible:

   ```python
   pruning_idxs = sorted(
       {
           idx
           for unit in zero_group.zero_units
           for idx in unit.pruning_idxs
       }
   )
   ```

4. Build one Torch-Pruning group from the canonical group root:

   ```python
   pruning_group, _ = self.get_prune_unit(group_id, first_unit_id)
   pruning_group.prune(idxs=pruning_idxs)
   ```

   If `get_prune_unit` returns a group already bound to the first unit's idxs,
   prefer adding a small private helper later:

   ```python
   self._get_pruning_group_for_unit_indices(group_id, pruning_idxs)
   ```

   That helper can reuse the same canonical member and dependency graph call
   without changing the public API.

5. After pruning, invalidate runtime caches:

   ```python
   self.calculations.clear()
   self._weighted_module_entries = None
   self._weighted_modules = None
   self._weighted_module_index = None
   ```

6. Rebuild dependency groups/canonical groups only if needed. This is the main
   implementation choice:

   - Minimal first version: prune in place and require a new `WeightTracker`
     after physical pruning.
   - Better version: rebuild `self.dependency_graph`, `self.groups`, and
     `self.canonical_groups` from stored `example_inputs`.

The better version is more ergonomic, but the minimal version is safer if the
current constructor/dependency rebuild path is still being changed.

## Important Implementation Detail

Do not call `group.prune()` once per zero canonical unit when several units
belong to the same canonical group. Physical pruning changes module dimensions,
so later unit indices in the same original coordinate system may shift after
the first prune.

Instead, merge zero units per canonical group and issue one pruning call per
group with the original source indices.

## Attention and Fused QKV

`view_zero_units` should not need special attention logic. It operates in
canonical-unit space.

`prune_zero_units` should also avoid special attention logic directly. It should
trust `get_prune_unit` / `CanonicalMember.pruning_source_indices` to provide
the physical root-handler IDs. For fused QKV-as-linear pruning, that means
semantic head/head-dim units must already be expanded to linear out-channel
rows before calling Torch-Pruning.

## Suggested Tests

1. Plain linear group:
   - Zero one output channel.
   - `view_zero_units()` reports the correct `group_id`, `unit_id`,
     `canonical_id`, and `pruning_idxs`.

2. No-zero case:
   - `view_zero_units()` returns no zero units.
   - `prune_zero_units(dry_run=True)` reports `pruned_units=0`.

3. Multi-unit same group:
   - Zero two units in one group.
   - `prune_zero_units()` merges them into one pruning call.

4. Fused QKV head group:
   - Zero one semantic head.
   - `view_zero_units()` reports the fused linear row IDs from
     `pruning_source_indices`.

5. Cache behavior:
   - After `prune_zero_units()`, stale calculations are not reused.

## Recommended First Implementation

Start with:

1. Dataclasses in `weight_tracker.py` or a small `pruning.py` module.
2. `WeightTracker.view_zero_units()`.
3. `WeightTracker.prune_zero_units(dry_run=True)`.
4. Then enable physical pruning with per-group merged indices.

Keep model/dependency-graph rebuilding as a follow-up if it complicates the
first patch. The core correctness issue is the selection and index mapping:

```text
UNIT_ACTIVE_MASK
-> zero canonical units
-> group_id + unit_id
-> get_prune_unit
-> physical Torch-Pruning idxs
-> one prune call per canonical group
```
