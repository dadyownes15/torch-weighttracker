# Reduction Compiler Optimization Spec

## Purpose

The reduction planner currently emits mapped calculation entries that are valid
and generic, but the calculation hot path executes each entry independently. If
multiple entries use equivalent reduction operations, the same source tensor can
be read and reduced multiple times in one forward pass.

This document specifies a compiler/runtime optimization that collapses repeated
reduction computations while preserving the existing public plan data types.

The target model is:

```text
unique reduction op result -> many output writes
```

The first implementation should optimize the private runtime representation of
`MappedReductionCalculation`, not replace the public plan ABI.

## Current Runtime Shape

The mapped reduction plan is represented by three entry types:

```python
SegmentEntry(op, start, length)
IndexedEntry(op, destination_indices)
IndexedGatherEntry(op, source_indices, destination_indices)
```

The calculation executes them approximately as:

```python
out.zero_()

for op, start, length in segment_entries:
    out.narrow(0, start, length).add_(op())

for op, dst in indexed_entries:
    out.index_add_(0, dst, op())

for op, src, dst in indexed_gather_entries:
    out.index_add_(0, dst, op().index_select(0, src))
```

This has a useful property: each entry uses the fastest known write shape for
that mapping.

It also has a cost: `op()` is called once per entry. If two entries have the
same `op.identity_key()`, they read the same source and apply the same
reduction. The second computation is redundant.

## Optimization Goals

1. Execute each unique reduction operation at most once per forward pass.
2. Preserve segment writes as segment writes.
3. Preserve indexed writes as indexed writes unless a better write path is
   proven safe.
4. Preserve indexed-gather writes as gather writes unless source and target are
   contiguous and can be narrowed.
5. Avoid changing public plan dataclasses in the first version.
6. Keep all caching per-forward only. Model weights can change between calls.
7. Keep validation metadata-only. Plan compilation must not execute tensor
   reductions.

## Non-Goals

The first implementation should not:

- Replace `SegmentEntry`, `IndexedEntry`, or `IndexedGatherEntry`.
- Convert segment entries to indexed or gather entries just to reduce entry
  count.
- Cache operation results across forward calls.
- Add model, module, group, QKV, or pruning-specific logic to the calculation.
- Require reductions to be pure beyond the existing `identity_key()` contract.

## Core Identity Contract

The compiler should group operations by an identity key:

```python
key = op.identity_key()
```

If an op does not expose `identity_key()`, use object identity:

```python
key = ("object", id(op))
```

Two operations with the same key are assumed to be semantically equivalent for
the current model structure:

- same source tensor reference
- same reduction function
- same output spec
- same output length

The compiler must verify that grouped ops agree on `output_spec` and
`output_length`. If they do not, compilation should raise a clear error because
the identity key is invalid or incomplete.

## Recommended V1: Private Grouped Runtime IR

Keep the public plan unchanged, but compile it into private runtime groups in
`MappedReductionCalculation.__init__`.

Conceptually:

```text
public plan entries
    -> intern equivalent ops by identity_key
    -> group writes by canonical op
    -> forward computes one op result per group
    -> forward applies all writes for that result
```

Private runtime structures can be simple dataclasses or tuples. The important
shape is:

```python
@dataclass(frozen=True)
class RuntimeOpGroup:
    op_index: int
    segment_writes: tuple[SegmentWrite, ...]
    indexed_writes: tuple[IndexedWrite, ...]
    gather_writes: tuple[GatherWrite, ...]


@dataclass(frozen=True)
class SegmentWrite:
    start: int
    length: int


@dataclass(frozen=True)
class IndexedWrite:
    destination_buffer_name: str


@dataclass(frozen=True)
class GatherWrite:
    source_buffer_name: str
    destination_buffer_name: str
```

Forward then becomes:

```python
out = self.accumulator
out.zero_()

for group in self.runtime_groups:
    values = self.ops[group.op_index]()

    for write in group.segment_writes:
        out.narrow(0, write.start, write.length).add_(values)

    for write in group.indexed_writes:
        dst = getattr(self, write.destination_buffer_name)
        out.index_add_(0, dst, values)

    for write in group.gather_writes:
        src = getattr(self, write.source_buffer_name)
        dst = getattr(self, write.destination_buffer_name)
        out.index_add_(0, dst, values.index_select(0, src))

return out
```

This is the key optimization: multiple segment writes can share one computed
`values` tensor without being converted to indexed writes.

Example:

```text
before:
  SegmentEntry(op_a, start=0,  length=L)
  SegmentEntry(op_b, start=L,  length=L)
  SegmentEntry(op_c, start=2L, length=L)

where:
  op_a.identity_key() == op_b.identity_key() == op_c.identity_key()

runtime:
  RuntimeOpGroup(
      op=op_a,
      segment_writes=((0, L), (L, L), (2L, L)),
  )

forward:
  values = op_a()
  out[0:L]     += values
  out[L:2L]    += values
  out[2L:3L]   += values
```

This collapses three reduction computations into one while preserving the fast
contiguous write path.

## Why Not Always Collapse To IndexedGatherEntry?

The public dataclasses can represent "one op result to many arbitrary
destinations" using `IndexedGatherEntry`:

```text
source_indices = (0, 1, 0, 1)
destination_indices = (0, 1, 4, 5)
```

That is functionally correct, but it can be slower than preserving segment
writes. A large segment write uses a view:

```python
out.narrow(0, start, length).add_(values)
```

The equivalent gather path materializes selected values and scatters them:

```python
out.index_add_(0, dst, values.index_select(0, src))
```

For expensive reductions, the saved `op()` call often dominates. For cheap large
outputs, the gather allocation and scatter can dominate and cause a regression.

Therefore V1 must not use "convert segments to gather" as the general collapse
strategy.

## Compiler Algorithm

### Step 1: Intern Operations

Build a mapping from identity key to canonical op index:

```python
key_to_op_index: dict[Hashable, int] = {}
ops = nn.ModuleList()
```

For each public entry:

1. Compute key.
2. If key is new, append the op to `ops`.
3. If key exists, validate compatibility with the canonical op.
4. Attach the entry write to the canonical op group.

Compatibility checks:

- `op.output_length == canonical.output_length`
- `op.output_spec == canonical.output_spec`

The compiler should prefer the first op as the canonical module. Later
equivalent ops are not registered in the runtime `ModuleList`, which avoids
duplicated modules and duplicated calls.

### Step 2: Preserve Write Kind

When compiling entries:

- `SegmentEntry` becomes `SegmentWrite`.
- `IndexedEntry` becomes `IndexedWrite`.
- `IndexedGatherEntry` becomes `GatherWrite`.

Do not lower segment writes to indexed writes.

### Step 3: Register Index Buffers Once

Destination and source index tensors should still be registered as non-persistent
buffers during construction, not rebuilt in `forward`.

For V1, each indexed/gather write may keep its own buffer. A later optimization
can deduplicate identical index buffers.

Buffer naming can remain implementation-private:

```text
dst_0
gather_src_0
gather_dst_0
```

or move to group-aware names:

```text
group_0_dst_0
group_0_gather_src_0
group_0_gather_dst_0
```

The public `destination_indices` property should continue to return all
registered destination buffers used by indexed and gather writes.

### Step 4: Execute Groups

Forward should compute each group once:

```python
for group in self.runtime_groups:
    values = self.ops[group.op_index]()
    ...
```

No dictionary cache is required inside `forward` if the runtime groups are
already unique by op. This keeps forward simple and avoids per-call key lookup.

If the implementation keeps the old flat runtime entries instead of grouping,
then a per-forward cache is required. The grouped IR is preferable because it
does the lookup once at construction time.

## Option A: Runtime CSE Only

This option keeps flat runtime entries but interns ops by identity key. Forward
uses a small per-call cache:

```python
cache = {}

for op_index, start, length in segment_entries:
    values = cache.get(op_index)
    if values is None:
        values = self.ops[op_index]()
        cache[op_index] = values
    out.narrow(0, start, length).add_(values)
```

Benefits:

- Smallest code change.
- Keeps public plan unchanged.
- Prevents duplicate `op()` calls.

Costs:

- Adds dictionary work in `forward`.
- Does not make the compiled runtime shape explicit.
- Harder to add segment fusion later.

This is acceptable as a minimal patch, but not the preferred architecture.

## Option B: Grouped Runtime IR

This is the recommended V1.

It groups writes by unique op during construction and forward simply iterates
groups.

Benefits:

- One `op()` call per unique op.
- No per-forward identity lookup.
- Preserves segment/indexed/gather write paths.
- Easy to extend with segment fusion.
- Keeps public plan unchanged.

Costs:

- Requires a new private runtime representation.
- Slightly larger construction logic.

This option gives the right performance shape without changing plan dataclasses.

## Option C: Segment Group Fusion

After grouped runtime IR exists, optimize repeated full-output segment writes.

If a group has equal-length adjacent segment writes:

```text
(0, L), (L, L), (2L, L), (3L, L)
```

then forward can apply the same values across a 2D view:

```python
block = out.narrow(0, 0, 4 * L).view(4, L)
block.add_(values)
```

This reduces Python loop overhead and can reduce kernel launch overhead on
accelerators.

Correctness requirements:

- All segment lengths must equal `values.numel()`.
- Starts must be exactly adjacent.
- Segments must not overlap unless overlap accumulation semantics are preserved.
- Output storage must support the view shape.

V1 should not depend on this. It should be a phase 2 optimization.

## Option D: Public Grouped Entry Types

Add public plan types:

```python
GroupedSegmentEntry(op, segments)
GroupedIndexedEntry(op, destination_indices_groups)
GroupedGatherEntry(op, source_destination_groups)
```

Benefits:

- Compression is visible in the plan.
- Validation can reason directly about grouped entries.
- Calculation construction becomes simpler.

Costs:

- Changes the public ABI.
- Requires broader test, docs, and migration work.
- Forces callers and validators to understand more entry types.

This is not recommended for the first implementation because the current goal is
to avoid changing existing data types.

## Option E: Full Source/Write IR

Replace mapped entries with a two-level graph:

```python
OpSource(id, op)
Write(source_id, source_selection, target_selection)
```

Benefits:

- Cleanest long-term model.
- Reuse is explicit.
- Future compiler passes become natural.
- Works for pipelines and more complex calculations.

Costs:

- Largest refactor.
- Existing entry dataclasses become legacy compatibility types.
- Requires a migration path.

This is a good long-term direction, but it is not necessary for the current
speedup.

## Additional Optimizations

### Contiguous Gather To Narrow

Some gather writes are actually contiguous:

```text
source_indices      = (5, 6, 7)
destination_indices = (20, 21, 22)
```

These can use narrow views:

```python
out.narrow(0, 20, 3).add_(values.narrow(0, 5, 3))
```

This avoids source and destination index tensors in the hot path. The compiler
can detect strictly increasing contiguous ranges and emit a private
`NarrowGatherWrite`.

This is phase 2.

### Index Buffer Deduplication

If many indexed writes use identical destination indices, register one buffer
and reference it from multiple writes.

This saves memory and buffer registration overhead, but usually does not change
the dominant runtime cost.

This is optional.

### Single-Destination Accumulation

The builder supports a singleton indexed target as an accumulation sink for a
longer source selection by repeating the destination index. The grouped runtime
should preserve this behavior exactly.

Example:

```text
source_indices      = (0, 1)
destination_indices = (0, 0)
```

This remains:

```python
out.index_add_(0, dst, values.index_select(0, src))
```

where `dst` contains repeated destination indices.

### Avoid Repeating Values

Do not create `values.repeat(...)` or `values.expand(...).reshape(...)` just to
merge writes. That introduces extra allocation or view complexity and can be
slower than applying multiple writes with one cached `values` tensor.

## Validation Rules

The existing plan validation rules still apply:

- plan output length is non-negative
- plan output spec shape matches output length
- entry op output specs are one-dimensional
- segment length matches op output length
- indexed destination count matches op output length
- gather source and destination counts match
- indices are non-negative and within bounds
- dtype and device are consistent across entries

Grouped runtime compilation adds:

- grouped ops with the same identity key must have matching output specs
- grouped ops with the same identity key must have matching output lengths
- each segment write in a group must still match group output length
- each indexed write in a group must still match group output length
- each gather source index must still be valid for group output length

Validation and runtime compilation must not call `op()`.

## Correctness Semantics

The optimized runtime must produce the same tensor as the unoptimized runtime:

```python
out = zeros(output_spec)
for entry in public_plan_entries_in_original_order:
    apply(entry)
```

Because all writes are additive, reordering writes across equivalent op groups
does not change results under ordinary floating point associativity assumptions.
However, to minimize numerical differences, the implementation can preserve
group creation order based on the first occurrence of each canonical op, and
preserve write order within each group.

Duplicate destination indices must continue to accumulate, not overwrite.

## Performance Expectations

This optimization helps most when:

- the same module/source appears in multiple mapped entries
- QKV semantic reductions are reused
- reductions traverse many parameters
- many dependency-group members map to the same source op

It is neutral or mildly helpful when:

- ops are cheap but repeated
- outputs are small

It should avoid regressions when:

- outputs are large and segment writes are cheap
- segment writes can stay as `narrow(...).add_(values)`

The key guardrail is: do not collapse segment writes by turning them into
indexed/gather writes.

## Testing Plan

### Unit Tests

Add tests for `MappedReductionCalculation`:

1. Repeated segment ops with the same identity key call the op once per forward.
2. Repeated indexed ops with the same identity key call the op once per forward.
3. Repeated indexed-gather ops with the same identity key call the op once per
   forward.
4. A mixed group with segment, indexed, and gather writes calls the op once per
   forward and produces the expected output.
5. Different identity keys do not share computation.
6. Same identity key with different output specs raises during construction.
7. Same identity key with different output lengths raises during construction.
8. Destination and source index buffers are registered once during construction
   and are not rebuilt in `forward`.
9. Duplicate destination indices still accumulate.
10. Output tensor data pointer remains stable across forwards.

### Builder Tests

The builder should continue to emit valid public entries. Add tests only if
builder-level canonicalization is introduced. For V1 grouped runtime IR, builder
tests should remain mostly unchanged.

### Regression Tests

Use a counting reduction operation:

```python
class CountingOp(nn.Module):
    def __init__(self, values, key):
        self.call_count = 0
        self.key = key

    def identity_key(self):
        return self.key

    def forward(self):
        self.call_count += 1
        return values
```

Assert one call per unique key per forward, not one call per entry.

### Benchmark Checks

Add or update a small synthetic benchmark with:

- duplicated expensive row reductions
- duplicated cheap large segment values
- mixed segment and gather writes

Acceptance criteria:

- duplicated expensive row reductions are materially faster
- duplicated cheap large segment values do not regress compared with current
  segment writes
- results match the old runtime within normal `torch.testing.assert_close`
  tolerances

## Implementation Notes

The best location is `torch_structracker/calculations/mapped_reduction.py`,
because every mapped calculation inherits the same hot path.

Suggested private helpers:

```python
def _op_identity_key(op) -> Hashable:
    if hasattr(op, "identity_key"):
        return op.identity_key()
    return ("object", id(op))


def _validate_equivalent_ops(canonical, candidate) -> None:
    if canonical.output_length != candidate.output_length:
        raise ValueError(...)
    if canonical.output_spec != candidate.output_spec:
        raise ValueError(...)
```

Suggested construction flow:

```python
self.ops = nn.ModuleList()
groups_by_key = {}
groups = []

def intern_op(op):
    key = _op_identity_key(op)
    if key not in groups_by_key:
        op_index = len(self.ops)
        self.ops.append(op)
        group = MutableRuntimeOpGroup(op_index)
        groups_by_key[key] = (op, group)
        groups.append(group)
        return group

    canonical, group = groups_by_key[key]
    _validate_equivalent_ops(canonical, op)
    return group
```

Then attach writes from each plan entry to the returned group.

The final `runtime_groups` should be immutable tuples so `forward` has a stable,
simple structure.

## Recommended Rollout

1. Implement grouped runtime IR in `MappedReductionCalculation`.
2. Keep public plan dataclasses unchanged.
3. Add counting-op tests for segment, indexed, gather, and mixed writes.
4. Run the existing reduction builder and mapped calculation tests.
5. Add optional benchmark checks for repeated heavy reductions and large cheap
   segment writes.
6. Consider segment group fusion only after V1 is correct and measured.

## Final Recommendation

Implement Option B first: private grouped runtime IR.

It gives the core optimization:

```text
many equivalent reduction entries -> one reduction op execution
```

while preserving the most important performance invariant:

```text
segment writes remain segment writes
```

This matches the current architecture, keeps the public data types stable, and
sets up later compiler passes without forcing a large refactor now.
