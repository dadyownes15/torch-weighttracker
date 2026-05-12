# Simplified Member Reduction Rule

This note describes a narrower alternative to the generic reduction planner for
the common structured pruning case:

1. Iterate through pruning groups.
2. Iterate through each group member.
3. Create the member-specific weight reduction operation.
4. Map the member output back to the group's structured unit indices.
5. Accumulate all member contributions into one output vector.

The key idea is that member reduction does not need the full extractor / mapper /
generic rule stack at first. A pruning group already carries the information we
need:

- the module to read parameters from
- the pruning handler, which tells us how that module is reduced
- the member root indices, which tell us which group units this member belongs to

## Target Behavior

For a group with `N` prunable units, the output owns `N` accumulator slots. Each
member emits a 1-D tensor of reduced parameter values. Those values are added to
the accumulator positions for that member's `root_idxs`.

Conceptually:

```python
out = torch.zeros(num_group_units)

for group in groups:
    for member in group:
        reducer = reducer_for_member(member)
        values = reducer()
        out.index_add_(0, member.root_idxs, values)
```

For the full plan compiler, `out` is not computed immediately. Instead, we first
describe the executable reducer plus the destination indices:

```python
ReductionRecord(
    op=reducer_for_member(member),
    target=IndexedTarget(destination_indices=member_output_indices),
)
```

At runtime, `StructuredUnitSum` can execute each reducer once and accumulate with
`index_add_`.

Important: `ReductionPlanBuilder.add(...)` should not be treated as a smart
compiler step. In the current builder shape, `add()` mainly:

- validates/touches output length
- lowers segment writes into indexed writes when needed
- appends `SegmentEntry`, `IndexedEntry`, or `IndexedGatherEntry`

It does not avoid duplicate reducer calculations by itself. If the same logical
operation is added twice, the runtime plan will contain two entries unless a
separate compile/finalize step coalesces them.

## Simple Compiler Shape

A member-specific compiler can be written directly around groups:

```python
def compile_member_unit_sum_plan(
    groups,
    *,
    operation_type=WeightOperationType.SUM,
) -> ReductionPlan:
    builder = ReductionPlanBuilder(output_length=count_group_units(groups))
    group_offset = 0

    for group in groups:
        group_size = len(group[0].root_idxs)

        for member in group:
            op = operation_for_member(member, operation_type)
            reducer = WeightReducer(
                parameter_extractor=ParameterExtractor(member.dep.target.module),
                operation=op,
            )

            destination_indices = tuple(
                group_offset + int(index) for index in member.root_idxs
            )

            builder.add(
                ReductionRecord(
                    op=reducer,
                    target=IndexedTarget(destination_indices),
                )
            )

        group_offset += group_size

    return builder.finalize()
```

This is intentionally specific:

- It assumes the source is pruning groups.
- It assumes the output layout is group units in group order.
- It assumes each member reduction output aligns with `member.root_idxs`.
- It only handles member parameter reductions.

That makes it easier to reason about than the generic planner while we stabilize
the structured unit sum path.

## Where Deduplication Belongs

Avoiding duplicate calculations is a compile concern, not an `add()` concern.
The simple compiler can maintain a dictionary keyed by the reducer identity:

```python
records_by_op: dict[Hashable, CompiledMemberRecord] = {}
```

Each member still contributes destination indices, but equal reducer operations
are grouped before the final plan is emitted:

```python
key = reducer.identity_key()
record = records_by_op.setdefault(key, CompiledMemberRecord(op=reducer))
record.add_mapping(
    source_indices=source_indices,
    destination_indices=destination_indices,
)
```

Then finalization emits one runtime reducer entry per unique operation/mapping
shape. That gives us the important property:

```python
values = reducer()  # computed once

for source_indices, destination_indices in record.mappings:
    selected = values if source_indices is None else values.index_select(0, source)
    out.index_add_(0, destination_indices, selected)
```

So the responsibilities should be:

- `add()`: append a concrete low-level mapping
- member compiler: detect equal logical reductions and group their mappings
- runtime calculation: execute each unique reducer once and accumulate all mapped
  contributions

If we do not add that grouping pass, the implementation is still correct, but it
can compute the same reducer multiple times.

## Handling Source Selection

Some modules may return more reduced values than the member should contribute.
For example, a fused QKV operation might reduce rows for `q`, `k`, and `v`, while
the member only maps a subset to the structured units.

In that case, the same simple rule still works by adding `source_indices`:

```python
builder.add(
    ReductionRecord(
        op=reducer,
        source_indices=member_source_indices,
        target=IndexedTarget(destination_indices),
    )
)
```

The runtime then does:

```python
values = reducer().index_select(0, source_indices)
out.index_add_(0, destination_indices, values)
```

So the compiler only has to decide two tuples:

- `source_indices`: which values to take from the reducer output
- `destination_indices`: where those values should be added in the global output

For simple linear, convolution, batchnorm, and layernorm members, these are often
the same logical positions. For attention modules, they may differ.

## Why This Is Enough For Member Unit Sum

`StructuredUnitSum` only needs an executable plan:

- reducer operations that produce flat tensors
- indexed destinations for accumulation
- optional source indices for partial reducer outputs
- total output length

A member-specific rule can produce exactly that without introducing separate
`MemberContext`, `MemberWeightExtractor`, and `MemberUnitMapper` objects.

Those abstractions are still useful later if we want one generic planner for many
different element types. For the immediate pruning member sum, a direct compiler
is simpler and has fewer moving parts.

## Suggested Public API

The narrow API could live next to `StructuredUnitSum` or in `reducer_plan.py`:

```python
plan = compile_member_unit_sum_plan(
    groups,
    operation_type=WeightOperationType.SUM,
)

calculation = StructuredUnitSum(plan)
```

Or, if we want to make the rule object explicit:

```python
rule = MemberUnitSumReductionRule(operation_type=WeightOperationType.SUM)
plan = rule.compile(groups)
calculation = StructuredUnitSum(plan)
```

The function form is probably enough for now. The rule object only becomes useful
if we expect multiple member operations with shared configuration.
