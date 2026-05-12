Hmm no, not like this. The idea is that we create a generic planer, which can generate computation steps across any list. For example it could be modules, it could pruneMembers, our it could module.parameterizations. For each element it should specify an operation, and a way to index. 


This then allows calculations to use it by 

    @torch.no_grad()
    def forward(self) -> torch.Tensor:
        out = self.accumulator
        out.zero_()

        for reducer, start, length in self.segment_entries:
            out.narrow(0, start, length).add_(reducer())

        for reducer, dst in self.indexed_entries:
            out.index_add_(0, dst, reducer())

        for reducer, src, dst in self.indexed_gather_entries:
            out.index_add_(0, dst, reducer().index_select(0, src))

        return out


This allows the calculations to abstract away any knowlegde of what the element is, or how to map it. It simply execute one or multiple plans


Yes — that is the abstraction.

You do **not** want every plan compiler to know about groups, modules, parametrizations, MHA, offsets, and targets. You want one generic planner that says:

```text
for each element:
    build one or more zero-arg operations
    build the mapping target for each operation result
    add the step to a generic reduction plan
```

Then calculations execute the final plan and do not know what an “element” was.

The clean split is:

```text
GenericReductionPlanner:
    iterates any element list

ReductionRule:
    knows how to turn one element into operation + target records

ReductionPlanBuilder:
    buckets records into segment / indexed / indexed-gather entries

Calculation:
    blindly executes the plan
```

This matches your spec: plans describe the recipe, while calculations own buffers and execute hot tensor work. 

---

## 1. Generic final plan

This plan is not weight-specific.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Protocol

import torch
import torch.nn as nn


class ReductionOp(Protocol):
    """
    Final executable operation used by calculations.

    It can internally read a module, parameter, parametrization, activation buffer,
    quantizer, or anything else.

    Contract:
        op() returns a flat 1-D tensor.
    """

    def __call__(self) -> torch.Tensor:
        ...

    def identity_key(self) -> Hashable:
        ...

    def first_tensor(self) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class SegmentTarget:
    start: int
    length: int


@dataclass(frozen=True)
class IndexedTarget:
    destination_indices: tuple[int, ...]


@dataclass(frozen=True)
class ReductionRecord:
    op: ReductionOp
    target: SegmentTarget | IndexedTarget

    # Optional selection from op() before writing.
    # None means use full op output.
    source_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SegmentEntry:
    op: ReductionOp
    start: int
    length: int


@dataclass(frozen=True)
class IndexedEntry:
    op: ReductionOp
    destination_indices: tuple[int, ...]


@dataclass(frozen=True)
class IndexedGatherEntry:
    op: ReductionOp
    source_indices: tuple[int, ...]
    destination_indices: tuple[int, ...]


@dataclass(frozen=True)
class ReductionPlan:
    output_length: int
    segment_entries: tuple[SegmentEntry, ...] = ()
    indexed_entries: tuple[IndexedEntry, ...] = ()
    indexed_gather_entries: tuple[IndexedGatherEntry, ...] = ()
    output_labels: tuple[str, ...] | None = None
```

This gives calculations exactly the shape you want:

```python
@torch.no_grad()
def forward(self) -> torch.Tensor:
    out = self.accumulator
    out.zero_()

    for op, start, length in self.segment_entries:
        out.narrow(0, start, length).add_(op())

    for op, dst in self.indexed_entries:
        out.index_add_(0, dst, op())

    for op, src, dst in self.indexed_gather_entries:
        out.index_add_(0, dst, op().index_select(0, src))

    return out
```

No calculation knows whether the op came from:

```text
module.weight
pruning group member
module parametrization
quantizer
buffer
custom extractor
```

---

## 2. Plan builder

The builder is the small object that converts generic records into the three execution buckets.

```python
class ReductionPlanBuilder:
    def __init__(self, output_length: int | None = None):
        self._fixed_output_length = output_length is not None
        self.output_length = 0 if output_length is None else int(output_length)

        self.segment_entries: list[SegmentEntry] = []
        self.indexed_entries: list[IndexedEntry] = []
        self.indexed_gather_entries: list[IndexedGatherEntry] = []

        self.output_labels: list[str] | None = None

    def reserve_segment(self, length: int) -> SegmentTarget:
        """
        Useful for module-wise plans where output is built sequentially.
        """
        if self._fixed_output_length:
            raise RuntimeError("Cannot reserve into a fixed-output builder.")

        start = self.output_length
        length = int(length)
        self.output_length += length
        return SegmentTarget(start=start, length=length)

    def add(self, record: ReductionRecord) -> None:
        target = record.target
        source = record.source_indices

        if isinstance(target, SegmentTarget):
            self._touch_output(target.start + target.length)

            if source is None:
                self.segment_entries.append(
                    SegmentEntry(
                        op=record.op,
                        start=target.start,
                        length=target.length,
                    )
                )
                return

            # Lower selected segment writes into indexed-gather.
            # This keeps the calculation forward simple.
            self.indexed_gather_entries.append(
                IndexedGatherEntry(
                    op=record.op,
                    source_indices=tuple(int(i) for i in source),
                    destination_indices=tuple(
                        range(target.start, target.start + target.length)
                    ),
                )
            )
            return

        if isinstance(target, IndexedTarget):
            if len(target.destination_indices) > 0:
                self._touch_output(max(target.destination_indices) + 1)

            if source is None:
                self.indexed_entries.append(
                    IndexedEntry(
                        op=record.op,
                        destination_indices=tuple(
                            int(i) for i in target.destination_indices
                        ),
                    )
                )
                return

            self.indexed_gather_entries.append(
                IndexedGatherEntry(
                    op=record.op,
                    source_indices=tuple(int(i) for i in source),
                    destination_indices=tuple(
                        int(i) for i in target.destination_indices
                    ),
                )
            )
            return

        raise TypeError(f"Unknown target type: {type(target)!r}")

    def finalize(self) -> ReductionPlan:
        return ReductionPlan(
            output_length=self.output_length,
            segment_entries=tuple(self.segment_entries),
            indexed_entries=tuple(self.indexed_entries),
            indexed_gather_entries=tuple(self.indexed_gather_entries),
            output_labels=None if self.output_labels is None else tuple(self.output_labels),
        )

    def _touch_output(self, required_length: int) -> None:
        required_length = int(required_length)

        if self._fixed_output_length:
            if required_length > self.output_length:
                raise ValueError(
                    f"Target exceeds fixed output length: "
                    f"{required_length} > {self.output_length}."
                )
            return

        self.output_length = max(self.output_length, required_length)
```

This builder gives you one central place for:

```text
segment vs indexed choice
source selection lowering
output length validation
```

---

## 3. Generic planner

This is the generic abstraction you are describing.

```python
from collections.abc import Iterable
from typing import Generic, Protocol, TypeVar


Element = TypeVar("Element")


class ReductionRule(Protocol[Element]):
    def emit(
        self,
        element: Element,
        builder: ReductionPlanBuilder,
    ) -> Iterable[ReductionRecord]:
        ...


class GenericReductionPlanner(Generic[Element]):
    """
    Generic compiler over any element list.

    Elements can be:
        - modules
        - pruning-group members
        - MemberContext objects
        - module parametrizations
        - quantizers
        - custom objects
    """

    def __init__(
        self,
        elements: Iterable[Element],
        *,
        output_length: int | None = None,
    ):
        self.elements = tuple(elements)
        self.output_length = output_length

    def compile(self, rule: ReductionRule[Element]) -> ReductionPlan:
        builder = ReductionPlanBuilder(output_length=self.output_length)

        for element in self.elements:
            for record in rule.emit(element, builder):
                builder.add(record)

        return builder.finalize()
```

That is the common planner.

It does not know about:

```text
weights
groups
MHA
num_heads
modules
parametrizations
```

It only knows:

```text
iterate elements
ask rule for records
build final execution plan
```

---

## 4. Example: module-wise plan

Element list:

```python
model.named_modules()
```

Rule:

```python
class ModuleWeightReductionRule:
    def __init__(
        self,
        *,
        operation_type,
        parameter_name: str = "weight",
        include_module=None,
    ):
        self.operation_type = operation_type
        self.parameter_name = parameter_name
        self.include_module = include_module

    def emit(self, element, builder: ReductionPlanBuilder):
        name, module = element

        if self.include_module is not None and not self.include_module(name, module):
            return

        if not hasattr(module, self.parameter_name):
            return

        parameter = getattr(module, self.parameter_name)
        if parameter is None:
            return

        op = ModuleWeightOp(
            module=module,
            parameter_name=self.parameter_name,
            operation_type=self.operation_type,
        )

        with torch.no_grad():
            value_count = int(op().numel())

        target = builder.reserve_segment(value_count)

        yield ReductionRecord(
            op=op,
            target=target,
        )
```

Then:

```python
plan = GenericReductionPlanner(
    model.named_modules(),
).compile(
    ModuleWeightReductionRule(
        operation_type=WeightOperationType.SUM,
    )
)
```

The calculation does not care that the elements were modules.

---

## 5. Example: pruning-member plan

Here the elements should not be raw members if attention layout matters. The structure analyzer/factory should create `MemberContext` objects first.

```python
@dataclass(frozen=True)
class MemberContext:
    group_offset: int
    group_length: int
    attention_config: object | None
    member: object
    module: object
    handler: object
    unit_indices: tuple[int, ...]
```

Then the rule is:

```python
class PruneMemberWeightReductionRule:
    def __init__(
        self,
        *,
        operation_type,
        parameter_name: str = "weight",
    ):
        self.operation_type = operation_type
        self.parameter_name = parameter_name

    def emit(self, ctx: MemberContext, builder: ReductionPlanBuilder):
        module = ctx.module

        if isinstance(module, nn.MultiheadAttention):
            yield from self._emit_mha(ctx)
            return

        if not isinstance(module, nn.Module):
            return

        if not hasattr(module, self.parameter_name):
            return

        if getattr(module, self.parameter_name) is None:
            return

        op = PruneMemberWeightOp(
            member=ctx.member,
            module=module,
            parameter_name=self.parameter_name,
            operation_type=self.operation_type,
        )

        yield ReductionRecord(
            op=op,
            target=indexed_or_segment_target(ctx.unit_indices),
        )

    def _emit_mha(self, ctx: MemberContext):
        op = MHAQKVReductionOp(
            module=ctx.module,
            operation_type=self.operation_type,
            attention_config=ctx.attention_config,
        )

        if ctx.attention_config is None:
            length = int(ctx.module.embed_dim)
        else:
            length = int(ctx.attention_config.output_length)

        yield ReductionRecord(
            op=op,
            target=SegmentTarget(
                start=ctx.group_offset,
                length=length,
            ),
        )
```

Then:

```python
plan = GenericReductionPlanner(
    factory.member_contexts(),
    output_length=factory.num_units,
).compile(
    PruneMemberWeightReductionRule(
        operation_type=WeightOperationType.SUM,
    )
)
```

Again, the calculation does not know about members or groups.

---

## 6. The structure-specific class becomes a context producer

This is where `num_heads`, `prune_dim`, and `prune_num_heads` belong.

```python
class StructuredGroupContext:
    """
    Converts torch-pruning groups into reusable MemberContext objects.

    Responsibilities:
        - attention config
        - group unit layout
        - root index -> unit index
        - module naming if needed

    Not responsible for reducing tensors.
    Not responsible for calculation buffers.
    """

    def __init__(
        self,
        groups,
        *,
        num_heads=None,
        prune_dim=False,
        prune_num_heads=False,
        module_name_getter=None,
    ):
        self.groups = tuple(groups)
        self.num_heads = {} if num_heads is None else dict(num_heads)
        self.prune_dim = bool(prune_dim)
        self.prune_num_heads = bool(prune_num_heads)
        self.module_name_getter = module_name_getter

        self.layouts = self._build_layouts()
        self.num_units = (
            0
            if len(self.layouts) == 0
            else self.layouts[-1].unit_offset + self.layouts[-1].unit_length
        )

    def member_contexts(self) -> tuple[MemberContext, ...]:
        contexts = []

        for layout in self.layouts:
            for member in layout.group.items:
                contexts.append(
                    MemberContext(
                        group_offset=layout.unit_offset,
                        group_length=layout.unit_length,
                        attention_config=layout.attention_config,
                        member=member,
                        module=member.dep.target.module,
                        handler=member.dep.handler,
                        unit_indices=self._unit_indices_for_member(layout, member),
                    )
                )

        return tuple(contexts)

    def reduction_plan(self, rule: ReductionRule[MemberContext]) -> ReductionPlan:
        return GenericReductionPlanner(
            self.member_contexts(),
            output_length=self.num_units,
        ).compile(rule)
```

Then `StructureAnalyzer` owns this:

```python
self.group_context = StructuredGroupContext(
    groups=tuple(self.iter_groups()),
    num_heads=self.config.num_heads,
    prune_dim=self.config.prune_head_dims,
    prune_num_heads=self.config.prune_num_heads,
    module_name_getter=self._module_name,
)
```

And creates plans:

```python
sum_plan = self.group_context.reduction_plan(
    PruneMemberWeightReductionRule(
        operation_type=WeightOperationType.SUM,
    )
)
```

---

## 7. Generic calculation

The calculation now accepts only a `ReductionPlan`.

```python
class MappedReductionCalculation(nn.Module):
    def __init__(self, plan: ReductionPlan, *, validate: bool = True):
        super().__init__()

        if validate:
            validate_reduction_plan(plan)

        first_op = first_plan_op(plan)

        with torch.no_grad():
            prototype = first_op()

        self.register_buffer(
            "accumulator",
            prototype.new_zeros(plan.output_length),
            persistent=False,
        )

        self.segment_ops = nn.ModuleList(
            entry.op for entry in plan.segment_entries
        )
        self.indexed_ops = nn.ModuleList(
            entry.op for entry in plan.indexed_entries
        )
        self.indexed_gather_ops = nn.ModuleList(
            entry.op for entry in plan.indexed_gather_entries
        )

        self.segment_specs = tuple(
            (entry.start, entry.length)
            for entry in plan.segment_entries
        )

        self._register_index_buffers(plan, prototype.device)
        self._compile_runtime_entries()

    @torch.no_grad()
    def forward(self) -> torch.Tensor:
        out = self.accumulator
        out.zero_()

        for op, start, length in self.segment_entries:
            out.narrow(0, start, length).add_(op())

        for op, dst in self.indexed_entries:
            out.index_add_(0, dst, op())

        for op, src, dst in self.indexed_gather_entries:
            out.index_add_(0, dst, op().index_select(0, src))

        return out
```

Then `StructuredUnitSum` is just a named wrapper if you still want the calculation type:

```python
class StructuredUnitSum(MappedReductionCalculation, BaseCalculation):
    calculation_type = CalculationType.STRUCTURED_UNIT_SUM
```

No `from_groups`.
No `num_heads`.
No `prune_dim`.
No structural logic.

---

## 8. Why this is the right abstraction

You get exactly this:

```text
Generic planner:
    any list of elements

Rule:
    element -> operation + mapping

Plan:
    executable segment/indexed/indexed-gather entries

Calculation:
    execute one or multiple plans without knowing element type
```

So you can build plans from:

```text
modules
pruning members
member contexts
module parametrizations
quantizer modules
custom layer records
```

by only changing the rule.

The calculation does not change.

---

## 9. Naming

With this abstraction, the generic name is valid:

```text
ReductionPlan
ReducerPlan
MappedReductionPlan
```

I would probably use:

```python
MappedReductionPlan
```

because the plan is specifically:

```text
operation result -> mapped output tensor
```

Then:

```python
MappedReductionPlanner
MappedReductionCalculation
ReductionRecord
SegmentEntry
IndexedEntry
IndexedGatherEntry
```

The old `WeightReducerPlan` name is no longer appropriate because the operation can read anything.

---

## 10. Final shape

The system becomes:

```text
StructuredGroupContext:
    torch-pruning groups -> MemberContext list

GenericReductionPlanner:
    any element list + rule -> MappedReductionPlan

Rules:
    ModuleWeightReductionRule
    PruneMemberWeightReductionRule
    ParametrizationReductionRule
    BitrateReductionRule
    etc.

Calculations:
    MappedReductionCalculation(plan)
    StructuredUnitSum(plan)
    StructuredUnitNorm(plan)
```

That is the architecture you are describing: the plan abstracts over **what the element is**, while the calculation only knows **how to execute mapped tensor operations**.


This specific part in side ModuleWeightReductionRUl, i personally find offesnive:

     with torch.no_grad():
            value_count = int(op().numel())

        target = builder.reserve_segment(value_count)

        yield ReductionRecord(
            op=op,
            target=target,
        )


why not include in the Op an output length / shape requirements? this way we can also validate the plan before hand, with out performing the op


Yes. That part is wrong for a proper planner.

This:

```python
with torch.no_grad():
    value_count = int(op().numel())

target = builder.reserve_segment(value_count)
```

means plan compilation is already doing the calculation. That defeats the point of having a plan.

The better contract is:

```text
ReductionOp knows its output shape/length statically.
Planner uses op.output_length.
Calculation later calls op().
```

So the operation should expose metadata like:

```python
op.output_shape
op.output_length
```

and `validate_plan()` should validate against those, not by running the op.

This fits the lifecycle rule: once the model structure changes, plans/calculations are invalid and must be rebuilt. The plan can assume source shapes are stable after construction. 

---

## 1. Generic operation protocol

I would define the executable operation contract like this:

```python
from typing import Hashable, Protocol
import torch


class ReductionOp(Protocol):
    @property
    def output_shape(self) -> tuple[int, ...]:
        ...

    @property
    def output_length(self) -> int:
        ...

    def __call__(self) -> torch.Tensor:
        """
        Must return a flat 1-D tensor with length == output_length.
        """
        ...

    def identity_key(self) -> Hashable:
        ...

    def first_tensor(self) -> torch.Tensor:
        ...
```

The important part is:

```python
op.output_length
```

is available without:

```python
op()
```

---

## 2. Extractors should expose shape metadata

To support this cleanly, extractors should also expose shape without doing reduction.

```python
class TensorExtractor(Protocol):
    def get(self) -> TensorValue:
        ...

    def shape(self) -> tuple[int, ...] | tuple[tuple[int, ...], ...]:
        ...

    def identity_key(self) -> Hashable:
        ...

    def first_tensor(self) -> torch.Tensor:
        ...
```

For `ParameterExtractor`:

```python
class ParameterExtractor(BaseExtractor):
    ...

    def shape(self) -> tuple[int, ...]:
        return tuple(self.get().shape)
```

For `SeparateQKVExtractor`:

```python
class SeparateQKVExtractor(BaseExtractor):
    ...

    def shape(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (
            tuple(self.q_extractor.get().shape),
            tuple(self.k_extractor.get().shape),
            tuple(self.v_extractor.get().shape),
        )
```

This reads tensor metadata, but it does **not** execute the reduction.

---

## 3. Axis reduction op with static output length

Example:

```python
from math import prod
import torch
import torch.nn as nn


class AxisReductionOp(nn.Module):
    def __init__(
        self,
        extractor: TensorExtractor,
        *,
        operation_type: WeightOperationType | str,
        dim=None,
        keepdim: bool = False,
    ):
        super().__init__()

        self.extractor = extractor
        self.operation_type = WeightOperationType(operation_type)
        self.dim = dim
        self.keepdim = keepdim

        source_shape = extractor.shape()
        if not _is_single_shape(source_shape):
            raise TypeError("AxisReductionOp expects a single tensor extractor.")

        self._output_shape = _reduced_shape(
            tuple(source_shape),
            dim=dim,
            keepdim=keepdim,
        )
        self._output_length = int(prod(self._output_shape)) if self._output_shape else 1

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    @property
    def output_length(self) -> int:
        return self._output_length

    def first_tensor(self) -> torch.Tensor:
        return self.extractor.first_tensor()

    def identity_key(self):
        return (
            type(self),
            self.extractor.identity_key(),
            self.operation_type,
            self.dim,
            self.keepdim,
        )

    def forward(self) -> torch.Tensor:
        weight = self.extractor.get()

        if self.operation_type == WeightOperationType.SUM:
            if self.dim == ():
                return weight.reshape(-1)
            return weight.sum(dim=self.dim, keepdim=self.keepdim).reshape(-1)

        if self.operation_type == WeightOperationType.MEAN:
            if self.dim == ():
                return weight.reshape(-1)
            return weight.mean(dim=self.dim, keepdim=self.keepdim).reshape(-1)

        if self.operation_type == WeightOperationType.COUNT:
            ones = torch.ones_like(weight)
            if self.dim == ():
                return ones.reshape(-1)
            return ones.sum(dim=self.dim, keepdim=self.keepdim).reshape(-1)

        if self.operation_type == WeightOperationType.L1:
            if self.dim == ():
                return weight.abs().reshape(-1)
            return weight.abs().sum(dim=self.dim, keepdim=self.keepdim).reshape(-1)

        if self.operation_type == WeightOperationType.L2:
            if self.dim == ():
                return weight.abs().reshape(-1)
            return torch.sqrt((weight ** 2).sum(dim=self.dim, keepdim=self.keepdim)).reshape(-1)

        raise ValueError(f"Unsupported operation: {self.operation_type}")
```

Helper:

```python
def _is_single_shape(shape) -> bool:
    return (
        isinstance(shape, tuple)
        and all(isinstance(dim, int) for dim in shape)
    )


def _normalize_dims(dim, ndim: int) -> tuple[int, ...] | None:
    if dim is None:
        return None

    if dim == ():
        return ()

    if isinstance(dim, int):
        dims = (dim,)
    else:
        dims = tuple(dim)

    normalized = []
    for item in dims:
        item = int(item)
        if item < 0:
            item += ndim
        if item < 0 or item >= ndim:
            raise ValueError(f"Reduction dim {item} is outside tensor ndim {ndim}.")
        normalized.append(item)

    return tuple(sorted(set(normalized)))


def _reduced_shape(
    shape: tuple[int, ...],
    *,
    dim,
    keepdim: bool,
) -> tuple[int, ...]:
    dims = _normalize_dims(dim, len(shape))

    if dims is None:
        return ()

    if dims == ():
        return shape

    if keepdim:
        return tuple(1 if index in dims else size for index, size in enumerate(shape))

    return tuple(size for index, size in enumerate(shape) if index not in dims)
```

---

## 4. QKV semantic op with static output length

For QKV, output length is known from metadata:

```python
class QKVSemanticReductionOp(nn.Module):
    def __init__(
        self,
        extractor: TensorExtractor,
        *,
        operation_type: WeightOperationType | str,
        embed_dim: int,
        num_heads: int | None = None,
        mode: str = "channel",
    ):
        super().__init__()

        self.extractor = extractor
        self.operation_type = WeightOperationType(operation_type)
        self.embed_dim = int(embed_dim)
        self.num_heads = None if num_heads is None else int(num_heads)
        self.mode = mode

        if self.mode == "channel":
            self._output_shape = (self.embed_dim,)
        elif self.mode == "head":
            if self.num_heads is None:
                raise ValueError("mode='head' requires num_heads.")
            self._output_shape = (self.num_heads,)
        elif self.mode == "head_dim":
            if self.num_heads is None:
                raise ValueError("mode='head_dim' requires num_heads.")
            if self.embed_dim % self.num_heads != 0:
                raise ValueError("embed_dim must be divisible by num_heads.")
            self._output_shape = (self.embed_dim // self.num_heads,)
        else:
            raise ValueError(f"Unknown QKV mode: {self.mode}")

        self._output_length = self._output_shape[0]

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    @property
    def output_length(self) -> int:
        return self._output_length

    def first_tensor(self) -> torch.Tensor:
        return self.extractor.first_tensor()

    def identity_key(self):
        return (
            type(self),
            self.extractor.identity_key(),
            self.operation_type,
            self.embed_dim,
            self.num_heads,
            self.mode,
        )

    def forward(self) -> torch.Tensor:
        qkv = self._as_qkv_tensor(self.extractor.get())
        row_values = self._reduce_rows(qkv)

        if self.mode == "channel":
            return row_values.sum(dim=0).reshape(-1)

        head_dim = self.embed_dim // self.num_heads
        row_values = row_values.reshape(3, self.num_heads, head_dim)

        if self.mode == "head":
            return row_values.sum(dim=(0, 2)).reshape(-1)

        if self.mode == "head_dim":
            return row_values.sum(dim=(0, 1)).reshape(-1)

        raise ValueError(f"Unknown QKV mode: {self.mode}")

    def _as_qkv_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value.reshape(3, self.embed_dim, *value.shape[1:])

        if len(value) != 3:
            raise ValueError("QKV reduction expects fused QKV or q, k, v tensors.")

        q, k, v = value
        return torch.stack((q, k, v), dim=0)

    def _reduce_rows(self, qkv: torch.Tensor) -> torch.Tensor:
        flat = qkv.flatten(start_dim=2)

        if self.operation_type == WeightOperationType.SUM:
            return flat.sum(dim=2)

        if self.operation_type == WeightOperationType.MEAN:
            return flat.mean(dim=2)

        if self.operation_type == WeightOperationType.COUNT:
            return torch.ones_like(flat).sum(dim=2)

        if self.operation_type == WeightOperationType.L1:
            return flat.abs().sum(dim=2)

        if self.operation_type == WeightOperationType.L2:
            return torch.sqrt((flat ** 2).sum(dim=2))

        raise ValueError(f"Unsupported QKV operation: {self.operation_type}")
```

Now the planner can use:

```python
op.output_length
```

without calling:

```python
op()
```

---

## 5. The planner becomes clean

The offensive part becomes:

```python
op = ModuleWeightReductionOp(...)

target = builder.reserve_segment(op.output_length)

yield ReductionRecord(
    op=op,
    target=target,
)
```

No `torch.no_grad()`.
No operation execution.
No temporary computation just to get shape.

For module-wise plans:

```python
class ModuleWeightReductionRule:
    def __init__(
        self,
        *,
        operation_type,
        parameter_name: str = "weight",
        include_module=None,
    ):
        self.operation_type = operation_type
        self.parameter_name = parameter_name
        self.include_module = include_module

    def emit(self, element, builder: ReductionPlanBuilder):
        name, module = element

        if self.include_module is not None and not self.include_module(name, module):
            return

        if not hasattr(module, self.parameter_name):
            return

        parameter = getattr(module, self.parameter_name)
        if parameter is None:
            return

        op = AxisReductionOp(
            extractor=ParameterExtractor(module, self.parameter_name),
            operation_type=self.operation_type,
            dim=None,
        )

        yield ReductionRecord(
            op=op,
            target=builder.reserve_segment(op.output_length),
        )
```

For prune members:

```python
op = AxisReductionOp(
    extractor=ParameterExtractor(module, parameter_name),
    operation_type=operation_type,
    dim=reduction_dim_for_member(member),
)

yield ReductionRecord(
    op=op,
    target=indexed_or_segment_target(ctx.unit_indices),
)
```

---

## 6. Plan validation without executing operations

Validation should use static metadata:

```python
def validate_reduction_plan(plan: ReductionPlan) -> None:
    if plan.output_length < 0:
        raise ValueError("ReductionPlan.output_length must be non-negative.")

    for index, entry in enumerate(plan.segment_entries):
        if entry.start < 0:
            raise ValueError(f"Segment entry {index} has negative start.")

        if entry.length < 0:
            raise ValueError(f"Segment entry {index} has negative length.")

        if entry.start + entry.length > plan.output_length:
            raise ValueError(f"Segment entry {index} exceeds output length.")

        if entry.op.output_length != entry.length:
            raise ValueError(
                f"Segment entry {index} expects {entry.length} values, "
                f"but op outputs {entry.op.output_length}."
            )

    for index, entry in enumerate(plan.indexed_entries):
        destinations = entry.destination_indices

        if entry.op.output_length != len(destinations):
            raise ValueError(
                f"Indexed entry {index} has {len(destinations)} destinations, "
                f"but op outputs {entry.op.output_length}."
            )

        _validate_destinations(destinations, plan.output_length, f"Indexed entry {index}")

    for index, entry in enumerate(plan.indexed_gather_entries):
        sources = entry.source_indices
        destinations = entry.destination_indices

        if len(sources) != len(destinations):
            raise ValueError(
                f"Indexed-gather entry {index} source/destination lengths differ."
            )

        if len(sources) > 0:
            min_source = min(sources)
            max_source = max(sources)

            if min_source < 0 or max_source >= entry.op.output_length:
                raise ValueError(
                    f"Indexed-gather entry {index} has source indices outside "
                    f"[0, {entry.op.output_length})."
                )

        _validate_destinations(
            destinations,
            plan.output_length,
            f"Indexed-gather entry {index}",
        )


def _validate_destinations(
    destinations: tuple[int, ...],
    output_length: int,
    label: str,
) -> None:
    if len(destinations) == 0:
        return

    min_destination = min(destinations)
    max_destination = max(destinations)

    if min_destination < 0 or max_destination >= output_length:
        raise ValueError(
            f"{label} has destinations outside [0, {output_length})."
        )
```

No op execution.

---

## 7. Calculation allocation also does not need op execution

Previously we used:

```python
with torch.no_grad():
    prototype = first_op()
```

That is also unnecessary.

Use:

```python
prototype = first_op.first_tensor()
```

Then:

```python
self.register_buffer(
    "accumulator",
    prototype.new_zeros(plan.output_length),
    persistent=False,
)
```

If you want output dtype different from source dtype, that should be explicit in the calculation or op metadata. But for the default path, `first_tensor()` is enough.

---

## 8. Summary

The operation should not only be executable. It should also be shape-aware.

The contract should be:

```text
ReductionOp:
    output_shape
    output_length
    first_tensor()
    identity_key()
    __call__() -> flat tensor
```

Then the planner can do:

```python
target = builder.reserve_segment(op.output_length)
```

and validation can do:

```python
entry.op.output_length == target.length
```

without calling the operation.

That gives you the architecture you want:

```text
plan compilation:
    metadata only

plan validation:
    metadata only

calculation construction:
    use first_tensor for device/dtype only

calculation forward:
    execute operations
```

Yes. You are right.

The `QKVSemanticReductionOp.forward()` I sketched is **not ideal**, because it leaves two dispatch decisions in the hot path:

```python
if self.mode == ...
if self.operation_type == ...
```

Even if those branches are small compared with GPU tensor kernels, they are still conceptually wrong for the architecture you want. The blueprint should resolve those choices once.

The better split is:

```text
TensorExtractor:
    knows how to read source tensor(s)

ReductionFunction:
    knows one concrete tensor operation
    has static output shape/length logic
    no mode/type branching in __call__

ReductionOp:
    extractor + reduction_function
    forward is just:
        return reduction_function(extractor.get())
```

So `ReductionOp` is generic, but the reduction function is specialized.

---

## The shape I would use

```python
from typing import Hashable, Protocol, TypeAlias
import torch
import torch.nn as nn


TensorValue: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]


class TensorReduction(Protocol):
    def __call__(self, value: TensorValue) -> torch.Tensor:
        ...

    def output_shape(self, source_shape) -> tuple[int, ...]:
        ...

    def identity_key(self) -> Hashable:
        ...


class ReductionOp(nn.Module):
    def __init__(self, extractor: TensorExtractor, reduction: TensorReduction):
        super().__init__()
        self.extractor = extractor
        self.reduction = reduction

        self._output_shape = reduction.output_shape(extractor.shape())
        self._output_length = _numel(self._output_shape)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    @property
    def output_length(self) -> int:
        return self._output_length

    def first_tensor(self) -> torch.Tensor:
        return self.extractor.first_tensor()

    def identity_key(self):
        return (
            self.extractor.identity_key(),
            self.reduction.identity_key(),
        )

    def forward(self) -> torch.Tensor:
        return self.reduction(self.extractor.get()).reshape(-1)


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for size in shape:
        total *= int(size)
    return total
```

Now `ReductionOp.forward()` has no semantic dispatch:

```python
return self.reduction(self.extractor.get()).reshape(-1)
```

The hot-path branch decisions are moved into blueprint construction.

---

## Simple example: axis reductions

Instead of one class with this inside `forward()`:

```python
if operation_type == SUM:
    ...
elif operation_type == L2:
    ...
```

make concrete reduction functions:

```python
class SumDim:
    def __init__(self, dim=None, keepdim: bool = False):
        self.dim = dim
        self.keepdim = keepdim

    def __call__(self, value: TensorValue) -> torch.Tensor:
        if self.dim == ():
            return value
        return value.sum(dim=self.dim, keepdim=self.keepdim)

    def output_shape(self, source_shape) -> tuple[int, ...]:
        return reduced_shape(source_shape, dim=self.dim, keepdim=self.keepdim)

    def identity_key(self):
        return (type(self), self.dim, self.keepdim)


class L2Dim:
    def __init__(self, dim=None, keepdim: bool = False):
        self.dim = dim
        self.keepdim = keepdim

    def __call__(self, value: TensorValue) -> torch.Tensor:
        if self.dim == ():
            return value.abs()
        return torch.sqrt((value ** 2).sum(dim=self.dim, keepdim=self.keepdim))

    def output_shape(self, source_shape) -> tuple[int, ...]:
        return reduced_shape(source_shape, dim=self.dim, keepdim=self.keepdim)

    def identity_key(self):
        return (type(self), self.dim, self.keepdim)
```

The only small branch left is `self.dim == ()`, but even that can be removed by having separate identity/no-reduction functions if you care.

Factory resolves once:

```python
def reduction_for_type(operation_type, *, dim=None, keepdim=False) -> TensorReduction:
    operation_type = WeightOperationType(operation_type)

    if operation_type == WeightOperationType.SUM:
        return SumDim(dim=dim, keepdim=keepdim)

    if operation_type == WeightOperationType.L2:
        return L2Dim(dim=dim, keepdim=keepdim)

    ...
```

This factory is **not** in the hot path.

---

## QKV should be specialized even more

For QKV, I would not use one `QKVSemanticReduction` with `mode` and `operation_type` checks.

Use a factory that returns a concrete function.

For example:

```python
def qkv_reduction_for(
    *,
    operation_type: WeightOperationType | str,
    qkv_kind: str,  # "fused" or "separate"
    mode: str,      # "channel", "head", "head_dim"
    embed_dim: int,
    num_heads: int | None,
) -> TensorReduction:
    operation_type = WeightOperationType(operation_type)

    if qkv_kind == "fused" and operation_type == WeightOperationType.L2 and mode == "head":
        return FusedQKVL2Head(embed_dim=embed_dim, num_heads=num_heads)

    if qkv_kind == "fused" and operation_type == WeightOperationType.SUM and mode == "head":
        return FusedQKVSumHead(embed_dim=embed_dim, num_heads=num_heads)

    if qkv_kind == "separate" and operation_type == WeightOperationType.L2 and mode == "head":
        return SeparateQKVL2Head(embed_dim=embed_dim, num_heads=num_heads)

    ...
```

Then each concrete QKV reduction has one straight-line body.

Example:

```python
class FusedQKVL2Head:
    def __init__(self, *, embed_dim: int, num_heads: int):
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads

    def __call__(self, value: TensorValue) -> torch.Tensor:
        qkv = value.reshape(3, self.embed_dim, *value.shape[1:])
        flat = qkv.flatten(start_dim=2)
        row_l2 = torch.sqrt((flat ** 2).sum(dim=2))
        return row_l2.reshape(3, self.num_heads, self.head_dim).sum(dim=(0, 2))

    def output_shape(self, source_shape) -> tuple[int, ...]:
        return (self.num_heads,)

    def identity_key(self):
        return (
            type(self),
            self.embed_dim,
            self.num_heads,
        )
```

Separate QKV:

```python
class SeparateQKVL2Head:
    def __init__(self, *, embed_dim: int, num_heads: int):
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads

    def __call__(self, value: TensorValue) -> torch.Tensor:
        q, k, v = value

        q = torch.sqrt((q.flatten(start_dim=1) ** 2).sum(dim=1))
        k = torch.sqrt((k.flatten(start_dim=1) ** 2).sum(dim=1))
        v = torch.sqrt((v.flatten(start_dim=1) ** 2).sum(dim=1))

        row_l2 = torch.stack((q, k, v), dim=0)
        return row_l2.reshape(3, self.num_heads, self.head_dim).sum(dim=(0, 2))

    def output_shape(self, source_shape) -> tuple[int, ...]:
        return (self.num_heads,)

    def identity_key(self):
        return (
            type(self),
            self.embed_dim,
            self.num_heads,
        )
```

Now the operation is fully specialized. No `mode` branch. No `operation_type` branch. No fused/separate branch.

---

## Then blueprint construction does the dispatch

In the member rule:

```python
if module.in_proj_weight is not None:
    extractor = FusedQKVExtractor(module, "in_proj_weight")
    qkv_kind = "fused"
else:
    extractor = SeparateQKVExtractor(module)
    qkv_kind = "separate"

reduction = qkv_reduction_for(
    operation_type=self.operation_type,
    qkv_kind=qkv_kind,
    mode=mode,
    embed_dim=embed_dim,
    num_heads=num_heads,
)

op = ReductionOp(
    extractor=extractor,
    reduction=reduction,
)

yield ReductionStep(
    op=op,
    target=segment_target(ctx.group_offset, op.output_length),
)
```

This is exactly the right place for `if` statements: blueprint compilation, not execution.

---

## Why your instinct is right

You want the runtime call to look like this:

```python
op()
```

and inside:

```python
return fixed_function(fixed_extractor.get())
```

not:

```python
if self.mode == ...
if self.operation_type == ...
```

That is better because:

```text
all structural choices are compiled once
runtime has stable straight-line tensor code
output length is known without executing
identity_key is precise
plan validation can be metadata-only
```

---

## What I would not do

I would not make `ReductionOp` a giant class hierarchy for every case.

I would make **reduction functions** concrete and small, then use one generic wrapper:

```python
ReductionOp(extractor, reduction_function)
```

So the number of classes grows where it should grow:

```text
SumDim
L2Dim
FusedQKVL2Head
FusedQKVL2HeadDim
SeparateQKVSumChannel
...
```

But the plan and calculation stay generic.

---

## Final architecture

```text
Blueprint/rule phase:
    decide extractor
    decide concrete TensorReduction function
    create ReductionOp(extractor, function)
    create target using op.output_length

Plan:
    stores ReductionOp + target entries

Calculation forward:
    for op, start, length:
        out.narrow(...).add_(op())

    for op, dst:
        out.index_add_(..., op())

    for op, src, dst:
        out.index_add_(..., op().index_select(...))
```

So yes: your design of `ReductionOp = TensorExtractor + concrete operation function` is the cleaner version. The operation function should be selected during blueprint construction and should have no semantic dispatch in its hot-path `__call__`.


we wont need the parameter_name do we? Also, we should have a operation_type for each reduction rule and a tensorExtractor. The tensor Extractor is responsbile for extracting the correct tensor fromt the element. The tensor Extractor should the take the element. This should be typed. For example a ParameterExtractor should take a module right
