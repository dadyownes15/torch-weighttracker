# Generic Reduction Planner Spec

## Purpose

The planner should compile fast, generic tensor reduction plans across any list
of elements. An element can be a module, a pruning-group member context, a
module parametrization, a quantizer, or another domain object.

The compiled plan must hide element-specific structure from calculations.
Calculations should only execute mapped zero-argument operations:

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

The calculation does not know whether an operation came from `module.weight`,
an MHA QKV projection, a pruning-group member, a parametrization, or another
source.

## Design Decision

`parameter_name` should not live on the generic reduction rule.

The rule should own:

- the `operation_type`, such as `sum`, `l1`, or `l2`
- a typed tensor extractor for the element type
- a target mapper that maps operation output positions into the final output

The extractor is responsible for deciding which tensor value to read from the
element. For example, a module weight extractor should take an `nn.Module` and
return a `TensorSourceRef` for that module's weight tensor. A QKV extractor
should take an MHA module or member context and return refs for the correct
fused or separate QKV tensors. A parametrization extractor should take the
parametrization element and return a ref for the appropriate tensor value.

Concrete extractors may still have internal configuration. For example,
`ParameterByNameExtractor("bias")` can exist when a caller explicitly wants
`bias`, but the generic rule should not expose `parameter_name`. Most common
uses should be named extractors such as `ModuleWeightExtractor`,
`ModuleBiasExtractor`, `FusedQKVExtractor`, and `SeparateQKVExtractor`.

## Core Split

```text
GenericReductionPlanner
    iterates any list of typed elements

ReductionRule[Element]
    converts one element into operation plus mapping records

ElementTensorExtractor[Element]
    resolves an element to a TensorSourceRef

TensorReduction
    one concrete tensor computation with static output metadata

ReductionOp
    zero-argument executable wrapper around TensorSourceRef + reduction

ReductionPlanBuilder
    buckets records into segment / indexed / indexed-gather execution entries

MappedReductionCalculation
    owns accumulator buffers and blindly executes the plan
```

Structural choices happen during planning. Runtime execution should be stable
and branch-light.

## Types

### Tensor Values

```python
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, Hashable, Protocol, TypeAlias, TypeVar

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TensorSpec:
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


TensorValue: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]
SourceSpec: TypeAlias = TensorSpec | tuple[TensorSpec, ...]

Element = TypeVar("Element")
```

Use `torch.Size` for shape metadata. PyTorch does not expose a `torch.Shape`
type in the supported runtime.

### Tensor Source Ref

The source ref is used at runtime by `ReductionOp`. It is a thin, non-owning
link to one live tensor source or tuple of live tensor sources, so `get()` is
zero-argument.

```python
class TensorSourceRef(Protocol):
    def get(self) -> TensorValue:
        ...

    def source_spec(self) -> SourceSpec:
        ...

    def identity_key(self) -> Hashable:
        ...
```

`source_spec()` must return source metadata without executing the reduction. It
may inspect tensor shape, dtype, and device metadata from the source tensor or
parameter. Tuple refs return one `TensorSpec` per tensor.

The ref should not discover or validate whether an element is eligible for a
plan. It should only remember how to read the already-selected live source.
For performance, source discovery should happen once in `bind()`. Runtime
`get()` should be a direct attribute lookup, tuple read, or cheap getter call.

### Element Tensor Extractor

The element extractor is typed by the element it accepts. It is used during
plan compilation.

```python
class ElementTensorExtractor(Protocol[Element]):
    def bind(self, element: Element) -> TensorSourceRef | None:
        ...
```

Returning `None` means the element does not expose a tensor source for this
extractor and should be skipped by the rule.

This split keeps the planner typed while preserving zero-argument runtime
operations.

### Example: Module Weight Extractor

```python
@dataclass(frozen=True)
class ModuleParameterRef:
    module: nn.Module
    name: str

    def get(self) -> torch.Tensor:
        return getattr(self.module, self.name)

    def source_spec(self) -> TensorSpec:
        value = self.get()
        return TensorSpec(
            shape=value.shape,
            dtype=value.dtype,
            device=value.device,
        )

    def identity_key(self) -> Hashable:
        return ("module_parameter", id(self.module), self.name)
```

`ModuleParameterRef` is deliberately simple: it stores `module + name` and
reads the live value. The extractor performs the existence and type checks
before creating the ref.

```python
class ModuleWeightExtractor(ElementTensorExtractor[nn.Module]):
    def bind(self, element: nn.Module) -> TensorSourceRef | None:
        if not isinstance(element, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(element).__name__}")

        if not isinstance(getattr(element, "weight", None), torch.Tensor):
            return None

        return ModuleParameterRef(element, "weight")
```

The generic rule does not need `parameter_name`. The knowledge that this
extractor reads `weight` is contained in the extractor.

If a truly dynamic parameter choice is needed, use a specific extractor:

```python
class ModuleParameterExtractor(ElementTensorExtractor[nn.Module]):
    def __init__(self, name: str):
        self.name = name

    def bind(self, element: nn.Module) -> TensorSourceRef | None:
        if not isinstance(getattr(element, self.name, None), torch.Tensor):
            return None
        return ModuleParameterRef(element, self.name)
```

That configuration is extractor-specific, not part of the generic rule API.

### Example: Named Module Extractor

When the element list is `model.named_modules()`, the element type is not
`nn.Module`; it is `tuple[str, nn.Module]`. That should be explicit:

```python
NamedModule = tuple[str, nn.Module]


class NamedModuleWeightExtractor(ElementTensorExtractor[NamedModule]):
    def bind(self, element: NamedModule) -> TensorSourceRef | None:
        _, module = element
        return ModuleWeightExtractor().bind(module)
```

This is preferable to an untyped rule that assumes every element can be unpacked
as `(name, module)`.

### Example: Member Context Extractor

Pruning-group plans should operate on a typed context object rather than raw
torch-pruning members when additional layout metadata is needed.

```python
@dataclass(frozen=True)
class MemberContext:
    group_offset: int
    group_length: int
    unit_indices: tuple[int, ...]
    member: object
    module: nn.Module
    handler: object
    attention_config: object | None = None


class MemberWeightExtractor(ElementTensorExtractor[MemberContext]):
    def bind(self, element: MemberContext) -> TensorSourceRef | None:
        if not isinstance(getattr(element.module, "weight", None), torch.Tensor):
            return None
        return ModuleParameterRef(element.module, "weight")
```

MHA-specific extractors can also be typed to `MemberContext` and use
`attention_config` to select fused or separate QKV sources.

## Tensor Reductions

The tensor reduction is one concrete operation function. It should not dispatch
on `operation_type`, MHA mode, fused/separate layout, or pruning mode in the hot
path.

```python
class TensorReduction(Protocol):
    def __call__(self, value: TensorValue) -> torch.Tensor:
        ...

    def output_spec(self, source_spec: SourceSpec) -> TensorSpec:
        ...

    def identity_key(self) -> Hashable:
        ...
```

Examples:

- `SumDim(dim=None, keepdim=False)`
- `L1Dim(dim=0, keepdim=False)`
- `L2Dim(dim=1, keepdim=False)`
- `FusedQKVL2Head(embed_dim, num_heads)`
- `SeparateQKVSumChannel(embed_dim)`

Factories may dispatch during planning:

```python
def reduction_for_type(
    operation_type: WeightOperationType | str,
    *,
    dim=None,
    keepdim: bool = False,
) -> TensorReduction:
    operation_type = WeightOperationType(operation_type)

    if operation_type == WeightOperationType.SUM:
        return SumDim(dim=dim, keepdim=keepdim)

    if operation_type == WeightOperationType.L1:
        return L1Dim(dim=dim, keepdim=keepdim)

    if operation_type == WeightOperationType.L2:
        return L2Dim(dim=dim, keepdim=keepdim)

    raise ValueError(f"Unsupported operation type: {operation_type}")
```

The factory is not in the calculation hot path.

The factory also does not compute output metadata. It only chooses a concrete
reduction object and captures static configuration such as `dim` and `keepdim`.
The output metadata is computed later, after an extractor has bound an element
to a `TensorSourceRef` and `ReductionOp` has access to the source spec.

For generic single-tensor reductions, factor shared metadata rules into helpers:

```python
def single_tensor_spec(source_spec: SourceSpec) -> TensorSpec:
    if isinstance(source_spec, tuple):
        raise TypeError("This reduction expects one tensor source.")
    return source_spec


def reduced_shape(
    shape: torch.Size,
    dim=None,
    keepdim: bool = False,
) -> torch.Size:
    if dim is None:
        return torch.Size([1] * len(shape)) if keepdim else torch.Size([])

    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    rank = len(shape)
    normalized = tuple(sorted(d + rank if d < 0 else d for d in dims))

    if any(d < 0 or d >= rank for d in normalized):
        raise IndexError(f"Reduction dim out of range for shape {tuple(shape)}.")

    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate reduction dims: {dims!r}.")

    if keepdim:
        out = list(shape)
        for d in normalized:
            out[d] = 1
        return torch.Size(out)

    return torch.Size(size for i, size in enumerate(shape) if i not in normalized)
```

Then a concrete reduction owns both runtime execution and output metadata:

```python
class SumDim:
    def __init__(self, dim=None, keepdim: bool = False) -> None:
        self.dim = dim
        self.keepdim = keepdim

    def __call__(self, value: TensorValue) -> torch.Tensor:
        if isinstance(value, tuple):
            raise TypeError("SumDim expects one tensor source.")
        return value.sum(dim=self.dim, keepdim=self.keepdim)

    def output_spec(self, source_spec: SourceSpec) -> TensorSpec:
        spec = single_tensor_spec(source_spec)
        return TensorSpec(
            shape=reduced_shape(spec.shape, dim=self.dim, keepdim=self.keepdim),
            dtype=spec.dtype,
            device=spec.device,
        )

    def identity_key(self) -> Hashable:
        return (type(self), self.dim, self.keepdim)
```

`L1Dim` and `L2Dim` can share the same output metadata logic while differing in
`__call__`. Tuple-source reductions such as fused/separate QKV reductions should
implement their own `output_spec()` because their source metadata and output
layout are reduction-specific.

If a mapped plan needs flattened outputs, flattening should be an explicit
reduction or reduction adapter, not hidden inside `ReductionOp`:

```python
class FlattenedReduction:
    def __init__(self, inner: TensorReduction) -> None:
        self.inner = inner

    def __call__(self, value: TensorValue) -> torch.Tensor:
        return self.inner(value).reshape(-1)

    def output_spec(self, source_spec: SourceSpec) -> TensorSpec:
        inner_spec = self.inner.output_spec(source_spec)
        return TensorSpec(
            shape=torch.Size([numel(inner_spec.shape)]),
            dtype=inner_spec.dtype,
            device=inner_spec.device,
        )

    def identity_key(self) -> Hashable:
        return ("flattened", self.inner.identity_key())
```

## Reduction Operation

`ReductionOp` combines a source ref and one concrete reduction. It exposes
output metadata without executing the reduction.

```python
class ReductionOp(nn.Module):
    def __init__(
        self,
        source_ref: TensorSourceRef,
        reduction: TensorReduction,
    ) -> None:
        super().__init__()
        self.source_ref = source_ref
        self.reduction = reduction

        self._source_spec = source_ref.source_spec()
        self._output_spec = reduction.output_spec(self._source_spec)
        self._output_length = numel(self._output_spec.shape)

    @property
    def source_spec(self) -> SourceSpec:
        return self._source_spec

    @property
    def output_spec(self) -> TensorSpec:
        return self._output_spec

    @property
    def output_shape(self) -> torch.Size:
        return self._output_spec.shape

    @property
    def output_length(self) -> int:
        return self._output_length

    def identity_key(self) -> Hashable:
        return (
            self.source_ref.identity_key(),
            self.reduction.identity_key(),
        )

    def forward(self) -> torch.Tensor:
        return self.reduction(self.source_ref.get())


def numel(shape: torch.Size) -> int:
    total = 1
    for size in shape:
        total *= int(size)
    return total
```

Planning and validation must use `op.output_spec` and `op.output_length`; they
must not call `op()`.

## Selections, Mappings, And Records

```python
@dataclass(frozen=True)
class FullSelection:
    pass


@dataclass(frozen=True)
class SegmentSelection:
    start: int
    length: int


@dataclass(frozen=True)
class IndexSelection:
    indices: tuple[int, ...]


Selection = FullSelection | SegmentSelection | IndexSelection


@dataclass(frozen=True)
class ReductionMapping:
    source: Selection
    target: Selection


@dataclass(frozen=True)
class ReductionRecord:
    op: ReductionOp
    mapping: ReductionMapping


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
class MappedReductionPlan:
    output_length: int
    output_spec: TensorSpec
    segment_entries: tuple[SegmentEntry, ...] = ()
    indexed_entries: tuple[IndexedEntry, ...] = ()
    indexed_gather_entries: tuple[IndexedGatherEntry, ...] = ()
    output_labels: tuple[str, ...] | None = None
```

`ReductionMapping` is the planning API. It describes which values from an op
output are written to which plan output positions. Single-index selections use
`IndexSelection((index,))`. A singleton indexed target can also be used as an
accumulation sink for a longer source selection; the builder lowers it by
repeating that destination index for every selected source value.

The builder lowers records into one of three execution buckets:

- segment: contiguous full write
- indexed: full output written to arbitrary destination indices
- indexed-gather: selected source indices written to arbitrary destinations

Mapped plans use a one-dimensional accumulator and their targets address flat
output positions. Therefore any reduction used with segment, indexed, or
indexed-gather writes must produce the one-dimensional representation the mapper
expects. If a natural reduction produces a scalar, matrix, or other structured
tensor, planning must select a concrete reduction or adapter that returns the
intended flat representation. `ReductionOp` must not flatten or reshape
implicitly.

The builder should derive `plan.output_spec` from the operation specs unless an
explicit output spec is provided. By default, all operation output specs in one
plan must agree on dtype and device. Mixed dtype or mixed device plans should be
an explicit feature rather than accidental behavior.

## Reduction Rule

The generic rule owns operation construction and mapping selection.

```python
class ReductionRule(Protocol[Element]):
    def emit(
        self,
        element: Element,
        builder: ReductionPlanBuilder,
    ) -> Iterable[ReductionRecord]:
        ...
```

A reusable rule can be implemented by composing a typed extractor, reduction
factory, and mapping strategy:

```python
class ElementReductionRule(Generic[Element]):
    def __init__(
        self,
        *,
        extractor: ElementTensorExtractor[Element],
        operation_type: WeightOperationType | str,
        mapping_strategy: MappingStrategy[Element],
        dim=None,
        keepdim: bool = False,
        flatten_output: bool = True,
    ) -> None:
        self.extractor = extractor
        self.operation_type = WeightOperationType(operation_type)
        self.mapping_strategy = mapping_strategy
        self.dim = dim
        self.keepdim = keepdim
        self.flatten_output = flatten_output

    def emit(
        self,
        element: Element,
        builder: ReductionPlanBuilder,
    ) -> Iterable[ReductionRecord]:
        source_ref = self.extractor.bind(element)
        if source_ref is None:
            return

        reduction = reduction_for_type(
            self.operation_type,
            dim=self.dim,
            keepdim=self.keepdim,
        )
        if self.flatten_output:
            reduction = FlattenedReduction(reduction)

        op = ReductionOp(source_ref, reduction)
        mapping = self.mapping_strategy.map(element, op, builder)

        yield ReductionRecord(
            op=op,
            mapping=mapping,
        )
```

The rule does not know how to read a module parameter. The extractor does that.
The rule does not know how to execute `sum` or `l2`. The concrete reduction
does that. The rule only wires the element-specific pieces into a plan record.

## Mapping Strategy

Mapping selection should also be typed by element.

```python
class MappingStrategy(Protocol[Element]):
    def map(
        self,
        element: Element,
        op: ReductionOp,
        builder: ReductionPlanBuilder,
    ) -> ReductionMapping:
        ...
```

Examples:

```python
class SequentialSegmentMapper(MappingStrategy[Element]):
    def map(
        self,
        element: Element,
        op: ReductionOp,
        builder: ReductionPlanBuilder,
    ) -> ReductionMapping:
        return ReductionMapping(
            source=FullSelection(),
            target=builder.reserve_segment(op.output_length),
        )
```

```python
class MemberUnitMapper(MappingStrategy[MemberContext]):
    def map(
        self,
        element: MemberContext,
        op: ReductionOp,
        builder: ReductionPlanBuilder,
    ) -> ReductionMapping:
        if op.output_length == element.group_length:
            return ReductionMapping(
                source=FullSelection(),
                target=SegmentSelection(
                    start=element.group_offset,
                    length=op.output_length,
                ),
            )

        if op.output_length == len(element.unit_indices):
            return ReductionMapping(
                source=FullSelection(),
                target=IndexSelection(element.unit_indices),
            )

        source_indices = source_indices_for_member(element, op)
        if len(source_indices) != len(element.unit_indices):
            raise ValueError(
                "Member source and destination mapping lengths must match."
            )

        return ReductionMapping(
            source=IndexSelection(source_indices),
            target=IndexSelection(element.unit_indices),
        )
```

The exact member mapping logic may differ, but it belongs in a mapper or
context producer, not in the calculation. `source_indices_for_member` is a
placeholder for the structure-specific logic that converts member-local source
positions into the operation output positions.

## Generic Planner

```python
class GenericReductionPlanner(Generic[Element]):
    def __init__(
        self,
        elements: Iterable[Element],
        *,
        output_length: int | None = None,
    ) -> None:
        self.elements = tuple(elements)
        self.output_length = output_length

    def compile(self, rule: ReductionRule[Element]) -> MappedReductionPlan:
        builder = ReductionPlanBuilder(output_length=self.output_length)

        for element in self.elements:
            for record in rule.emit(element, builder):
                builder.add(record)

        return builder.finalize()
```

This planner knows nothing about modules, groups, parametrizations, offsets,
QKV, or parameter names.

## Plan Validation

Plan validation must be metadata-only.

Required checks:

- `plan.output_length >= 0`
- `plan.output_spec.shape == torch.Size([plan.output_length])`
- every mapped entry's `op.output_spec.shape == torch.Size([op.output_length])`
- segment entries stay inside `[0, output_length)`
- `segment.length == segment.op.output_length`
- indexed destination length equals `indexed.op.output_length`
- indexed destinations stay inside `[0, output_length)`
- indexed-gather source and destination lengths match
- indexed-gather source indices stay inside `[0, op.output_length)`
- indexed-gather destinations stay inside `[0, output_length)`
- every entry's `op.output_spec.dtype` matches `plan.output_spec.dtype`
- every entry's `op.output_spec.device` matches `plan.output_spec.device`

Validation must not call `op()`.

Calculation construction should allocate from `plan.output_spec`:

```python
spec = plan.output_spec
self.register_buffer(
    "accumulator",
    torch.zeros(
        plan.output_length,
        device=spec.device,
        dtype=spec.dtype,
    ),
    persistent=False,
)
```

If output dtype, device, shape, or layout should differ from the source, the
reduction should expose that through `output_spec()`, and the builder should
reflect it in `plan.output_spec`.

## Examples

### Module-Wise Weight Sum

```python
planner = GenericReductionPlanner(model.modules())

plan = planner.compile(
    ElementReductionRule[nn.Module](
        extractor=ModuleWeightExtractor(),
        operation_type=WeightOperationType.SUM,
        mapping_strategy=SequentialSegmentMapper(),
        dim=None,
    )
)
```

The output is a sequential concatenation of all module weight reductions. No
rule argument named `parameter_name` is needed.

### Named Module Weight Sum

```python
planner = GenericReductionPlanner(model.named_modules())

plan = planner.compile(
    ElementReductionRule[tuple[str, nn.Module]](
        extractor=NamedModuleWeightExtractor(),
        operation_type=WeightOperationType.SUM,
        mapping_strategy=SequentialSegmentMapper(),
        dim=None,
    )
)
```

Filtering by name should be a wrapper extractor or an element filter, not a
special case inside the planner.

### Pruning Member Unit Sum

```python
contexts = StructuredGroupContext(groups, num_heads=num_heads).member_contexts()

planner = GenericReductionPlanner(
    contexts,
    output_length=StructuredGroupContext.num_units,
)

plan = planner.compile(
    ElementReductionRule[MemberContext](
        extractor=MemberWeightExtractor(),
        operation_type=WeightOperationType.SUM,
        mapping_strategy=MemberUnitMapper(),
        dim=member_reduction_dim,
    )
)
```

If the member is MHA-specific, use a different extractor and reduction factory:

```python
plan = planner.compile(
    ElementReductionRule[MemberContext](
        extractor=MemberQKVExtractor(),
        operation_type=WeightOperationType.L2,
        mapping_strategy=MemberUnitMapper(),
        dim=None,
    )
)
```

The MHA extractor and QKV reduction factory decide fused/separate QKV and
head/channel/head-dim semantics during planning.

### Parametrization Reduction

```python
plan = GenericReductionPlanner(module.parametrizations.weight).compile(
    ElementReductionRule[nn.Module](
        extractor=ParametrizationTensorExtractor(),
        operation_type=WeightOperationType.L1,
        mapping_strategy=SequentialSegmentMapper(),
    )
)
```

The same planner shape works because only the element type and extractor
changed.

## Migration Notes

1. Rename the generic plan to `MappedReductionPlan` or keep `ReductionPlan` if
   no weight-specific names remain.
2. Move any `parameter_name` argument out of reduction rules and into concrete
   extractors.
3. Replace untyped element unpacking in rules with typed extractors for
   `nn.Module`, `tuple[str, nn.Module]`, `MemberContext`, and parametrization
   elements.
4. Use `TensorSourceRef` implementations and add
   `output_spec`, `output_shape`, and `output_length` to `ReductionOp`.
5. Add `output_spec` to the plan so calculations allocate without inspecting
   source tensors or executing operations.
6. Change plan compilation and validation to use `op.output_length`, never
   `op().numel()`.
7. Keep structural logic in context producers and mapping strategies.
8. Keep calculations limited to buffer allocation, index buffer registration,
   and execution of segment/indexed/indexed-gather entries.

## Non-Goals

- The generic planner should not know about torch-pruning group internals.
- The generic planner should not know about module parameter names.
- The calculation should not know about modules, members, parametrizations, MHA,
  offsets, or source tensor layout.
- Plan compilation should not execute tensor reductions.

## Final Shape

```text
elements: Iterable[Element]
    -> ElementTensorExtractor[Element].bind(element)
    -> TensorSourceRef
    -> TensorReduction selected from operation_type
    -> ReductionOp(source_ref, reduction)
    -> MappingStrategy[Element].map(element, op, builder)
    -> ReductionRecord(op, ReductionMapping(source, target))
    -> ReductionPlanBuilder.add(record)
    -> MappedReductionPlan
    -> MappedReductionCalculation.forward()
```

This gives the intended abstraction: the planner can generate computation steps
across any typed element list, while calculations execute one or more mapped
plans without knowing what the original elements were.
