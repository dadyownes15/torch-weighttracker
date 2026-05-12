# Generic Reduction Types And Parameter Sum Spec

## Purpose

This spec defines the generic type shape for the reduction planner and gives a
complete example of compiling a parameter-sum plan.

The planner must work over any typed element list. An element can be an
`nn.Module`, a `(name, module)` pair, a pruning member context, a
parametrization, a quantizer, or another domain object.

The important split is:

- extractors resolve elements into `TensorSourceRef` objects
- source refs know how to read live tensor values
- reductions know how to turn tensors into output values
- mappers know where each output value belongs in the final plan output
- calculations only execute the compiled plan

The generic rule should not expose `parameter_name`. If a plan reads
`module.weight`, that knowledge belongs in the extractor.

## Type Overview

```text
GenericReductionPlanner[ElementT]
    owns an Iterable[ElementT]

ReductionRule[ElementT]
    emits mapped reduction records for one element

ElementTensorExtractor[ElementT]
    resolves one typed element to a TensorSourceRef

TensorSourceRef
    thin runtime link to the live tensor source

TensorReduction
    performs one concrete tensor reduction such as sum, l1, or l2

ReductionOp
    combines a TensorSourceRef with a concrete reduction

TargetMapper[ElementT]
    maps each operation output into the final plan output

ReductionPlanBuilder
    lowers records into segment, indexed, and indexed-gather entries

MappedReductionCalculation
    allocates buffers and executes entries without element-specific logic
```

## Generic Types

These are the core type contracts the generic planner should depend on.

```python
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Hashable, Protocol, TypeAlias, TypeVar

import torch


ElementT = TypeVar("ElementT")

ReductionDim: TypeAlias = int | tuple[int, ...] | None
TensorValue: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]
```

`TensorReductionType` is intentionally generic. The planner should not need a
weight-specific enum name for reductions that can apply to any tensor source.

```python
class TensorReductionType(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    L1 = "l1"
    L2 = "l2"
```

Tensor metadata must be available without running the reduction.

```python
@dataclass(frozen=True)
class TensorSpec:
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


SourceSpec: TypeAlias = TensorSpec | tuple[TensorSpec, ...]
```

## Extractors

The element extractor is typed by the element it accepts. It runs during plan
compilation and either returns a `TensorSourceRef` or skips the element by
returning `None`.

```python
class TensorSourceRef(Protocol):
    def get(self) -> TensorValue:
        ...

    def source_spec(self) -> SourceSpec:
        ...

    def identity_key(self) -> Hashable:
        ...


class ElementTensorExtractor(Protocol[ElementT]):
    def bind(self, element: ElementT) -> TensorSourceRef | None:
        ...
```

`TensorSourceRef` is intentionally thin. It should not discover sources,
validate element eligibility, choose reductions, map outputs, clone tensors, or
own model state. The element extractor does source discovery and validation
before creating the ref. For performance, runtime `get()` should be a direct
attribute lookup, tuple read, or cheap getter call.

Example source ref for a module parameter:

```python
import torch.nn as nn


@dataclass(frozen=True)
class ModuleParameterRef:
    module: nn.Module
    parameter_name: str

    def get(self) -> torch.Tensor:
        return getattr(self.module, self.parameter_name)

    def source_spec(self) -> TensorSpec:
        value = self.get()
        return TensorSpec(
            shape=value.shape,
            dtype=value.dtype,
            device=value.device,
        )

    def identity_key(self) -> Hashable:
        return ("module_parameter", id(self.module), self.parameter_name)
```

Example typed extractor for named modules:

```python
NamedModule: TypeAlias = tuple[str, nn.Module]


class NamedModuleParameterExtractor(ElementTensorExtractor[NamedModule]):
    def __init__(self, parameter_name: str) -> None:
        self.parameter_name = parameter_name

    def bind(self, element: NamedModule) -> TensorSourceRef | None:
        _, module = element

        if not isinstance(getattr(module, self.parameter_name, None), torch.Tensor):
            return None

        return ModuleParameterRef(module, self.parameter_name)
```

This is where `parameter_name="weight"` belongs. The reduction rule remains
generic.

## Reductions

A `TensorReduction` is one concrete operation. Runtime execution should not
branch on operation names.

```python
class TensorReduction(Protocol):
    def __call__(self, value: TensorValue) -> torch.Tensor:
        ...

    def output_shape(self, source_spec: SourceSpec) -> torch.Size:
        ...

    def output_dtype(self, source_spec: SourceSpec) -> torch.dtype:
        ...

    def identity_key(self) -> Hashable:
        ...
```

Example reduction:

```python
class SumDim:
    def __init__(
        self,
        dim: ReductionDim = None,
        keepdim: bool = False,
    ) -> None:
        self.dim = dim
        self.keepdim = keepdim

    def __call__(self, value: TensorValue) -> torch.Tensor:
        if isinstance(value, tuple):
            raise TypeError("SumDim expects one tensor, not a tensor tuple.")
        return value.sum(dim=self.dim, keepdim=self.keepdim)

    def output_shape(self, source_spec: SourceSpec) -> torch.Size:
        if not isinstance(source_spec, TensorSpec):
            raise TypeError("SumDim expects one source spec.")

        if self.dim is None:
            return torch.Size([])

        dims = (self.dim,) if isinstance(self.dim, int) else tuple(self.dim)
        rank = len(source_spec.shape)
        normalized_dims = tuple(dim if dim >= 0 else rank + dim for dim in dims)

        output = []
        for index, size in enumerate(source_spec.shape):
            if index in normalized_dims:
                if self.keepdim:
                    output.append(1)
                continue
            output.append(size)

        return torch.Size(output)

    def output_dtype(self, source_spec: SourceSpec) -> torch.dtype:
        if not isinstance(source_spec, TensorSpec):
            raise TypeError("SumDim expects one source spec.")
        return source_spec.dtype

    def identity_key(self) -> Hashable:
        return (type(self), self.dim, self.keepdim)
```

The factory can branch during planning. The calculation never sees this branch.

```python
def reduction_for_type(
    operation_type: TensorReductionType | str,
    *,
    dim: ReductionDim = None,
    keepdim: bool = False,
) -> TensorReduction:
    operation_type = TensorReductionType(operation_type)

    if operation_type == TensorReductionType.SUM:
        return SumDim(dim=dim, keepdim=keepdim)

    if operation_type == TensorReductionType.MEAN:
        return MeanDim(dim=dim, keepdim=keepdim)

    if operation_type == TensorReductionType.COUNT:
        return CountDim(dim=dim, keepdim=keepdim)

    if operation_type == TensorReductionType.L1:
        return L1Dim(dim=dim, keepdim=keepdim)

    if operation_type == TensorReductionType.L2:
        return L2Dim(dim=dim, keepdim=keepdim)

    raise ValueError(f"Unsupported reduction type: {operation_type}")
```

`MeanDim`, `CountDim`, `L1Dim`, and `L2Dim` follow the same interface as
`SumDim`.

## Reduction Operation

`ReductionOp` binds a concrete source to a concrete reduction. Planning and
validation use `output_spec` and `output_length`; they must not call `op()`.

```python
class ReductionOp(torch.nn.Module):
    def __init__(
        self,
        source_ref: TensorSourceRef,
        reduction: TensorReduction,
    ) -> None:
        super().__init__()
        self.source_ref = source_ref
        self.reduction = reduction

        source_spec = source_ref.source_spec()
        self._output_spec = TensorSpec(
            shape=reduction.output_shape(source_spec),
            dtype=reduction.output_dtype(source_spec),
            device=common_source_device(source_spec),
        )
        self._output_length = numel(self._output_spec.shape)

    @property
    def output_spec(self) -> TensorSpec:
        return self._output_spec

    @property
    def output_length(self) -> int:
        return self._output_length

    def identity_key(self) -> Hashable:
        return (
            self.source_ref.identity_key(),
            self.reduction.identity_key(),
        )

    def forward(self) -> torch.Tensor:
        return self.reduction(self.source_ref.get()).reshape(-1)
```

Helpers:

```python
def common_source_device(source_spec: SourceSpec) -> torch.device:
    if isinstance(source_spec, TensorSpec):
        return source_spec.device

    if len(source_spec) == 0:
        raise ValueError("Tuple source spec must not be empty.")

    device = source_spec[0].device
    if any(spec.device != device for spec in source_spec):
        raise ValueError("Tuple source tensors must live on the same device.")

    return device


def numel(shape: torch.Size) -> int:
    total = 1
    for size in shape:
        total *= int(size)
    return total
```

## Targets And Plan Records

The plan supports three write patterns:

- segment: write a full op output into one contiguous output range
- indexed: write a full op output into arbitrary destination indices
- indexed-gather: gather selected op output positions before indexed write

```python
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
class MappedReductionPlan:
    output_length: int
    output_spec: TensorSpec
    segment_entries: tuple[SegmentEntry, ...] = ()
    indexed_entries: tuple[IndexedEntry, ...] = ()
    indexed_gather_entries: tuple[IndexedGatherEntry, ...] = ()
    output_labels: tuple[str, ...] | None = None
```

## Builder API

The full builder implementation can live in the planner module. The generic
contracts only need this API:

```python
class ReductionPlanBuilder:
    def __init__(self, output_length: int | None = None) -> None:
        ...

    def reserve_segment(self, length: int) -> SegmentTarget:
        ...

    def add(self, record: ReductionRecord) -> None:
        ...

    def finalize(self) -> MappedReductionPlan:
        ...
```

For inferred-output plans, `reserve_segment()` appends a contiguous output
range and grows `output_length`. For fixed-output plans, mappers should return
explicit `SegmentTarget` or `IndexedTarget` values instead.

## Rules And Mappers

The rule composes an extractor, a reduction type, and a target mapper.

```python
class TargetMapper(Protocol[ElementT]):
    def map(
        self,
        element: ElementT,
        op: ReductionOp,
        builder: ReductionPlanBuilder,
    ) -> tuple[SegmentTarget | IndexedTarget, tuple[int, ...] | None]:
        ...


class ReductionRule(Protocol[ElementT]):
    def emit(
        self,
        element: ElementT,
        builder: ReductionPlanBuilder,
    ) -> Iterable[ReductionRecord]:
        ...
```

Reusable rule:

```python
class ElementReductionRule(Generic[ElementT]):
    def __init__(
        self,
        *,
        extractor: ElementTensorExtractor[ElementT],
        operation_type: TensorReductionType | str,
        target_mapper: TargetMapper[ElementT],
        dim: ReductionDim = None,
        keepdim: bool = False,
    ) -> None:
        self.extractor = extractor
        self.operation_type = TensorReductionType(operation_type)
        self.target_mapper = target_mapper
        self.dim = dim
        self.keepdim = keepdim

    def emit(
        self,
        element: ElementT,
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
        op = ReductionOp(source_ref, reduction)
        target, source_indices = self.target_mapper.map(element, op, builder)

        yield ReductionRecord(
            op=op,
            target=target,
            source_indices=source_indices,
        )
```

Sequential mapper:

```python
class SequentialSegmentMapper(TargetMapper[ElementT]):
    def map(
        self,
        element: ElementT,
        op: ReductionOp,
        builder: ReductionPlanBuilder,
    ) -> tuple[SegmentTarget, None]:
        return builder.reserve_segment(op.output_length), None
```

Planner:

```python
class GenericReductionPlanner(Generic[ElementT]):
    def __init__(
        self,
        elements: Iterable[ElementT],
        *,
        output_length: int | None = None,
    ) -> None:
        self.elements = tuple(elements)
        self.output_length = output_length

    def compile(self, rule: ReductionRule[ElementT]) -> MappedReductionPlan:
        builder = ReductionPlanBuilder(output_length=self.output_length)

        for element in self.elements:
            for record in rule.emit(element, builder):
                builder.add(record)

        return builder.finalize()
```

## Full Example: Parameter Sum Plan

This example builds a plan that reads each `nn.Linear.weight`, sums each row,
and concatenates those row sums into one output vector.

For a linear weight shaped `[out_features, in_features]`, `dim=1` means:

```text
one output value per output unit
```

That makes the plan suitable for a structured-unit-sum calculation. A
parameter-sum tracker can then compute the scalar parameter sum by summing the
calculation output.

### Model

```python
import torch
import torch.nn as nn


model = nn.Sequential(
    nn.Linear(2, 3, bias=False),
    nn.Linear(3, 2, bias=False),
)

with torch.no_grad():
    model[0].weight.copy_(
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
    )
    model[1].weight.copy_(
        torch.tensor(
            [
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        )
    )
```

### Elements

The element type is explicit: a named module is `tuple[str, nn.Module]`.

```python
linear_modules: list[NamedModule] = [
    (name, module)
    for name, module in model.named_modules()
    if isinstance(module, nn.Linear)
]
```

For this model, the elements are:

```text
("0", model[0])
("1", model[1])
```

### Plan Creation

The rule uses `TensorReductionType.SUM`. The extractor selects `weight`, and
the rule stays independent of parameter names.

```python
plan = GenericReductionPlanner[NamedModule](linear_modules).compile(
    ElementReductionRule[NamedModule](
        extractor=NamedModuleParameterExtractor("weight"),
        operation_type=TensorReductionType.SUM,
        target_mapper=SequentialSegmentMapper(),
        dim=1,
    )
)
```

What happens during compilation:

```text
element ("0", model[0])
    -> extractor binds model[0].weight
    -> SumDim(dim=1) output shape is [3]
    -> mapper reserves segment start=0, length=3

element ("1", model[1])
    -> extractor binds model[1].weight
    -> SumDim(dim=1) output shape is [2]
    -> mapper reserves segment start=3, length=2
```

The finalized plan has:

```text
output_length = 5
segment_entries = (
    SegmentEntry(op=sum(model[0].weight, dim=1), start=0, length=3),
    SegmentEntry(op=sum(model[1].weight, dim=1), start=3, length=2),
)
indexed_entries = ()
indexed_gather_entries = ()
```

### Runtime Result

At runtime, the calculation executes the plan without knowing about modules,
parameter names, or `TensorReductionType.SUM`.

```python
calculation = MappedReductionCalculation(plan)
structured_unit_sum = calculation()
parameter_sum = structured_unit_sum.sum()
```

The values are:

```text
model[0].weight.sum(dim=1) = [3.0, 7.0, 11.0]
model[1].weight.sum(dim=1) = [24.0, 33.0]

structured_unit_sum = [3.0, 7.0, 11.0, 24.0, 33.0]
parameter_sum = 78.0
```

The parameter-sum tracker can expose both values:

```python
{
    "structured_unit_sum": structured_unit_sum,
    "parameter_sum": float(parameter_sum.item()),
}
```

## Why This Shape Works

The planner is generic because every domain-specific decision is pushed to a
typed component:

- `NamedModuleParameterExtractor("weight")` chooses which parameter to read
- `TensorReductionType.SUM` chooses the reduction during planning
- `SumDim(dim=1)` performs the concrete tensor operation
- `SequentialSegmentMapper` chooses where each result is written
- `MappedReductionCalculation` only executes compiled entries

That keeps the hot path small:

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

## Implementation Notes

- Use `TensorReductionType` or another generic enum name in the generic planner
  layer. Keep weight-specific names as compatibility aliases if needed.
- Keep `parameter_name` inside extractors such as
  `NamedModuleParameterExtractor("weight")`.
- Make every successful `ElementTensorExtractor.bind(...)` return a
  `TensorSourceRef`. Do source discovery once during binding; keep runtime
  source refs thin.
- Use `op.output_length` and `op.output_spec` during planning and validation.
  Do not call `op()` just to discover output sizes.
- Derive calculation buffers from `plan.output_spec`, not from the first source
  tensor.
- Keep pruning-group offsets and source-index logic inside typed target mappers
  or context producers.
