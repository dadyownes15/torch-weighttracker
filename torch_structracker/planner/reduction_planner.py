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




## This is used to compile a reducion plan, that can be computed efficiently using calculations. 
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