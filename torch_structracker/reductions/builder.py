from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

import torch

from torch_structracker.extractors.extractor import TensorSpec
from torch_structracker.reductions.ops import ReductionOp


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

Selection: TypeAlias = FullSelection | SegmentSelection | IndexSelection


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
class _BuildEvaluation:
    output_length: int
    output_spec: TensorSpec
    segment_entries: tuple[SegmentEntry, ...]
    indexed_entries: tuple[IndexedEntry, ...]
    indexed_gather_entries: tuple[IndexedGatherEntry, ...]


@dataclass(frozen=True)
class ComputationPlan(ABC):
    output_length: int
    output_spec: TensorSpec
    segment_entries: tuple[SegmentEntry, ...] = ()
    indexed_entries: tuple[IndexedEntry, ...] = ()
    indexed_gather_entries: tuple[IndexedGatherEntry, ...] = ()
    output_labels: tuple[str, ...] | None = None

    @abstractmethod
    def kind(self) -> str:
        ...


@dataclass(frozen=True, kw_only=True)
class MappedReductionPlan(ComputationPlan):
    def kind(self) -> str:
        return "mapped_reduction_plan"


@dataclass(frozen=True, kw_only=True)
class PipelinePlan(ComputationPlan):
    input_spec: TensorSpec

    def kind(self) -> str:
        return "pipeline_plan"


class ReductionPlanBuilder:
    def __init__(self, output_length: int | None = None):
        self._fixed_output_length = output_length is not None
        self.output_length = 0 if output_length is None else int(output_length)

        self.segment_entries: list[SegmentEntry] = []
        self.indexed_entries: list[IndexedEntry] = []
        self.indexed_gather_entries: list[IndexedGatherEntry] = []

        self.output_labels: list[str] | None = None

    def reserve_segment(self, length: int) -> SegmentSelection:
        """
        Useful for module-wise plans where output is built sequentially.
        """
        if self._fixed_output_length:
            raise RuntimeError("Cannot reserve into a fixed-output builder.")

        start = self.output_length
        length = int(length)
        self.output_length += length
        return SegmentSelection(start=start, length=length)

    def add(self, record: ReductionRecord) -> None:
        source_indices = self._lower_source_selection(
            record.mapping.source,
            op_output_length=record.op.output_length,
        )
        source_length = (
            int(record.op.output_length)
            if source_indices is None
            else len(source_indices)
        )
        if source_indices is not None:
            _validate_indices(
                source_indices,
                upper_bound=int(record.op.output_length),
                label="Reduction mapping source indices",
                upper_bound_label="reduction op output length",
            )

        target = self._normalize_target_selection(
            record.mapping.target,
            source_length=source_length,
        )

        if isinstance(target, SegmentSelection):
            start = int(target.start)
            length = int(target.length)
            self._validate_segment_selection(target, label="Segment target")
            self._validate_selection_lengths(
                source_length=source_length,
                target_length=length,
            )
            self._touch_output(start + length)

            if source_indices is not None:
                self.indexed_gather_entries.append(
                    IndexedGatherEntry(
                        op=record.op,
                        source_indices=source_indices,
                        destination_indices=tuple(range(start, start + length)),
                    )
                )
                return

            self.segment_entries.append(
                SegmentEntry(
                    op=record.op,
                    start=start,
                    length=length,
                )
            )
            return

        if isinstance(target, IndexSelection):
            destination_indices = self._lower_index_target_selection(
                target,
                source_length=source_length,
            )
            _validate_indices_lower_bound(
                destination_indices,
                label="Reduction mapping target indices",
            )
            if len(destination_indices) > 0:
                self._touch_output(max(destination_indices) + 1)
            _validate_indices(
                destination_indices,
                upper_bound=self.output_length,
                label="Reduction mapping target indices",
                upper_bound_label="reduction plan output length",
            )

            if source_indices is not None:
                self.indexed_gather_entries.append(
                    IndexedGatherEntry(
                        op=record.op,
                        source_indices=source_indices,
                        destination_indices=destination_indices,
                    )
                )
                return

            self.indexed_entries.append(
                IndexedEntry(
                    op=record.op,
                    destination_indices=destination_indices,
                )
            )
            return

        raise TypeError(f"Unknown target type: {type(target)!r}")

    def finalize(self, input_spec: TensorSpec | None = None) -> ComputationPlan:
        evaluation = self._evaluate_build()
        if input_spec is None:
            return MappedReductionPlan(
                output_length=evaluation.output_length,
                segment_entries=evaluation.segment_entries,
                indexed_entries=evaluation.indexed_entries,
                indexed_gather_entries=evaluation.indexed_gather_entries,
                output_labels=(
                    None if self.output_labels is None else tuple(self.output_labels)
                ),
                output_spec=evaluation.output_spec,
            )

        return PipelinePlan(
            output_length=evaluation.output_length,
            segment_entries=evaluation.segment_entries,
            indexed_entries=evaluation.indexed_entries,
            indexed_gather_entries=evaluation.indexed_gather_entries,
            output_labels=(
                None if self.output_labels is None else tuple(self.output_labels)
            ),
            output_spec=evaluation.output_spec,
            input_spec=input_spec,
        )

    def _normalize_target_selection(
        self,
        selection: Selection,
        *,
        source_length: int,
    ) -> SegmentSelection | IndexSelection:
        if isinstance(selection, FullSelection):
            length = self.output_length
            if length == 0 and not self._fixed_output_length:
                length = source_length
                self._touch_output(length)
            return SegmentSelection(start=0, length=length)

        if isinstance(selection, (SegmentSelection, IndexSelection)):
            return selection

        raise TypeError(f"Unknown target type: {type(selection)!r}")

    @staticmethod
    def _lower_source_selection(
        selection: Selection,
        *,
        op_output_length: int,
    ) -> tuple[int, ...] | None:
        if isinstance(selection, FullSelection):
            return None

        if isinstance(selection, SegmentSelection):
            start = int(selection.start)
            length = int(selection.length)
            if length < 0:
                raise ValueError("Segment source length must be non-negative.")
            return tuple(range(start, start + length))

        if isinstance(selection, IndexSelection):
            return ReductionPlanBuilder._indices_tuple(selection.indices)

        raise TypeError(f"Unknown source type: {type(selection)!r}")

    @staticmethod
    def _indices_tuple(indices: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(int(index) for index in indices)

    @staticmethod
    def _validate_segment_selection(
        selection: SegmentSelection,
        *,
        label: str,
    ) -> None:
        if int(selection.start) < 0:
            raise ValueError(f"{label} start must be non-negative.")

        if int(selection.length) < 0:
            raise ValueError(f"{label} length must be non-negative.")

    @staticmethod
    def _validate_selection_lengths(
        *,
        source_length: int,
        target_length: int,
    ) -> None:
        if source_length != target_length:
            raise ValueError(
                "Reduction mapping source and target lengths must match."
            )

    @staticmethod
    def _lower_index_target_selection(
        selection: IndexSelection,
        *,
        source_length: int,
    ) -> tuple[int, ...]:
        destination_indices = ReductionPlanBuilder._indices_tuple(selection.indices)
        if len(destination_indices) == source_length:
            return destination_indices

        if len(destination_indices) == 1 and source_length > 1:
            return destination_indices * source_length

        ReductionPlanBuilder._validate_selection_lengths(
            source_length=source_length,
            target_length=len(destination_indices),
        )
        return destination_indices

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

    def _evaluate_build(self) -> _BuildEvaluation:
        output_length = int(self.output_length)
        segment_entries = tuple(self.segment_entries)
        indexed_entries = tuple(self.indexed_entries)
        indexed_gather_entries = tuple(self.indexed_gather_entries)

        if output_length < 0:
            raise ValueError("Reduction plan output length must be non-negative.")

        if (
            len(segment_entries) == 0
            and len(indexed_entries) == 0
            and len(indexed_gather_entries) == 0
        ):
            raise ValueError("Cannot finalize an empty reduction plan.")

        dtype: torch.dtype | None = None
        device: torch.device | None = None

        for entry in segment_entries:
            spec = self._validate_entry_output_spec(entry.op)
            dtype, device = self._merge_entry_spec(dtype, device, spec)
            self._validate_segment_entry(entry, output_length)

        for entry in indexed_entries:
            spec = self._validate_entry_output_spec(entry.op)
            dtype, device = self._merge_entry_spec(dtype, device, spec)
            self._validate_indexed_entry(entry, output_length)

        for entry in indexed_gather_entries:
            spec = self._validate_entry_output_spec(entry.op)
            dtype, device = self._merge_entry_spec(dtype, device, spec)
            self._validate_indexed_gather_entry(entry, output_length)

        if dtype is None or device is None:
            raise ValueError("Cannot infer output spec for an empty reduction plan.")

        return _BuildEvaluation(
            output_length=output_length,
            output_spec=TensorSpec(
                shape=torch.Size([output_length]),
                dtype=dtype,
                device=device,
            ),
            segment_entries=segment_entries,
            indexed_entries=indexed_entries,
            indexed_gather_entries=indexed_gather_entries,
        )

    @staticmethod
    def _validate_entry_output_spec(op: ReductionOp) -> TensorSpec:
        spec = op.output_spec
        if len(spec.shape) != 1:
            raise ValueError("Reduction op output spec must be one-dimensional.")

        spec_length = int(spec.shape[0])
        if spec_length != int(op.output_length):
            raise ValueError(
                "Reduction op output_length must match its output spec shape."
            )

        return spec

    @staticmethod
    def _merge_entry_spec(
        dtype: torch.dtype | None,
        device: torch.device | None,
        spec: TensorSpec,
    ) -> tuple[torch.dtype, torch.device]:
        if dtype is None or device is None:
            return spec.dtype, spec.device

        if spec.dtype != dtype:
            raise ValueError("All reduction entries must have matching dtype.")

        if spec.device != device:
            raise ValueError("All reduction entries must have matching device.")

        return dtype, device

    @staticmethod
    def _validate_segment_entry(entry: SegmentEntry, output_length: int) -> None:
        ReductionPlanBuilder._validate_segment_selection(
            SegmentSelection(start=entry.start, length=entry.length),
            label="Segment target",
        )

        if entry.start + entry.length > output_length:
            raise ValueError("Segment target exceeds reduction plan output length.")

        if entry.length != int(entry.op.output_length):
            raise ValueError(
                "Segment target length must match reduction op output length."
            )

    @staticmethod
    def _validate_indexed_entry(entry: IndexedEntry, output_length: int) -> None:
        if len(entry.destination_indices) != int(entry.op.output_length):
            raise ValueError(
                "Indexed target destination count must match reduction op "
                "output length."
            )

        _validate_indices(
            entry.destination_indices,
            upper_bound=output_length,
            label="Indexed target destinations",
            upper_bound_label="reduction plan output length",
        )

    @staticmethod
    def _validate_indexed_gather_entry(
        entry: IndexedGatherEntry,
        output_length: int,
    ) -> None:
        if len(entry.source_indices) != len(entry.destination_indices):
            raise ValueError(
                "Indexed gather source and destination counts must match."
            )

        _validate_indices(
            entry.source_indices,
            upper_bound=int(entry.op.output_length),
            label="Indexed gather sources",
            upper_bound_label="reduction op output length",
        )
        _validate_indices(
            entry.destination_indices,
            upper_bound=output_length,
            label="Indexed gather destinations",
            upper_bound_label="reduction plan output length",
        )


def _validate_indices(
    indices: tuple[int, ...],
    *,
    upper_bound: int,
    label: str,
    upper_bound_label: str,
) -> None:
    if len(indices) == 0:
        return

    _validate_indices_lower_bound(indices, label=label)

    max_index = max(indices)
    if max_index >= upper_bound:
        raise ValueError(f"{label} exceeds {upper_bound_label}.")


def _validate_indices_lower_bound(
    indices: tuple[int, ...],
    *,
    label: str,
) -> None:
    if len(indices) == 0:
        return

    min_index = min(indices)
    if min_index < 0:
        raise ValueError(f"{label} must be non-negative.")
