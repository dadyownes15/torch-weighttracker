import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.reductions.backend import ReductionBackend
from torch_weighttracker.reductions.builder import ReductionPlan
from torch_weighttracker.reductions.sparse import select_backend


class ReductionCalc(nn.Module):
    def __init__(
        self,
        plan: ReductionPlan,
        *,
        calculation_type: CalcType | str | None = None,
    ) -> None:
        if not isinstance(plan, ReductionPlan):
            raise TypeError(
                "ReductionCalc requires a ReductionPlan, "
                f"got {type(plan).__name__}."
            )

        super().__init__()
        self.calculation_type = (
            None if calculation_type is None else CalcType(calculation_type)
        )
        self.plan = plan
        self.output_length = int(plan.output_length)
        self.output_shape = torch.Size(plan.output_spec.shape)
        self.ops = nn.ModuleList()

        self._segment_specs: list[tuple[int, int, int]] = []
        self._indexed_specs: list[tuple[int, str]] = []
        self._indexed_gather_specs: list[tuple[int, str, str]] = []

        self.register_buffer(
            "output_anchor",
            torch.empty(
                (),
                dtype=plan.output_spec.dtype,
                device=plan.output_spec.device,
            ),
            persistent=False,
        )

        for index, entry in enumerate(plan.segment_entries):
            op_index = self._add_op(entry.op)
            self._segment_specs.append((op_index, int(entry.start), int(entry.length)))

        for index, entry in enumerate(plan.indexed_entries):
            op_index = self._add_op(entry.op)
            dst_name = f"dst_{index}"
            self.register_buffer(
                dst_name,
                torch.as_tensor(
                    entry.destination_indices,
                    dtype=torch.long,
                    device=plan.output_spec.device,
                ),
                persistent=False,
            )
            self._indexed_specs.append((op_index, dst_name))

        for index, entry in enumerate(plan.indexed_gather_entries):
            op_index = self._add_op(entry.op)
            src_name = f"gather_src_{index}"
            dst_name = f"gather_dst_{index}"
            self.register_buffer(
                src_name,
                torch.as_tensor(
                    entry.source_indices,
                    dtype=torch.long,
                    device=plan.output_spec.device,
                ),
                persistent=False,
            )
            self.register_buffer(
                dst_name,
                torch.as_tensor(
                    entry.destination_indices,
                    dtype=torch.long,
                    device=plan.output_spec.device,
                ),
                persistent=False,
            )
            self._indexed_gather_specs.append((op_index, src_name, dst_name))

        self._compile_runtime_entries()
        self.sparse_executor, self._sparse_flags = select_backend(
            plan,
            input_spec=None,
            loop_output=self._loop_output,
            label=type(self).__name__,
        )
        self._compile_runtime_sections()

    @property
    def backend(self) -> ReductionBackend:
        if self.sparse_executor is None:
            return ReductionBackend.LOOP
        return ReductionBackend.SPARSE

    def _add_op(self, op) -> int:
        index = len(self.ops)
        self.ops.append(op)
        return index

    @property
    def destination_indices(self) -> tuple[torch.Tensor, ...]:
        indexed = tuple(getattr(self, dst) for _, dst in self._indexed_specs)
        gathered = tuple(
            getattr(self, dst) for _, _, dst in self._indexed_gather_specs
        )
        return (*indexed, *gathered)

    def _compile_runtime_entries(self) -> None:
        self.segment_entries = tuple(
            (self.ops[op_index], start, length)
            for op_index, start, length in self._segment_specs
        )
        self.indexed_entries = tuple(
            (self.ops[op_index], getattr(self, dst_name))
            for op_index, dst_name in self._indexed_specs
        )
        self.indexed_gather_entries = tuple(
            (
                self.ops[op_index],
                getattr(self, src_name),
                getattr(self, dst_name),
            )
            for op_index, src_name, dst_name in self._indexed_gather_specs
        )

    def _split_flags(
        self,
        flags: tuple[bool, ...],
    ) -> tuple[tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]]:
        segment_count = len(self.segment_entries)
        indexed_count = len(self.indexed_entries)
        return (
            flags[:segment_count],
            flags[segment_count : segment_count + indexed_count],
            flags[segment_count + indexed_count :],
        )

    def _compile_runtime_sections(self) -> None:
        segment, indexed, gathered = self._split_flags(self._sparse_flags)
        self._loop_segment_entries = tuple(
            entry
            for entry, lowered in zip(self.segment_entries, segment, strict=True)
            if not lowered
        )
        self._loop_indexed_entries = tuple(
            entry
            for entry, lowered in zip(self.indexed_entries, indexed, strict=True)
            if not lowered
        )
        self._loop_indexed_gather_entries = tuple(
            entry
            for entry, lowered in zip(
                self.indexed_gather_entries, gathered, strict=True
            )
            if not lowered
        )

        sections = []
        if self._loop_segment_entries:
            sections.append(self._run_segment_entries)
        if self._loop_indexed_entries:
            sections.append(self._run_indexed_entries)
        if self._loop_indexed_gather_entries:
            sections.append(self._run_indexed_gather_entries)
        self._runtime_sections = tuple(sections)

    def _new_output(self) -> torch.Tensor:
        return self.output_anchor.new_zeros(self.output_shape)

    def forward(self) -> torch.Tensor:
        if self.sparse_executor is None:
            out = self._new_output()
        else:
            out = self.sparse_executor().reshape(self.output_shape)
        for run_section in self._runtime_sections:
            run_section(out)
        return out

    def _loop_output(
        self,
        flags: tuple[bool, ...],
        _input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Loop-path output restricted to the entries whose flag is set."""
        segment, indexed, gathered = self._split_flags(flags)
        out = self._new_output()
        for (op, start, length), keep in zip(
            self.segment_entries, segment, strict=True
        ):
            if keep:
                out.narrow(0, start, length).add_(op())
        for (op, dst), keep in zip(self.indexed_entries, indexed, strict=True):
            if keep:
                out.index_add_(0, dst, op())
        for (op, src, dst), keep in zip(
            self.indexed_gather_entries, gathered, strict=True
        ):
            if keep:
                out.index_add_(0, dst, op().index_select(0, src))
        return out

    def _run_segment_entries(self, out: torch.Tensor) -> None:
        for op, start, length in self._loop_segment_entries:
            out.narrow(0, start, length).add_(op())

    def _run_indexed_entries(self, out: torch.Tensor) -> None:
        for op, dst in self._loop_indexed_entries:
            out.index_add_(0, dst, op())

    def _run_indexed_gather_entries(self, out: torch.Tensor) -> None:
        for op, src, dst in self._loop_indexed_gather_entries:
            out.index_add_(0, dst, op().index_select(0, src))
