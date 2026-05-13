import torch
import torch.nn as nn

from torch_structracker.calculations.base import CalcType
from torch_structracker.reductions.builder import MappedReductionPlan


class ReductionCalc(nn.Module):
    def __init__(
        self,
        plan: MappedReductionPlan,
        *,
        calculation_type: CalcType | str | None = None,
    ) -> None:
        if not isinstance(plan, MappedReductionPlan):
            raise TypeError(
                "MappedReductionCalculation requires a MappedReductionPlan, "
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
        self._compile_runtime_sections()

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

    def _compile_runtime_sections(self) -> None:
        sections = []
        if self.segment_entries:
            sections.append(self._run_segment_entries)
        if self.indexed_entries:
            sections.append(self._run_indexed_entries)
        if self.indexed_gather_entries:
            sections.append(self._run_indexed_gather_entries)
        self._runtime_sections = tuple(sections)

    def _new_output(self) -> torch.Tensor:
        return self.output_anchor.new_zeros(self.output_shape)

    def forward(self) -> torch.Tensor:
        out = self._new_output()
        for run_section in self._runtime_sections:
            run_section(out)
        return out

    def _run_segment_entries(self, out: torch.Tensor) -> None:
        for op, start, length in self.segment_entries:
            out.narrow(0, start, length).add_(op())

    def _run_indexed_entries(self, out: torch.Tensor) -> None:
        for op, dst in self.indexed_entries:
            out.index_add_(0, dst, op())

    def _run_indexed_gather_entries(self, out: torch.Tensor) -> None:
        for op, src, dst in self.indexed_gather_entries:
            out.index_add_(0, dst, op().index_select(0, src))


MappedReductionCalculation = ReductionCalc
