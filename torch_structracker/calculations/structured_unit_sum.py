import torch
import torch.nn as nn

from torch_structracker.calculations.base import BaseCalculation, CalculationType
from torch_structracker.operations import WeightOperationType
from torch_structracker.reducer_plan import (
    IndexedTarget,
    ReducerPlan,
    SegmentTarget,
    compile_reducer_plan_from_groups,
    validate_reducer_plan,
)


class StructuredUnitSum(BaseCalculation):
    calculation_type = CalculationType.STRUCTURED_UNIT_SUM

    @classmethod
    def from_groups(
        cls,
        groups,
        *,
        device=None,
        dtype=None,
        num_heads=None,
        prune_dim=None,
        prune_num_heads=False,
        **kwargs,
    ):
        return cls(
            groups,
            num_heads=num_heads,
            prune_dim=prune_dim,
            prune_num_heads=prune_num_heads,
            **kwargs,
        )

    def __init__(
        self,
        groups=None,
        *,
        plan: ReducerPlan | None = None,
        num_heads=None,
        prune_dim=None,
        prune_num_heads=False,
        validate: bool = True,
    ) -> None:
        super().__init__()

        if plan is None and isinstance(groups, ReducerPlan):
            plan = groups
            groups = None

        if plan is None:
            if groups is None:
                raise ValueError("StructuredUnitSum requires either groups or plan.")

            plan = compile_reducer_plan_from_groups(
                groups,
                operation_type=WeightOperationType.SUM,
                num_heads=num_heads,
                prune_dim=prune_dim,
                prune_num_heads=prune_num_heads,
            )

        if len(plan.mappings) == 0:
            raise ValueError("StructuredUnitSum requires at least one reducer mapping.")

        if validate:
            validate_reducer_plan(plan)

        self.output_length = int(plan.output_length)

        self.reducers = nn.ModuleList()

        # These are used to create the mappings from the reduction output, to our acc. We have three, one for each type of possible output from the plan, for example segmnet, index, and index_gather_specs. They are only registered in the buffer when actually filled, but allows compatiablility with any sumation plan
        self._segment_specs: list[tuple[int, int, int]] = []
        self._indexed_specs: list[tuple[int, str]] = []
        self._indexed_gather_specs: list[tuple[int, str, str]] = []

        prototype = plan.mappings[0].reducer.parameter_extractor.first_tensor()

        self.register_buffer(
            "accumulator",
            prototype.new_zeros(self.output_length),
            persistent=False,
        )

        for mapping_index, mapping in enumerate(plan.mappings):
            reducer_index = len(self.reducers)
            self.reducers.append(mapping.reducer)

            target = mapping.target

            if isinstance(target, SegmentTarget):
                self._segment_specs.append(
                    (
                        reducer_index,
                        int(target.start),
                        int(target.length),
                    )
                )
                continue

            if isinstance(target, IndexedTarget):
                dst_name = f"dst_{mapping_index}"

                self.register_buffer(
                    dst_name,
                    torch.as_tensor(
                        target.destination_indices,
                        device=prototype.device,
                        dtype=torch.long,
                    ),
                    persistent=False,
                )

                if target.source_indices is None:
                    self._indexed_specs.append((reducer_index, dst_name))
                    continue

                src_name = f"src_{mapping_index}"

                self.register_buffer(
                    src_name,
                    torch.as_tensor(
                        target.source_indices,
                        device=prototype.device,
                        dtype=torch.long,
                    ),
                    persistent=False,
                )

                self._indexed_gather_specs.append(
                    (
                        reducer_index,
                        src_name,
                        dst_name,
                    )
                )
                continue

            raise TypeError(f"Unknown reducer target type: {type(target)!r}")

        self._compile_runtime_entries()

    @property
    def destination_indices(self) -> tuple[torch.Tensor, ...]:
        indexed_destinations = tuple(
            getattr(self, dst_name) for _, dst_name in self._indexed_specs
        )
        gather_destinations = tuple(
            getattr(self, dst_name) for _, _, dst_name in self._indexed_gather_specs
        )
        return (*indexed_destinations, *gather_destinations)

    def _compile_runtime_entries(self) -> None:
        self.segment_entries = tuple(
            (self.reducers[reducer_index], start, length)
            for reducer_index, start, length in self._segment_specs
        )

        self.indexed_entries = tuple(
            (self.reducers[reducer_index], getattr(self, dst_name))
            for reducer_index, dst_name in self._indexed_specs
        )

        self.indexed_gather_entries = tuple(
            (
                self.reducers[reducer_index],
                getattr(self, src_name),
                getattr(self, dst_name),
            )
            for reducer_index, src_name, dst_name in self._indexed_gather_specs
        )

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
