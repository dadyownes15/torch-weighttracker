import torch
import torch.nn as nn

from torch_structracker.calculations.base import BaseCalculation, CalculationType
from torch_structracker.operations import WeightOperationType
from torch_structracker.reducer_plan import (
    ReducerPlan,
    compile_reducer_plan_from_groups,
    validate_reducer_plan,
)
plan = compile_reducer_plan_from_groups(

            groups,
            operation_type=WeightOperationType.SUM,
            num_heads=num_heads,
            prune_dim=prune_dim,
            prune_num_heads=prune_num_heads,
        )

class StructuredUnitSum(BaseCalculation):
    calculation_type = CalculationType.STRUCTURED_UNIT_SUM

    @classmethod
    def from_groups(
        cls,
        groups,
        device=None,
        dtype=None,
        num_heads=None,
        prune_dim=None,
        prune_num_heads=False,
    ):
        plan = compile_reducer_plan_from_groups(
            groups,
            operation_type=WeightOperationType.SUM,
            num_heads=num_heads,
            prune_dim=prune_dim,
            prune_num_heads=prune_num_heads,
        )
        validate_reducer_plan(plan)
        return cls(plan, device=device, dtype=dtype)

    def __init__(self, plan: ReducerPlan, device=None, dtype=None) -> None:
        super().__init__()

        if len(plan.mappings) == 0:
            raise ValueError("StructuredUnitSum requires at least one reducer mapping.")

        self.output_length = plan.output_length
        self.reducers = nn.ModuleList()
        self._dst_names: list[str] = []

        first_reducer = plan.mappings[0].reducer
        first_weight = _first_tensor(first_reducer.parameter_extractor.get())

        device = first_weight.device if device is None else device
        dtype = first_weight.dtype if dtype is None else dtype

        for i, mapping in enumerate(plan.mappings):
            self.reducers.append(mapping.reducer)

            if len(mapping.destination_indices) == 0:
                raise ValueError(f"Empty mapping for reducer {i}")

            self.register_buffer(
                f"dst_{i}",
                torch.tensor(
                    mapping.destination_indices,
                    dtype=torch.long,
                    device=device,
                ),
                persistent=False,
            )

            self._dst_names.append(f"dst_{i}")

        self.register_buffer(
            "accumulator",
            torch.zeros(plan.output_length, device=device, dtype=dtype),
            persistent=False,
        )

        self.compile_indices()

    def compile_indices(self):
        self.destination_indices = tuple(
            getattr(self, name) for name in self._dst_names
        )
        return self

    @torch.no_grad()
    def forward(self):
        acc = self.accumulator
        acc.zero_()
        for reducer, dst in zip(self.reducers, self.destination_indices):
            acc.index_add_(0, dst, reducer().reshape(-1))

        return acc


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value

    return value[0]
