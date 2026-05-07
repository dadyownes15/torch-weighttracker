from __future__ import annotations

import torch
import torch.nn as nn

from torch_structracker.reducer_plan import ReducerPlan


class NaiveStructuredUnitSum(nn.Module):
    """Allocation-heavy reference runner for a structured unit sum plan.

    This intentionally uses the same reducer plan as ``StructuredUnitSum`` but
    rebuilds destination tensors and the accumulator every time. It is useful as
    a small baseline for measuring the runtime value of the precompiled buffers
    in the optimized implementation.
    """

    def __init__(self, plan: ReducerPlan) -> None:
        super().__init__()
        if len(plan.mappings) == 0:
            raise ValueError("NaiveStructuredUnitSum requires at least one mapping.")

        self.output_length = int(plan.output_length)
        self.mappings = tuple(plan.mappings)

    @property
    def destination_tensor_allocations_per_call(self) -> int:
        return len(self.mappings)

    @torch.no_grad()
    def forward(self) -> torch.Tensor:
        accumulator = None

        for mapping in self.mappings:
            values = mapping.reducer().reshape(-1)
            if accumulator is None:
                accumulator = torch.zeros(
                    self.output_length,
                    device=values.device,
                    dtype=values.dtype,
                )

            destination_indices = torch.tensor(
                mapping.destination_indices,
                dtype=torch.long,
                device=values.device,
            )
            accumulator.index_add_(0, destination_indices, values)

        if accumulator is None:
            raise RuntimeError("NaiveStructuredUnitSum cannot run an empty plan.")

        return accumulator
