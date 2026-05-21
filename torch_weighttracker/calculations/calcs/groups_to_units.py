from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import group_input_spec
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.canonical_units import CanonicalUnitGroup
from torch_weighttracker.extractors.extractor import TensorSpec


class GroupsToUnits(nn.Module):
    calculation_type = CalcType.GROUPS_TO_UNITS

    def __init__(
        self,
        groups: Iterable[CanonicalUnitGroup],
        *,
        input_spec: TensorSpec,
    ) -> None:
        super().__init__()
        canonical_groups = tuple(groups)
        source_indices = [
            int(group.group_id)
            for group in canonical_groups
            for _ in range(int(group.length))
        ]
        self.output_length = len(source_indices)
        self.output_shape = torch.Size([self.output_length])

        self.register_buffer(
            "source_indices",
            torch.tensor(
                source_indices,
                dtype=torch.long,
                device=input_spec.device,
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1).index_select(0, self.source_indices)


def create_groups_to_units_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    input_spec: TensorSpec,
) -> GroupsToUnits:
    return GroupsToUnits(groups, input_spec=input_spec)


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.GROUPS_TO_UNITS,
    create=lambda ctx, deps: create_groups_to_units_calc(
        ctx.canonical_groups,
        input_spec=group_input_spec(ctx),
    ),
)
