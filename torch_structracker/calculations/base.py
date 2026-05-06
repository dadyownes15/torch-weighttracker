from abc import ABC, abstractmethod
from enum import Enum

import torch
import torch.nn as nn


class CalculationType(str, Enum):
    STRUCTURED_UNIT_SUM = "structured_unit_sum"
    STRUCTURED_UNIT_NORM = "structured_unit_norm"
    STRUCTURED_UNIT_COUNT_FROM_NORM = "structured_unit_count"


class BaseCalculation(nn.Module, ABC):
    calculation_type: CalculationType

    @classmethod
    def from_groups(cls, groups, device=None, dtype=None, **kwargs):
        return cls(groups, device=device, dtype=dtype)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


def calculation_class_for_type(calculation_type: CalculationType):
    from torch_structracker.calculations.structured_unit_norm import StructuredUnitNorm
    from torch_structracker.calculations.structured_unit_sum import StructuredUnitSum

    calculation_type = CalculationType(calculation_type)
    calculation_classes = {
        CalculationType.STRUCTURED_UNIT_SUM: StructuredUnitSum,
        CalculationType.STRUCTURED_UNIT_NORM: StructuredUnitNorm,
    }

    if calculation_type not in calculation_classes:
        raise ValueError(f"Calculation type is not registered: {calculation_type}")

    return calculation_classes[calculation_type]


def create_calculation(
    calculation_type: CalculationType,
    groups,
    device=None,
    dtype=None,
    num_heads=None,
    prune_dim=None,
    prune_num_heads=False,
):
    calculation_type = CalculationType(calculation_type)

    if calculation_type == CalculationType.STRUCTURED_UNIT_COUNT_FROM_NORM:
        raise NotImplementedError(
            "StructuredUnitCount creation needs initial count inputs and is not "
            "wired into StructTracker yet."
        )

    calculation_cls = calculation_class_for_type(calculation_type)
    return calculation_cls.from_groups(
        groups,
        device=device,
        dtype=dtype,
        num_heads=num_heads,
        prune_dim=prune_dim,
        prune_num_heads=prune_num_heads,
    )
