from abc import ABC, abstractmethod
from enum import Enum

import torch
import torch.nn as nn


class CalculationType(str, Enum):
    STRUCTURED_UNIT_SUM = "structured_unit_sum"
    ACTIVE_UNITS = "active_units"
    BITRATE_PR_MODULE = "bitrate_pr_module"
    UNITS_TO_MODULE = "units_to_module"


class BaseCalculation(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


def calculation_class_for_type(calculation_type: CalculationType):
    from torch_structracker.calculations.bit_rate_pr_module import BitRatePrModule
    from torch_structracker.calculations.structured_unit_sum import StructuredUnitSum

    calculation_type = CalculationType(calculation_type)
    calculation_classes = {
        CalculationType.BITRATE_PR_MODULE: BitRatePrModule,
        CalculationType.STRUCTURED_UNIT_SUM: StructuredUnitSum,
    }

    if calculation_type not in calculation_classes:
        raise ValueError(f"Calculation type is not registered: {calculation_type}")

    return calculation_classes[calculation_type]


def create_calculation(
    calculation_type: CalculationType,
    groups,
    model=None,
    device=None,
    dtype=None,
    num_heads=None,
    prune_dim=None,
    prune_num_heads=False,
):
    calculation_type = CalculationType(calculation_type)

    calculation_cls = calculation_class_for_type(calculation_type)
    if calculation_type == CalculationType.BITRATE_PR_MODULE:
        if model is None:
            raise ValueError("BITRATE_PR_MODULE requires a model.")
        return calculation_cls.from_model(
            model,
            device=device,
            dtype=dtype,
        )

    return calculation_cls.from_groups(
        groups,
        device=device,
        dtype=dtype,
        num_heads=num_heads,
        prune_dim=prune_dim,
        prune_num_heads=prune_num_heads,
    )
