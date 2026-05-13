from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn


ReductionDim = Optional[int | tuple[int, ...]]


class WeightOperationType(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    ACTIVE = "active"
    L1 = "l1"
    L2 = "l2"


class WeightOperation(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def identity_key(self):
        return (type(self),)

    @staticmethod
    def create(
        operation: WeightOperationType | str,
        dim: ReductionDim = None,
        keepdim: bool = False,
    ) -> "WeightOperation":
        from torch_structracker.operations.generic import create_generic_operation

        return create_generic_operation(operation, dim=dim, keepdim=keepdim)
