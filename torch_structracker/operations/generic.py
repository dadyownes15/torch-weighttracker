import torch

from torch_structracker.operations.base import (
    ReductionDim,
    WeightOperation,
    WeightOperationType,
)


class _DimensionalWeightOperation(WeightOperation):
    def __init__(self, dim: ReductionDim = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def identity_key(self):
        return (type(self), self.dim, self.keepdim)


class SumWeight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.sum(dim=self.dim, keepdim=self.keepdim)


class MeanWeight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.mean(dim=self.dim, keepdim=self.keepdim)


class CountWeight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(weight)
        return ones.sum(dim=self.dim, keepdim=self.keepdim)


class L1Weight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.abs().sum(dim=self.dim, keepdim=self.keepdim)


class L2Weight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((weight**2).sum(dim=self.dim, keepdim=self.keepdim))


def create_generic_operation(
    operation: WeightOperationType | str,
    dim: ReductionDim = None,
    keepdim: bool = False,
) -> WeightOperation:
    operation = WeightOperationType(operation)

    if operation == WeightOperationType.SUM:
        return SumWeight(dim=dim, keepdim=keepdim)

    if operation == WeightOperationType.MEAN:
        return MeanWeight(dim=dim, keepdim=keepdim)

    if operation == WeightOperationType.COUNT:
        return CountWeight(dim=dim, keepdim=keepdim)

    if operation == WeightOperationType.L1:
        return L1Weight(dim=dim, keepdim=keepdim)

    if operation == WeightOperationType.L2:
        return L2Weight(dim=dim, keepdim=keepdim)

    raise ValueError(f"Unknown weight operation: {operation}")
