import torch

from torch_weighttracker.extractors.extractor import SourceSpec, TensorSpec
from torch_weighttracker.operations.base import (
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

    def output_spec(self, source_spec: SourceSpec) -> TensorSpec:
        if not isinstance(source_spec, TensorSpec):
            raise TypeError(f"{type(self).__name__} expects one source tensor.")

        output_shape = _reduced_shape(
            source_spec.shape,
            dim=self.dim,
            keepdim=self.keepdim,
        )
        return TensorSpec(
            shape=torch.Size([_numel(output_shape)]),
            dtype=source_spec.dtype,
            device=source_spec.device,
        )


class SumWeight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.sum(dim=self.dim, keepdim=self.keepdim)


class ElementwiseSumWeight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight


class SquaredSumWeight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.square().sum(dim=self.dim, keepdim=self.keepdim)


class ElementwiseSquaredSumWeight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.square()


class MeanWeight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.mean(dim=self.dim, keepdim=self.keepdim)


class CountWeight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(weight)
        return ones.sum(dim=self.dim, keepdim=self.keepdim)


class ActiveWeight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        active = weight.ne(0)
        if self.dim is None:
            return active.any().to(dtype=weight.dtype)

        if self.dim == ():
            return active.to(dtype=weight.dtype)

        return active.any(dim=self.dim, keepdim=self.keepdim).to(dtype=weight.dtype)


class L1Weight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.abs().sum(dim=self.dim, keepdim=self.keepdim)


class L2Weight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(
            weight,
            ord=2,
            dim=self.dim,
            keepdim=self.keepdim,
        )


class ElementwiseL2Weight(_DimensionalWeightOperation):
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.abs()


def create_generic_operation(
    operation: WeightOperationType | str,
    dim: ReductionDim = None,
    keepdim: bool = False,
) -> WeightOperation:
    operation = WeightOperationType(operation)

    if operation == WeightOperationType.SUM:
        if dim == ():
            return ElementwiseSumWeight(dim=dim, keepdim=keepdim)
        return SumWeight(dim=dim, keepdim=keepdim)

    if operation == WeightOperationType.SQUARED_SUM:
        if dim == ():
            return ElementwiseSquaredSumWeight(dim=dim, keepdim=keepdim)
        return SquaredSumWeight(dim=dim, keepdim=keepdim)

    if operation == WeightOperationType.MEAN:
        return MeanWeight(dim=dim, keepdim=keepdim)

    if operation == WeightOperationType.COUNT:
        return CountWeight(dim=dim, keepdim=keepdim)

    if operation == WeightOperationType.ACTIVE:
        return ActiveWeight(dim=dim, keepdim=keepdim)

    if operation == WeightOperationType.L1:
        return L1Weight(dim=dim, keepdim=keepdim)

    if operation == WeightOperationType.L2:
        if dim == ():
            return ElementwiseL2Weight(dim=dim, keepdim=keepdim)
        return L2Weight(dim=dim, keepdim=keepdim)

    raise ValueError(f"Unknown weight operation: {operation}")


def _reduced_shape(
    shape: torch.Size,
    *,
    dim: ReductionDim,
    keepdim: bool,
) -> torch.Size:
    if dim is None:
        return torch.Size([])

    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    if len(dims) == 0:
        return torch.Size(shape)

    rank = len(shape)
    normalized_dims = set()
    for item in dims:
        normalized = int(item)
        if normalized < 0:
            normalized += rank
        if normalized < 0 or normalized >= rank:
            raise ValueError(
                f"Reduction dim {item} is outside tensor rank {rank}."
            )
        normalized_dims.add(normalized)

    output: list[int] = []
    for index, size in enumerate(shape):
        if index in normalized_dims:
            if keepdim:
                output.append(1)
            continue
        output.append(int(size))

    return torch.Size(output)


def _numel(shape: torch.Size) -> int:
    total = 1
    for size in shape:
        total *= int(size)
    return total
