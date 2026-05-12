from typing import Hashable, Protocol
import torch
import torch.nn as nn

from torch_structracker.extractor import TensorExtractor, TensorValue


class TensorReduction(Protocol):
    def __call__(self, value: TensorValue) -> torch.Tensor:
        ...

    def output_shape(self, source_shape) -> tuple[int, ...]:
        ...

    def identity_key(self) -> Hashable:
        ...     


class ReductionOp(nn.Module):
    def __init__(self, extractor: TensorExtractor, reduction: TensorReduction):
        super().__init__()
        self.extractor = extractor
        self.reduction = reduction

        self._output_shape = reduction.output_shape(extractor.out())
        self._output_length = torch.numel(self._output_shape)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    @property
    def output_length(self) -> int:
        return self._output_length

    def first_tensor(self) -> torch.Tensor:
        return self.extractor.first_tensor()

    def identity_key(self):
        return (
            self.extractor.identity_key(),
            self.reduction.identity_key(),
        )

    def forward(self) -> torch.Tensor:
        return self.reduction(self.extractor.get()).reshape(-1)
