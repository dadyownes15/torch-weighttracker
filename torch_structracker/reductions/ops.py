from typing import Hashable, Protocol, TypeAlias
import torch
import torch.nn as nn

from torch_structracker.extractors.extractor import (
    TensorSourceRef,
    TensorSpec,
    TensorValue,
)

SourceSpec: TypeAlias = TensorSpec | tuple[TensorSpec, ...]

class TensorReduction(Protocol):
    def __call__(self, value: TensorValue) -> torch.Tensor:
        ...

    def output_spec(self, source_spec: SourceSpec) -> TensorSpec:
        ...

    def identity_key(self) -> Hashable:
        ...


class IdentityTensorReduction:
    def __call__(self, value: TensorValue) -> torch.Tensor:
         return value.reshape(-1)

    def output_spec(self, source_spec: SourceSpec) -> TensorSpec:
        if not isinstance(source_spec, TensorSpec):
            raise TypeError("IdentityTensorReduction expects one source tensor.")

        return TensorSpec(
            shape=torch.Size([numel(source_spec.shape)]),
            dtype=source_spec.dtype,
            device=source_spec.device,
        )

    def identity_key(self) -> Hashable:
        return (type(self),)


class ReductionOp(nn.Module):
    def __init__(
        self,
        source_ref: TensorSourceRef,
        reduction: TensorReduction,
    ) -> None:
        super().__init__()
        self.source_ref = source_ref
        self.reduction = reduction

        self._source_spec = source_ref.source_spec()
        self._output_spec = reduction.output_spec(self._source_spec)
        self._output_length = numel(self._output_spec.shape)
    @property
    def source_spec(self) -> SourceSpec:
        return self._source_spec

    @property
    def output_spec(self) -> TensorSpec:
        return self._output_spec

    @property
    def output_length(self) -> int:
        return self._output_length
        
    def identity_key(self) -> Hashable:
        return (
            self.source_ref.identity_key(),
            self.reduction.identity_key(),
        )

    def forward(self) -> torch.Tensor:
        return self.reduction(self.source_ref.get()).reshape(-1)



def common_source_device(source_spec: SourceSpec) -> torch.device:
    if isinstance(source_spec, TensorSpec):
        return source_spec.device

    if len(source_spec) == 0:
        raise ValueError("Tuple source spec must not be empty.")

    device = source_spec[0].device
    if any(spec.device != device for spec in source_spec):
        raise ValueError("Tuple source tensors must live on the same device.")

    return device


def numel(shape: torch.Size) -> int:
    total = 1
    for size in shape:
        total *= int(size)
    return total
