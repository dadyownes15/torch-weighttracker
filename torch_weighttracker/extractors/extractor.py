
from dataclasses import dataclass
from typing import Hashable, Protocol, TypeAlias, TypeVar

import torch
import torch.nn as nn


TensorValue: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]
Element = TypeVar("Element")

@dataclass(frozen=True)
class TensorSpec:
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


SourceSpec: TypeAlias = TensorSpec | tuple[TensorSpec, ...]


class TensorSourceRef(Protocol):
    def get(self) -> TensorValue:
        ...

    def source_spec(self) -> SourceSpec:
        ...

    def identity_key(self) -> Hashable:
        ...


class TensorExtractor(Protocol):
    def get(self) -> TensorValue:
        ...

    def identity_key(self) -> Hashable:
        ...


class ElementTensorExtractor(Protocol[Element]):
    def bind(self, element: Element) -> TensorSourceRef | None:
        ...

@dataclass(frozen=True)
class ValueTensorRef:
    value: torch.Tensor
    spec: TensorSpec

    def get(self) -> torch.Tensor:
        return self.value
        
    def source_spec(self) -> TensorSpec:
        return self.spec
        
    def identity_key(self) -> Hashable:
        return id(self) 
    
     
@dataclass(frozen=True)
class ModuleParameterRef:
    module: nn.Module
    name: str

    def get(self) -> torch.Tensor:
        return getattr(self.module, self.name)

    def source_spec(self) -> TensorSpec:
        value = self.get()
        return TensorSpec(
            shape=value.shape,
            dtype=value.dtype,
            device=value.device,
        )

    def identity_key(self) -> Hashable:
        return ("module_parameter", id(self.module), self.name)


@dataclass(frozen=True)
class ModuleParameterTupleRef:
    refs: tuple[ModuleParameterRef, ...]

    def __post_init__(self) -> None:
        if len(self.refs) == 0:
            raise ValueError("ModuleParameterTupleRef requires at least one ref.")

    def get(self) -> tuple[torch.Tensor, ...]:
        return tuple(ref.get() for ref in self.refs)

    def source_spec(self) -> tuple[TensorSpec, ...]:
        return tuple(ref.source_spec() for ref in self.refs)

    def identity_key(self) -> Hashable:
        return ("module_parameter_tuple", tuple(ref.identity_key() for ref in self.refs))


class ModuleWeightExtractor(ElementTensorExtractor[nn.Module]):
    def bind(self, element: nn.Module) -> TensorSourceRef | None:
        if not isinstance(element, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(element).__name__}")

        if not isinstance(getattr(element, "weight", None), torch.Tensor):
            return None

        return ModuleParameterRef(element, "weight")
