from typing import Hashable, Protocol, TypeAlias

import torch
import torch.nn as nn


TensorValue: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]


class TensorExtractor(Protocol):
    def get(self) -> TensorValue:
        ...

    def identity_key(self) -> Hashable:
        ...
    def output_shape(self) -> torch.Size:
        ...



class ParameterExtractor(TensorExtractor):
    """
    Non-owning reference to a parameter or tensor attribute on an existing module.

    Useful when another object needs access to a model parameter without
    registering that parameter as its own.
    """

    def __init__(self, module: nn.Module, name: str = "weight"):
        if not isinstance(module, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(module).__name__}")

        if not isinstance(name, str):
            raise TypeError(
                f"Expected parameter name as str, got {type(name).__name__}"
            )

        if not hasattr(module, name):
            raise ValueError(
                f"{module.__class__.__name__} has no attribute '{name}'"
            )

        self.module = module
        self.name = name

    def get(self) -> torch.Tensor:
        value = getattr(self.module, self.name, None)

        if value is None:
            raise RuntimeError(
                f"{self.module.__class__.__name__}.{self.name} is None"
            )

        return value

    def identity_key(self):
        return (id(self.module), self.name)


class ParameterTupleExtractor(TensorExtractor):
    """Non-owning extractor for operations that consume multiple parameters."""

    def __init__(self, *extractors: ParameterExtractor):
        if len(extractors) == 0:
            raise ValueError(
                "ParameterTupleExtractor requires at least one extractor."
            )

        self.extractors = tuple(extractors)

    def get(self) -> tuple[torch.Tensor, ...]:
        return tuple(extractor.get() for extractor in self.extractors)

    def identity_key(self):
        return tuple(extractor.identity_key() for extractor in self.extractors)


class FusedQKVExtractor(ParameterExtractor):
    """Extractor for packed QKV tensors such as in_proj_weight or qkv.weight."""

    def identity_key(self):
        return ("fused_qkv", *super().identity_key())


class SeparateQKVExtractor(TensorExtractor):
    """Extractor for separate q_proj_weight, k_proj_weight, and v_proj_weight."""

    def __init__(
        self,
        module: nn.Module,
        q_name: str = "q_proj_weight",
        k_name: str = "k_proj_weight",
        v_name: str = "v_proj_weight",
    ):
        self.module = module
        self.q_extractor = ParameterExtractor(module, q_name)
        self.k_extractor = ParameterExtractor(module, k_name)
        self.v_extractor = ParameterExtractor(module, v_name)
        
    def get(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.q_extractor.get(),
            self.k_extractor.get(),
            self.v_extractor.get(),
        )

    def identity_key(self):
        return (
            "separate_qkv",
            self.q_extractor.identity_key(),
            self.k_extractor.identity_key(),
            self.v_extractor.identity_key(),
        )
