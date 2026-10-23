from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
import torch.nn as nn

from torch_weighttracker.extractors.extractor import (
    ElementTensorExtractor,
    TensorSpec,
)

BitwidthFn = Callable[[str, nn.Module], tuple[Any, Any]]
"""`bitwidth_fn(name, module) -> (activation_bits, weight_bits)`.

See the README section "Exposing bit-widths".
"""

DEFAULT_BITWIDTH = 32


@dataclass(frozen=True)
class LayerBitwidth:
    name: str
    activation_bits: int
    weight_bits: int
    source: str


@dataclass(frozen=True)
class BitwidthView:
    layers: tuple[LayerBitwidth, ...]


@dataclass(frozen=True, eq=False)
class ModuleBitwidthRef:
    """Runtime source ref that re-resolves a module's bitwidths on every `get()`."""

    extractor: ModuleBitwidthExtractor
    module: nn.Module
    name: str
    device: torch.device

    def get(self) -> torch.Tensor:
        activation_bits, weight_bits = self.extractor.resolve(self.name, self.module)
        return torch.tensor(
            (float(activation_bits), float(weight_bits)),
            device=self.device,
            dtype=self.extractor.dtype,
        )

    def source_spec(self) -> TensorSpec:
        return TensorSpec(
            shape=torch.Size([2]),
            dtype=self.extractor.dtype,
            device=self.device,
        )

    def identity_key(self) -> Hashable:
        return (
            "module_bitwidth",
            id(self.module),
            id(self.extractor.bitwidth_fn),
            self.device,
            self.extractor.dtype,
        )


class ModuleBitwidthExtractor(ElementTensorExtractor[nn.Module]):
    """Bind weighted modules to live `(activation_bits, weight_bits)` source refs.

    Bitwidths come only from `bitwidth_fn`; without it every module is
    `(32, 32)`.
    """

    def __init__(
        self,
        *,
        bitwidth_fn: BitwidthFn | None = None,
        module_names: Mapping[nn.Module, str] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if bitwidth_fn is not None and not callable(bitwidth_fn):
            raise TypeError("bitwidth_fn must be callable or None.")
        self.bitwidth_fn = bitwidth_fn
        self.module_names = {} if module_names is None else dict(module_names)
        self.device = device
        self.dtype = torch.float32 if dtype is None else dtype

    @property
    def source(self) -> str:
        return "default" if self.bitwidth_fn is None else "bitwidth_fn"

    def bind(self, element: nn.Module) -> ModuleBitwidthRef | None:
        if not isinstance(element, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(element).__name__}")

        if not isinstance(self._weight_tensor_for(element), torch.Tensor):
            return None

        ref = ModuleBitwidthRef(
            extractor=self,
            module=element,
            name=self.module_names.get(
                element,
                f"<unnamed:{element.__class__.__name__}>",
            ),
            device=self._device_for(element),
        )
        ref.get()
        return ref

    @torch.no_grad()
    def resolve(self, name: str, module: nn.Module) -> tuple[int, int]:
        if self.bitwidth_fn is None:
            return DEFAULT_BITWIDTH, DEFAULT_BITWIDTH

        try:
            result = self.bitwidth_fn(name, module)
        except Exception as error:
            raise RuntimeError(f"bitwidth_fn failed for layer {name!r}.") from error

        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise TypeError(
                f"bitwidth_fn must return (activation_bits, weight_bits) for layer "
                f"{name!r}, got {result!r}."
            )

        activation_bits, weight_bits = result
        return (
            _validated_bitwidth(activation_bits, name, "activation"),
            _validated_bitwidth(weight_bits, name, "weight"),
        )

    def _device_for(self, module: nn.Module) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)

        weight = self._weight_tensor_for(module)
        if isinstance(weight, torch.Tensor):
            return weight.device

        return torch.device("cpu")

    @staticmethod
    def _weight_tensor_for(module: nn.Module):
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight

        if isinstance(module, nn.MultiheadAttention):
            in_proj_weight = getattr(module, "in_proj_weight", None)
            if isinstance(in_proj_weight, torch.Tensor):
                return in_proj_weight

            q_proj_weight = getattr(module, "q_proj_weight", None)
            if isinstance(q_proj_weight, torch.Tensor):
                return q_proj_weight

        return None

    @staticmethod
    def weighted_modules(model: nn.Module):
        for name, module in model.named_modules():
            weight = ModuleBitwidthExtractor._weight_tensor_for(module)
            if not isinstance(weight, torch.Tensor):
                continue

            label = name if name else module.__class__.__name__
            yield label, module


def _validated_bitwidth(value, name: str, side: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"Layer {name!r}: {side} bitwidth must be a scalar, got tensor "
                f"with shape {tuple(value.shape)}."
            )
        value = value.item()

    if isinstance(value, bool):
        raise ValueError(
            f"Layer {name!r}: {side} bitwidth must be a whole number >= 1, "
            f"got {value!r}."
        )
    if not isinstance(value, Real):
        raise TypeError(
            f"Layer {name!r}: {side} bitwidth must be a number or scalar tensor, "
            f"got {type(value).__name__}."
        )

    number = float(value)
    if not number.is_integer() or number < 1:
        raise ValueError(
            f"Layer {name!r}: {side} bitwidth must be a whole number >= 1, "
            f"got {value!r}."
        )
    return int(number)
