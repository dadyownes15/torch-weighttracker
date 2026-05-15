from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Hashable

import torch
import torch.nn as nn

from torch_weighttracker.extractors.extractor import (
    ElementTensorExtractor,
    TensorSpec,
)


@dataclass(frozen=True)
class ModuleBitrateRef:
    """Runtime source ref for a resolved bitrate tensor."""

    value: torch.Tensor
    key: Hashable

    def __post_init__(self) -> None:
        if not isinstance(self.value, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(self.value).__name__}")

    def get(self) -> torch.Tensor:
        return self.value

    def source_spec(self) -> TensorSpec:
        return TensorSpec(
            shape=self.value.shape,
            dtype=self.value.dtype,
            device=self.value.device,
        )

    def identity_key(self) -> Hashable:
        return self.key


class ModuleBitrateExtractor(ElementTensorExtractor[nn.Module]):
    """Bind weighted modules to bitrate source refs."""

    activation_attribute_names = ("activation_bitrate", "act_bitrate")
    weight_attribute_names = ("weight_bitrate",)
    shared_attribute_names = ("bitrate",)

    def __init__(
        self,
        *,
        activation_default: float = 32.0,
        weight_default: float = 32.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.activation_default = float(activation_default)
        self.weight_default = float(weight_default)
        self.device = device
        self.dtype = torch.float32 if dtype is None else dtype

    @torch.no_grad()
    def bind(self, element: nn.Module) -> ModuleBitrateRef | None:
        if not isinstance(element, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(element).__name__}")

        weight = self._weight_tensor_for(element)
        if not isinstance(weight, torch.Tensor):
            return None

        device = self._device_for(element)
        value = torch.stack(
            (
                self._activation_bitrate(element, device),
                self._weight_bitrate(element, device),
            )
        )

        return ModuleBitrateRef(
            value=value,
            key=(
                "module_bitrate",
                id(element),
                self.activation_default,
                self.weight_default,
                device,
                self.dtype,
            ),
        )

    def _activation_bitrate(
        self,
        module: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        value = self._first_module_attribute(module, self.activation_attribute_names)
        if value is not None:
            return self._scalar_tensor(value, "activation bitrate", device)

        value = self._first_module_attribute(module, self.shared_attribute_names)
        if value is not None:
            return self._scalar_tensor(value, "activation bitrate", device)

        return self._scalar_tensor(self.activation_default, "activation bitrate", device)

    def _weight_bitrate(
        self,
        module: nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        value = self._codeq_weight_bitrate(module)
        if value is not None:
            return self._scalar_tensor(value, "weight bitrate", device)

        value = self._first_module_attribute(module, self.weight_attribute_names)
        if value is not None:
            return self._scalar_tensor(value, "weight bitrate", device)

        value = self._first_module_attribute(module, self.shared_attribute_names)
        if value is not None:
            return self._scalar_tensor(value, "weight bitrate", device)

        return self._scalar_tensor(self.weight_default, "weight bitrate", device)

    def _codeq_weight_bitrate(self, module: nn.Module):
        providers = self._codeq_weight_bitrate_providers(module)
        if len(providers) == 0:
            return None

        if len(providers) > 1:
            raise ValueError(
                f"{module.__class__.__name__} has multiple weight bitwidth "
                "providers in its weight parametrizations."
            )

        return providers[0]()

    def _codeq_weight_bitrate_providers(self, module: nn.Module):
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is None:
            return ()

        weight_parametrizations = getattr(parametrizations, "weight", None)
        if weight_parametrizations is None:
            return ()

        providers = []
        for parametrization in weight_parametrizations:
            quantizer = getattr(parametrization, "quantizer", None)
            if quantizer is not None and callable(
                getattr(quantizer, "get_bitwidth", None)
            ):
                providers.append(quantizer.get_bitwidth)
                continue

            if callable(getattr(parametrization, "get_bitwidth", None)):
                providers.append(parametrization.get_bitwidth)

        return tuple(providers)

    @staticmethod
    def _first_module_attribute(module: nn.Module, names: tuple[str, ...]):
        for name in names:
            if hasattr(module, name):
                return getattr(module, name)
        return None

    def _scalar_tensor(
        self,
        value,
        label: str,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"{label} must be scalar, got tensor with shape "
                    f"{tuple(value.shape)}."
                )

            return value.detach().to(device=device, dtype=self.dtype).reshape(())

        if isinstance(value, Real):
            return torch.tensor(float(value), device=device, dtype=self.dtype)

        raise TypeError(f"{label} must be a scalar number or tensor.")

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
            weight = ModuleBitrateExtractor._weight_tensor_for(module)
            if not isinstance(weight, torch.Tensor):
                continue

            label = name if name else module.__class__.__name__
            yield label, module
