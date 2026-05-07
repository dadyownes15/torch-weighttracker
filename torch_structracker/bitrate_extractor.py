from __future__ import annotations

from numbers import Real

import torch
import torch.nn as nn


class ModuleBitrateExtractor:
    """Extract per-module activation and weight bitrates from weighted modules."""

    activation_attribute_names = ("activation_bitrate", "act_bitrate")
    weight_attribute_names = ("weight_bitrate",)
    shared_attribute_names = ("bitrate",)

    def __init__(
        self,
        model: nn.Module,
        *,
        activation_default: float = 32.0,
        weight_default: float = 32.0,
        device=None,
        dtype=None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(model).__name__}")

        self.model = model
        self.activation_default = float(activation_default)
        self.weight_default = float(weight_default)
        self.device = device
        self.dtype = torch.float32 if dtype is None else dtype
        self._entries = tuple(self._weighted_modules(model))
        self.module_names = tuple(name for name, _ in self._entries)

        if self.device is None:
            self.device = self._infer_device()

    @torch.no_grad()
    def extract(self) -> torch.Tensor:
        rows = [
            torch.stack(
                (
                    self._activation_bitrate(module),
                    self._weight_bitrate(module),
                )
            )
            for _, module in self._entries
        ]

        if len(rows) == 0:
            return torch.empty((0, 2), device=self.device, dtype=self.dtype)

        return torch.stack(rows)

    def _activation_bitrate(self, module: nn.Module) -> torch.Tensor:
        value = self._first_module_attribute(module, self.activation_attribute_names)
        if value is not None:
            return self._scalar_tensor(value, "activation bitrate")

        value = self._first_module_attribute(module, self.shared_attribute_names)
        if value is not None:
            return self._scalar_tensor(value, "activation bitrate")

        return self._scalar_tensor(self.activation_default, "activation bitrate")

    def _weight_bitrate(self, module: nn.Module) -> torch.Tensor:
        value = self._codeq_weight_bitrate(module)
        if value is not None:
            return self._scalar_tensor(value, "weight bitrate")

        value = self._first_module_attribute(module, self.weight_attribute_names)
        if value is not None:
            return self._scalar_tensor(value, "weight bitrate")

        value = self._first_module_attribute(module, self.shared_attribute_names)
        if value is not None:
            return self._scalar_tensor(value, "weight bitrate")

        return self._scalar_tensor(self.weight_default, "weight bitrate")

    def _codeq_weight_bitrate(self, module: nn.Module):
        providers = list(self._codeq_weight_bitrate_providers(module))
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
            return

        weight_parametrizations = getattr(parametrizations, "weight", None)
        if weight_parametrizations is None:
            return

        for parametrization in weight_parametrizations:
            quantizer = getattr(parametrization, "quantizer", None)
            if quantizer is not None and callable(
                getattr(quantizer, "get_bitwidth", None)
            ):
                yield quantizer.get_bitwidth
                continue

            if callable(getattr(parametrization, "get_bitwidth", None)):
                yield parametrization.get_bitwidth

    def _first_module_attribute(self, module: nn.Module, names: tuple[str, ...]):
        for name in names:
            if hasattr(module, name):
                return getattr(module, name)
        return None

    def _scalar_tensor(self, value, label: str) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"{label} must be scalar, got tensor with shape "
                    f"{tuple(value.shape)}."
                )

            return value.detach().to(device=self.device, dtype=self.dtype).reshape(())

        if isinstance(value, Real):
            return torch.tensor(float(value), device=self.device, dtype=self.dtype)

        raise TypeError(f"{label} must be a scalar number or tensor.")

    def _infer_device(self):
        for _, module in self._entries:
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor):
                return weight.device

        return torch.device("cpu")

    @staticmethod
    def _weighted_modules(model: nn.Module):
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor):
                continue

            label = name if name else module.__class__.__name__
            yield label, module
