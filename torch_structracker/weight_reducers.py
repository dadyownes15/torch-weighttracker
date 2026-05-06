import torch
from abc import ABC, abstractmethod
import torch.nn as nn

from torch_structracker.torch_pruning.dependency.group import Group
from torch_structracker.weight_operations import WeightOperation

class ParameterExtractor:
    """
    Non-owning reference to a parameter or tensor attribute on an existing module.

    Useful when another object needs access to a model parameter without
    registering that parameter as its own.
    """

    def __init__(self, module: nn.Module, name: str = "weight"):
        if not isinstance(module, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(module).__name__}")

        if not isinstance(name, str):
            raise TypeError(f"Expected parameter name as str, got {type(name).__name__}")

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

class WeightReducer(nn.Module):
    def __init__(self, parameter_extractor: ParameterExtractor, operation: WeightOperation):
        super().__init__()
        self.operation = operation
        self.parameter_extractor = parameter_extractor

    def forward(self):
        return self.operation(self.parameter_extractor.get()) 
    


