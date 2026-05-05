import torch
from abc import ABC, abstractmethod
import torch.nn as nn

from torch_structracker.torch_pruning.dependency.group import Group

class WeightOperation(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) 

    @abstractmethod
    def forward(self,):
        pass

class WeightExtractor(ABC):
    """
    Reference to a parameter on an existing model/module.

    This avoids assigning borrowed Parameters directly as attributes
    on the regularizer module, which would register them as parameters
    of the regularizer.
    """
    def __init__(self, module: nn.Module):
        self.module = module

    def get(self):
        pass


class WeightReducer(nn.Module):
    def __init__(self, weight_extractor: WeightExtractor, operation: WeightOperation):
        super().__init__()
        self.operation = operation
        self.weight_extractor = weight_extractor

    def forward(self):
        return self.operation(self.weight_extractor.get()) 
    




class LinearOut(nn.Module):
    def forward(self, weight: torch.Tensor):
        return weight.sqrt().sum(dim=1)

class LinearIn(nn.Module):
    def forward(self, weight: torch.Tensor):
        return weight.sqrt().sum(dim=0)



class UnitParameterSummer(nn.Module):
    def __init__(self, groups, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.groups = groups

