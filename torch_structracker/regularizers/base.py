from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import Enum

import torch
from torch import nn

from torch_structracker.calculations import CalcType


class RegularizerType(Enum):
    GROUP_LASSO = "group_lasso"


class BaseRegularizer(nn.Module, ABC):
    regularizer_type: RegularizerType
    required_calculations: tuple[CalcType, ...] = ()

    def __init__(
        self,
        calculations: Mapping[CalcType, nn.Module],
    ) -> None:
        super().__init__()

        missing = [
            calc_type
            for calc_type in self.required_calculations
            if calc_type not in calculations
        ]

        if missing:
            raise ValueError(
                f"{self.__class__.__name__} is missing required calculations: {missing}"
            )

        self.calculations = nn.ModuleDict(
            {calc_type.name: module for calc_type, module in calculations.items()}
        )

    @abstractmethod
    def forward(self) -> torch.Tensor:
        raise NotImplementedError

    def calc(self, calc_type: CalcType) -> nn.Module:
        return self.calculations[calc_type.name]
    
    def compute(self, calc_type: CalcType, *args, **kwargs) -> torch.Tensor:
        return self.calc(calc_type)(*args, **kwargs)