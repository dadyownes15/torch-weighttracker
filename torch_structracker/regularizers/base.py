from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from enum import Enum

import torch
from torch import nn

from torch_structracker.calculations import CalcType, CalculationContext
from torch_structracker.consumer_ignore import IgnoreItem


class RegularizerType(Enum):
    GROUP_LASSO = "group_lasso"
    GROUP_LASSO_WITH_BITRATE = "group_lasso_with_bitrate"


class BaseRegularizer(nn.Module, ABC):
    regularizer_type: RegularizerType
    required_calculations: tuple[CalcType, ...] = ()

    @classmethod
    def calculation_context(
        cls,
        owner,
        *,
        ignore: Iterable[IgnoreItem] = (),
        **kwargs,
    ) -> CalculationContext | None:
        return None

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
        calc_type = CalcType(calc_type)
        return self.calculations[calc_type.name]

    def compute(self, calc_type: CalcType, *args, **kwargs) -> torch.Tensor:
        return self.calc(calc_type)(*args, **kwargs)


def regularizer_class_for_type(regularizer_type: RegularizerType | str):
    regularizer_type = RegularizerType(regularizer_type)

    if regularizer_type == RegularizerType.GROUP_LASSO:
        from torch_structracker.regularizers.group_lasso import GroupLasso

        return GroupLasso

    if regularizer_type == RegularizerType.GROUP_LASSO_WITH_BITRATE:
        from torch_structracker.regularizers.group_lasso_with_bitrate import (
            GroupLassoWithBitrate,
        )

        return GroupLassoWithBitrate

    raise ValueError(f"Unknown regularizer type: {regularizer_type.value}")
