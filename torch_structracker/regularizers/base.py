from abc import ABC, abstractmethod
from enum import Enum

import torch
import torch.nn as nn

from torch_structracker.calculations import CalculationType


class RegularizerType(str, Enum):
    GROUP_LASSO = "group_lasso"
    GROUP_LASSO_WITH_BITRATE = "group_lasso_with_bitrate"


class BaseRegularizer(nn.Module, ABC):
    regularizer_type: RegularizerType
    required_calculations: tuple[CalculationType, ...] = ()

    def __init__(self, calculations=None) -> None:
        super().__init__()
        self.calculations = {} if calculations is None else calculations

    @abstractmethod
    def forward(self) -> torch.Tensor:
        raise NotImplementedError


def regularizer_class_for_type(regularizer_type: RegularizerType):
    from torch_structracker.regularizers.group_lasso import GroupLasso
    from torch_structracker.regularizers.group_lasso_with_bitrate import (
        GroupLassoWithBitrate,
    )

    regularizer_type = RegularizerType(regularizer_type)
    regularizer_classes = {
        RegularizerType.GROUP_LASSO: GroupLasso,
        RegularizerType.GROUP_LASSO_WITH_BITRATE: GroupLassoWithBitrate,
    }
    return regularizer_classes[regularizer_type]
