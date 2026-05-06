
import torch

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn

from torch_structracker.torch_pruning.pruner.function import prune_linear_out_channels, prune_linear_in_channels 

class WeightOperationType(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    L1 = "l1"
    L2 = "l2"


class WeightOperation(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def create(
        operation: WeightOperationType | str,
        dim: Optional[int] = None,
        keepdim: bool = False,
    ) -> "WeightOperation":
        operation = WeightOperationType(operation)

        if operation == WeightOperationType.SUM:
            return SumWeight(dim=dim, keepdim=keepdim)

        if operation == WeightOperationType.MEAN:
            return MeanWeight(dim=dim, keepdim=keepdim)

        if operation == WeightOperationType.COUNT:
            return CountWeight(dim=dim, keepdim=keepdim)

        if operation == WeightOperationType.L1:
            return L1Weight(dim=dim, keepdim=keepdim)

        if operation == WeightOperationType.L2:
            return L2Weight(dim=dim, keepdim=keepdim)

        raise ValueError(f"Unknown weight operation: {operation}")


class SumWeight(WeightOperation):
    def __init__(self, dim: Optional[int] = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.sum(dim=self.dim, keepdim=self.keepdim)


class MeanWeight(WeightOperation):
    def __init__(self, dim: Optional[int] = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.mean(dim=self.dim, keepdim=self.keepdim)


class CountWeight(WeightOperation):
    def __init__(self, dim: Optional[int] = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(weight)
        return ones.sum(dim=self.dim, keepdim=self.keepdim)


class L1Weight(WeightOperation):
    def __init__(self, dim: Optional[int] = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight.abs().sum(dim=self.dim, keepdim=self.keepdim)


class L2Weight(WeightOperation):
    def __init__(self, dim: Optional[int] = None, keepdim: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((weight ** 2).sum(dim=self.dim, keepdim=self.keepdim))


def pruner_to_operation_map(handler, task) -> WeightOperation:
    if handler == prune_linear_in_channels:
        return WeightOperation.create(WeightOperationType.SUM,dim=0)
    elif handler == prune_linear_out_channels:
        return WeightOperation.create(WeightOperationType.SUM,dim=1)
    else:
        raise ValueError("Un implemented handler in weight op mathcing")

"""
prune_conv_out_channels = PrunerBox[ops.OPTYPE.CONV].prune_out_channels
prune_conv_in_channels = PrunerBox[ops.OPTYPE.CONV].prune_in_channels

prune_depthwise_conv_out_channels = PrunerBox[ops.OPTYPE.DEPTHWISE_CONV].prune_out_channels
prune_depthwise_conv_in_channels = PrunerBox[ops.OPTYPE.DEPTHWISE_CONV].prune_in_channels

prune_batchnorm_out_channels = PrunerBox[ops.OPTYPE.BN].prune_out_channels
prune_batchnorm_in_channels = PrunerBox[ops.OPTYPE.BN].prune_in_channels

prune_linear_out_channels = PrunerBox[ops.OPTYPE.LINEAR].prune_out_channels
prune_linear_in_channels = PrunerBox[ops.OPTYPE.LINEAR].prune_in_channels

prune_prelu_out_channels = PrunerBox[ops.OPTYPE.PRELU].prune_out_channels
prune_prelu_in_channels = PrunerBox[ops.OPTYPE.PRELU].prune_in_channels

prune_layernorm_out_channels = PrunerBox[ops.OPTYPE.LN].prune_out_channels
prune_layernorm_in_channels = PrunerBox[ops.OPTYPE.LN].prune_in_channels

prune_embedding_out_channels = PrunerBox[ops.OPTYPE.EMBED].prune_out_channels
prune_embedding_in_channels = PrunerBox[ops.OPTYPE.EMBED].prune_in_channels

prune_parameter_out_channels = PrunerBox[ops.OPTYPE.PARAMETER].prune_out_channels
prune_parameter_in_channels = PrunerBox[ops.OPTYPE.PARAMETER].prune_in_channels

prune_multihead_attention_out_channels = PrunerBox[ops.OPTYPE.MHA].prune_out_channels
prune_multihead_attention_in_channels = PrunerBox[ops.OPTYPE.MHA].prune_in_channels

prune_lstm_out_channels = PrunerBox[ops.OPTYPE.LSTM].prune_out_channels
prune_lstm_in_channels = PrunerBox[ops.OPTYPE.LSTM].prune_in_channels

prune_groupnorm_out_channels = PrunerBox[ops.OPTYPE.GN].prune_out_channels
prune_groupnorm_in_channels = PrunerBox[ops.OPTYPE.GN].prune_in_channels

prune_instancenorm_out_channels = PrunerBox[ops.OPTYPE.IN].prune_out_channels
prune_instancenorm_in_channels = PrunerBox[ops.OPTYPE.IN].prune_in_channels
"""
