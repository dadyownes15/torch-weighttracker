import torch.nn as nn

from torch_structracker.extractor import (
    FusedQKVExtractor,
    ParameterExtractor,
    ParameterTupleExtractor,
    SeparateQKVExtractor,
)
from torch_structracker.operations import WeightOperation


class WeightReducer(nn.Module):
    def __init__(
        self,
        parameter_extractor: (
            ParameterExtractor
            | ParameterTupleExtractor
            | FusedQKVExtractor
            | SeparateQKVExtractor
        ),
        operation: WeightOperation,
    ):
        super().__init__()
        self.operation = operation
        self.parameter_extractor = parameter_extractor

    def forward(self):
        return self.operation(self.parameter_extractor.get())

    def __key(self):
        return (
            self.parameter_extractor.identity_key(),
            self.operation.identity_key(),
        )

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if not isinstance(other, WeightReducer):
            return NotImplemented
        return self.__key() == other.__key()
