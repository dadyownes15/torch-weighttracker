import torch.nn as nn

from torch_structracker.extractor import (
    FusedQKVExtractor,
    ParameterExtractor,
    ParameterTupleExtractor,
    SeparateQKVExtractor,
    TensorExtractor,
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
            | TensorExtractor
        ),
        operation: WeightOperation,
    ):
        super().__init__()
        self.operation = operation
        self.parameter_extractor = parameter_extractor

    def forward(self):
        return self.operation(self.parameter_extractor.get()).reshape(-1)

    def identity_key(self):
        return (
            self.parameter_extractor.identity_key(),
            self.operation.identity_key(),
        )

    def __hash__(self):
        return hash(self.identity_key())

    def __eq__(self, other):
        if not isinstance(other, WeightReducer):
            return NotImplemented
        return self.identity_key() == other.identity_key()
