import torch.nn as nn

from torch_structracker.operations.base import WeightOperationType
from torch_structracker.operations.generic import create_generic_operation
from torch_structracker.torch_pruning.pruner.function import (
    prune_linear_in_channels,
    prune_linear_out_channels,
)


def operation_for_linear(
    module: nn.Linear,
    handler,
    operation_type: WeightOperationType | str,
):
    if handler == prune_linear_out_channels:
        return create_generic_operation(operation_type, dim=1)

    if handler == prune_linear_in_channels:
        return create_generic_operation(operation_type, dim=0)

    raise ValueError(
        f"Unsupported Linear pruning handler: {getattr(handler, '__name__', handler)}"
    )
