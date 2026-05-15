import torch.nn as nn

from torch_weighttracker.operations.base import WeightOperationType
from torch_weighttracker.operations.generic import create_generic_operation
from torch_weighttracker.torch_pruning.pruner.function import (
    prune_conv_in_channels,
    prune_conv_out_channels,
)


def operation_for_conv2d(
    module: nn.Conv2d,
    handler,
    operation_type: WeightOperationType | str,
):
    if module.groups != 1:
        raise ValueError(
            "Grouped and depthwise Conv2d reducer mappings are not implemented yet."
        )

    if handler == prune_conv_out_channels:
        return create_generic_operation(operation_type, dim=(1, 2, 3))

    if handler == prune_conv_in_channels:
        return create_generic_operation(operation_type, dim=(0, 2, 3))

    raise ValueError(
        f"Unsupported Conv2d pruning handler: {getattr(handler, '__name__', handler)}"
    )
