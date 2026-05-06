import torch.nn as nn

from torch_structracker.operations.base import WeightOperationType
from torch_structracker.operations.generic import create_generic_operation
from torch_structracker.torch_pruning.pruner.function import (
    prune_batchnorm_in_channels,
    prune_batchnorm_out_channels,
    prune_layernorm_in_channels,
    prune_layernorm_out_channels,
)


def operation_for_batchnorm(
    module: nn.modules.batchnorm._BatchNorm,
    handler,
    operation_type: WeightOperationType | str,
):
    if handler in {prune_batchnorm_out_channels, prune_batchnorm_in_channels}:
        return create_generic_operation(operation_type, dim=())

    handler_name = getattr(handler, "__name__", handler)
    raise ValueError(f"Unsupported BatchNorm pruning handler: {handler_name}")


def operation_for_layernorm(
    module: nn.LayerNorm,
    handler,
    operation_type: WeightOperationType | str,
):
    if len(module.normalized_shape) != 1:
        raise ValueError(
            "LayerNorm reducer mappings only support 1D normalized_shape."
        )

    if handler in {prune_layernorm_out_channels, prune_layernorm_in_channels}:
        return create_generic_operation(operation_type, dim=())

    handler_name = getattr(handler, "__name__", handler)
    raise ValueError(f"Unsupported LayerNorm pruning handler: {handler_name}")
