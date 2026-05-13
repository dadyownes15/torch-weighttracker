import torch.nn as nn

from torch_structracker.operations.base import WeightOperationType
from torch_structracker.operations.conv import operation_for_conv2d
from torch_structracker.operations.generic import create_generic_operation
from torch_structracker.operations.linear import operation_for_linear
from torch_structracker.operations.mha import raise_mha_not_implemented
from torch_structracker.operations.norm import (
    operation_for_batchnorm,
    operation_for_layernorm,
)

# TODO:
# We can replace the imports from torch pruning and the logic that uses the handler, by creating a maping that uses the canonical group.

def operation_for_member(member, operation_type: WeightOperationType | str):
    module = member.dep.target.module
    handler = member.dep.handler

    if isinstance(module, nn.MultiheadAttention):
        raise_mha_not_implemented(module)

    if isinstance(module, nn.Linear):
        return operation_for_linear(module, handler, operation_type)

    if isinstance(module, nn.Conv2d):
        return operation_for_conv2d(module, handler, operation_type)

    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return operation_for_batchnorm(module, handler, operation_type)

    if isinstance(module, nn.LayerNorm):
        return operation_for_layernorm(module, handler, operation_type)

    raise ValueError(
        f"Reducer operation is not implemented for {module.__class__.__name__}."
    )


def operation_for_module(module: nn.Module, operation_type: WeightOperationType | str):
    if isinstance(module, nn.MultiheadAttention):
        raise_mha_not_implemented(module)

    return create_generic_operation(operation_type, dim=None)
