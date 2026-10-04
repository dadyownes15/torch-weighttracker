import torch.nn as nn

from torch_weighttracker.canonical_units import UnitAxis
from torch_weighttracker.operations.base import WeightOperationType
from torch_weighttracker.operations.conv import operation_for_conv2d
from torch_weighttracker.operations.generic import create_generic_operation
from torch_weighttracker.operations.linear import operation_for_linear
from torch_weighttracker.operations.mha import raise_mha_not_implemented
from torch_weighttracker.operations.norm import (
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


def operation_for_module_axis(
    module: nn.Module,
    unit_axis: UnitAxis,
    operation_type: WeightOperationType | str,
):
    """Resolve plain Conv/Linear reductions from canonical axis semantics.

    Custom pruning handlers are intentionally distinct from Torch-Pruning's
    built-in function objects. Canonicalization has already classified those
    handlers as input or output axes, so downstream reductions must consume
    that classification instead of comparing handler identity again.
    """
    if isinstance(module, nn.Conv2d):
        if module.groups != 1:
            raise ValueError(
                "Grouped and depthwise Conv2d reducer mappings are not "
                "implemented yet."
            )
        if unit_axis == UnitAxis.OUT_CHANNEL:
            return create_generic_operation(operation_type, dim=(1, 2, 3))
        if unit_axis == UnitAxis.IN_CHANNEL:
            return create_generic_operation(operation_type, dim=(0, 2, 3))

    if isinstance(module, nn.Linear):
        if unit_axis == UnitAxis.OUT_CHANNEL:
            return create_generic_operation(operation_type, dim=1)
        if unit_axis == UnitAxis.IN_CHANNEL:
            return create_generic_operation(operation_type, dim=0)

    return None


def operation_for_module(module: nn.Module, operation_type: WeightOperationType | str):
    if isinstance(module, nn.MultiheadAttention):
        raise_mha_not_implemented(module)

    return create_generic_operation(operation_type, dim=None)
