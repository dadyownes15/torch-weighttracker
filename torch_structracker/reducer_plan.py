from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_structracker.extractor import (
    FusedQKVExtractor,
    ParameterExtractor,
    SeparateQKVExtractor,
)
from torch_structracker.operations import (
    QKVSourceOperation,
    WeightOperationType,
    operation_for_member,
    operation_for_module,
)
from torch_structracker.reducers import WeightReducer
from torch_structracker.torch_pruning.pruner.function import (
    prune_linear_out_channels,
    prune_multihead_attention_out_channels,
)


@dataclass(frozen=True)
class ReducerMapping:
    reducer: WeightReducer
    destination_indices: tuple[int, ...]


@dataclass(frozen=True)
class ReducerPlan:
    output_length: int
    mappings: tuple[ReducerMapping, ...]
    output_labels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class _AttentionGroupConfig:
    embed_dim: int
    num_heads: int
    output_length: int
    mode: str


def add_mapping(
    mapping: ReducerMapping,
    mappings_by_reducer: dict[WeightReducer, ReducerMapping],
) -> None:
    existing = mappings_by_reducer.get(mapping.reducer)

    if existing is None:
        mappings_by_reducer[mapping.reducer] = mapping
        return
    # TODO: If so see this, please ensure to add a test so that it does not ducplicate destination indexes, such that it is a proper set. 
    mappings_by_reducer[mapping.reducer] = ReducerMapping(
        reducer=mapping.reducer,
        destination_indices=(
            *existing.destination_indices,
            *mapping.destination_indices,
        ),
    )


def compile_reducer_plan_from_groups(
    groups,
    operation_type: WeightOperationType | str,
    parameter_name: str = "weight",
    num_heads: dict[nn.Module, int] | None = None,
    prune_dim: bool | None = None,
    prune_num_heads: bool = False,
) -> ReducerPlan:
    if prune_dim and prune_num_heads:
        raise ValueError("prune_dim and prune_num_heads cannot both be enabled.")

    num_heads = {} if num_heads is None else dict(num_heads)
    mappings_by_reducer: dict[WeightReducer, ReducerMapping] = {}
    output_length = 0

    for group in groups:
        group_offset = output_length
        attention_config = _attention_group_config(
            group,
            num_heads=num_heads,
            prune_dim=prune_dim,
            prune_num_heads=prune_num_heads,
        )

        for member in group.items:
            for mapping in create_reducer_mappings_for_member(
                member=member,
                group_offset=group_offset,
                operation_type=operation_type,
                parameter_name=parameter_name,
                attention_config=attention_config,
            ):
                add_mapping(mapping, mappings_by_reducer)

        if attention_config is None:
            output_length += len(group[0].root_idxs)
        else:
            output_length += attention_config.output_length

    return ReducerPlan(
        output_length=output_length,
        mappings=tuple(mappings_by_reducer.values()),
    )

# This function creates a reducer plan, which essentially entails the ability to specif a weight operation, which gets executed, on all modules

# These are the fundamental operations we use to get measurements from the actual modules. It automatically maps the appropiate operation, to the the correct type of module. For example MHA, likely needs some additional properties, to be able to function the same weight operation as a simple linear layer

# TODO: We should definetly work on a more unified approach, being aple to includ and exclude modules from this. This also is important for propgating the inputs fo the struct tracker down to the actual calcs. Furthermore, we cannot specify custom extractors
def compile_reducer_plan_from_modules(
    model: nn.Module,
    operation_type: WeightOperationType | str,
    parameter_name: str = "weight",
) -> ReducerPlan:
    mappings: list[ReducerMapping] = []
    output_labels: list[str] = []
    output_offset = 0

    for name, module in model.named_modules():
        if not _has_live_parameter(module, parameter_name):
            continue

        reducer = WeightReducer(
            parameter_extractor=ParameterExtractor(module, parameter_name),
            operation=operation_for_module(module, operation_type),
        )

        with torch.no_grad():
            value_count = reducer().reshape(-1).numel()

        destination_indices = tuple(range(output_offset, output_offset + value_count))
        mappings.append(
            ReducerMapping(
                reducer=reducer,
                destination_indices=destination_indices,
            )
        )

        label = name if name else module.__class__.__name__
        if value_count == 1:
            output_labels.append(label)
        else:
            output_labels.extend(f"{label}[{i}]" for i in range(value_count))

        output_offset += value_count

    return ReducerPlan(
        output_length=output_offset,
        mappings=tuple(mappings),
        output_labels=tuple(output_labels),
    )


def create_reducer_mappings_for_member(
    member,
    group_offset: int,
    operation_type: WeightOperationType | str,
    parameter_name: str = "weight",
    attention_config: _AttentionGroupConfig | None = None,
) -> tuple[ReducerMapping, ...]:
    module = member.dep.target.module

    if isinstance(module, nn.MultiheadAttention):
        return _create_mha_reducer_mapping(
            module=module,
            group_offset=group_offset,
            operation_type=operation_type,
            attention_config=attention_config,
        )

    if not isinstance(module, nn.Module):
        return ()

    if _is_attention_qkv_linear_member(member, module, attention_config):
        return _create_fused_qkv_linear_mapping(
            module=module,
            group_offset=group_offset,
            operation_type=operation_type,
            attention_config=attention_config,
        )

    if not _has_live_parameter(module, parameter_name):
        return ()

    if member.root_idxs is None:
        raise ValueError("Dependency group member is missing root indices.")

    reducer = WeightReducer(
        parameter_extractor=ParameterExtractor(module, parameter_name),
        operation=operation_for_member(member, operation_type),
    )

    if attention_config is None:
        destination_indices = tuple(group_offset + int(idx) for idx in member.root_idxs)
    else:
        destination_indices = _semantic_destinations_for_indices(
            member.root_idxs,
            group_offset=group_offset,
            attention_config=attention_config,
        )

    return (
        ReducerMapping(
            reducer=reducer,
            destination_indices=destination_indices,
        ),
    )


@torch.no_grad()
def validate_reducer_plan(plan: ReducerPlan) -> None:
    if plan.output_length < 0:
        raise ValueError("ReducerPlan.output_length must be non-negative.")

    if plan.output_labels is not None and len(plan.output_labels) != plan.output_length:
        raise ValueError(
            "ReducerPlan.output_labels must match ReducerPlan.output_length."
        )

    for index, mapping in enumerate(plan.mappings):
        values = mapping.reducer().reshape(-1)
        destinations = mapping.destination_indices

        if values.numel() != len(destinations):
            raise ValueError(
                f"Reducer mapping {index} produced {values.numel()} values, "
                f"but has {len(destinations)} destination indices."
            )

        if len(destinations) == 0:
            continue

        min_destination = min(destinations)
        max_destination = max(destinations)

        if min_destination < 0 or max_destination >= plan.output_length:
            raise ValueError(
                f"Reducer mapping {index} has destination indices outside "
                f"[0, {plan.output_length})."
            )


def _has_live_parameter(module: nn.Module, parameter_name: str) -> bool:
    return hasattr(module, parameter_name) and getattr(module, parameter_name) is not None

# TODO:
# Move this out into structuretracker, not here. Make sure the dep graph is proper setup, before passing groups. We shoud make all the reducers assume that groups are in the correct format
def _attention_group_config(
    group,
    num_heads: dict[nn.Module, int],
    prune_dim: bool | None,
    prune_num_heads: bool,
) -> _AttentionGroupConfig | None:
    if not prune_dim and not prune_num_heads:
        return None

    for member in group.items:
        module = member.dep.target.module
        if module not in num_heads:
            continue

        if not _is_attention_out_member(member, module):
            continue

        embed_dim = _attention_embed_dim(module)
        heads = int(num_heads[module])
        if heads <= 0:
            raise ValueError("num_heads values must be positive.")
        if embed_dim % heads != 0:
            raise ValueError("Attention embed_dim must be divisible by num_heads.")

        if prune_dim:
            return _AttentionGroupConfig(
                embed_dim=embed_dim,
                num_heads=heads,
                output_length=embed_dim // heads,
                mode="head_dim",
            )

        return _AttentionGroupConfig(
            embed_dim=embed_dim,
            num_heads=heads,
            output_length=heads,
            mode="head",
        )

    return None


def _is_attention_out_member(member, module: nn.Module) -> bool:
    handler = member.dep.handler

    if isinstance(module, nn.MultiheadAttention):
        return handler == prune_multihead_attention_out_channels

    return (
        isinstance(module, nn.Linear)
        and handler == prune_linear_out_channels
        and module.out_features % 3 == 0
    )


def _attention_embed_dim(module: nn.Module) -> int:
    if isinstance(module, nn.MultiheadAttention):
        return int(module.embed_dim)

    if isinstance(module, nn.Linear) and module.out_features % 3 == 0:
        return int(module.out_features // 3)

    raise ValueError(
        f"{module.__class__.__name__} cannot be interpreted as a QKV module."
    )


def _is_attention_qkv_linear_member(
    member,
    module: nn.Module,
    attention_config: _AttentionGroupConfig | None,
) -> bool:
    return (
        attention_config is not None
        and isinstance(module, nn.Linear)
        and member.dep.handler == prune_linear_out_channels
        and module.out_features == 3 * attention_config.embed_dim
    )


def _create_mha_reducer_mapping(
    module: nn.MultiheadAttention,
    group_offset: int,
    operation_type: WeightOperationType | str,
    attention_config: _AttentionGroupConfig | None,
) -> tuple[ReducerMapping, ...]:
    embed_dim = int(module.embed_dim)

    if module.in_proj_weight is not None:
        qkv_extractor = FusedQKVExtractor(module, "in_proj_weight")
    else:
        qkv_extractor = SeparateQKVExtractor(module)

    qkv_reducer = WeightReducer(
        parameter_extractor=qkv_extractor,
        operation=QKVSourceOperation(operation_type),
    )
    destination_indices = _qkv_destinations(
        embed_dim=embed_dim,
        group_offset=group_offset,
        attention_config=attention_config,
    )

    return (
        ReducerMapping(
            reducer=qkv_reducer,
            destination_indices=destination_indices,
        ),
    )


def _create_fused_qkv_linear_mapping(
    module: nn.Linear,
    group_offset: int,
    operation_type: WeightOperationType | str,
    attention_config: _AttentionGroupConfig,
) -> tuple[ReducerMapping, ...]:
    qkv_reducer = WeightReducer(
        parameter_extractor=FusedQKVExtractor(module, "weight"),
        operation=QKVSourceOperation(operation_type),
    )
    destination_indices = _qkv_destinations(
        embed_dim=attention_config.embed_dim,
        group_offset=group_offset,
        attention_config=attention_config,
    )

    return (
        ReducerMapping(
            reducer=qkv_reducer,
            destination_indices=destination_indices,
        ),
    )


def _qkv_destinations(
    embed_dim: int,
    group_offset: int,
    attention_config: _AttentionGroupConfig | None,
) -> tuple[int, ...]:
    source_indices = range(3 * embed_dim)

    if attention_config is None:
        return tuple(group_offset + (index % embed_dim) for index in source_indices)

    return _semantic_destinations_for_indices(
        source_indices,
        group_offset=group_offset,
        attention_config=attention_config,
    )


def _semantic_destinations_for_indices(
    indices,
    group_offset: int,
    attention_config: _AttentionGroupConfig,
) -> tuple[int, ...]:
    head_dim = attention_config.embed_dim // attention_config.num_heads
    destinations: list[int] = []

    for index in indices:
        embed_index = int(index) % attention_config.embed_dim

        if attention_config.mode == "head_dim":
            destination = embed_index % head_dim
        elif attention_config.mode == "head":
            destination = embed_index // head_dim
        else:
            raise ValueError(f"Unknown attention mode: {attention_config.mode}")

        destinations.append(group_offset + destination)

    return tuple(destinations)
