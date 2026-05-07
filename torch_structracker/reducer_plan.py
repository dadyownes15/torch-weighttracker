from dataclasses import dataclass
from typing import Callable, TypeAlias

import torch
import torch.nn as nn

from torch_structracker.extractor import (
    FusedQKVExtractor,
    ParameterExtractor,
    SeparateQKVExtractor,
    TensorExtractor,
)
from torch_structracker.operations import (
    QKVSemanticOperation,
    WeightOperationType,
    operation_for_member,
    operation_for_module,
)
from torch_structracker.reducers import WeightReducer
from torch_structracker.torch_pruning.pruner.function import (
    prune_linear_out_channels,
    prune_multihead_attention_out_channels,
)


ModulePredicate: TypeAlias = Callable[[str, nn.Module], bool]
ExtractorFactory: TypeAlias = Callable[[str, nn.Module, str], TensorExtractor | None]



@dataclass(frozen=True)
class SegmentTarget:
    start: int
    length: int


@dataclass(frozen=True)
class IndexedTarget:
    destination_indices: tuple[int, ...]
    source_indices: tuple[int, ...] | None = None


ReducerTarget: TypeAlias = SegmentTarget | IndexedTarget


@dataclass(frozen=True, init=False)
class ReducerMapping:
    reducer: WeightReducer
    target: ReducerTarget

    def __init__(
        self,
        reducer: WeightReducer,
        target: ReducerTarget | None = None,
        *,
        destination_indices: tuple[int, ...] | None = None,
        source_indices: tuple[int, ...] | None = None,
    ) -> None:
        if target is None:
            if destination_indices is None:
                raise TypeError("ReducerMapping requires a target.")
            target = indexed_or_segment_target(
                destination_indices,
                source_indices=source_indices,
            )
        elif destination_indices is not None or source_indices is not None:
            raise TypeError(
                "ReducerMapping accepts either target or destination/source indices."
            )

        object.__setattr__(self, "reducer", reducer)
        object.__setattr__(self, "target", target)

    @property
    def destination_indices(self) -> tuple[int, ...]:
        if isinstance(self.target, SegmentTarget):
            return tuple(range(self.target.start, self.target.start + self.target.length))

        return self.target.destination_indices

    @property
    def source_indices(self) -> tuple[int, ...] | None:
        if isinstance(self.target, IndexedTarget):
            return self.target.source_indices

        return None


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


def segment_target(start: int, length: int) -> SegmentTarget:
    start = int(start)
    length = int(length)

    if start < 0:
        raise ValueError("SegmentTarget.start must be non-negative.")

    if length < 0:
        raise ValueError("SegmentTarget.length must be non-negative.")

    return SegmentTarget(start=start, length=length)


def indexed_or_segment_target(
    destination_indices,
    *,
    source_indices=None,
) -> ReducerTarget:
    destinations = tuple(int(index) for index in destination_indices)
    sources = None
    if source_indices is not None:
        sources = tuple(int(index) for index in source_indices)

    if sources is None:
        segment = _as_contiguous_segment(destinations)
        if segment is not None:
            start, length = segment
            return segment_target(start=start, length=length)

    return IndexedTarget(destination_indices=destinations, source_indices=sources)


def _as_contiguous_segment(indices: tuple[int, ...]) -> tuple[int, int] | None:
    if len(indices) == 0:
        return None

    start = int(indices[0])

    for offset, index in enumerate(indices):
        if int(index) != start + offset:
            return None

    return start, len(indices)


#TODO:
# Major improvement of speed possible here: We can deduplicate reduction operations, if we can group the mappings, however how does one group segment indexing and direct indeixing?

# Finally, we can experiment in the future grouping weight operations into more efficient operations that uses groups of tensors with the same widths. My experiments have shown up to 2x on 1024 w tensors, 8x on 256 w*w tensors, assumming the same size. However, we need to test our experimetns with larger before commiting.

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
    mappings: list[ReducerMapping] = []
    output_length = 0

    for group in groups:
        attention_config = _attention_group_config(
            group,
            num_heads=num_heads,
            prune_dim=prune_dim,
            prune_num_heads=prune_num_heads,
        )
        group_offset = output_length
        group_length = _group_output_length(group, attention_config)
        root_to_position = _root_position_map(group)

        for member in group.items:
            mappings.extend(
                create_reducer_mappings_for_member(
                    member=member,
                    group_offset=group_offset,
                    operation_type=operation_type,
                    parameter_name=parameter_name,
                    attention_config=attention_config,
                    group_length=group_length,
                    root_to_position=root_to_position,
                )
            )

        output_length += group_length

    return ReducerPlan(
        output_length=output_length,
        mappings=tuple(mappings),
    )


def compile_reducer_plan_from_modules(
    model: nn.Module,
    operation_type: WeightOperationType | str,
    parameter_name: str = "weight",
    custom_param_extractor: TensorExtractor | None = None,
    *,
    include_module: ModulePredicate | None = None,
    extractor_factory: ExtractorFactory | None = None,
) -> ReducerPlan:
    if custom_param_extractor is not None and extractor_factory is not None:
        raise ValueError(
            "custom_param_extractor and extractor_factory cannot both be provided."
        )

    mappings: list[ReducerMapping] = []
    output_labels: list[str] = []
    output_offset = 0

    for name, module in model.named_modules():
        if include_module is not None and not include_module(name, module):
            continue

        if extractor_factory is not None:
            extractor = extractor_factory(name, module, parameter_name)
            if extractor is None:
                continue
        elif custom_param_extractor is not None:
            extractor = custom_param_extractor
        else:
            if not _has_live_parameter(module, parameter_name):
                continue
            extractor = ParameterExtractor(module, parameter_name)

        reducer = WeightReducer(
            parameter_extractor=extractor,
            operation=operation_for_module(module, operation_type),
        )

        with torch.no_grad():
            values = reducer()

        if values.ndim != 1:
            raise ValueError("Module reducer must return a flat 1-D tensor.")

        value_count = int(values.numel())
        mappings.append(
            ReducerMapping(
                reducer=reducer,
                target=segment_target(
                    start=output_offset,
                    length=value_count,
                ),
            )
        )

        label = name if name else module.__class__.__name__
        if value_count == 1:
            output_labels.append(label)
        else:
            output_labels.extend(f"{label}[{index}]" for index in range(value_count))

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
    *,
    group_length: int | None = None,
    root_to_position: dict[int, int] | None = None,
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

    root_idxs = _member_root_indices(member)
    reducer = WeightReducer(
        parameter_extractor=ParameterExtractor(module, parameter_name),
        operation=operation_for_member(member, operation_type),
    )

    if attention_config is None:
        destination_indices = _destinations_from_root_positions(
            root_idxs,
            group_offset=group_offset,
            root_to_position=root_to_position,
        )
    else:
        destination_indices = _semantic_destinations_for_indices(
            root_idxs,
            group_offset=group_offset,
            attention_config=attention_config,
        )

    if group_length is not None:
        _validate_member_destinations_within_group(
            destination_indices,
            group_offset=group_offset,
            group_length=group_length,
        )

    return (
        ReducerMapping(
            reducer=reducer,
            target=indexed_or_segment_target(destination_indices),
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
        values = mapping.reducer()

        if values.ndim != 1:
            raise ValueError(
                f"Reducer mapping {index} must return a flat 1-D tensor, "
                f"but returned shape {tuple(values.shape)}."
            )

        target = mapping.target

        if isinstance(target, SegmentTarget):
            if target.start < 0:
                raise ValueError(f"Reducer mapping {index} has negative start.")

            if target.length < 0:
                raise ValueError(f"Reducer mapping {index} has negative length.")

            if target.start + target.length > plan.output_length:
                raise ValueError(
                    f"Reducer mapping {index} segment exceeds output length."
                )

            if values.numel() != target.length:
                raise ValueError(
                    f"Reducer mapping {index} produced {values.numel()} values, "
                    f"but segment length is {target.length}."
                )

            continue

        if isinstance(target, IndexedTarget):
            destinations = target.destination_indices

            if target.source_indices is None:
                expected_values = len(destinations)
            else:
                expected_values = len(target.source_indices)

                if len(target.source_indices) != len(destinations):
                    raise ValueError(
                        f"Reducer mapping {index} source_indices and "
                        f"destination_indices must have the same length."
                    )

                if len(target.source_indices) > 0:
                    min_source = min(target.source_indices)
                    max_source = max(target.source_indices)

                    if min_source < 0 or max_source >= values.numel():
                        raise ValueError(
                            f"Reducer mapping {index} has source indices outside "
                            f"[0, {values.numel()})."
                        )

            if values.numel() != expected_values and target.source_indices is None:
                raise ValueError(
                    f"Reducer mapping {index} produced {values.numel()} values, "
                    f"but has {len(destinations)} destination indices."
                )

            if len(destinations) > 0:
                min_destination = min(destinations)
                max_destination = max(destinations)

                if min_destination < 0 or max_destination >= plan.output_length:
                    raise ValueError(
                        f"Reducer mapping {index} has destination indices outside "
                        f"[0, {plan.output_length})."
                    )

            continue

        raise TypeError(f"Unknown reducer target type: {type(target)!r}")


def _has_live_parameter(module: nn.Module, parameter_name: str) -> bool:
    return hasattr(module, parameter_name) and getattr(module, parameter_name) is not None


def _group_output_length(
    group,
    attention_config: _AttentionGroupConfig | None,
) -> int:
    if attention_config is not None:
        return int(attention_config.output_length)

    return len(_member_root_indices(group[0]))


def _root_position_map(group) -> dict[int, int]:
    return {
        int(root_idx): position
        for position, root_idx in enumerate(_member_root_indices(group[0]))
    }


def _member_root_indices(member) -> tuple[int, ...]:
    root_idxs = getattr(member, "root_idxs", None)
    if root_idxs is None:
        raise ValueError("Dependency group member is missing root indices.")

    return tuple(int(root_idx) for root_idx in root_idxs)


def _destinations_from_root_positions(
    root_idxs: tuple[int, ...],
    *,
    group_offset: int,
    root_to_position: dict[int, int] | None,
) -> tuple[int, ...]:
    if root_to_position is None:
        return tuple(int(group_offset) + int(root_idx) for root_idx in root_idxs)

    destinations: list[int] = []
    for root_idx in root_idxs:
        try:
            position = root_to_position[int(root_idx)]
        except KeyError as error:
            raise ValueError(
                f"Root index {root_idx} is not present in the dependency group root."
            ) from error
        destinations.append(int(group_offset) + position)

    return tuple(destinations)


def _validate_member_destinations_within_group(
    destination_indices: tuple[int, ...],
    *,
    group_offset: int,
    group_length: int,
) -> None:
    if len(destination_indices) == 0:
        return

    group_start = int(group_offset)
    group_stop = group_start + int(group_length)

    for destination in destination_indices:
        if destination < group_start or destination >= group_stop:
            raise ValueError(
                f"Destination index {destination} is outside group range "
                f"[{group_start}, {group_stop})."
            )


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
        if not _is_attention_out_member(member, module):
            continue

        heads = _attention_num_heads(module, num_heads)
        if heads is None:
            continue

        embed_dim = _attention_embed_dim(module)
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


def _attention_num_heads(
    module: nn.Module,
    num_heads: dict[nn.Module, int],
) -> int | None:
    if module in num_heads:
        return int(num_heads[module])

    heads = getattr(module, "num_heads", None)
    if heads is not None:
        return int(heads)

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

    if attention_config is None:
        mode = "channel"
        num_heads = None
        output_length = embed_dim
    else:
        mode = attention_config.mode
        num_heads = attention_config.num_heads
        output_length = attention_config.output_length

    qkv_reducer = WeightReducer(
        parameter_extractor=qkv_extractor,
        operation=QKVSemanticOperation(
            operation_type=operation_type,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mode=mode,
        ),
    )

    return (
        ReducerMapping(
            reducer=qkv_reducer,
            target=segment_target(
                start=group_offset,
                length=output_length,
            ),
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
        operation=QKVSemanticOperation(
            operation_type=operation_type,
            embed_dim=attention_config.embed_dim,
            num_heads=attention_config.num_heads,
            mode=attention_config.mode,
        ),
    )

    return (
        ReducerMapping(
            reducer=qkv_reducer,
            target=segment_target(
                start=group_offset,
                length=attention_config.output_length,
            ),
        ),
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
