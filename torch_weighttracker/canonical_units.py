from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import Enum

import torch.nn as nn

from torch_weighttracker.reductions.builder import IndexSelection, SegmentSelection
from torch_weighttracker.torch_pruning.pruner.function import (
    prune_batchnorm_in_channels,
    prune_batchnorm_out_channels,
    prune_conv_in_channels,
    prune_conv_out_channels,
    prune_layernorm_in_channels,
    prune_layernorm_out_channels,
    prune_linear_in_channels,
    prune_linear_out_channels,
    prune_multihead_attention_in_channels,
    prune_multihead_attention_out_channels,
)


class SourceLayout(str, Enum):
    PLAIN = "plain"
    FUSED_QKV = "fused_qkv"
    SEPARATE_QKV = "separate_qkv"


class PruningIndexLayout(str, Enum):
    EMBED_SPACE = "embed_space"
    FUSED_QKV_ROW_SPACE = "fused_qkv_row_space"


class UnitAxis(str, Enum):
    OUT_CHANNEL = "out_channel"
    IN_CHANNEL = "in_channel"
    FEATURE = "feature"
    QKV_CHANNEL = "qkv_channel"
    QKV_HEAD = "qkv_head"
    QKV_HEAD_DIM = "qkv_head_dim"


class UnitKind(str, Enum):
    CHANNEL = "channel"
    HEAD = "head"
    HEAD_DIM = "head_dim"


@dataclass(frozen=True)
class SeparateQKVAttentionSpec:
    """Describe one attention block built from separate Linear projections.

    ``prune_heads`` receives the owning attention module and head positions in
    the *current* compact projection layout. Framework integrations may map
    those positions back to stable/original head identifiers before pruning.
    """

    query_projection: nn.Linear
    key_projection: nn.Linear
    value_projection: nn.Linear
    output_projection: nn.Linear
    attention_module: nn.Module
    num_heads: int
    prune_heads: Callable[[nn.Module, tuple[int, ...]], None]

    @property
    def projections(self) -> tuple[nn.Linear, nn.Linear, nn.Linear, nn.Linear]:
        return (
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.output_projection,
        )

    @property
    def head_dim(self) -> int:
        return int(self.query_projection.out_features) // int(self.num_heads)


@dataclass(frozen=True)
class AttentionUnitConfig:
    source_module: nn.Module
    source_layout: SourceLayout
    projection_out_features: int
    projection_in_features: int
    num_heads: int | None
    unit_axis: UnitAxis
    output_length: int
    pruning_index_layout: PruningIndexLayout
    separate_qkv_spec: SeparateQKVAttentionSpec | None = None

    @property
    def head_dim(self) -> int | None:
        if self.num_heads is None:
            return None
        return self.projection_out_features // self.num_heads

    @property
    def embed_dim(self) -> int:
        return self.projection_out_features


@dataclass(frozen=True)
class CanonicalMember:
    group_id: int
    group_offset: int
    group_length: int
    member: object
    module: nn.Module
    handler: object
    source_layout: SourceLayout
    unit_axis: UnitAxis
    destination: SegmentSelection | IndexSelection
    calculation_source_indices: tuple[int, ...] | None = None
    pruning_indices_by_unit: tuple[tuple[int, ...], ...] | None = None
    projection_out_features: int | None = None
    projection_in_features: int | None = None
    pruning_index_layout: PruningIndexLayout | None = None
    num_heads: int | None = None
    head_dim: int | None = None

    @property
    def unit_indices(self) -> tuple[int, ...]:
        if isinstance(self.destination, SegmentSelection):
            return tuple(
                range(
                    self.destination.start,
                    self.destination.start + self.destination.length,
                )
            )
        return self.destination.indices

    @property
    def source_indices(self) -> tuple[int, ...] | None:
        return self.calculation_source_indices

    @property
    def pruning_source_indices(self) -> tuple[tuple[int, ...], ...] | None:
        return self.pruning_indices_by_unit

    @property
    def embed_dim(self) -> int | None:
        return self.projection_out_features


@dataclass(frozen=True)
class CanonicalUnitGroup:
    group_id: int
    offset: int
    length: int
    unit_kind: UnitKind
    members: tuple[CanonicalMember, ...]
    raw_group: object
    attention_spec: SeparateQKVAttentionSpec | None = None


def canonicalize_groups(
    groups: Iterable[object],
    *,
    num_heads: dict[nn.Module, int] | None = None,
    prune_dim: bool | None = None,
    prune_num_heads: bool = False,
    customized_pruners: dict[object, object] | None = None,
    attention_specs: Iterable[SeparateQKVAttentionSpec] = (),
    canonical_group_filter: Callable[[CanonicalUnitGroup], bool] | None = None,
) -> tuple[CanonicalUnitGroup, ...]:
    if prune_dim and prune_num_heads:
        raise ValueError("prune_dim and prune_num_heads cannot both be enabled.")

    num_heads = {} if num_heads is None else dict(num_heads)
    attention_specs = tuple(attention_specs)
    _validate_separate_qkv_attention_specs(attention_specs)
    matched_attention_specs: set[int] = set()
    canonical_groups: list[CanonicalUnitGroup] = []
    group_offset = 0

    for group_id, group in enumerate(groups):
        items = tuple(group_items(group))
        if len(items) == 0:
            continue
        attention = attention_unit_config(
            items,
            num_heads=num_heads,
            prune_dim=prune_dim,
            prune_num_heads=prune_num_heads,
            attention_specs=attention_specs,
        )
        if attention is not None and attention.separate_qkv_spec is not None:
            spec_id = id(attention.separate_qkv_spec)
            if spec_id in matched_attention_specs:
                raise ValueError(
                    "SeparateQKVAttentionSpec matched more than one dependency "
                    "group. Each projection set must own exactly one group."
                )
            matched_attention_specs.add(spec_id)

        group_length = (
            attention.output_length
            if attention is not None
            else len(member_root_indices(items[0]))
        )

        unit_kind = unit_kind_for_attention(attention)
        root_to_position = root_position_map(items[0])

        members = tuple(
            canonical_member_for_raw_member(
                member,
                group_id=group_id,
                group_offset=group_offset,
                group_length=group_length,
                root_to_position=root_to_position,
                attention=attention,
                customized_pruners=customized_pruners,
            )
            for member in items
        )

        members = tuple(member for member in members if member is not None)
        canonical_groups.append(
            CanonicalUnitGroup(
                group_id=group_id,
                offset=group_offset,
                length=group_length,
                unit_kind=unit_kind,
                members=members,
                raw_group=group,
                attention_spec=(
                    None if attention is None else attention.separate_qkv_spec
                ),
            )
        )
        group_offset += group_length

    missing_specs = tuple(
        spec for spec in attention_specs if id(spec) not in matched_attention_specs
    )
    if missing_specs:
        raise ValueError(
            "SeparateQKVAttentionSpec did not match a dependency group containing "
            "Q/K/V output axes and the output-projection input axis."
        )

    result = tuple(canonical_groups)
    if canonical_group_filter is not None:
        result = _filter_and_reindex_groups(result, canonical_group_filter)
    return result


def canonical_members(
    groups: Iterable[CanonicalUnitGroup],
) -> tuple[CanonicalMember, ...]:
    return tuple(member for group in groups for member in group.members)


def group_items(group) -> tuple[object, ...]:
    return tuple(group.items if hasattr(group, "items") else group)


def member_root_indices(member) -> tuple[int, ...]:
    root_indices = getattr(member, "root_idxs", None)
    if root_indices is None:
        raise ValueError("Dependency group member is missing root indices.")
    return tuple(int(index) for index in root_indices)


def member_local_indices(member) -> tuple[int, ...]:
    indices = getattr(member, "idxs", None)
    if indices is None:
        return ()
    return tuple(int(index) for index in indices)


def root_position_map(root_member) -> dict[int, int]:
    return {
        int(root_idx): position
        for position, root_idx in enumerate(member_root_indices(root_member))
    }


def canonical_member_for_raw_member(
    member,
    *,
    group_id: int,
    group_offset: int,
    group_length: int,
    root_to_position: dict[int, int],
    attention: AttentionUnitConfig | None,
    customized_pruners: dict[object, object] | None = None,
) -> CanonicalMember | None:
    module = member.dep.target.module
    handler = member.dep.handler

    if not isinstance(module, nn.Module):
        return None

    qkv_layout = qkv_source_layout_for_member(member, module, attention)
    if qkv_layout is not None:
        return CanonicalMember(
            group_id=group_id,
            group_offset=group_offset,
            group_length=group_length,
            member=member,
            module=module,
            handler=handler,
            source_layout=qkv_layout,
            calculation_source_indices=member_local_indices(member),
            pruning_indices_by_unit=pruning_indices_by_unit_for_attention(
                attention,
            ),
            unit_axis=attention.unit_axis,
            destination=SegmentSelection(group_offset, group_length),
            projection_out_features=attention.projection_out_features,
            projection_in_features=attention.projection_in_features,
            pruning_index_layout=attention.pruning_index_layout,
            num_heads=attention.num_heads,
            head_dim=attention.head_dim,
        )

    destination = destination_for_member(
        member,
        group_offset=group_offset,
        root_to_position=root_to_position,
        attention=attention,
    )
    unit_axis = unit_axis_for_plain_member(
        module,
        handler,
        customized_pruners=customized_pruners,
    )
    if unit_axis is None:
        return None

    return CanonicalMember(
        group_id=group_id,
        group_offset=group_offset,
        group_length=group_length,
        member=member,
        module=module,
        handler=handler,
        source_layout=SourceLayout.PLAIN,
        unit_axis=unit_axis,
        destination=destination,
        projection_out_features=(
            None if attention is None else attention.projection_out_features
        ),
        projection_in_features=(
            None if attention is None else attention.projection_in_features
        ),
        pruning_index_layout=(
            None if attention is None else attention.pruning_index_layout
        ),
        num_heads=None if attention is None else attention.num_heads,
        head_dim=None if attention is None else attention.head_dim,
    )


def destination_for_member(
    member,
    *,
    group_offset: int,
    root_to_position: dict[int, int],
    attention: AttentionUnitConfig | None,
) -> IndexSelection:
    root_indices = member_root_indices(member)

    if attention is not None:
        return IndexSelection(
            semantic_destinations_for_indices(
                root_indices,
                group_offset=group_offset,
                attention=attention,
            )
        )

    destinations: list[int] = []
    for root_index in root_indices:
        try:
            position = root_to_position[int(root_index)]
        except KeyError as error:
            raise ValueError(
                f"Root index {root_index} is not present in the dependency group root."
            ) from error
        destinations.append(int(group_offset) + position)
    return IndexSelection(tuple(destinations))


def attention_unit_config(
    items: tuple[object, ...],
    *,
    num_heads: dict[nn.Module, int],
    prune_dim: bool | None,
    prune_num_heads: bool,
    attention_specs: tuple[SeparateQKVAttentionSpec, ...] = (),
) -> AttentionUnitConfig | None:
    separate_spec = _separate_qkv_spec_for_group(items, attention_specs)
    if separate_spec is not None:
        projection = separate_spec.query_projection
        return AttentionUnitConfig(
            source_module=separate_spec.attention_module,
            source_layout=SourceLayout.PLAIN,
            projection_out_features=int(projection.out_features),
            projection_in_features=int(projection.in_features),
            num_heads=int(separate_spec.num_heads),
            unit_axis=UnitAxis.QKV_HEAD,
            output_length=int(separate_spec.num_heads),
            pruning_index_layout=PruningIndexLayout.EMBED_SPACE,
            separate_qkv_spec=separate_spec,
        )

    for member in items:
        module = member.dep.target.module
        handler = member.dep.handler

        if isinstance(module, nn.MultiheadAttention) and handler in {
            prune_multihead_attention_out_channels,
            prune_multihead_attention_in_channels,
        }:
            heads = int(num_heads.get(module, module.num_heads))
            return make_attention_config(
                source_module=module,
                source_layout=(
                    SourceLayout.FUSED_QKV
                    if module.in_proj_weight is not None
                    else SourceLayout.SEPARATE_QKV
                ),
                embed_dim=int(module.embed_dim),
                projection_in_features=int(
                    module.in_proj_weight.shape[1]
                    if module.in_proj_weight is not None
                    else module.embed_dim
                ),
                num_heads=heads,
                prune_dim=prune_dim,
                prune_num_heads=prune_num_heads,
            )

        if (
            isinstance(module, nn.Linear)
            and module in num_heads
            and handler == prune_linear_out_channels
            and module.out_features % 3 == 0
        ):
            return make_attention_config(
                source_module=module,
                source_layout=SourceLayout.FUSED_QKV,
                embed_dim=int(module.out_features // 3),
                projection_in_features=int(module.in_features),
                num_heads=int(num_heads[module]),
                prune_dim=prune_dim,
                prune_num_heads=prune_num_heads,
            )

    return None


def make_attention_config(
    *,
    source_module: nn.Module,
    source_layout: SourceLayout,
    embed_dim: int,
    projection_in_features: int,
    num_heads: int,
    prune_dim: bool | None,
    prune_num_heads: bool,
) -> AttentionUnitConfig:
    if num_heads <= 0:
        raise ValueError("num_heads must be positive for attention groups.")
    if embed_dim % num_heads != 0:
        raise ValueError("Attention embed_dim must be divisible by num_heads.")

    pruning_index_layout = (
        PruningIndexLayout.EMBED_SPACE
        if isinstance(source_module, nn.MultiheadAttention)
        else PruningIndexLayout.FUSED_QKV_ROW_SPACE
    )

    head_dim = embed_dim // num_heads
    if prune_num_heads:
        return AttentionUnitConfig(
            source_module=source_module,
            source_layout=source_layout,
            projection_out_features=embed_dim,
            projection_in_features=projection_in_features,
            num_heads=num_heads,
            unit_axis=UnitAxis.QKV_HEAD,
            output_length=num_heads,
            pruning_index_layout=pruning_index_layout,
        )

    if prune_dim:
        return AttentionUnitConfig(
            source_module=source_module,
            source_layout=source_layout,
            projection_out_features=embed_dim,
            projection_in_features=projection_in_features,
            num_heads=num_heads,
            unit_axis=UnitAxis.QKV_HEAD_DIM,
            output_length=head_dim,
            pruning_index_layout=pruning_index_layout,
        )

    return AttentionUnitConfig(
        source_module=source_module,
        source_layout=source_layout,
        projection_out_features=embed_dim,
        projection_in_features=projection_in_features,
        num_heads=num_heads,
        unit_axis=UnitAxis.QKV_CHANNEL,
        output_length=embed_dim,
        pruning_index_layout=pruning_index_layout,
    )


def qkv_source_layout_for_member(
    member,
    module: nn.Module,
    attention: AttentionUnitConfig | None,
) -> SourceLayout | None:
    if attention is None or module is not attention.source_module:
        return None

    if isinstance(module, nn.MultiheadAttention):
        return attention.source_layout

    if (
        isinstance(module, nn.Linear)
        and member.dep.handler == prune_linear_out_channels
    ):
        return attention.source_layout

    return None


def unit_axis_for_plain_member(
    module: nn.Module,
    handler,
    *,
    customized_pruners: dict[object, object] | None = None,
) -> UnitAxis | None:
    custom_axis = custom_unit_axis_for_plain_member(
        module,
        handler,
        customized_pruners=customized_pruners,
    )
    if custom_axis is not None:
        return custom_axis

    if isinstance(module, nn.Linear):
        if handler == prune_linear_out_channels:
            return UnitAxis.OUT_CHANNEL
        if handler == prune_linear_in_channels:
            return UnitAxis.IN_CHANNEL

    if isinstance(module, nn.Conv2d):
        if handler == prune_conv_out_channels:
            return UnitAxis.OUT_CHANNEL
        if handler == prune_conv_in_channels:
            return UnitAxis.IN_CHANNEL

    if isinstance(module, nn.modules.batchnorm._BatchNorm) and handler in {
        prune_batchnorm_out_channels,
        prune_batchnorm_in_channels,
    }:
        return UnitAxis.FEATURE

    if isinstance(module, nn.LayerNorm) and handler in {
        prune_layernorm_out_channels,
        prune_layernorm_in_channels,
    }:
        return UnitAxis.FEATURE

    return None


def custom_unit_axis_for_plain_member(
    module: nn.Module,
    handler,
    *,
    customized_pruners: dict[object, object] | None,
) -> UnitAxis | None:
    if not isinstance(module, (nn.Linear, nn.Conv2d)):
        return None

    custom_pruner = custom_pruner_for_module(module, customized_pruners)
    if custom_pruner is None:
        return None

    is_out = same_method(handler, custom_pruner.prune_out_channels)
    is_in = same_method(handler, custom_pruner.prune_in_channels)

    if is_out and not is_in:
        return UnitAxis.OUT_CHANNEL
    if is_in and not is_out:
        return UnitAxis.IN_CHANNEL
    return None


def custom_pruner_for_module(
    module: nn.Module,
    customized_pruners: dict[object, object] | None,
):
    if not customized_pruners:
        return None

    custom_pruner = customized_pruners.get(module)
    if custom_pruner is not None:
        return custom_pruner
    return customized_pruners.get(module.__class__)


def same_method(left, right) -> bool:
    if left == right:
        return True

    left_self = getattr(left, "__self__", None)
    right_self = getattr(right, "__self__", None)
    left_func = getattr(left, "__func__", None)
    right_func = getattr(right, "__func__", None)

    return (
        left_self is not None
        and left_self is right_self
        and left_func is not None
        and left_func is right_func
    )


def pruning_indices_by_unit_for_attention(
    attention: AttentionUnitConfig,
) -> tuple[tuple[int, ...], ...]:
    if attention.unit_axis == UnitAxis.QKV_CHANNEL:
        return tuple((index,) for index in range(attention.projection_out_features))

    if attention.num_heads is None or attention.head_dim is None:
        raise ValueError("Head-based attention pruning indices require num_heads.")

    if attention.pruning_index_layout == PruningIndexLayout.EMBED_SPACE:
        qkv_offsets = (0,)
    else:
        width = attention.projection_out_features
        qkv_offsets = (0, width, 2 * width)

    if attention.unit_axis == UnitAxis.QKV_HEAD:
        return tuple(
            tuple(
                qkv_offset + channel
                for qkv_offset in qkv_offsets
                for channel in range(
                    head * attention.head_dim,
                    (head + 1) * attention.head_dim,
                )
            )
            for head in range(attention.num_heads)
        )

    if attention.unit_axis == UnitAxis.QKV_HEAD_DIM:
        return tuple(
            tuple(
                qkv_offset + head * attention.head_dim + dim
                for qkv_offset in qkv_offsets
                for head in range(attention.num_heads)
            )
            for dim in range(attention.head_dim)
        )

    raise ValueError(f"Unsupported attention unit axis: {attention.unit_axis}")


def semantic_destinations_for_indices(
    indices: Iterable[int],
    *,
    group_offset: int,
    attention: AttentionUnitConfig,
) -> tuple[int, ...]:
    if attention.unit_axis == UnitAxis.QKV_CHANNEL:
        return tuple(
            int(group_offset) + (int(index) % attention.projection_out_features)
            for index in indices
        )

    if attention.num_heads is None or attention.head_dim is None:
        raise ValueError("Head-based attention destinations require num_heads.")

    destinations: list[int] = []
    for index in indices:
        embed_index = int(index) % attention.projection_out_features
        if attention.unit_axis == UnitAxis.QKV_HEAD:
            destination = embed_index // attention.head_dim
        elif attention.unit_axis == UnitAxis.QKV_HEAD_DIM:
            destination = embed_index % attention.head_dim
        else:
            raise ValueError(f"Unsupported attention unit axis: {attention.unit_axis}")
        destinations.append(int(group_offset) + destination)
    return tuple(destinations)


def unit_kind_for_attention(attention: AttentionUnitConfig | None) -> UnitKind:
    if attention is None or attention.unit_axis == UnitAxis.QKV_CHANNEL:
        return UnitKind.CHANNEL
    if attention.unit_axis == UnitAxis.QKV_HEAD:
        return UnitKind.HEAD
    if attention.unit_axis == UnitAxis.QKV_HEAD_DIM:
        return UnitKind.HEAD_DIM
    raise ValueError(f"Unsupported attention unit axis: {attention.unit_axis}")


def _validate_separate_qkv_attention_specs(
    specs: tuple[SeparateQKVAttentionSpec, ...],
) -> None:
    owned_projection_ids: set[int] = set()
    owned_attention_ids: set[int] = set()

    for spec in specs:
        if not isinstance(spec, SeparateQKVAttentionSpec):
            raise TypeError(
                "attention_specs entries must be SeparateQKVAttentionSpec "
                f"instances, got {type(spec).__name__}."
            )
        if not isinstance(spec.attention_module, nn.Module):
            raise TypeError("attention_module must be an nn.Module instance.")
        if not callable(spec.prune_heads):
            raise TypeError("SeparateQKVAttentionSpec.prune_heads must be callable.")

        projections = spec.projections
        if not all(isinstance(module, nn.Linear) for module in projections):
            raise TypeError("Separate Q/K/V/output projections must be nn.Linear.")
        if len({id(module) for module in projections}) != len(projections):
            raise ValueError("Separate Q/K/V/output projections must be distinct.")
        owned_module_ids = {id(module) for module in spec.attention_module.modules()}
        if any(id(module) not in owned_module_ids for module in projections):
            raise ValueError(
                "SeparateQKVAttentionSpec.attention_module must own all Q/K/V/output "
                "projections."
            )

        query, key, value, output = projections
        qkv_shapes = {
            (int(module.in_features), int(module.out_features))
            for module in (query, key, value)
        }
        if len(qkv_shapes) != 1:
            raise ValueError("Separate Q/K/V projections must have matching shapes.")
        if int(output.in_features) != int(query.out_features):
            raise ValueError(
                "The attention output projection input width must match the Q/K/V "
                "output width."
            )
        if int(spec.num_heads) <= 0:
            raise ValueError("SeparateQKVAttentionSpec.num_heads must be positive.")
        if int(query.out_features) % int(spec.num_heads) != 0:
            raise ValueError("Q/K/V output width must be divisible by num_heads.")

        projection_ids = {id(module) for module in projections}
        if owned_projection_ids.intersection(projection_ids):
            raise ValueError("SeparateQKVAttentionSpec projections must not overlap.")
        if id(spec.attention_module) in owned_attention_ids:
            raise ValueError(
                "Only one SeparateQKVAttentionSpec may own an attention module."
            )
        owned_projection_ids.update(projection_ids)
        owned_attention_ids.add(id(spec.attention_module))


def _separate_qkv_spec_for_group(
    items: tuple[object, ...],
    specs: tuple[SeparateQKVAttentionSpec, ...],
) -> SeparateQKVAttentionSpec | None:
    if not specs:
        return None

    item_pairs = tuple(
        (member.dep.target.module, member.dep.handler) for member in items
    )
    matches: list[SeparateQKVAttentionSpec] = []
    for spec in specs:
        required = {
            (spec.query_projection, prune_linear_out_channels),
            (spec.key_projection, prune_linear_out_channels),
            (spec.value_projection, prune_linear_out_channels),
            (spec.output_projection, prune_linear_in_channels),
        }
        if required.issubset(set(item_pairs)):
            matches.append(spec)

    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            "Multiple SeparateQKVAttentionSpec entries matched one dependency group."
        )

    match = matches[0]
    owned_modules = set(match.projections)
    unexpected_linear_members = tuple(
        (module, handler)
        for module, handler in item_pairs
        if isinstance(module, nn.Linear) and module not in owned_modules
    )
    if unexpected_linear_members:
        raise ValueError(
            "A SeparateQKVAttentionSpec dependency group contains additional "
            "Linear members, so replacing it with a semantic head group would "
            "discard coupled structure."
        )
    return match


def _filter_and_reindex_groups(
    groups: tuple[CanonicalUnitGroup, ...],
    predicate: Callable[[CanonicalUnitGroup], bool],
) -> tuple[CanonicalUnitGroup, ...]:
    filtered: list[CanonicalUnitGroup] = []
    offset = 0

    for group in groups:
        if not predicate(group):
            continue

        group_id = len(filtered)
        delta = offset - int(group.offset)
        members = tuple(
            replace(
                member,
                group_id=group_id,
                group_offset=offset,
                destination=_shift_selection(member.destination, delta),
            )
            for member in group.members
        )
        filtered.append(
            replace(
                group,
                group_id=group_id,
                offset=offset,
                members=members,
            )
        )
        offset += int(group.length)

    return tuple(filtered)


def _shift_selection(
    selection: SegmentSelection | IndexSelection,
    delta: int,
) -> SegmentSelection | IndexSelection:
    if isinstance(selection, SegmentSelection):
        return SegmentSelection(
            start=int(selection.start) + int(delta),
            length=int(selection.length),
        )
    if isinstance(selection, IndexSelection):
        return IndexSelection(
            tuple(int(index) + int(delta) for index in selection.indices)
        )
    raise TypeError(f"Unsupported canonical selection: {type(selection).__name__}.")
