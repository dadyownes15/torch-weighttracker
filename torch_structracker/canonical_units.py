from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch.nn as nn

from torch_structracker.reductions.builder import IndexSelection, SegmentSelection
from torch_structracker.torch_pruning.pruner.function import (
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
class AttentionUnitConfig:
    source_module: nn.Module
    source_layout: SourceLayout
    embed_dim: int
    num_heads: int | None
    unit_axis: UnitAxis
    output_length: int

    @property
    def head_dim(self) -> int | None:
        if self.num_heads is None:
            return None
        return self.embed_dim // self.num_heads


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
    source_indices: tuple[int, ...] | None = None
    embed_dim: int | None = None
    num_heads: int | None = None
    head_dim: int | None = None

    @property
    def unit_indices(self) -> tuple[int, ...]:
        if isinstance(self.destination, SegmentSelection):
            return tuple(
                range(self.destination.start, self.destination.start + self.destination.length)
            )
        return self.destination.indices


@dataclass(frozen=True)
class CanonicalUnitGroup:
    group_id: int
    offset: int
    length: int
    unit_kind: UnitKind
    members: tuple[CanonicalMember, ...]
    raw_group: object


def canonicalize_groups(
    groups: Iterable[object],
    *,
    num_heads: dict[nn.Module, int] | None = None,
    prune_dim: bool | None = None,
    prune_num_heads: bool = False,
) -> tuple[CanonicalUnitGroup, ...]:
    if prune_dim and prune_num_heads:
        raise ValueError("prune_dim and prune_num_heads cannot both be enabled.")

    num_heads = {} if num_heads is None else dict(num_heads)
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
        )
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
            )
        )
        group_offset += group_length

    return tuple(canonical_groups)


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
            unit_axis=attention.unit_axis,
            destination=SegmentSelection(group_offset, group_length),
            embed_dim=attention.embed_dim,
            num_heads=attention.num_heads,
            head_dim=attention.head_dim,
        )

    destination = destination_for_member(
        member,
        group_offset=group_offset,
        root_to_position=root_to_position,
        attention=attention,
    )
    unit_axis = unit_axis_for_plain_member(module, handler)
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
        embed_dim=None if attention is None else attention.embed_dim,
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
) -> AttentionUnitConfig | None:
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
    num_heads: int,
    prune_dim: bool | None,
    prune_num_heads: bool,
) -> AttentionUnitConfig:
    if num_heads <= 0:
        raise ValueError("num_heads must be positive for attention groups.")
    if embed_dim % num_heads != 0:
        raise ValueError("Attention embed_dim must be divisible by num_heads.")

    head_dim = embed_dim // num_heads
    if prune_num_heads:
        return AttentionUnitConfig(
            source_module=source_module,
            source_layout=source_layout,
            embed_dim=embed_dim,
            num_heads=num_heads,
            unit_axis=UnitAxis.QKV_HEAD,
            output_length=num_heads,
        )

    if prune_dim:
        return AttentionUnitConfig(
            source_module=source_module,
            source_layout=source_layout,
            embed_dim=embed_dim,
            num_heads=num_heads,
            unit_axis=UnitAxis.QKV_HEAD_DIM,
            output_length=head_dim,
        )

    return AttentionUnitConfig(
        source_module=source_module,
        source_layout=source_layout,
        embed_dim=embed_dim,
        num_heads=num_heads,
        unit_axis=UnitAxis.QKV_CHANNEL,
        output_length=embed_dim,
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

    if isinstance(module, nn.Linear) and member.dep.handler == prune_linear_out_channels:
        return attention.source_layout

    return None


def unit_axis_for_plain_member(module: nn.Module, handler) -> UnitAxis | None:
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


def semantic_destinations_for_indices(
    indices: Iterable[int],
    *,
    group_offset: int,
    attention: AttentionUnitConfig,
) -> tuple[int, ...]:
    if attention.unit_axis == UnitAxis.QKV_CHANNEL:
        return tuple(
            int(group_offset) + (int(index) % attention.embed_dim)
            for index in indices
        )

    if attention.num_heads is None or attention.head_dim is None:
        raise ValueError("Head-based attention destinations require num_heads.")

    destinations: list[int] = []
    for index in indices:
        embed_index = int(index) % attention.embed_dim
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
