from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

from torch_weighttracker.canonical_units import (
    CanonicalMember,
    CanonicalUnitGroup,
    PruningIndexLayout,
    SourceLayout,
    UnitAxis,
    member_local_indices,
)
from torch_weighttracker.reductions.builder import IndexSelection, SegmentSelection


@dataclass(frozen=True)
class FakePruneUnitResult:
    group_id: int
    unit_id: int
    zeroed_members: int
    zeroed_parameters: tuple[str, ...]
    prune_bias: bool


def fake_prune_canonical_unit(
    model: nn.Module,
    canonical_groups: tuple[CanonicalUnitGroup, ...],
    group_id: int,
    unit_id: int,
    *,
    prune_bias: bool = True,
) -> FakePruneUnitResult:
    group = canonical_groups[group_id]
    if len(group.members) == 0:
        raise ValueError(f"Canonical group {group_id} has no members.")
    if unit_id < 0 or unit_id >= group.length:
        raise IndexError(
            f"unit_id {unit_id} is outside canonical group {group_id} "
            f"length {group.length}."
        )

    member_indices = tuple(
        (member, member_indices_for_unit(member, unit_id)) for member in group.members
    )
    module_names = _module_name_map(model)
    zeroed_parameters: list[str] = []
    zeroed_members = 0

    with torch.no_grad():
        for member, indices in member_indices:
            zeroed = _zero_member_unit(
                member,
                indices,
                prune_bias=prune_bias,
                module_names=module_names,
            )
            if len(zeroed) == 0:
                continue
            zeroed_members += 1
            for parameter_name in zeroed:
                if parameter_name not in zeroed_parameters:
                    zeroed_parameters.append(parameter_name)

    return FakePruneUnitResult(
        group_id=group_id,
        unit_id=unit_id,
        zeroed_members=zeroed_members,
        zeroed_parameters=tuple(zeroed_parameters),
        prune_bias=prune_bias,
    )


def member_indices_for_unit(
    member: CanonicalMember,
    unit_id: int,
) -> tuple[int, ...]:
    if member.pruning_indices_by_unit is not None:
        if unit_id >= len(member.pruning_indices_by_unit):
            raise IndexError(
                f"unit_id {unit_id} is outside member pruning index length "
                f"{len(member.pruning_indices_by_unit)}."
            )
        return tuple(int(index) for index in member.pruning_indices_by_unit[unit_id])

    canonical_id = int(member.group_offset) + int(unit_id)
    local_indices = _member_local_source_indices(member)

    if isinstance(member.destination, SegmentSelection):
        start = int(member.destination.start)
        stop = start + int(member.destination.length)
        if canonical_id < start or canonical_id >= stop:
            return ()
        position = canonical_id - start
        if len(local_indices) == 0:
            return (position,)
        if position >= len(local_indices):
            raise IndexError(
                f"unit_id {unit_id} maps to source position {position}, outside "
                f"member local index length {len(local_indices)}."
            )
        return (int(local_indices[position]),)

    if isinstance(member.destination, IndexSelection):
        positions = tuple(
            position
            for position, destination in enumerate(member.destination.indices)
            if int(destination) == canonical_id
        )
        if len(local_indices) == 0:
            return tuple(int(position) for position in positions)
        return tuple(int(local_indices[position]) for position in positions)

    raise TypeError(
        f"Unsupported canonical member destination: "
        f"{type(member.destination).__name__}."
    )


def _member_local_source_indices(member: CanonicalMember) -> tuple[int, ...]:
    if member.calculation_source_indices is not None:
        return tuple(int(index) for index in member.calculation_source_indices)
    return tuple(int(index) for index in member_local_indices(member.member))


def _zero_member_unit(
    member: CanonicalMember,
    indices: tuple[int, ...],
    *,
    prune_bias: bool,
    module_names: Mapping[nn.Module, str],
) -> tuple[str, ...]:
    if len(indices) == 0:
        return ()

    if member.source_layout == SourceLayout.SEPARATE_QKV:
        return _zero_separate_qkv_member(
            member,
            indices,
            prune_bias=prune_bias,
            module_names=module_names,
        )

    if member.source_layout == SourceLayout.FUSED_QKV and isinstance(
        member.module, nn.MultiheadAttention
    ):
        indices = _mha_qkv_row_indices(member, indices)
        return _zero_multihead_attention_qkv(
            member.module,
            indices,
            prune_bias=prune_bias,
            module_names=module_names,
        )

    return _zero_plain_member(
        member,
        indices,
        prune_bias=prune_bias,
        module_names=module_names,
    )


def _mha_qkv_row_indices(
    member: CanonicalMember,
    indices: tuple[int, ...],
) -> tuple[int, ...]:
    if member.pruning_index_layout != PruningIndexLayout.EMBED_SPACE:
        return indices
    width = int(member.projection_out_features or member.module.embed_dim)
    return tuple(
        offset + int(index) for offset in (0, width, 2 * width) for index in indices
    )


def _zero_plain_member(
    member: CanonicalMember,
    indices: tuple[int, ...],
    *,
    prune_bias: bool,
    module_names: Mapping[nn.Module, str],
) -> tuple[str, ...]:
    module = member.module
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            f"Cannot fake prune {module.__class__.__name__}: missing weight."
        )

    if member.unit_axis in {
        UnitAxis.OUT_CHANNEL,
        UnitAxis.QKV_CHANNEL,
        UnitAxis.QKV_HEAD,
        UnitAxis.QKV_HEAD_DIM,
        UnitAxis.FEATURE,
    }:
        return _zero_axis_zero_member(
            module,
            indices,
            prune_bias=prune_bias,
            module_names=module_names,
        )

    if member.unit_axis == UnitAxis.IN_CHANNEL:
        _zero_tensor_indices(weight, axis=1, indices=indices)
        return (_parameter_display_name(module_names, module, "weight"),)

    raise ValueError(f"Unsupported fake prune unit axis: {member.unit_axis}.")


def _zero_axis_zero_member(
    module: nn.Module,
    indices: tuple[int, ...],
    *,
    prune_bias: bool,
    module_names: Mapping[nn.Module, str],
) -> tuple[str, ...]:
    zeroed = [_parameter_display_name(module_names, module, "weight")]
    _zero_tensor_indices(module.weight, axis=0, indices=indices)

    bias = getattr(module, "bias", None)
    if prune_bias and isinstance(bias, torch.Tensor):
        _zero_tensor_indices(bias, axis=0, indices=indices)
        zeroed.append(_parameter_display_name(module_names, module, "bias"))

    return tuple(zeroed)


def _zero_multihead_attention_qkv(
    module: nn.MultiheadAttention,
    indices: tuple[int, ...],
    *,
    prune_bias: bool,
    module_names: Mapping[nn.Module, str],
) -> tuple[str, ...]:
    zeroed: list[str] = []
    if isinstance(module.in_proj_weight, torch.Tensor):
        _zero_tensor_indices(module.in_proj_weight, axis=0, indices=indices)
        zeroed.append(_parameter_display_name(module_names, module, "in_proj_weight"))
    if prune_bias and isinstance(module.in_proj_bias, torch.Tensor):
        _zero_tensor_indices(module.in_proj_bias, axis=0, indices=indices)
        zeroed.append(_parameter_display_name(module_names, module, "in_proj_bias"))
    if len(zeroed) == 0:
        raise TypeError(
            f"Cannot fake prune {module.__class__.__name__}: missing QKV parameters."
        )
    return tuple(zeroed)


def _zero_separate_qkv_member(
    member: CanonicalMember,
    indices: tuple[int, ...],
    *,
    prune_bias: bool,
    module_names: Mapping[nn.Module, str],
) -> tuple[str, ...]:
    module = member.module
    zeroed: list[str] = []
    for parameter_name in ("q_proj_weight", "k_proj_weight", "v_proj_weight"):
        parameter = getattr(module, parameter_name, None)
        if not isinstance(parameter, torch.Tensor):
            continue
        _zero_tensor_indices(parameter, axis=0, indices=indices)
        zeroed.append(_parameter_display_name(module_names, module, parameter_name))

    in_proj_bias = getattr(module, "in_proj_bias", None)
    if prune_bias and isinstance(in_proj_bias, torch.Tensor):
        embed_dim = int(module.embed_dim)
        bias_indices = tuple(
            offset + index
            for offset in (0, embed_dim, 2 * embed_dim)
            for index in indices
        )
        _zero_tensor_indices(in_proj_bias, axis=0, indices=bias_indices)
        zeroed.append(_parameter_display_name(module_names, module, "in_proj_bias"))

    if len(zeroed) == 0:
        raise TypeError(
            f"Cannot fake prune {module.__class__.__name__}: missing separate "
            "QKV parameters."
        )
    return tuple(zeroed)


def _zero_tensor_indices(
    tensor: torch.Tensor,
    *,
    axis: int,
    indices: tuple[int, ...],
) -> None:
    if tensor.ndim <= axis:
        raise ValueError(
            f"Cannot zero axis {axis} of tensor with shape {tuple(tensor.shape)}."
        )
    index = torch.as_tensor(indices, dtype=torch.long, device=tensor.device)
    tensor.index_fill_(axis, index, 0)


def _parameter_display_name(
    module_names: Mapping[nn.Module, str],
    module: nn.Module,
    parameter_name: str,
) -> str:
    module_name = module_names.get(module, f"<unnamed:{module.__class__.__name__}>")
    return f"{module_name}.{parameter_name}"


def _module_name_map(model: nn.Module) -> dict[nn.Module, str]:
    names = {}
    for name, module in model.named_modules():
        names[module] = name if name else "<root>"
    return names
