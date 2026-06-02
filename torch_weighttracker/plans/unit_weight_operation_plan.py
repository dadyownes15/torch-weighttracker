from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import torch
import torch.nn as nn

from torch_weighttracker.canonical_units import (
    CanonicalMember,
    CanonicalUnitGroup,
    SourceLayout,
    UnitAxis,
    canonical_members,
    member_local_indices,
    member_root_indices,
)
from torch_weighttracker.extractors.extractor import (
    ElementTensorExtractor,
    ModuleParameterRef,
    ModuleParameterTupleRef,
    TensorSourceRef,
)
from torch_weighttracker.operations import (
    MultiheadAttentionSemanticOperation,
    QKVSemanticOperation,
    WeightOperationType,
)
from torch_weighttracker.operations.resolver import operation_for_member
from torch_weighttracker.reductions.builder import (
    FullSelection,
    IndexSelection,
    ReductionMapping,
    ReductionPlan,
    ReductionPlanBuilder,
    SegmentSelection,
)
from torch_weighttracker.reductions.compiler import (
    GenericReductionPlanner,
    MappingStrategy,
)
from torch_weighttracker.reductions.helpers import ElementReductionRule
from torch_weighttracker.reductions.ops import ReductionOp, TensorReduction

MemberContext = CanonicalMember


class CanonicalMemberTensorExtractor(ElementTensorExtractor[CanonicalMember]):
    def bind(self, element: CanonicalMember) -> TensorSourceRef | None:
        if element.source_layout == SourceLayout.PLAIN:
            return _bind_module_parameter(element.module, "weight")

        if element.source_layout == SourceLayout.FUSED_QKV:
            if isinstance(element.module, nn.MultiheadAttention):
                in_proj = _bind_module_parameter(element.module, "in_proj_weight")
                out_proj = _bind_mha_out_proj_weight(element.module)
                if in_proj is None:
                    return None
                if out_proj is None:
                    return in_proj
                return ModuleParameterTupleRef((in_proj, out_proj))
            return _bind_module_parameter(element.module, "weight")

        if element.source_layout == SourceLayout.SEPARATE_QKV:
            refs = [
                _bind_module_parameter(element.module, "q_proj_weight"),
                _bind_module_parameter(element.module, "k_proj_weight"),
                _bind_module_parameter(element.module, "v_proj_weight"),
            ]
            if isinstance(element.module, nn.MultiheadAttention):
                refs.append(_bind_mha_out_proj_weight(element.module))
            if any(ref is None for ref in refs):
                return None
            return ModuleParameterTupleRef(tuple(refs))  # type: ignore[arg-type]

        raise ValueError(f"Unsupported source layout: {element.source_layout}")


MemberWeightExtractor = CanonicalMemberTensorExtractor


class UnitWeightReductionMapper:
    def __init__(self, operation_type: WeightOperationType | str) -> None:
        self.operation_type = WeightOperationType(operation_type)

    def __call__(self, element: CanonicalMember) -> TensorReduction:
        if element.source_layout in {
            SourceLayout.FUSED_QKV,
            SourceLayout.SEPARATE_QKV,
        }:
            return self._qkv_reduction(element)

        return operation_for_member(element.member, self.operation_type)

    def _qkv_reduction(self, element: CanonicalMember) -> TensorReduction:
        if element.embed_dim is None:
            raise ValueError("QKV reductions require embed_dim.")

        if element.unit_axis == UnitAxis.QKV_CHANNEL:
            mode = "channel"
            num_heads = None
        elif element.unit_axis == UnitAxis.QKV_HEAD:
            mode = "head"
            num_heads = element.num_heads
        elif element.unit_axis == UnitAxis.QKV_HEAD_DIM:
            mode = "head_dim"
            num_heads = element.num_heads
        else:
            raise ValueError(f"Unsupported QKV unit axis: {element.unit_axis}")

        if isinstance(element.module, nn.MultiheadAttention):
            return MultiheadAttentionSemanticOperation(
                operation_type=self.operation_type,
                embed_dim=element.embed_dim,
                num_heads=num_heads,
                mode=mode,
            )

        return QKVSemanticOperation(
            operation_type=self.operation_type,
            embed_dim=element.embed_dim,
            num_heads=num_heads,
            mode=mode,
        )


class MemberUnitMapper(MappingStrategy[CanonicalMember]):
    def map(
        self,
        element: CanonicalMember,
        op: ReductionOp,
        builder: ReductionPlanBuilder,
    ) -> ReductionMapping:
        target = element.destination

        if isinstance(target, SegmentSelection):
            if op.output_length == target.length:
                return ReductionMapping(source=FullSelection(), target=target)
            source_indices = source_indices_for_member(element, op)
            return ReductionMapping(
                source=IndexSelection(source_indices),
                target=target,
            )

        destination_indices = tuple(int(index) for index in target.indices)
        if op.output_length == len(destination_indices):
            return ReductionMapping(source=FullSelection(), target=target)

        source_indices = source_indices_for_member(element, op)
        if len(source_indices) != len(destination_indices):
            raise ValueError(
                "Member source and destination mapping lengths must match."
            )

        return ReductionMapping(
            source=IndexSelection(source_indices),
            target=IndexSelection(destination_indices),
        )


def create_group_member_plan(
    groups: Iterable[CanonicalUnitGroup],
    operation_type: WeightOperationType | str,
) -> ReductionPlan:
    canonical_groups = tuple(groups)
    elements = canonical_members(canonical_groups)
    output_length = sum(group.length for group in canonical_groups)

    planner = GenericReductionPlanner[CanonicalMember](
        elements=elements,
        output_length=output_length,
    )
    compiled = planner.compile(
        ElementReductionRule[CanonicalMember](
            extractor=CanonicalMemberTensorExtractor(),
            reduction_mapper=UnitWeightReductionMapper(operation_type),
            mapping_strategy=MemberUnitMapper(),
        )
    )

    return cast(ReductionPlan, compiled)


def count_group_units(groups: Iterable[CanonicalUnitGroup]) -> int:
    return sum(group.length for group in groups)


def source_indices_for_member(
    element: CanonicalMember,
    op: ReductionOp,
) -> tuple[int, ...]:
    if element.source_indices is not None:
        local_indices = tuple(int(index) for index in element.source_indices)
    else:
        local_indices = member_local_indices(element.member)

    if len(local_indices) == 0:
        return ()

    if min(local_indices) < 0:
        raise ValueError("Member source indices must be non-negative.")

    if max(local_indices) >= op.output_length:
        raise ValueError(
            f"Member source index {max(local_indices)} is outside op output "
            f"length {op.output_length}."
        )

    return local_indices


def _bind_module_parameter(
    module: nn.Module,
    name: str,
) -> ModuleParameterRef | None:
    value = getattr(module, name, None)
    if not isinstance(value, torch.Tensor):
        return None
    return ModuleParameterRef(module, name)


def _bind_mha_out_proj_weight(
    module: nn.MultiheadAttention,
) -> ModuleParameterRef | None:
    out_proj = getattr(module, "out_proj", None)
    if not isinstance(out_proj, nn.Module):
        return None
    return _bind_module_parameter(out_proj, "weight")


def group_items(group):
    return group.items if hasattr(group, "items") else group


def destination_indices_for_member(
    member,
    *,
    group_offset: int,
    root_to_position: dict[int, int],
) -> tuple[int, ...]:
    destinations: list[int] = []
    for root_index in member_root_indices(member):
        try:
            position = root_to_position[int(root_index)]
        except KeyError as error:
            raise ValueError(
                f"Root index {root_index} is not present in the dependency group root."
            ) from error
        destinations.append(int(group_offset) + position)
    return tuple(destinations)


def has_live_parameter(module: nn.Module, parameter_name: str) -> bool:
    return (
        hasattr(module, parameter_name) and getattr(module, parameter_name) is not None
    )
