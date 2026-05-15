from __future__ import annotations

from collections.abc import Hashable, Iterable

import torch

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import calculation_device, calculation_dtype
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.calculations.static_calc import StaticCalc
from torch_weighttracker.canonical_units import CanonicalMember, CanonicalUnitGroup, canonical_members
from torch_weighttracker.operations import WeightOperationType
from torch_weighttracker.plans.unit_weight_operation_plan import (
    CanonicalMemberTensorExtractor,
    UnitWeightReductionMapper,
    source_indices_for_member,
)
from torch_weighttracker.reductions.builder import IndexSelection, SegmentSelection
from torch_weighttracker.reductions.ops import ReductionOp

BaselineContributionKey = tuple[int, Hashable, tuple[int, ...]]


class BaselineParamPrUnitPrGroup(StaticCalc):
    calculation_type = CalcType.BASELINE_PARAM_PR_UNIT_PR_GROUP


def create_baseline_param_pr_unit_pr_group_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> BaselineParamPrUnitPrGroup:
    canonical_groups = tuple(groups)
    values = torch.zeros(len(canonical_groups), device=device, dtype=dtype)
    extractor = CanonicalMemberTensorExtractor()
    reduction_mapper = UnitWeightReductionMapper(WeightOperationType.COUNT)
    seen: set[BaselineContributionKey] = set()

    for member in canonical_members(canonical_groups):
        for key, contribution in _baseline_param_contributions(
            member,
            extractor=extractor,
            reduction_mapper=reduction_mapper,
        ):
            if key in seen:
                continue
            seen.add(key)
            values[int(member.group_id)] += contribution

    return BaselineParamPrUnitPrGroup(values)


def _baseline_param_contributions(
    member: CanonicalMember,
    *,
    extractor: CanonicalMemberTensorExtractor,
    reduction_mapper: UnitWeightReductionMapper,
) -> Iterable[tuple[BaselineContributionKey, float]]:
    source_ref = extractor.bind(member)
    if source_ref is None:
        return

    op = ReductionOp(source_ref, reduction_mapper(member))
    source_indices = _representative_source_indices(member, op)
    if len(source_indices) == 0:
        return

    with torch.no_grad():
        values = op().detach().reshape(-1)
        if values.numel() == 1 and op.output_length > 1:
            contribution = values.new_tensor(
                float(values.item()) / float(op.output_length)
            ) * len(source_indices)
        else:
            index = torch.tensor(
                source_indices,
                device=values.device,
                dtype=torch.long,
            )
            contribution = values.index_select(0, index).sum()

    key = (
        int(member.group_id),
        op.identity_key(),
        source_indices,
    )
    yield key, float(contribution.item())


def _representative_source_indices(
    member: CanonicalMember,
    op: ReductionOp,
) -> tuple[int, ...]:
    representative = int(member.group_offset)
    target = member.destination

    if isinstance(target, SegmentSelection):
        start = int(target.start)
        stop = start + int(target.length)
        if representative < start or representative >= stop:
            return ()

        position = representative - start
        if op.output_length == target.length:
            return (position,)

        member_source_indices = source_indices_for_member(member, op)
        if position >= len(member_source_indices):
            return ()
        return (member_source_indices[position],)

    if isinstance(target, IndexSelection):
        destination_indices = tuple(int(index) for index in target.indices)
        if op.output_length == len(destination_indices):
            return tuple(
                source_index
                for source_index, destination in enumerate(destination_indices)
                if destination == representative
            )

        member_source_indices = source_indices_for_member(member, op)
        return tuple(
            source_index
            for source_index, destination in zip(
                member_source_indices,
                destination_indices,
                strict=True,
            )
            if destination == representative
        )

    raise TypeError(f"Unknown member destination: {type(target)!r}")


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.BASELINE_PARAM_PR_UNIT_PR_GROUP,
    cache_constant=True,
    create=lambda ctx, deps: create_baseline_param_pr_unit_pr_group_calc(
        ctx.canonical_groups,
        device=calculation_device(ctx),
        dtype=calculation_dtype(ctx),
    ),
)
