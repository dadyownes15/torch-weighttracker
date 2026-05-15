from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.reduction_calc import ReductionCalc
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.canonical_units import (
    CanonicalMember,
    CanonicalUnitGroup,
    canonical_members,
)
from torch_weighttracker.operations import WeightOperationType
from torch_weighttracker.plans.unit_weight_operation_plan import (
    CanonicalMemberTensorExtractor,
    UnitWeightReductionMapper,
    source_indices_for_member,
)
from torch_weighttracker.reductions.builder import (
    IndexSelection,
    ReductionMapping,
    ReductionPlan,
    ReductionPlanBuilder,
    ReductionRecord,
    SegmentSelection,
)
from torch_weighttracker.reductions.compiler import GenericReductionPlanner
from torch_weighttracker.reductions.ops import ReductionOp


def create_group_change_effect_calc(
    groups: Iterable[CanonicalUnitGroup],
) -> ReductionCalc:
    """
    Returns the structural parameter-count effect of changing one unit per group.

    Output: 1D tensor with length `len(canonical_groups)`.
    Input: none.
    """
    return ReductionCalc(
        _build_group_change_effect_plan(groups),
        calculation_type=CalcType.GROUP_CHANGE_EFFECT,
    )


def _build_group_change_effect_plan(
    groups: Iterable[CanonicalUnitGroup],
) -> ReductionPlan:
    canonical_groups = tuple(groups)
    planner = GenericReductionPlanner[CanonicalMember](
        elements=canonical_members(canonical_groups),
        output_length=len(canonical_groups),
    )
    compiled = planner.compile(_RepresentativeUnitToGroupRule())
    return cast(ReductionPlan, compiled)


class _RepresentativeUnitToGroupRule:
    def __init__(self) -> None:
        self.extractor = CanonicalMemberTensorExtractor()
        self.reduction_mapper = UnitWeightReductionMapper(WeightOperationType.COUNT)

    def emit(
        self,
        element: CanonicalMember,
        builder: ReductionPlanBuilder,
    ) -> Iterable[ReductionRecord]:
        source_ref = self.extractor.bind(element)
        if source_ref is None:
            return ()

        reduction = self.reduction_mapper(element)
        op = ReductionOp(source_ref, reduction)
        source_indices = _representative_source_indices(element, op)
        if len(source_indices) == 0:
            return ()

        return (
            ReductionRecord(
                op=op,
                mapping=ReductionMapping(
                    source=IndexSelection(source_indices),
                    target=IndexSelection((element.group_id,) * len(source_indices)),
                ),
            ),
        )


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


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.GROUP_CHANGE_EFFECT,
    cache_constant=True,
    create=lambda ctx, deps: create_group_change_effect_calc(ctx.canonical_groups),
)
