from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import cast

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import group_input_spec
from torch_weighttracker.calculations.pipeline_calc import PipelineCalc
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.canonical_units import CanonicalMember, CanonicalUnitGroup, canonical_members
from torch_weighttracker.extractors.extractor import SourceSpec, TensorSpec, TensorValue, ValueTensorRef
from torch_weighttracker.calculations.calcs.unit_delta_to_module_axis import (
    axis_multiplier_for_member,
)
from torch_weighttracker.plans.mapping_plan import module_axis_for_member
from torch_weighttracker.reductions.builder import (
    IndexSelection,
    PipelinePlan,
    ReductionMapping,
    ReductionPlanBuilder,
    ReductionRecord,
)
from torch_weighttracker.reductions.ops import ReductionOp, numel

GroupUnitParamChangeKey = tuple[int, int, int, int, int, float]


def create_group_unit_param_change_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    input_spec: TensorSpec,
) -> PipelineCalc:
    return PipelineCalc(
        create_group_unit_param_change_plan(groups, input_spec=input_spec),
        calculation_type=CalcType.GROUP_UNIT_PARAM_CHANGE,
    )


def create_group_unit_param_change_plan(
    groups: Iterable[CanonicalUnitGroup],
    *,
    input_spec: TensorSpec,
) -> PipelinePlan:
    canonical_groups = tuple(groups)
    input_ref = ValueTensorRef(
        value=torch.empty(
            input_spec.shape,
            dtype=input_spec.dtype,
            device=input_spec.device,
        ),
        spec=input_spec,
    )
    builder = ReductionPlanBuilder(output_length=len(canonical_groups))
    axis_group_ids = _module_axis_group_ids(canonical_groups)
    seen: set[GroupUnitParamChangeKey] = set()
    record_count = 0

    for target_group in canonical_groups:
        for member in target_group.members:
            if not _has_opposite_axis(member):
                continue

            target_group_id = int(target_group.group_id)
            target_axis = module_axis_for_member(member)
            source_axis = 1 - target_axis
            edge_scale = member_axis_scale(member, target_axis=target_axis)

            for source_group_id in axis_group_ids.get((member.module, source_axis), ()):
                source_group_id = int(source_group_id)
                if source_group_id == target_group_id:
                    continue

                key = (
                    target_group_id,
                    source_group_id,
                    id(member.module),
                    target_axis,
                    source_axis,
                    edge_scale,
                )
                if key in seen:
                    continue
                seen.add(key)

                builder.add(
                    ReductionRecord(
                        op=ReductionOp(input_ref, ScaleTensorReduction(edge_scale)),
                        mapping=ReductionMapping(
                            source=IndexSelection((source_group_id,)),
                            target=IndexSelection((target_group_id,)),
                        ),
                    )
                )
                record_count += 1

    if record_count == 0:
        builder.add(
            ReductionRecord(
                op=ReductionOp(input_ref, ScaleTensorReduction(0.0)),
                mapping=ReductionMapping(
                    source=IndexSelection(()),
                    target=IndexSelection(()),
                ),
            )
        )

    return cast(PipelinePlan, builder.finalize(input_ref.source_spec()))


class ScaleTensorReduction:
    def __init__(self, scale: float) -> None:
        self.scale = float(scale)

    def __call__(self, value: TensorValue) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("ScaleTensorReduction expects one tensor.")
        return value.reshape(-1) * self.scale

    def output_spec(self, source_spec: SourceSpec) -> TensorSpec:
        if not isinstance(source_spec, TensorSpec):
            raise TypeError("ScaleTensorReduction expects one source tensor.")

        return TensorSpec(
            shape=torch.Size([numel(source_spec.shape)]),
            dtype=source_spec.dtype,
            device=source_spec.device,
        )

    def identity_key(self) -> Hashable:
        return ("scale", self.scale)


def _module_axis_group_ids(
    groups: Iterable[CanonicalUnitGroup],
) -> dict[tuple[nn.Module, int], tuple[int, ...]]:
    result: dict[tuple[nn.Module, int], set[int]] = {}

    for member in canonical_members(groups):
        if not _has_opposite_axis(member):
            continue
        axis = module_axis_for_member(member)
        result.setdefault((member.module, axis), set()).add(int(member.group_id))

    return {key: tuple(sorted(group_ids)) for key, group_ids in result.items()}


def _has_opposite_axis(member: CanonicalMember) -> bool:
    return isinstance(member.module, (nn.Linear, nn.Conv2d))


def member_axis_scale(member: CanonicalMember, *, target_axis: int) -> float:
    scale = float(axis_multiplier_for_member(member, axis=target_axis))
    module = member.module

    if isinstance(module, nn.Conv2d):
        if module.groups != 1:
            raise ValueError("GROUP_UNIT_PARAM_CHANGE supports only Conv2d(groups=1).")
        kernel_h, kernel_w = module.kernel_size
        return scale * float(kernel_h * kernel_w)

    if isinstance(module, nn.Linear):
        return scale

    raise ValueError(
        "GROUP_UNIT_PARAM_CHANGE is not implemented for "
        f"{module.__class__.__name__}."
    )


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.GROUP_UNIT_PARAM_CHANGE,
    create=lambda ctx, deps: create_group_unit_param_change_calc(
        ctx.canonical_groups,
        input_spec=group_input_spec(ctx),
    ),
)
