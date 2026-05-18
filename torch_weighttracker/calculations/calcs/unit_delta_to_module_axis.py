from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import cast

import torch
import torch.nn as nn

from torch_weighttracker.calculations.base import CalcType
from torch_weighttracker.calculations.context import (
    calculation_device,
    calculation_dtype,
)
from torch_weighttracker.calculations.pipeline_calc import PipelineCalc
from torch_weighttracker.calculations.spec import CalculationSpec
from torch_weighttracker.canonical_units import (
    CanonicalMember,
    CanonicalUnitGroup,
    SourceLayout,
    UnitAxis,
    canonical_members,
)
from torch_weighttracker.extractors.extractor import TensorSpec
from torch_weighttracker.plans.mapping_plan import create_unit_input_ref
from torch_weighttracker.plans.module_axis_plan import module_axis_for_member
from torch_weighttracker.reductions.builder import (
    IndexSelection,
    PipelinePlan,
    ReductionMapping,
    ReductionPlanBuilder,
    ReductionRecord,
)
from torch_weighttracker.reductions.ops import ReductionOp, SourceSpec, numel


def create_unit_delta_to_module_axis_calc(
    groups: Iterable[CanonicalUnitGroup],
    *,
    weighted_module_index: Mapping[nn.Module, int],
    device: torch.device | str,
    dtype: torch.dtype,
) -> PipelineCalc:
    """
    Returns the active-unit delta for each weighted-module axis.

    Output: 1D tensor with length `2 * len(weighted_modules)`, ordered as
    `(input_axis, output_axis)` for each weighted module.
    Input: 1D active-unit tensor with length equal to the total canonical unit
    count. Element `i` is the active value for canonical unit `i`.
    """
    plan = _build_unit_delta_to_module_axis_plan(
        groups,
        weighted_module_index=weighted_module_index,
        device=device,
        dtype=dtype,
    )
    return PipelineCalc(plan, calculation_type=CalcType.UNIT_DELTA_TO_MODULE_AXIS)


class ActiveUnitAxisDeltaReduction:
    def __init__(self, multiplier: float) -> None:
        self.multiplier = float(multiplier)

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return (value.reshape(-1) - 1.0) * self.multiplier

    def output_spec(self, source_spec: SourceSpec) -> TensorSpec:
        if not isinstance(source_spec, TensorSpec):
            raise TypeError("ActiveUnitAxisDeltaReduction expects one source tensor.")

        return TensorSpec(
            shape=torch.Size([numel(source_spec.shape)]),
            dtype=source_spec.dtype,
            device=source_spec.device,
        )

    def identity_key(self):
        return ("active_unit_axis_delta", self.multiplier)


def _build_unit_delta_to_module_axis_plan(
    groups: Iterable[CanonicalUnitGroup],
    *,
    weighted_module_index: Mapping[nn.Module, int],
    device: torch.device | str,
    dtype: torch.dtype,
) -> PipelinePlan:
    canonical_groups = tuple(groups)
    input_ref = create_unit_input_ref(canonical_groups, device=device, dtype=dtype)
    builder = ReductionPlanBuilder(output_length=len(weighted_module_index) * 2)
    seen: set[tuple[int, int, int]] = set()
    record_count = 0

    for member in canonical_members(canonical_groups):
        module_index = weighted_module_index.get(member.module)
        if module_index is None:
            continue

        axis = module_axis_for_member(member)
        source_indices: list[int] = []
        for unit_index in member.unit_indices:
            key = (module_index, axis, int(unit_index))
            if key in seen:
                continue
            seen.add(key)
            source_indices.append(int(unit_index))

        if not source_indices:
            continue

        op = ReductionOp(
            input_ref,
            ActiveUnitAxisDeltaReduction(
                axis_multiplier_for_member(member, axis=axis),
            ),
        )
        target = module_index * 2 + axis
        builder.add(
            ReductionRecord(
                op=op,
                mapping=ReductionMapping(
                    source=IndexSelection(tuple(source_indices)),
                    target=IndexSelection((target,) * len(source_indices)),
                ),
            )
        )
        record_count += 1

    if record_count == 0:
        op = ReductionOp(input_ref, ActiveUnitAxisDeltaReduction(1.0))
        builder.add(
            ReductionRecord(
                op=op,
                mapping=ReductionMapping(
                    source=IndexSelection(()),
                    target=IndexSelection(()),
                ),
            )
        )

    return cast(PipelinePlan, builder.finalize(input_ref.source_spec()))


def axis_multiplier_for_member(member: CanonicalMember, *, axis: int) -> float:
    multiplier = units_to_embed_multiplier_for_member(member)

    if member.source_layout == SourceLayout.FUSED_QKV and axis == 1:
        return 3.0 * multiplier

    return multiplier


def units_to_embed_multiplier_for_member(member: CanonicalMember) -> float:
    if member.unit_axis == UnitAxis.QKV_CHANNEL:
        return 1.0

    if member.unit_axis == UnitAxis.QKV_HEAD:
        if member.head_dim is None:
            raise ValueError("QKV_HEAD members require head_dim metadata.")
        return float(member.head_dim)

    if member.unit_axis == UnitAxis.QKV_HEAD_DIM:
        if member.num_heads is None:
            raise ValueError("QKV_HEAD_DIM members require num_heads metadata.")
        return float(member.num_heads)

    if member.num_heads is not None and member.head_dim is not None:
        if member.group_length == member.num_heads:
            return float(member.head_dim)
        if member.group_length == member.head_dim:
            return float(member.num_heads)
        if member.embed_dim is not None and member.group_length == member.embed_dim:
            return 1.0

    return 1.0


CALCULATION_SPEC = CalculationSpec(
    calculation_type=CalcType.UNIT_DELTA_TO_MODULE_AXIS,
    create=lambda ctx, deps: create_unit_delta_to_module_axis_calc(
        ctx.canonical_groups,
        weighted_module_index=ctx.weighted_module_index,
        device=calculation_device(ctx),
        dtype=calculation_dtype(ctx),
    ),
)
