from __future__ import annotations

from typing import Generic, Hashable, Iterable, Protocol, TypeAlias, TypeVar, cast

import torch
import torch.nn as nn

from torch_weighttracker.extractors.extractor import TensorSpec
from torch_weighttracker.reductions.builder import (
    ComputationPlan,
    PipelinePlan,
    ReductionMapping,
    ReductionPlan,
    ReductionPlanBuilder,
    ReductionRecord,
)
from torch_weighttracker.reductions.ops import ReductionOp, SourceSpec


Element = TypeVar("Element")
TensorValue: TypeAlias = torch.Tensor | tuple[torch.Tensor, ...]


class TensorSourceRef(Protocol):
    def get(self) -> TensorValue:
        ...

    def source_spec(self) -> SourceSpec:
        ...

    def identity_key(self) -> Hashable:
        ...


class ReductionRule(Protocol[Element]):
    def emit(
        self,
        element: Element,
        builder: ReductionPlanBuilder,
    ) -> Iterable[ReductionRecord]:
        ...


class MappingStrategy(Protocol[Element]):
    def map(
        self,
        element: Element,
        op: ReductionOp,
        builder: ReductionPlanBuilder,
    ) -> ReductionMapping:
        ...


class GenericReductionPlanner(Generic[Element]):
    def __init__(
        self,
        elements: Iterable[Element],
        *,
        output_length: int | None = None,
    ) -> None:
        self.elements = tuple(elements)
        self.output_length = output_length

    def compile(
        self,
        rule: ReductionRule[Element],
        input_spec: TensorSpec | None = None,
    ) -> ComputationPlan:
        builder = ReductionPlanBuilder(output_length=self.output_length)

        for element in self.elements:
            for record in rule.emit(element, builder):
                builder.add(record)

        plan = builder.finalize(input_spec)
        return plan


def create_module_plan(
    modules: list[nn.Module],
    rule: ReductionRule[nn.Module],
    *,
    output_length: int | None = None,
) -> ReductionPlan:
    planner = GenericReductionPlanner[nn.Module](
        elements=modules,
        output_length=output_length,
    )

    compiled = planner.compile(rule=rule)

    return cast(ReductionPlan, compiled)
