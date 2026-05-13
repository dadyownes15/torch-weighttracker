from __future__ import annotations

from typing import Callable, Generic, Iterable, TypeVar

from torch_structracker.extractors.extractor import (
    ElementTensorExtractor,
    TensorSourceRef,
)
from torch_structracker.reductions.builder import ReductionPlanBuilder, ReductionRecord
from torch_structracker.reductions.compiler import MappingStrategy
from torch_structracker.reductions.ops import ReductionOp, TensorReduction


Element = TypeVar("Element")
TensorReductionMapper = Callable[[Element], TensorReduction]


class ElementReductionRule(Generic[Element]):
    """
    Creates reduction records for elements with element-bound tensor sources.
    """

    def __init__(
        self,
        *,
        extractor: ElementTensorExtractor[Element],
        reduction_mapper: TensorReductionMapper[Element],
        mapping_strategy: MappingStrategy[Element],
    ) -> None:
        self.extractor = extractor
        self.reduction_mapper = reduction_mapper
        self.mapping_strategy = mapping_strategy

    def emit(
        self,
        element: Element,
        builder: ReductionPlanBuilder,
    ) -> Iterable[ReductionRecord]:
        source_ref = self.extractor.bind(element)
        if source_ref is None:
            return

        reduction = self.reduction_mapper(element)
        op = ReductionOp(source_ref, reduction)
        mapping = self.mapping_strategy.map(element, op, builder)

        yield ReductionRecord(op=op, mapping=mapping)


class PipelineReductionRule(Generic[Element]):
    """
    Creates reduction records for elements that all read from one input source.
    """

    def __init__(
        self,
        *,
        input: TensorSourceRef,
        reduction_mapper: TensorReductionMapper[Element],
        mapping_strategy: MappingStrategy[Element],
    ) -> None:
        self.input = input
        self.reduction_mapper = reduction_mapper
        self.mapping_strategy = mapping_strategy

    def emit(
        self,
        element: Element,
        builder: ReductionPlanBuilder,
    ) -> Iterable[ReductionRecord]:
        reduction = self.reduction_mapper(element)
        op = ReductionOp(self.input, reduction)
        mapping = self.mapping_strategy.map(element, op, builder)

        yield ReductionRecord(op=op, mapping=mapping)
