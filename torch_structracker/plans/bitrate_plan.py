from typing import Iterable

import torch.nn as nn

from torch_structracker.extractors.codeq_bitrate_extractor import ModuleBitrateExtractor
from torch_structracker.reductions.builder import (
    FullSelection,
    MappedReductionPlan,
    ReductionMapping,
    ReductionPlanBuilder,
    ReductionRecord,
)
from torch_structracker.reductions.compiler import create_module_plan
from torch_structracker.reductions.ops import IdentityTensorReduction, ReductionOp


class CodeQBitrateRule:
    def __init__(self, extractor: ModuleBitrateExtractor | None = None) -> None:
        self.extractor = ModuleBitrateExtractor() if extractor is None else extractor
        self.reduction = IdentityTensorReduction()

    def emit(
        self,
        element: nn.Module,
        builder: ReductionPlanBuilder,
    ) -> Iterable[ReductionRecord]:
        bitrate_tensor = self.extractor.bind(element)
        if bitrate_tensor is None:
            return ()

        op = ReductionOp(
            source_ref=bitrate_tensor,
            reduction=self.reduction,
        )
        target = builder.reserve_segment(op.output_length)

        return (
            ReductionRecord(
                op=op,
                mapping=ReductionMapping(
                    source=FullSelection(),
                    target=target,
                ),
            ),
        )


def create_codeq_bitrates(modules: list[nn.Module]) -> MappedReductionPlan:
    return 
