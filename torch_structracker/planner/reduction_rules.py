from collections.abc import Iterable
from typing import Generic, Protocol, TypeVar

from torch_structracker.reducer_plan import ReductionPlanBuilder, ReductionRecord


Element = TypeVar("Element")


class ReductionRule(Protocol[Element]):
    def emit(
        self,
        element: Element,
        builder: ReductionPlanBuilder,
    ) -> Iterable[ReductionRecord]:
        ...


class ModuleWeightReductionRule:
    def __init__(
        self,
        *,
        operation_type,
        parameter_name: str = "weight",
        include_module=None,
    ):
        self.operation_type = operation_type
        self.parameter_name = parameter_name
        self.include_module = include_module

    def emit(self, element, builder: ReductionPlanBuilder):
        name, module = element

        if self.include_module is not None and not self.include_module(name, module):
            return

        if not hasattr(module, self.parameter_name):
            return

        parameter = getattr(module, self.parameter_name)
        if parameter is None:
            return

        op = ModuleWeightOp(
            module=module,
            parameter_name=self.parameter_name,
            operation_type=self.operation_type,
        )

        with torch.no_grad():
            value_count = int(op().numel())

        target = builder.reserve_segment(value_count)

        yield ReductionRecord(
            op=op,
            target=target,
        )