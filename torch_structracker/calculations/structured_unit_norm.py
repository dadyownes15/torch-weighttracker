from torch_structracker.calculations.base import BaseCalculation, CalculationType


class StructuredUnitNorm(BaseCalculation):
    calculation_type = CalculationType.STRUCTURED_UNIT_NORM

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("StructuredUnitNorm is not implemented yet.")
