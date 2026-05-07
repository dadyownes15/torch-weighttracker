import torch

from torch_structracker.calculations import CalculationType
from torch_structracker.trackers.base import BaseTracker


class ParameterSumTracker(BaseTracker):
    required_calculations = (CalculationType.STRUCTURED_UNIT_SUM)

    @torch.no_grad()
    def compute(self):
        calculation = self.calculations[CalculationType.STRUCTURED_UNIT_SUM]
        return calculation()

    def toMetric(self, result):
        return {
            "structured_unit_sum": result,
            "parameter_sum": float(result.sum().item()),
        }
