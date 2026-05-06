from torch_structracker.calculations import CalculationType
from torch_structracker.trackers.base import BaseTracker, TrackerType


class ParameterSumTracker(BaseTracker):
    tracker_type = TrackerType.PARAMETER_SUM
    required_calculations = (CalculationType.STRUCTURED_UNIT_SUM,)

    def _compute(self, calculations):
        return calculations[CalculationType.STRUCTURED_UNIT_SUM]()

    def toMetric(self, result):
        return {
            "parameter_sum": result.sum().item(),
            "structured_unit_sum": result,
        }
