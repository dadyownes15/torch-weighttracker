from torch_structracker.calculations import CalculationType
from torch_structracker.trackers.base import BaseTracker, TrackerType


class StructuredSparsityTracker(BaseTracker):
    tracker_type = TrackerType.STRUCTURED_SPARSITY
    required_calculations = (
        CalculationType.STRUCTURED_UNIT_NORM,
        CalculationType.STRUCTURED_UNIT_COUNT_FROM_NORM,
    )

    def _compute(self, calculations):
        raise NotImplementedError("StructuredSparsityTracker is not implemented yet.")

    def toMetric(self, result):
        raise NotImplementedError("StructuredSparsityTracker metrics are not implemented yet.")
