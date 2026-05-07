from abc import ABC, abstractmethod
from enum import Enum

import torch

from torch_structracker.calculations import CalculationType


class TrackerType(str, Enum):
    PARAMETER_SUM = "parameter_sum"
    STRUCTURED_SPARSITY = "structured_sparsity"
    BOBS_TRACKER = "bobs_tracker"


class BaseTracker(ABC):
    required_calculations: tuple[CalculationType, ...] = ()

    def __init__(self, calculations=None) -> None:
        self.calculations = {} if calculations is None else calculations

    @abstractmethod
    def compute(self, ):
        raise NotImplementedError

    @abstractmethod
    def toMetric(self, result):
        raise NotImplementedError

    def track(self):
        return self.toMetric(self.compute())


def tracker_class_for_type(tracker_type: TrackerType):
    from torch_structracker.trackers.bobs_tracker import BobsTracker
    from torch_structracker.trackers.parameter_sum import ParameterSumTracker
    from torch_structracker.trackers.structured_sparsity import (
        StructuredSparsityTracker,
    )

    tracker_type = TrackerType(tracker_type)
    tracker_classes = {
        TrackerType.PARAMETER_SUM: ParameterSumTracker,
        TrackerType.STRUCTURED_SPARSITY: StructuredSparsityTracker,
        TrackerType.BOBS_TRACKER: BobsTracker,
    }
    return tracker_classes[tracker_type]
