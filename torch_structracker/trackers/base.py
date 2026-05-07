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
    tracker_type = TrackerType(tracker_type)

    if tracker_type == TrackerType.PARAMETER_SUM:
        from torch_structracker.trackers.parameter_sum import ParameterSumTracker

        return ParameterSumTracker

    raise ValueError(f"Tracker type is not registered yet: {tracker_type}")
