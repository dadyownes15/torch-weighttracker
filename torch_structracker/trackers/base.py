from abc import ABC, abstractmethod
from enum import Enum

import torch

from torch_structracker.calculations import CalcType


class TrackerType(str, Enum):
    STRUCTURED_BOPS = "structured_bops"


class BaseTracker(ABC):
    required_calculations: tuple[CalcType, ...] = ()

    def __init__(self, calculations=None) -> None:
        self.calculations = {} if calculations is None else calculations

    @abstractmethod
    def compute(
        self,
    ):
        raise NotImplementedError

    @abstractmethod
    def toMetric(self, result):
        raise NotImplementedError

    def track(self):
        with torch.no_grad():
            return self.toMetric(self.compute())
